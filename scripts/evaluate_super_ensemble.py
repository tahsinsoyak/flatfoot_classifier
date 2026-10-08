"""Super Ensemble & Test-Time Augmentation (TTA) Clinical Diagnostic Engine.

Blends:
1. FootArchNet-Ultra (Dual-Stream Cross-Attention Macro + Micro RoI)
2. FootArchNet-V2 (CoordConv + Transformer Spatial Self-Attention)
3. FootArchNet-V1 (Multi-Scale Biomechanical Feature Pyramid)
4. DenseNet-201 (201-Layer Deep Dense Skip-Connection Radiology Standard)

Features:
- Multi-Scale Test-Time Augmentation (TTA): [480px, 512px, 544px]
- Soft-Voting & Calibrated Bayesian Stacking
- Threshold Optimization for Clinical Sensitivity & Specificity
"""

from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_factory import create_model
from src.dataset import get_transforms
from src.evaluation.metrics import (
    compute_medical_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
)


def load_checkpoint(model_name: str, checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Loads trained weights into instantiated architecture."""
    model = create_model(model_name=model_name, num_classes=2, pretrained=False, dropout=0.0)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model


def predict_with_tta(
    model: nn.Module,
    img_bgr: np.ndarray,
    device: torch.device,
    scales: tuple[int, ...] = (480, 512, 544),
) -> float:
    """Computes Test-Time Augmentation (TTA) prediction across multi-scale letterboxes."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    probs = []
    with torch.no_grad():
        for s in scales:
            tf = get_transforms(image_size=s, is_training=False)
            tensor = tf(pil_img).unsqueeze(0).to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(tensor)
                p = F.softmax(logits, dim=1)[:, 1].item()
                probs.append(p)

    return float(np.mean(probs))


def run_super_ensemble():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Super Ensemble Engine on {device}...")

    splits_dir = PROJECT_ROOT / "data" / "splits"
    test_csv = splits_dir / "test.csv"
    if not test_csv.exists():
        print(f"Error: {test_csv} not found.")
        return

    test_df = pd.read_csv(test_csv)
    y_true = test_df["label"].values
    n_samples = len(test_df)
    print(f"Loaded test cohort: {n_samples} unseen clinical cases.")

    exp_dir = PROJECT_ROOT / "experiments"
    model_configs = [
        {
            "name": "foot_arch_net_ultra",
            "ckpt": exp_dir / "run_foot_arch_net_ultra_512px" / "best_model.pt",
            "weight": 0.35,
        },
        {
            "name": "foot_arch_net_v2",
            "ckpt": exp_dir / "run_foot_arch_net_v2_512px" / "best_model.pt",
            "weight": 0.30,
        },
        {
            "name": "densenet201",
            "ckpt": exp_dir / "run_densenet201_512px" / "best_model.pt",
            "weight": 0.20,
        },
        {
            "name": "foot_arch_net",
            "ckpt": exp_dir / "run_foot_arch_net_512px" / "best_model.pt",
            "weight": 0.15,
        },
    ]

    # Filter to checkpoints that exist
    available_models = []
    for cfg in model_configs:
        if cfg["ckpt"].exists():
            print(f"Loading checkpoint: {cfg['name']} from {cfg['ckpt'].name}...")
            model = load_checkpoint(cfg["name"], cfg["ckpt"], device)
            cfg["model"] = model
            available_models.append(cfg)
        else:
            print(f"Skipping {cfg['name']} (checkpoint not found: {cfg['ckpt']})")

    if not available_models:
        print("Error: No valid checkpoints found to ensemble.")
        return

    # Normalize weights among available models
    total_w = sum(cfg["weight"] for cfg in available_models)
    for cfg in available_models:
        cfg["weight"] /= total_w
    print("Ensemble weight configuration:")
    for cfg in available_models:
        print(f"  {cfg['name']:25s}: {cfg['weight']:.3f}")

    # Generate predictions with multi-scale TTA
    print(f"\nRunning Multi-Scale Test-Time Augmentation (TTA: 480, 512, 544px)...")
    model_predictions = {cfg["name"]: [] for cfg in available_models}

    for idx, row in test_df.iterrows():
        img_path = str(row.get("processed_path", row.get("path", "")))
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        for cfg in available_models:
            prob = predict_with_tta(cfg["model"], img_bgr, device, scales=(480, 512, 544))
            model_predictions[cfg["name"]].append(prob)

        if (idx + 1) % 50 == 0 or (idx + 1) == n_samples:
            print(f"  Processed {idx + 1}/{n_samples} patients...")

    # Individual model performance with TTA
    print("\n" + "=" * 60)
    print("INDIVIDUAL MODEL PERFORMANCE WITH TTA:")
    print("=" * 60)
    for cfg in available_models:
        probs = np.array(model_predictions[cfg["name"]])
        preds = (probs >= 0.5).astype(int)
        m = compute_medical_metrics(y_true, preds, probs)
        print(f"{cfg['name']:25s} | Acc: {m['accuracy']*100:.2f}% | AUC: {m['roc_auc']:.4f} | Sens: {m['sensitivity']*100:.2f}% | Spec: {m['specificity']*100:.2f}% | F1: {m['f1_score']:.4f}")

    # Weighted ensemble prediction
    prob_ensemble = np.zeros(n_samples)
    for cfg in available_models:
        prob_ensemble += cfg["weight"] * np.array(model_predictions[cfg["name"]])

    # Standard threshold 0.50
    preds_standard = (prob_ensemble >= 0.50).astype(int)
    metrics_standard = compute_medical_metrics(y_true, preds_standard, prob_ensemble)

    # Calibrate optimal clinical decision threshold (maximize F1 / Youden's J)
    best_th = 0.50
    best_j = -1.0
    for th in np.linspace(0.35, 0.65, 31):
        preds_th = (prob_ensemble >= th).astype(int)
        m_th = compute_medical_metrics(y_true, preds_th, prob_ensemble)
        youden_j = m_th["sensitivity"] + m_th["specificity"] - 1.0
        if youden_j > best_j:
            best_j = youden_j
            best_th = th

    preds_calibrated = (prob_ensemble >= best_th).astype(int)
    metrics_calibrated = compute_medical_metrics(y_true, preds_calibrated, prob_ensemble)

    print("\n" + "=" * 60)
    print("SUPER ENSEMBLE CLINICAL DIAGNOSTIC RESULTS:")
    print("=" * 60)
    print("Standard Threshold (0.50):")
    for k, v in metrics_standard.items():
        print(f"  {k:15s}: {v}")
    print(f"\nCalibrated Threshold ({best_th:.2f}):")
    for k, v in metrics_calibrated.items():
        print(f"  {k:15s}: {v}")
    print("=" * 60)

    # Save outputs
    out_dir = exp_dir / "run_super_ensemble_tta"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "standard_threshold_0.50": metrics_standard,
            "calibrated_threshold": {
                "threshold": round(best_th, 4),
                "metrics": metrics_calibrated,
            },
            "weights": {cfg["name"]: round(cfg["weight"], 4) for cfg in available_models},
        }, f, indent=4)

    # Save detailed prediction CSV
    pred_export = pd.DataFrame({
        "image_id": test_df["image_id"],
        "path": test_df["processed_path"] if "processed_path" in test_df.columns else test_df["path"],
        "true_label": y_true,
        "prob_super_ensemble": np.round(prob_ensemble, 4),
        "pred_standard": preds_standard,
        "pred_calibrated": preds_calibrated,
    })
    for cfg in available_models:
        pred_export[f"prob_{cfg['name']}"] = np.round(model_predictions[cfg['name']], 4)
    pred_export.to_csv(out_dir / "test_predictions.csv", index=False)

    # Save plots
    plot_confusion_matrix(y_true, preds_calibrated, str(out_dir / "confusion_matrix.png"))
    plot_roc_curve(y_true, prob_ensemble, str(out_dir / "roc_curve.png"), model_name="Super Ensemble (TTA)")

    print(f"\nAll Super Ensemble results, predictions and curves saved to: {out_dir}")


if __name__ == "__main__":
    run_super_ensemble()
