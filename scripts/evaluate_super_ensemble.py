"""Super Ensemble V2 & Multi-Scale Test-Time Augmentation (TTA) Diagnostic Engine.

Architectures:
1. FootArchNet-Ultra (Dual-Stream Cross-Attention Macro+Micro RoI)
2. DenseNet-201 (201-Layer Deep Dense Skip-Connection Radiology Standard)
3. Swin-T (Shifted-Window Hierarchical Vision Transformer)
4. FootArchNet-V2 (CoordConv + TransArchAttention Spatial Self-Attention)
5. FootArchNet-V1 (Multi-Scale Biomechanical Feature Pyramid)

Features:
- Multi-Scale Test-Time Augmentation (TTA): [480px, 512px, 544px]
- Mathematically Optimized Blending Weights via Validation-Set Stacking (Nelder-Mead)
- Zero-Leakage Threshold Calibration on Validation Set
- Medical Metrics & ROC-AUC Optimization
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
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    print(f"Executing Super Ensemble V2 Engine on {device}...")

    splits_dir = PROJECT_ROOT / "data" / "splits"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    if not test_csv.exists() or not val_csv.exists():
        print(f"Error: Dataset splits not found in {splits_dir}")
        return

    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    y_val = val_df["label"].values
    y_test = test_df["label"].values
    n_test = len(test_df)
    n_val = len(val_df)
    print(f"Loaded cohorts: Validation (n={n_val}), Test (n={n_test}).")

    exp_dir = PROJECT_ROOT / "experiments"
    model_configs = [
        {
            "name": "foot_arch_net_ultra",
            "ckpt": exp_dir / "run_foot_arch_net_ultra_512px" / "best_model.pt",
            "init_weight": 0.35,
        },
        {
            "name": "densenet201",
            "ckpt": exp_dir / "run_densenet201_512px" / "best_model.pt",
            "init_weight": 0.25,
        },
        {
            "name": "swin_t",
            "ckpt": exp_dir / "run_swin_t_512px" / "best_model.pt",
            "init_weight": 0.20,
        },
        {
            "name": "foot_arch_net_v2",
            "ckpt": exp_dir / "run_foot_arch_net_v2_512px" / "best_model.pt",
            "init_weight": 0.15,
        },
        {
            "name": "foot_arch_net",
            "ckpt": exp_dir / "run_foot_arch_net_512px" / "best_model.pt",
            "init_weight": 0.05,
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

    # 1. Validation Set Inference with TTA for Optimal Stacking Weights
    print(f"\nPhase 1: Computing Validation Set Predictions for Weight Calibration...")
    val_preds = {cfg["name"]: [] for cfg in available_models}
    for idx, row in val_df.iterrows():
        img_path = str(row.get("processed_path", row.get("path", "")))
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        for cfg in available_models:
            p = predict_with_tta(cfg["model"], img_bgr, device, scales=(480, 512, 544))
            val_preds[cfg["name"]].append(p)

    val_matrix = np.column_stack([np.array(val_preds[cfg["name"]]) for cfg in available_models])

    # Optimize weights on validation set
    def loss_func(weights):
        w = np.maximum(weights, 0)
        if w.sum() == 0:
            return 1.0
        w = w / w.sum()
        p_ens = val_matrix @ w
        # Combine negative AUC and log-loss for smooth convex calibration
        auc = roc_auc_score(y_val, p_ens)
        eps = 1e-6
        p_clip = np.clip(p_ens, eps, 1 - eps)
        log_loss = -np.mean(y_val * np.log(p_clip) + (1 - y_val) * np.log(1 - p_clip))
        return -auc * 2.0 + log_loss * 0.5

    init_w = [cfg["init_weight"] for cfg in available_models]
    opt_res = minimize(loss_func, init_w, method="Nelder-Mead")
    final_weights = np.maximum(opt_res.x, 0)
    final_weights /= final_weights.sum()

    for idx, cfg in enumerate(available_models):
        cfg["weight"] = float(final_weights[idx])

    print("\n" + "=" * 60)
    print("MATHEMATICALLY CALIBRATED ENSEMBLE WEIGHTS (from Validation Set):")
    print("=" * 60)
    for cfg in available_models:
        print(f"  {cfg['name']:25s}: {cfg['weight']:.4f}")

    # Determine optimal threshold from validation set
    val_ens_prob = val_matrix @ final_weights
    best_th = 0.50
    best_youden = -1.0
    for th in np.linspace(0.35, 0.65, 61):
        p_bin = (val_ens_prob >= th).astype(int)
        m = compute_medical_metrics(y_val, p_bin, val_ens_prob)
        youden = m["sensitivity"] + m["specificity"] - 1.0
        if youden > best_youden:
            best_youden = youden
            best_th = float(th)

    print(f"Optimal Decision Threshold (calibrated on val set): {best_th:.3f}")
    print("=" * 60)

    # 2. Test Set Inference with TTA
    print(f"\nPhase 2: Running Multi-Scale Test-Time Augmentation on 229 Unseen Clinical Test Cases...")
    test_preds = {cfg["name"]: [] for cfg in available_models}
    for idx, row in test_df.iterrows():
        img_path = str(row.get("processed_path", row.get("path", "")))
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        for cfg in available_models:
            prob = predict_with_tta(cfg["model"], img_bgr, device, scales=(480, 512, 544))
            test_preds[cfg["name"]].append(prob)

        if (idx + 1) % 50 == 0 or (idx + 1) == n_test:
            print(f"  Processed {idx + 1}/{n_test} patients...")

    test_matrix = np.column_stack([np.array(test_preds[cfg["name"]]) for cfg in available_models])

    # Individual model performance on test set
    print("\n" + "=" * 60)
    print("INDIVIDUAL MODEL PERFORMANCE WITH TTA (TEST SET):")
    print("=" * 60)
    for cfg in available_models:
        probs = np.array(test_preds[cfg["name"]])
        preds = (probs >= 0.5).astype(int)
        m = compute_medical_metrics(y_test, preds, probs)
        print(f"{cfg['name']:25s} | Acc: {m['accuracy']*100:.2f}% | AUC: {m['roc_auc']:.4f} | Sens: {m['sensitivity']*100:.2f}% | Spec: {m['specificity']*100:.2f}% | F1: {m['f1_score']:.4f}")

    # Weighted Ensemble on test set
    test_ens_prob = test_matrix @ final_weights

    # Standard threshold 0.50
    preds_standard = (test_ens_prob >= 0.50).astype(int)
    metrics_standard = compute_medical_metrics(y_test, preds_standard, test_ens_prob)

    # Calibrated threshold (from validation set)
    preds_calibrated = (test_ens_prob >= best_th).astype(int)
    metrics_calibrated = compute_medical_metrics(y_test, preds_calibrated, test_ens_prob)

    print("\n" + "=" * 60)
    print("SUPER ENSEMBLE V2 FINAL CLINICAL TEST RESULTS:")
    print("=" * 60)
    print("Standard Threshold (0.50):")
    for k, v in metrics_standard.items():
        print(f"  {k:15s}: {v}")
    print(f"\nCalibrated Threshold ({best_th:.3f}):")
    for k, v in metrics_calibrated.items():
        print(f"  {k:15s}: {v}")
    print("=" * 60)

    # Save outputs
    out_dir = exp_dir / "run_super_ensemble_tta"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "standard_threshold_0.50": metrics_standard,
            "calibrated_threshold": {
                "threshold": round(best_th, 4),
                "metrics": metrics_calibrated,
            },
            "weights": {cfg["name"]: round(cfg["weight"], 4) for cfg in available_models},
        }, f, indent=4)

    pred_export = pd.DataFrame({
        "image_id": test_df["image_id"],
        "path": test_df["processed_path"] if "processed_path" in test_df.columns else test_df["path"],
        "true_label": y_test,
        "prob_super_ensemble": np.round(test_ens_prob, 4),
        "pred_standard": preds_standard,
        "pred_calibrated": preds_calibrated,
    })
    for cfg in available_models:
        pred_export[f"prob_{cfg['name']}"] = np.round(test_preds[cfg['name']], 4)
    pred_export.to_csv(out_dir / "test_predictions.csv", index=False)

    plot_confusion_matrix(y_test, preds_calibrated, str(out_dir / "confusion_matrix.png"))
    plot_roc_curve(y_test, test_ens_prob, str(out_dir / "roc_curve.png"), model_name="Super Ensemble V2 (TTA)")

    print(f"\nAll Super Ensemble V2 results saved to: {out_dir}")


if __name__ == "__main__":
    run_super_ensemble()
