"""Stratified K-Fold Training & Ensemble Evaluation for Flatfoot Classification.

Designed for high-performance training on Google Colab / High-VRAM GPUs.
Trains K models across stratified folds and ensembles them on the held-out test cohort.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import FootRadiographDataset, get_transforms
from src.models.model_factory import create_model
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Stratified K-Fold Training for Flatfoot Classifier")
    parser.add_argument("--model", type=str, default="foot_arch_net_ultra", help="Model backbone architecture")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per fold (default: 20)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--img-size", type=int, default=512, help="Image resolution (default: 512)")
    parser.add_argument("--num-workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument("--use-ema", action="store_true", default=True, help="Use Model EMA")
    parser.add_argument("--output-dir", type=str, default="", help="Directory to save experiment results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("STRATIFIED K-FOLD ENSEMBLE TRAINING PIPELINE")
    print("=" * 70)
    print(f"Device:           {device}")
    if device.type == "cuda":
        print(f"GPU Name:         {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Model:            {args.model}")
    print(f"Number of Folds:  {args.folds}")
    print(f"Epochs per Fold:  {args.epochs}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Resolution:       {args.img_size}x{args.img_size}")
    print(f"Model EMA:        {args.use_ema}")
    print("=" * 70)

    # 1. Load data
    train_df = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "train.csv")
    val_df = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "val.csv")
    test_df = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "test.csv")

    # Combine train + val into total development pool for K-fold (1071 + 229 = 1300 samples)
    dev_df = pd.concat([train_df, val_df], ignore_index=True).reset_index(drop=True)
    print(f"\nDevelopment Cohort (for {args.folds}-Fold): {len(dev_df)} samples")
    print(f"Held-Out Clinical Test Cohort: {len(test_df)} samples (strictly untouched during training)")

    out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "experiments" / f"run_kfold_{args.model}_{args.folds}f"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup K-Fold split
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    fold_models = []

    # Calculate class weights
    n_normal = (dev_df["label"] == 0).sum()
    n_flat = (dev_df["label"] == 1).sum()
    total_dev = len(dev_df)
    w0 = total_dev / (2.0 * n_normal)
    w1 = total_dev / (2.0 * n_flat)
    class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    train_tf = get_transforms(args.img_size, is_training=True)
    val_tf = get_transforms(args.img_size, is_training=False)

    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(dev_df, dev_df["label"])):
        print("\n" + "#" * 50)
        print(f"TRAINING FOLD {fold_idx + 1} / {args.folds}")
        print("#" * 50)

        fold_train_df = dev_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = dev_df.iloc[val_idx].reset_index(drop=True)

        fold_train_csv = out_dir / f"fold_{fold_idx}_train.csv"
        fold_val_csv = out_dir / f"fold_{fold_idx}_val.csv"
        fold_train_df.to_csv(fold_train_csv, index=False)
        fold_val_df.to_csv(fold_val_csv, index=False)

        train_ds = FootRadiographDataset(fold_train_csv, transform=train_tf)
        val_ds = FootRadiographDataset(fold_val_csv, transform=val_tf)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )

        model = create_model(args.model, num_classes=2, pretrained=True, dropout=0.3).to(device)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

        fold_dir = out_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=device,
            use_amp=True,
            output_dir=fold_dir,
            patience=8,
            use_ema=args.use_ema,
            ema_decay=0.999
        )

        history = trainer.fit(train_loader, val_loader, epochs=args.epochs)
        best_ckpt = fold_dir / "best_model.pt"
        fold_models.append(best_ckpt)

    print("\n" + "=" * 70)
    print("ALL FOLDS TRAINED. EVALUATING 5-FOLD ENSEMBLE ON CLINICAL TEST SET (n=229)...")
    print("=" * 70)

    # Multi-Scale Test-Time Augmentation (TTA)
    scales = [args.img_size - 32, args.img_size, args.img_size + 32]
    all_fold_test_probs = []

    for fold_idx, ckpt_path in enumerate(fold_models):
        print(f"Evaluating Fold {fold_idx + 1} checkpoint with TTA across scales {scales}...")
        ckpt = torch.load(ckpt_path, map_location=device)
        model = create_model(args.model, num_classes=2, pretrained=False, dropout=0.0).to(device)
        
        # Check if EMA weights exist
        if "ema_state_dict" in ckpt and ckpt["ema_state_dict"] is not None:
            model.load_state_dict(ckpt["ema_state_dict"])
            print("  --> Loaded EMA weights.")
        else:
            model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        fold_scale_probs = []
        for s in scales:
            s_tf = get_transforms(s, is_training=False)
            test_ds = FootRadiographDataset(PROJECT_ROOT / "data" / "splits" / "test.csv", transform=s_tf)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

            probs_s = []
            with torch.no_grad():
                for imgs, _ in test_loader:
                    imgs = imgs.to(device)
                    with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                        logits = model(imgs)
                        p = torch.softmax(logits, dim=1)[:, 1]
                    probs_s.extend(p.cpu().numpy().tolist())
            fold_scale_probs.append(np.array(probs_s))

        fold_avg_prob = np.mean(fold_scale_probs, axis=0)
        all_fold_test_probs.append(fold_avg_prob)

    # Master ensemble probability: average across all K folds
    ensemble_probs = np.mean(all_fold_test_probs, axis=0)
    y_true = test_df["label"].to_numpy()

    # Default 0.50 threshold
    preds_05 = (ensemble_probs >= 0.50).astype(int)
    acc = accuracy_score(y_true, preds_05)
    prec = precision_score(y_true, preds_05, zero_division=0)
    sens = recall_score(y_true, preds_05, zero_division=0)
    f1 = f1_score(y_true, preds_05, zero_division=0)
    auc = roc_auc_score(y_true, ensemble_probs)
    cm = confusion_matrix(y_true, preds_05)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp)
    npv = tn / (tn + fn)

    # Calibrated optimal threshold via Youden's J statistic
    from sklearn.metrics import roc_curve
    fpr_curve, tpr_curve, thresholds = roc_curve(y_true, ensemble_probs)
    j_scores = tpr_curve - fpr_curve
    opt_idx = np.argmax(j_scores)
    opt_thresh = float(thresholds[opt_idx])

    preds_opt = (ensemble_probs >= opt_thresh).astype(int)
    acc_opt = accuracy_score(y_true, preds_opt)
    sens_opt = recall_score(y_true, preds_opt, zero_division=0)
    cm_opt = confusion_matrix(y_true, preds_opt)
    tn_opt, fp_opt, fn_opt, tp_opt = cm_opt.ravel()
    spec_opt = tn_opt / (tn_opt + fp_opt)
    prec_opt = precision_score(y_true, preds_opt, zero_division=0)
    npv_opt = tn_opt / (tn_opt + fn_opt)
    f1_opt = f1_score(y_true, preds_opt, zero_division=0)

    print("\n" + "=" * 70)
    print(f"MASTER {args.folds}-FOLD ENSEMBLE TEST SET RESULTS (n={len(test_df)})")
    print("=" * 70)
    print(f"--- Standard Threshold (0.50) ---")
    print(f"Accuracy:            {acc * 100:.2f}% ({tp + tn} / {len(test_df)} correct)")
    print(f"Sensitivity (Recall): {sens * 100:.2f}% ({tp} / {tp + fn} pes planus detected)")
    print(f"Specificity:         {spec * 100:.2f}% ({tn} / {tn + fp} normal detected)")
    print(f"Precision (PPV):     {prec * 100:.2f}%")
    print(f"ROC-AUC:             {auc:.4f}")
    print(f"Confusion Matrix:    TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"\n--- Calibrated Optimal Threshold ({opt_thresh:.3f}) ---")
    print(f"Calibrated Accuracy: {acc_opt * 100:.2f}% ({tp_opt + tn_opt} / {len(test_df)} correct)")
    print(f"Sensitivity:         {sens_opt * 100:.2f}%")
    print(f"Specificity:         {spec_opt * 100:.2f}% ({tn_opt} / {tn_opt + fp_opt} normal detected)")
    print(f"Precision (PPV):     {prec_opt * 100:.2f}%")
    print(f"F1-Score:            {f1_opt:.4f}")
    print(f"Confusion Matrix:    TP={tp_opt}, TN={tn_opt}, FP={fp_opt}, FN={fn_opt}")
    print("=" * 70)

    # Save results
    metrics = {
        "model": f"{args.model}_{args.folds}f_ensemble",
        "folds": args.folds,
        "epochs": args.epochs,
        "accuracy": float(acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "npv": float(npv),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }

    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pred_df = test_df.copy()
    pred_df["ensemble_prob"] = ensemble_probs
    pred_df["ensemble_pred"] = preds_05
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    # Confusion matrix plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Pes Planus"],
                yticklabels=["Normal", "Pes Planus"])
    plt.title(f"{args.folds}-Fold Ensemble Confusion Matrix (Acc: {acc*100:.2f}%)")
    plt.ylabel("True Clinical Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    print(f"\nAll artifacts, fold weights, and metrics saved in: {out_dir}")


if __name__ == "__main__":
    main()
