"""Train a deep learning model for flatfoot classification and evaluate on Test set."""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import FootRadiographDataset, get_transforms
from src.models.model_factory import create_model
from src.training.trainer import Trainer
from src.evaluation.metrics import (
    compute_medical_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
)
from src.models.gradcam import GradCAM


def get_target_layer_for_gradcam(model_name: str, model: nn.Module) -> nn.Module:
    model_name = model_name.lower().replace("-", "_")
    if "resnet" in model_name:
        return model.layer4[-1]
    elif "efficientnet" in model_name:
        return model.features[-1]
    elif "convnext" in model_name:
        return model.features[-1]
    elif "densenet" in model_name:
        return model.features.denseblock4
    else:
        # Default fallback to last module before fc
        return list(model.children())[-2]


def main():
    parser = argparse.ArgumentParser(description="Train Flatfoot Classifier")
    parser.add_argument("--model", type=str, default="resnet50", help="Model backbone name")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=512, help="Image resolution")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda or cpu or auto)")
    parser.add_argument("--num-workers", type=int, default=2, help="Data loader workers")
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Paths
    splits_dir = PROJECT_ROOT / "data" / "splits"
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    if not train_csv.exists():
        print(f"Error: {train_csv} not found. Please run scripts/split_dataset.py first.")
        sys.exit(1)

    exp_dir = PROJECT_ROOT / "experiments" / f"run_{args.model}_{args.img_size}px"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Transforms & Datasets
    train_tf = get_transforms(image_size=args.img_size, is_training=True)
    eval_tf = get_transforms(image_size=args.img_size, is_training=False)

    train_ds = FootRadiographDataset(train_csv, transform=train_tf)
    val_ds = FootRadiographDataset(val_csv, transform=eval_tf)
    test_ds = FootRadiographDataset(test_csv, transform=eval_tf, return_meta=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Class weights for loss
    train_df = pd.read_csv(train_csv)
    n_pes = int((train_df["label"] == 1).sum())
    n_norm = int((train_df["label"] == 0).sum())
    total_train = n_pes + n_norm
    # weight = total / (num_classes * count)
    weight_norm = total_train / (2.0 * n_norm)
    weight_pes = total_train / (2.0 * n_pes)
    class_weights = torch.tensor([weight_norm, weight_pes], dtype=torch.float32).to(device)
    print(f"Class counts - Normal: {n_norm}, Pes Planus: {n_pes}")
    print(f"Calculated loss weights - Normal: {weight_norm:.3f}, Pes Planus: {weight_pes:.3f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Create model
    print(f"Building model: {args.model}...")
    model = create_model(
        model_name=args.model,
        num_classes=2,
        pretrained=True,
        dropout=args.dropout,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=exp_dir,
        use_amp=(device.type == "cuda"),
        early_stopping_patience=8,
    )

    # Fit
    train_result = trainer.fit(num_epochs=args.epochs)

    # Load best model for evaluation on Test set
    best_path = exp_dir / "best_model.pt"
    print(f"\nLoading best checkpoint from {best_path} for final Test Set evaluation...")
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Test set inference
    test_loss = 0.0
    y_true_list, y_pred_list, y_prob_list = [], [], []
    meta_list = []

    with torch.no_grad():
        for images, labels, metas in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            y_true_list.extend(labels.cpu().numpy())
            y_pred_list.extend(preds)
            y_prob_list.extend(probs)

            for i in range(len(labels)):
                meta_list.append({
                    "image_id": metas["image_id"][i],
                    "class_name": metas["class_name"][i],
                    "true_label": labels[i].item(),
                    "pred_label": preds[i],
                    "prob_pes_planus": round(float(probs[i]), 4),
                    "path": metas["path"][i],
                })

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_prob = np.array(y_prob_list)

    test_metrics = compute_medical_metrics(y_true, y_pred, y_prob)
    test_metrics["test_loss"] = round(test_loss / len(test_ds), 4)

    # Save metrics JSON
    with open(exp_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4)

    # Save prediction CSV
    pred_df = pd.DataFrame(meta_list)
    pred_df.to_csv(exp_dir / "test_predictions.csv", index=False)

    # Save Plots
    plot_confusion_matrix(y_true, y_pred, str(exp_dir / "confusion_matrix.png"))
    plot_roc_curve(y_true, y_prob, str(exp_dir / "roc_curve.png"), model_name=args.model)

    print("\n" + "=" * 50)
    print("FINAL TEST SET DIAGNOSTIC PERFORMANCE:")
    print("=" * 50)
    for k, v in test_metrics.items():
        print(f"  {k:15s}: {v}")
    print("=" * 50)

    # Explainability: Grad-CAM generation
    print("\nGenerating Grad-CAM anatomical saliency maps for representative test samples...")
    try:
        target_layer = get_target_layer_for_gradcam(args.model, model)
        gradcam = GradCAM(model, target_layer)

        # Select 3 Normal and 3 Pes Planus
        norm_samples = pred_df[pred_df["true_label"] == 0].head(3)
        pes_samples = pred_df[pred_df["true_label"] == 1].head(3)
        cam_samples = pd.concat([norm_samples, pes_samples])

        fig, axes = plt.subplots(len(cam_samples), 2, figsize=(10, 3 * len(cam_samples)))
        fig.suptitle(f"Grad-CAM Anatomical Saliency Maps ({args.model})", fontsize=14, y=0.99)

        for i, (_, row) in enumerate(cam_samples.iterrows()):
            img_path = row["path"]
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (args.img_size, args.img_size))

            # Transform for model
            tensor = eval_tf(from_cv2_to_pil(img_rgb)).unsqueeze(0).to(device)

            cam = gradcam.generate(tensor, target_class=row["true_label"])
            overlay = GradCAM.overlay_heatmap(img_resized, cam, alpha=0.45)

            status = "CORRECT" if row["true_label"] == row["pred_label"] else "MISCLASSIFIED"
            title_orig = f"True: {row['class_name']} ({row['image_id']})"
            title_cam = f"Pred: {('Pes Planus' if row['pred_label'] == 1 else 'Normal')} (p={row['prob_pes_planus']:.2f}) [{status}]"

            axes[i, 0].imshow(img_resized)
            axes[i, 0].set_title(title_orig, fontsize=9)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(overlay)
            axes[i, 1].set_title(title_cam, fontsize=9)
            axes[i, 1].axis("off")

        plt.tight_layout()
        gradcam_path = exp_dir / "gradcam_test_samples.png"
        plt.savefig(gradcam_path, dpi=300)
        plt.close()
        print(f"Saved Grad-CAM visualizations to {gradcam_path}")
    except Exception as e:
        print(f"Warning: Grad-CAM generation skipped due to: {e}")

    print(f"\nAll experiment outputs saved to: {exp_dir}")


def from_cv2_to_pil(img_rgb):
    from PIL import Image
    return Image.fromarray(img_rgb)


if __name__ == "__main__":
    main()
