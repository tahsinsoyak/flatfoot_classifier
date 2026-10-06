"""Standalone script to generate Grad-CAM interpretability visualizations for any trained run."""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import pandas as pd
from PIL import Image
import torch
import matplotlib.pyplot as plt

from src.models.model_factory import create_model
from src.models.gradcam import GradCAM
from src.dataset import get_transforms
from scripts.train_model import get_target_layer_for_gradcam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default="experiments/run_resnet18_512px")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--samples-per-class", type=int, default=3)
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / args.run_dir
    pred_path = run_dir / "test_predictions.csv"
    best_pt = run_dir / "best_model.pt"

    if not pred_path.exists() or not best_pt.exists():
        print(f"Missing predictions or checkpoint in {run_dir}")
        return

    pred_df = pd.read_csv(pred_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(args.model, num_classes=2, pretrained=False)
    chk = torch.load(best_pt, map_location=device)
    model.load_state_dict(chk["model_state_dict"])
    model.to(device).eval()

    target_layer = get_target_layer_for_gradcam(args.model, model)
    gradcam = GradCAM(model, target_layer)
    eval_tf = get_transforms(args.img_size, is_training=False)

    norm_samples = pred_df[pred_df["true_label"] == 0].head(args.samples_per_class)
    pes_samples = pred_df[pred_df["true_label"] == 1].head(args.samples_per_class)
    samples = pd.concat([norm_samples, pes_samples])

    fig, axes = plt.subplots(len(samples), 2, figsize=(10, 3.2 * len(samples)))
    fig.suptitle(f"Grad-CAM Anatomical Saliency Maps ({args.model.upper()})", fontsize=14, y=0.995)

    for i, (_, row) in enumerate(samples.iterrows()):
        img_bgr = cv2.imread(row["path"])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (args.img_size, args.img_size))

        tensor = eval_tf(Image.fromarray(img_rgb)).unsqueeze(0).to(device)
        cam = gradcam.generate(tensor, target_class=int(row["true_label"]))
        overlay = GradCAM.overlay_heatmap(img_resized, cam, alpha=0.45)

        pred_name = "Pes Planus" if row["pred_label"] == 1 else "Normal"
        status = "CORRECT" if row["true_label"] == row["pred_label"] else "MISCLASSIFIED"

        axes[i, 0].imshow(img_resized)
        axes[i, 0].set_title(f"True: {row['class_name']} ({row['image_id']})", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title(f"Pred: {pred_name} (p={row['prob_pes_planus']:.2f}) [{status}]", fontsize=9)
        axes[i, 1].axis("off")

    plt.tight_layout()
    out_path = run_dir / "gradcam_test_samples.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Successfully generated Grad-CAM to {out_path}")


if __name__ == "__main__":
    main()
