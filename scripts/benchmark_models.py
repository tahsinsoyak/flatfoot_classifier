"""Benchmark script to train and evaluate multiple standard deep learning architectures.

Compares:
- ResNet-50
- EfficientNet-B2
- ConvNeXt-Tiny

Generates:
- experiments/benchmark_summary.csv
- experiments/benchmark_summary.md
- experiments/benchmark_roc_comparison.png
"""

from __future__ import annotations
import os
import sys
import json
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    models_to_benchmark = ["resnet50", "efficientnet_b2", "convnext_tiny"]
    epochs = 20
    batch_size = 8
    img_size = 512

    results = []
    roc_data = {}

    print(f"Starting standard models benchmark: {models_to_benchmark}")
    python_exe = sys.executable

    for model_name in models_to_benchmark:
        print("\n" + "=" * 65)
        print(f"BENCHMARKING MODEL: {model_name.upper()} ({epochs} epochs, batch={batch_size})")
        print("=" * 65)

        cmd = [
            python_exe,
            str(PROJECT_ROOT / "scripts" / "train_model.py"),
            "--model", model_name,
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--img-size", str(img_size),
        ]
        res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if res.returncode != 0:
            print(f"Error during training of {model_name}")
            continue

        # Generate Grad-CAM for this model
        print(f"\nGenerating Grad-CAM for {model_name}...")
        exp_dir = PROJECT_ROOT / "experiments" / f"run_{model_name}_{img_size}px"
        gradcam_cmd = [
            python_exe,
            str(PROJECT_ROOT / "scripts" / "generate_gradcam.py"),
            "--run-dir", str(exp_dir.relative_to(PROJECT_ROOT)),
            "--model", model_name,
            "--img-size", str(img_size),
        ]
        subprocess.run(gradcam_cmd, cwd=str(PROJECT_ROOT))

        # Collect metrics
        metrics_file = exp_dir / "test_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["model"] = model_name
            results.append(data)

        # Collect ROC data
        preds_file = exp_dir / "test_predictions.csv"
        if preds_file.exists():
            pdf = pd.read_csv(preds_file)
            roc_data[model_name] = (pdf["true_label"].values, pdf["prob_pes_planus"].values)

    if not results:
        print("No benchmark results collected.")
        return

    df = pd.DataFrame(results)
    cols_order = [
        "model", "accuracy", "sensitivity", "specificity", "precision", "npv", "f1_score", "roc_auc", "test_loss"
    ]
    df = df[[c for c in cols_order if c in df.columns]]

    # Format percentages for markdown table
    df_display = df.copy()
    for col in ["accuracy", "sensitivity", "specificity", "precision", "npv", "f1_score", "roc_auc"]:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: f"{x*100:.2f}%" if col != "roc_auc" else f"{x:.4f}")

    # Save CSV & Markdown
    out_csv = PROJECT_ROOT / "experiments" / "benchmark_summary.csv"
    out_md = PROJECT_ROOT / "experiments" / "benchmark_summary.md"
    df.to_csv(out_csv, index=False)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Academic Benchmark Comparison (Test Set - 230 Cases)\n\n")
        f.write("Comparison of standard deep learning classification backbones on unseen clinical lateral radiographs:\n\n")
        f.write(df_display.to_markdown(index=False))
        f.write("\n\n*All models evaluated under identical stratified test split (137 Pes Planus, 93 Normal).* \n")

    # Generate Combined ROC Curve Plot
    if roc_data:
        plt.figure(figsize=(7, 6))
        colors = {"resnet50": "navy", "efficientnet_b2": "darkgreen", "convnext_tiny": "crimson"}
        for model_name, (y_true, y_prob) in roc_data.items():
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            c = colors.get(model_name, "black")
            plt.plot(fpr, tpr, color=c, lw=2, label=f"{model_name.upper()} (AUC = {auc:.3f})")

        plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
        plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
        plt.title("Comparison of ROC Curves (Unseen Test Set)", fontsize=12)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        roc_out = PROJECT_ROOT / "experiments" / "benchmark_roc_comparison.png"
        plt.savefig(roc_out, dpi=300)
        plt.close()
        print(f"Saved combined ROC curve to {roc_out}")

    print("\n" + "=" * 65)
    print("ALL BENCHMARKS COMPLETED SUCCESSFULLY!")
    print("=" * 65)
    print(df_display.to_string(index=False))
    print(f"\nResults saved to {out_csv} and {out_md}")


if __name__ == "__main__":
    main()
