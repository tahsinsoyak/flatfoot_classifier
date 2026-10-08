"""Script to aggregate all evaluated models and produce updated benchmark summary & ROC curve.
"""

from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    exp_dir = PROJECT_ROOT / "experiments"
    models = ["resnet50", "convnext_tiny", "efficientnet_b2", "foot_arch_net"]
    model_display_names = {
        "resnet50": "ResNet-50",
        "convnext_tiny": "ConvNeXt-Tiny",
        "efficientnet_b2": "EfficientNet-B2",
        "foot_arch_net": "FootArchNet (Proposed)",
    }

    results = []
    roc_data = {}

    for m in models:
        run_path = exp_dir / f"run_{m}_512px"
        metrics_file = run_path / "test_metrics.json"
        preds_file = run_path / "test_predictions.csv"
        if metrics_file.exists():
            with open(metrics_file, "r", encoding="utf-8") as f:
                d = json.load(f)
            d["model"] = model_display_names.get(m, m)
            results.append(d)
        if preds_file.exists():
            df_p = pd.read_csv(preds_file)
            roc_data[m] = (df_p["true_label"].values, df_p["prob_pes_planus"].values)

    df = pd.DataFrame(results)
    cols_order = ["model", "accuracy", "sensitivity", "specificity", "precision", "npv", "f1_score", "roc_auc", "test_loss"]
    df = df[[c for c in cols_order if c in df.columns]]
    df.to_csv(exp_dir / "benchmark_summary.csv", index=False)

    header = "| Model | Accuracy | Sensitivity | Specificity | Precision | NPV | F1-Score | ROC-AUC | Test Loss |"
    separator = "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    rows = []
    for _, row in df.iterrows():
        acc = f"{row['accuracy'] * 100:.2f}%"
        sens = f"{row['sensitivity'] * 100:.2f}%"
        spec = f"{row['specificity'] * 100:.2f}%"
        prec = f"{row['precision'] * 100:.2f}%"
        npv = f"{row['npv'] * 100:.2f}%"
        f1 = f"{row['f1_score'] * 100:.2f}%"
        auc = f"{row['roc_auc']:.4f}"
        loss = f"{row['test_loss']:.4f}"
        rows.append(f"| **{row['model']}** | {acc} | {sens} | {spec} | {prec} | {npv} | {f1} | {auc} | {loss} |")

    with open(exp_dir / "benchmark_summary.md", "w", encoding="utf-8") as f:
        f.write("# Academic Benchmark Comparison (Test Set)\n\n")
        f.write("Comparison of standard deep learning classification backbones versus the proposed FootArchNet architecture on unseen clinical lateral radiographs:\n\n")
        f.write(header + "\n" + separator + "\n" + "\n".join(rows) + "\n\n")
        f.write("*All models evaluated under stratified test split with letterbox standardized canonical orientation.* \n")

    plt.figure(figsize=(7.5, 6.5))
    colors = {
        "resnet50": "#1e40af",
        "convnext_tiny": "#b91c1c",
        "efficientnet_b2": "#059669",
        "foot_arch_net": "#7c3aed",
    }
    linestyles = {
        "resnet50": ":",
        "convnext_tiny": "-.",
        "efficientnet_b2": "--",
        "foot_arch_net": "-",
    }
    linewidths = {
        "resnet50": 2.0,
        "convnext_tiny": 2.0,
        "efficientnet_b2": 2.2,
        "foot_arch_net": 3.0,
    }

    for m in ["resnet50", "convnext_tiny", "efficientnet_b2", "foot_arch_net"]:
        if m in roc_data:
            y_true, y_prob = roc_data[m]
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            plt.plot(
                fpr, tpr,
                color=colors[m],
                linestyle=linestyles[m],
                linewidth=linewidths[m],
                label=f"{model_display_names[m]} (AUC = {auc:.4f})"
            )

    plt.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold", labelpad=8)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold", labelpad=8)
    plt.title("Receiver Operating Characteristic (ROC) Benchmark\nUnseen Patient Lateral Weight-Bearing Radiographs", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=10.5, frameon=True, framealpha=0.95, edgecolor="#cbd5e1")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    roc_out = exp_dir / "benchmark_roc_comparison.png"
    plt.savefig(roc_out, dpi=300)
    plt.close()
    print(f"Successfully generated {roc_out} and updated benchmark summary!")

if __name__ == "__main__":
    main()
