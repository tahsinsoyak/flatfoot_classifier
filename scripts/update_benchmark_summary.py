"""Script to aggregate all evaluated models and produce updated benchmark summary & ROC curve.
Includes FootArchNet-V2 and the Clinical Ensemble.
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
    models_config = [
        ("resnet50", "run_resnet50_512px", "ResNet-50"),
        ("convnext_tiny", "run_convnext_tiny_512px", "ConvNeXt-Tiny"),
        ("efficientnet_b2", "run_efficientnet_b2_512px", "EfficientNet-B2"),
        ("foot_arch_net", "run_foot_arch_net_512px", "FootArchNet-V1"),
        ("foot_arch_net_v2", "run_foot_arch_net_v2_512px", "FootArchNet-V2 (Proposed)"),
        ("clinical_ensemble", "run_clinical_ensemble", "Clinical Ensemble (V1+V2)"),
    ]

    results = []
    roc_data = {}

    for key, folder, display_name in models_config:
        run_path = exp_dir / folder
        metrics_file = run_path / "test_metrics.json"
        preds_file = run_path / "test_predictions.csv"
        if metrics_file.exists():
            with open(metrics_file, "r", encoding="utf-8") as f:
                d = json.load(f)
            d["model"] = display_name
            results.append(d)
        if preds_file.exists():
            df_p = pd.read_csv(preds_file)
            prob_col = "prob_ensemble" if "prob_ensemble" in df_p.columns else "prob_pes_planus"
            roc_data[key] = (df_p["true_label"].values, df_p[prob_col].values, display_name)

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
        loss = f"{row['test_loss']:.4f}" if "test_loss" in row and pd.notnull(row["test_loss"]) else "N/A"
        is_highlight = "Ensemble" in row["model"] or "Proposed" in row["model"]
        m_str = f"**{row['model']}**" if is_highlight else row["model"]
        rows.append(f"| {m_str} | {acc} | {sens} | {spec} | {prec} | {npv} | {f1} | {auc} | {loss} |")

    with open(exp_dir / "benchmark_summary.md", "w", encoding="utf-8") as f:
        f.write("# Academic Benchmark Comparison (Test Set - 229 Unseen Clinical Patients)\n\n")
        f.write("Comparison of standard deep backbones versus FootArchNet-V1, FootArchNet-V2, and the Clinical Ensemble:\n\n")
        f.write(header + "\n" + separator + "\n" + "\n".join(rows) + "\n\n")
        f.write("*All models evaluated on the standardized letterbox test cohort.* \n")

    plt.figure(figsize=(8.0, 7.0))
    plot_styles = {
        "resnet50": ("#94a3b8", ":", 1.8),
        "convnext_tiny": ("#64748b", "-.", 1.8),
        "efficientnet_b2": ("#0284c7", "--", 2.0),
        "foot_arch_net": ("#059669", "-.", 2.2),
        "foot_arch_net_v2": ("#7c3aed", "-", 2.8),
        "clinical_ensemble": ("#dc2626", "-", 3.4),
    }

    for key, folder, display_name in models_config:
        if key in roc_data:
            y_true, y_prob, name = roc_data[key]
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            color, ls, lw = plot_styles.get(key, ("black", "-", 1.5))
            plt.plot(
                fpr, tpr,
                color=color,
                linestyle=ls,
                linewidth=lw,
                label=f"{name} (AUC = {auc:.4f})"
            )

    plt.plot([0, 1], [0, 1], color="#cbd5e1", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold", labelpad=8)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold", labelpad=8)
    plt.title("Receiver Operating Characteristic (ROC) Benchmark\nUnseen Patient Lateral Weight-Bearing Radiographs", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=10.0, frameon=True, framealpha=0.95, edgecolor="#cbd5e1")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    roc_out = exp_dir / "benchmark_roc_comparison.png"
    plt.savefig(roc_out, dpi=300)
    plt.close()
    print(f"Successfully generated {roc_out} and updated benchmark summary!")

if __name__ == "__main__":
    main()
