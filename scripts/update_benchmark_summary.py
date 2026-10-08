"""Script to aggregate all evaluated models and produce updated benchmark summary & ROC curve.
Includes FootArchNet-V1, V2, Ultra, DenseNet-201, and Super Ensemble (TTA).
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
        ("resnet50", "run_resnet50_512px", "ResNet-50", "test_metrics.json", "prob_pes_planus"),
        ("efficientnet_b2", "run_efficientnet_b2_512px", "EfficientNet-B2", "test_metrics.json", "prob_pes_planus"),
        ("convnext_tiny", "run_convnext_tiny_512px", "ConvNeXt-Tiny", "test_metrics.json", "prob_pes_planus"),
        ("densenet201", "run_densenet201_512px", "DenseNet-201", "test_metrics.json", "prob_pes_planus"),
        ("foot_arch_net", "run_foot_arch_net_512px", "FootArchNet-V1", "test_metrics.json", "prob_pes_planus"),
        ("foot_arch_net_v2", "run_foot_arch_net_v2_512px", "FootArchNet-V2", "test_metrics.json", "prob_pes_planus"),
        ("foot_arch_net_ultra", "run_foot_arch_net_ultra_512px", "FootArchNet-Ultra", "test_metrics.json", "prob_pes_planus"),
        ("swin_t", "run_swin_t_512px", "Swin-T (Vision Transformer)", "test_metrics.json", "prob_pes_planus"),
        ("super_ensemble", "run_super_ensemble_tta", "Super Ensemble V2 (TTA, Calibrated)", "test_metrics.json", "prob_super_ensemble"),
    ]

    results = []
    roc_data = {}

    for key, folder, display_name, metrics_filename, prob_col_name in models_config:
        run_path = exp_dir / folder
        metrics_file = run_path / metrics_filename
        preds_file = run_path / "test_predictions.csv"

        if metrics_file.exists():
            with open(metrics_file, "r", encoding="utf-8") as f:
                d = json.load(f)

            if "calibrated_threshold" in d and "metrics" in d["calibrated_threshold"]:
                metric_entry = dict(d["calibrated_threshold"]["metrics"])
            else:
                metric_entry = dict(d)

            metric_entry["model"] = display_name
            results.append(metric_entry)

        if preds_file.exists():
            df_p = pd.read_csv(preds_file)
            prob_col = prob_col_name if prob_col_name in df_p.columns else ("prob_ensemble" if "prob_ensemble" in df_p.columns else "prob_pes_planus")
            roc_data[key] = (df_p["true_label"].values, df_p[prob_col].values, display_name)

    df = pd.DataFrame(results)
    cols_order = ["model", "accuracy", "sensitivity", "specificity", "precision", "npv", "f1_score", "roc_auc", "test_loss"]
    df = df[[c for c in cols_order if c in df.columns]]
    df.to_csv(exp_dir / "benchmark_summary.csv", index=False)

    header = "| Model | Accuracy | Sensitivity | Specificity | Precision | NPV | F1-Score | ROC-AUC |"
    separator = "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    rows = []
    for _, row in df.iterrows():
        acc = f"{row['accuracy'] * 100:.2f}%"
        sens = f"{row['sensitivity'] * 100:.2f}%"
        spec = f"{row['specificity'] * 100:.2f}%"
        prec = f"{row['precision'] * 100:.2f}%"
        npv = f"{row['npv'] * 100:.2f}%"
        f1 = f"{row['f1_score'] * 100:.2f}%"
        auc = f"{row['roc_auc']:.4f}"
        is_highlight = "Ensemble" in row["model"] or "Ultra" in row["model"] or "V2" in row["model"]
        m_str = f"**{row['model']}**" if is_highlight else row["model"]
        rows.append(f"| {m_str} | {acc} | {sens} | {spec} | {prec} | {npv} | {f1} | {auc} |")

    with open(exp_dir / "benchmark_summary.md", "w", encoding="utf-8") as f:
        f.write("# Academic Benchmark Comparison (Test Set - 229 Unseen Clinical Patients)\n\n")
        f.write("Evaluation of standard radiology architectures vs. dedicated FootArchNet architectures and Multi-Scale Super Ensemble:\n\n")
        f.write(header + "\n" + separator + "\n" + "\n".join(rows) + "\n\n")
        f.write("*All models evaluated on the standardized letterbox test cohort (136 pes planus, 93 normal controls).* \n")

    # High-quality academic ROC curve
    plt.figure(figsize=(8.5, 7.5))
    plot_styles = {
        "resnet50": ("#94a3b8", ":", 1.8),
        "efficientnet_b2": ("#0284c7", "--", 1.8),
        "convnext_tiny": ("#64748b", "-.", 1.8),
        "densenet201": ("#d97706", "-.", 2.2),
        "foot_arch_net": ("#059669", "--", 2.2),
        "foot_arch_net_v2": ("#7c3aed", "-", 2.5),
        "foot_arch_net_ultra": ("#2563eb", "-", 2.8),
        "swin_t": ("#0d9488", ":", 2.2),
        "super_ensemble": ("#dc2626", "-", 3.5),
    }

    for key, folder, display_name, _, _ in models_config:
        if key in roc_data:
            y_true, y_prob, name = roc_data[key]
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            color, ls, lw = plot_styles.get(key, ("black", "-", 1.5))
            label_name = name.replace(" (TTA, Calibrated)", " + TTA")
            plt.plot(
                fpr, tpr,
                color=color,
                linestyle=ls,
                linewidth=lw,
                label=f"{label_name} (AUC = {auc:.4f})"
            )

    plt.plot([0, 1], [0, 1], color="#cbd5e1", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold", labelpad=8)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold", labelpad=8)
    plt.title("Receiver Operating Characteristic (ROC) Benchmark\nUnseen Patient Lateral Weight-Bearing Radiographs", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10, shadow=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_roc = exp_dir / "benchmark_roc_comparison.png"
    plt.savefig(out_roc, dpi=300)
    plt.close()
    print(f"Updated benchmark summary CSV, MD, and ROC curve saved to {exp_dir}!")

if __name__ == "__main__":
    main()
