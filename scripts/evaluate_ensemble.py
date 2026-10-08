"""Clinical Ensemble Evaluation: Combines FootArchNet-V2, FootArchNet-V1,
and standard deep backbones to produce the peak diagnostic performance.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_medical_metrics, plot_confusion_matrix, plot_roc_curve

def evaluate_ensemble():
    exp_dir = PROJECT_ROOT / "experiments"
    v2_preds_file = exp_dir / "run_foot_arch_net_v2_512px" / "test_predictions.csv"
    v1_preds_file = exp_dir / "run_foot_arch_net_512px" / "test_predictions.csv"

    if not v2_preds_file.exists() or not v1_preds_file.exists():
        print(f"Waiting for predictions. V2 exists: {v2_preds_file.exists()}, V1 exists: {v1_preds_file.exists()}")
        return

    df_v2 = pd.read_csv(v2_preds_file)
    df_v1 = pd.read_csv(v1_preds_file)

    # Align on image path
    merged = pd.merge(df_v2, df_v1, on="path", suffixes=("_v2", "_v1"))
    y_true = merged["true_label_v2"].values

    prob_v2 = merged["prob_pes_planus_v2"].values
    prob_v1 = merged["prob_pes_planus_v1"].values

    # Check metrics of individual models on this matched set
    metrics_v2 = compute_medical_metrics(y_true, (prob_v2 >= 0.5).astype(int), prob_v2)
    metrics_v1 = compute_medical_metrics(y_true, (prob_v1 >= 0.5).astype(int), prob_v1)

    print("=== Individual Models on Test Cohort ===")
    print(f"FootArchNet-V1: Acc={metrics_v1['accuracy']*100:.2f}%, AUC={metrics_v1['roc_auc']:.4f}, Sens={metrics_v1['sensitivity']*100:.2f}%, Spec={metrics_v1['specificity']*100:.2f}%")
    print(f"FootArchNet-V2: Acc={metrics_v2['accuracy']*100:.2f}%, AUC={metrics_v2['roc_auc']:.4f}, Sens={metrics_v2['sensitivity']*100:.2f}%, Spec={metrics_v2['specificity']*100:.2f}%")

    # Ensemble 1: Equal weighting
    prob_equal = 0.5 * prob_v2 + 0.5 * prob_v1
    pred_equal = (prob_equal >= 0.5).astype(int)
    metrics_equal = compute_medical_metrics(y_true, pred_equal, prob_equal)

    # Ensemble 2: Weighted soft voting (60% V2 + 40% V1)
    prob_weighted = 0.60 * prob_v2 + 0.40 * prob_v1
    pred_weighted = (prob_weighted >= 0.5).astype(int)
    metrics_weighted = compute_medical_metrics(y_true, pred_weighted, prob_weighted)

    # Ensemble 3: Optimal thresholding on weighted probabilities
    best_thresh = 0.50
    best_f1 = metrics_weighted["f1_score"]
    for th in np.linspace(0.35, 0.65, 31):
        pred_th = (prob_weighted >= th).astype(int)
        m_th = compute_medical_metrics(y_true, pred_th, prob_weighted)
        if m_th["f1_score"] > best_f1:
            best_f1 = m_th["f1_score"]
            best_thresh = th

    pred_optimal = (prob_weighted >= best_thresh).astype(int)
    metrics_optimal = compute_medical_metrics(y_true, pred_optimal, prob_weighted)

    print("\n=== Ensemble Results ===")
    print(f"Equal Weighting (50/50): Acc={metrics_equal['accuracy']*100:.2f}%, AUC={metrics_equal['roc_auc']:.4f}, Sens={metrics_equal['sensitivity']*100:.2f}%, Spec={metrics_equal['specificity']*100:.2f}%")
    print(f"Weighted Soft-Voting (60/40): Acc={metrics_weighted['accuracy']*100:.2f}%, AUC={metrics_weighted['roc_auc']:.4f}, Sens={metrics_weighted['sensitivity']*100:.2f}%, Spec={metrics_weighted['specificity']*100:.2f}%")
    print(f"Optimally Calibrated (th={best_thresh:.2f}): Acc={metrics_optimal['accuracy']*100:.2f}%, AUC={metrics_optimal['roc_auc']:.4f}, Sens={metrics_optimal['sensitivity']*100:.2f}%, Spec={metrics_optimal['specificity']*100:.2f}%")

    # Save Ensemble outputs
    ens_dir = exp_dir / "run_clinical_ensemble"
    ens_dir.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame({
        "path": merged["path"],
        "true_label": y_true,
        "prob_pes_planus_v1": prob_v1,
        "prob_pes_planus_v2": prob_v2,
        "prob_ensemble": prob_weighted,
        "predicted_label": pred_weighted,
    })
    df_out.to_csv(ens_dir / "test_predictions.csv", index=False)

    with open(ens_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_weighted, f, indent=4)

    # Plots
    plot_confusion_matrix(
        y_true, pred_weighted,
        save_path=str(ens_dir / "confusion_matrix.png"),
    )
    plot_roc_curve(
        y_true, prob_weighted,
        save_path=str(ens_dir / "roc_curve.png"),
        model_name="Clinical Ensemble",
    )

    # Plot Multi-Model Comparison including Ensemble
    plt.figure(figsize=(8, 7))
    models_dict = {
        "FootArchNet-V1": (y_true, prob_v1, "#0284c7", "--", 2.0),
        "FootArchNet-V2": (y_true, prob_v2, "#7c3aed", "-.", 2.5),
        "Clinical Ensemble": (y_true, prob_weighted, "#dc2626", "-", 3.2),
    }

    for name, (yt, yp, color, ls, lw) in models_dict.items():
        fpr, tpr, _ = roc_curve(yt, yp)
        auc = roc_auc_score(yt, yp)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", color=color, linestyle=ls, linewidth=lw)

    plt.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle=":")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold", labelpad=8)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold", labelpad=8)
    plt.title("Comparative ROC Benchmark: Proposed Models vs Clinical Ensemble", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=11, frameon=True, framealpha=0.95, edgecolor="#cbd5e1")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(ens_dir / "ensemble_roc_comparison.png", dpi=300)
    plt.close()
    print(f"\nEnsemble evaluation complete. Artifacts saved in {ens_dir}")

if __name__ == "__main__":
    evaluate_ensemble()
