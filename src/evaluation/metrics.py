"""Medical Diagnostic Metrics calculation and visualization."""

from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def compute_medical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Calculate standard clinical diagnostic test metrics.

    Class 1: Pes Planus (Flatfoot) [Positive]
    Class 0: Normal Foot [Negative]
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = float(accuracy_score(y_true, y_pred))
    sensitivity = float(recall_score(y_true, y_pred, zero_division=0))  # True Positive Rate
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0      # True Negative Rate
    precision = float(precision_score(y_true, y_pred, zero_division=0)) # Positive Predictive Value
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0             # Negative Predictive Value
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5

    return {
        "accuracy": round(accuracy, 4),
        "sensitivity": round(sensitivity, 4),  # Recall
        "specificity": round(specificity, 4),
        "precision": round(precision, 4),      # PPV
        "npv": round(npv, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    class_names: Tuple[str, str] = ("Normal", "Pes Planus"),
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Diagnosis")
    plt.xlabel("Predicted Diagnosis")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str,
    model_name: str = "Model",
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"{model_name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
