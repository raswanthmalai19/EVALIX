import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    average_precision_score,
)


def plot_confusion_matrix(y_test, y_pred, model_name="Model", ax=None):
    cm = confusion_matrix(y_test, y_pred)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Rejected", "Approved"],
                yticklabels=["Rejected", "Approved"],
                linewidths=1, linecolor="black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} - Confusion Matrix", fontweight="bold")
    return cm


def plot_roc_curve(y_test, y_prob, model_name="Model", ax=None):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    return roc_auc


def plot_precision_recall(y_test, y_prob, model_name="Model", ax=None):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, linewidth=2, label=f"{model_name} (AP={ap:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    return ap


def classification_summary(y_test, y_pred, model_name="Model"):
    print(f"\n{'='*60}")
    print(f"Classification Report - {model_name}")
    print("=" * 60)
    print(classification_report(y_test, y_pred,
                                target_names=["Rejected", "Approved"]))


def compare_models_table(results_list):
    df = pd.DataFrame(results_list)
    df = df.sort_values("F1_Score", ascending=False).reset_index(drop=True)
    for col in ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if x is not None else "N/A")
    return df