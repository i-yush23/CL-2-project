import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def compute_metrics(y_true: list, y_pred: list, labels: list = None) -> dict:
    """Compute overall accuracy and macro-F1."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {"accuracy": float(acc), "macro_f1": float(macro_f1)}


def get_classification_report_df(
    y_true: list, y_pred: list, labels: list = None
) -> pd.DataFrame:
    """Return precision, recall, F1, and support per class as a DataFrame."""
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    df = pd.DataFrame(
        {
            "class": labels,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support,
        }
    )
    return df


def plot_confusion_matrix(
    y_true: list,
    y_pred: list,
    labels: list,
    output_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)
    fig.colorbar(cax)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="red" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=11,
            )

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="left")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, pad=20)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_dataframe_csv(df: pd.DataFrame, output_path: str) -> None:
    """Save DataFrame to CSV, ensuring parent directories exist."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
