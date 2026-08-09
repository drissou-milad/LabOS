import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def euclidean_distance(p1, p2):
    """
    Euclidean distance between two points.
    """

    return np.sqrt(np.sum((p1 - p2) ** 2))


def nearest_neighbor(point, candidates):
    """
    Find nearest point.
    """

    distances = np.linalg.norm(candidates - point, axis=1)

    idx = np.argmin(distances)

    return idx, distances[idx]


# --------------------------------------------------------------------------
# Classification metrics (CellCNN validation) — accuracy alone hides class
# imbalance and doesn't distinguish "confidently wrong" from "borderline."
# --------------------------------------------------------------------------

def classification_report(y_true, y_prob, threshold=0.5):
    """
    y_true: array of 0/1 labels
    y_prob: array of predicted probabilities in [0, 1] (i.e. sigmoid(logits),
            NOT raw logits)

    Returns a dict with precision, recall, f1, roc_auc, and a confusion
    matrix (as a nested list, so it's directly JSON-serializable).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    report = {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    try:
        report["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        # Happens if the validation split ended up with only one class present
        report["roc_auc"] = float("nan")

    report["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return report


def plot_confusion_matrix(cm, out_path, labels=("negative", "positive")):
    """Save a confusion matrix (2x2 nested list/array) as a PNG."""
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
