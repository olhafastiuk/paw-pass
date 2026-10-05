"""
Choosing the decision threshold, then scoring the final model once.

The two splits play different roles:

  validation  the threshold sweep runs here. Picking a threshold is a
              decision, and decisions may only be informed by validation.
  test        scored exactly once, at the threshold already chosen, to get
              an honest number. Nothing is tuned by looking at it.

Accuracy alone is not informative here: the majority-class baseline already
reaches 0.677 because of the class imbalance, so precision, recall and F1 are
reported alongside it.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from src.config import (ASSETS_DIR, BATCH_SIZE, MOBILENET_THRESHOLD,
                        MODEL_PATH)
from src.device import get_device
from src.training.data import get_dataloaders, to_binary_labels
from src.ml.model import load_trained_model

THRESHOLD_SWEEP = [0.5, 0.7, 0.8, 0.9, 0.95]  # the configured threshold is added too


def collect_predictions(model, loader, device):
    """Run the loader through the model and return (labels, cat_probabilities)."""
    all_labels, all_probs = [], []

    with torch.no_grad():
        for images, breed_labels in loader:
            labels = to_binary_labels(breed_labels, loader)
            outputs = model(images.to(device))
            # softmax -> [P(not cat), P(cat)]; keep the second column
            probs = F.softmax(outputs, dim=1)[:, 1]

            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())

    return np.array(all_labels), np.array(all_probs)


def print_metrics(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)
    print(f"\nMetrics at threshold {threshold}")
    print(f"  Accuracy:  {accuracy_score(labels, preds):.3f}")
    print(f"  Precision: {precision_score(labels, preds, zero_division=0):.3f}")
    print(f"  Recall:    {recall_score(labels, preds, zero_division=0):.3f}")
    print(f"  F1:        {f1_score(labels, preds, zero_division=0):.3f}")
    return preds


def print_threshold_sweep(labels, probs, chosen=MOBILENET_THRESHOLD):
    """Show how the precision / recall trade-off moves with the threshold."""
    thresholds = sorted(set(THRESHOLD_SWEEP) | {chosen})
    print(f"{'thr':>6} {'Precision':>10} {'Recall':>8} {'F1':>7} {'FP':>5} {'FN':>5}")
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        _, fp, fn, _ = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        print(f"{thr:>6.2f} "
              f"{precision_score(labels, preds, zero_division=0):>10.3f} "
              f"{recall_score(labels, preds, zero_division=0):>8.3f} "
              f"{f1_score(labels, preds, zero_division=0):>7.3f} "
              f"{fp:>5} {fn:>5}"
              f"{'   <- chosen' if thr == chosen else ''}")


def plot_confusion_matrix(labels, preds, threshold, save_path):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    names = ["not cat", "cat"]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(cm, cmap="Blues")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=18,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_xticks([0, 1], names)
    ax.set_yticks([0, 1], names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (threshold {threshold})")
    fig.tight_layout()

    save_path.parent.mkdir(exist_ok=True)
    fig.savefig(save_path, dpi=120)
    print(f"\nSaved: {save_path.name}")
    return fig


def evaluate(show: bool = True):
    device = get_device()
    _, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)
    model = load_trained_model(MODEL_PATH, device)

    # 1. Validation: where the threshold is chosen.
    val_labels, val_probs = collect_predictions(model, val_loader, device)
    print("\n=== Validation: threshold sweep ===")
    print_threshold_sweep(val_labels, val_probs)

    # 2. Test: scored once, at the threshold already fixed in config.
    test_labels, test_probs = collect_predictions(model, test_loader, device)
    print("\n=== Test: final score ===")
    preds = print_metrics(test_labels, test_probs, MOBILENET_THRESHOLD)

    tn, fp, fn, tp = confusion_matrix(test_labels, preds, labels=[0, 1]).ravel()
    print(f"\nTN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"{fp} images would have opened the door for a non-cat")
    print(f"{fn} real cats would have been left outside")

    plot_confusion_matrix(test_labels, preds, MOBILENET_THRESHOLD,
                          ASSETS_DIR / "confusion_matrix.png")
    if show:
        plt.show()


if __name__ == "__main__":
    evaluate()
