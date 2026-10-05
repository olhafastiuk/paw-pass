"""
Error analysis: inspect the individual images the model got wrong.

Runs on the validation split, not test: what the errors reveal feeds back
into decisions about data and training, and the test split has to stay
untouched by those decisions.
"""
import matplotlib.pyplot as plt

from src.config import (ASSETS_DIR, BATCH_SIZE, MOBILENET_THRESHOLD,
                        MODEL_PATH)
from src.device import get_device
from src.training.data import get_dataloaders, load_raw_dataset
from src.training.evaluate import collect_predictions
from src.ml.model import load_trained_model

EXAMPLES_PER_GROUP = 4


def _grid(raw, items, title, save_name):
    """items: list of (dataset_index, cat_probability)."""
    if not items:
        print(f"{title}: no examples")
        return

    items = items[:EXAMPLES_PER_GROUP]
    fig, axes = plt.subplots(1, len(items), figsize=(3.2 * len(items), 3.8))
    axes = [axes] if len(items) == 1 else list(axes)

    for ax, (idx, prob) in zip(axes, items):
        image, breed_label = raw[idx]
        ax.imshow(image)
        ax.set_title(f"{raw.classes[breed_label]}\nP(cat) = {prob:.2f}", fontsize=9)
        ax.axis("off")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    ASSETS_DIR.mkdir(exist_ok=True)
    path = ASSETS_DIR / save_name
    fig.savefig(path, dpi=110)
    print(f"Saved: {path.name}")


def error_analysis(show: bool = True):
    device = get_device()
    _, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)
    model = load_trained_model(MODEL_PATH, device)

    labels, probs = collect_predictions(model, val_loader, device)

    # shuffle=False, so the i-th prediction maps to the i-th Subset index.
    val_indices = val_loader.dataset.indices
    raw = load_raw_dataset()  # no transform: we want the original images

    false_positives, false_negatives, correct_cats = [], [], []

    for i, (label, prob) in enumerate(zip(labels, probs)):
        predicted_cat = prob >= MOBILENET_THRESHOLD
        item = (val_indices[i], prob)

        if predicted_cat and label == 0:
            false_positives.append(item)
        elif not predicted_cat and label == 1:
            false_negatives.append(item)
        elif predicted_cat and label == 1:
            correct_cats.append(item)

    # Worst failures first: the most confident false positives, and the true
    # cats the model was least willing to recognise.
    false_positives.sort(key=lambda x: -x[1])
    false_negatives.sort(key=lambda x: x[1])
    correct_cats.sort(key=lambda x: -x[1])

    print(f"False positives: {len(false_positives)}")
    print(f"False negatives: {len(false_negatives)}")
    print(f"True positives:  {len(correct_cats)}")

    _grid(raw, false_positives,
          "False positives - predicted CAT, actually not a cat",
          "errors_false_positive.png")
    _grid(raw, false_negatives,
          "False negatives - a cat the model missed",
          "errors_false_negative.png")
    _grid(raw, correct_cats,
          "True positives - correctly detected cats",
          "errors_true_positive.png")

    if show:
        plt.show()


if __name__ == "__main__":
    error_analysis()
