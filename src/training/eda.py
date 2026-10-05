"""
Exploratory data analysis.

Run before training to see what the data actually looks like: how much there
is, how balanced the classes are, what image sizes to expect, and what the
samples look like.
"""
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from src.config import ASSETS_DIR, RANDOM_STATE
from src.training.data import is_cat_breed, load_raw_dataset

SAMPLES_TO_SHOW = 8
SIZE_SAMPLE = 200


def class_balance(dataset):
    breed_labels = getattr(dataset, "_labels", None) or [
        dataset[i][1] for i in range(len(dataset))
    ]
    binary = [1 if is_cat_breed(dataset.classes[b]) else 0 for b in breed_labels]
    counts = Counter(binary)
    total = len(binary)

    print(f"Images:  {total}")
    print(f"Breeds:  {len(dataset.classes)}")
    print(f"  not cat (0): {counts[0]}")
    print(f"  cat     (1): {counts[1]}")

    # Majority-class baseline: always predict the larger class.
    majority = max(counts.values()) / total
    print(f"\nBaseline (always 'not cat'): accuracy = {majority:.3f}")


def size_stats(dataset, n=SIZE_SAMPLE):
    widths, heights = [], []
    for i in range(min(n, len(dataset))):
        image, _ = dataset[i]
        w, h = image.size
        widths.append(w)
        heights.append(h)

    print(f"\nImage sizes (first {len(widths)} images):")
    print(f"  width  - min {min(widths)}, max {max(widths)}, "
          f"mean {int(np.mean(widths))}")
    print(f"  height - min {min(heights)}, max {max(heights)}, "
          f"mean {int(np.mean(heights))}")


def sample_grid(dataset, n=SAMPLES_TO_SHOW, show=True):
    random.seed(RANDOM_STATE)
    indices = random.sample(range(len(dataset)), n)

    fig, axes = plt.subplots(2, n // 2, figsize=(12, 6))
    for ax, idx in zip(axes.flatten(), indices):
        image, breed_label = dataset[idx]
        breed = dataset.classes[breed_label]
        cat = is_cat_breed(breed)

        ax.imshow(image)
        ax.set_title(f"{'CAT' if cat else 'DOG'}\n{breed}",
                     color="green" if cat else "red", fontsize=9)
        ax.axis("off")

    fig.tight_layout()
    ASSETS_DIR.mkdir(exist_ok=True)
    path = ASSETS_DIR / "eda_samples.png"
    fig.savefig(path, dpi=110)
    print(f"\nSaved: {path.name}")

    if show:
        plt.show()


def main(show: bool = True):
    dataset = load_raw_dataset(download=True)
    class_balance(dataset)
    size_stats(dataset)
    sample_grid(dataset, show=show)


if __name__ == "__main__":
    main()
