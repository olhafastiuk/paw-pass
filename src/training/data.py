"""
Dataset loading and the train / validation / test split.

Oxford-IIIT Pet provides 37-class breed labels. This module maps them to the
binary target used by the project: 1 = cat, 0 = not cat.
"""
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import OxfordIIITPet

from src.config import (BATCH_SIZE, CAT_BREEDS, DATA_DIR, RANDOM_STATE,
                        VAL_TEST_FRACTION)
from src.ml.preprocessing import train_transform, val_transform


def load_raw_dataset(transform=None, download: bool = False) -> OxfordIIITPet:
    return OxfordIIITPet(
        root=str(DATA_DIR),
        split="trainval",
        target_types="category",
        transform=transform,
        download=download,
    )


def is_cat_breed(breed_name: str) -> bool:
    return breed_name in CAT_BREEDS


def to_binary_labels(breed_labels: torch.Tensor, loader: DataLoader) -> torch.Tensor:
    """
    Map a batch of breed labels to the binary target (1 = cat, 0 = not cat).

    """
    classes = loader.dataset.dataset.classes
    return torch.tensor(
        [1 if is_cat_breed(classes[i]) else 0 for i in breed_labels.tolist()],
        dtype=torch.long,
    )


def get_dataloaders(batch_size: int = BATCH_SIZE, verbose: bool = True):
    """
    Return (train_loader, val_loader, test_loader) split 70 / 15 / 15.

    """
    # download=True is a no-op when the files are already there, and lets
    # `python main.py train` work on a fresh machine without running eda first.
    raw = load_raw_dataset(download=True)

    indices = list(range(len(raw)))

    # raw[i] would decode a JPEG just to read one label, 3680 times.
    # OxfordIIITPet keeps the breed labels in _labels, so read them directly.
    breed_labels = getattr(raw, "_labels", None) or [raw[i][1] for i in indices]
    labels = [1 if is_cat_breed(raw.classes[b]) else 0 for b in breed_labels]

    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices, labels,
        test_size=VAL_TEST_FRACTION,
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=temp_labels,
        random_state=RANDOM_STATE,
    )

    if verbose:
        print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    train_base = load_raw_dataset(transform=train_transform)
    eval_base = load_raw_dataset(transform=val_transform)

    train_loader = DataLoader(
        Subset(train_base, train_idx), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        Subset(eval_base, val_idx), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        Subset(eval_base, test_idx), batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader
