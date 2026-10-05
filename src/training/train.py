"""
Training loop.

One epoch is a full pass over the training split (80 batches of 32).
Validation accuracy is measured after every epoch to track whether the model
is still generalising or starting to memorise the training data.
"""
import torch
import torch.nn as nn

from src.config import (BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_PATH,
                        MODELS_DIR)
from src.device import get_device
from src.training.data import get_dataloaders, to_binary_labels
from src.ml.model import get_model


def run_epoch(model, loader, device, criterion, optimizer=None):
    """
    Run one pass over the loader.

    Passing optimizer=None switches to evaluation mode: no gradients,
    no weight updates.
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_training):
        for images, breed_labels in loader:
            images = images.to(device)
            labels = to_binary_labels(breed_labels, loader).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                optimizer.zero_grad()  # clear gradients from the previous batch
                loss.backward()        # backpropagation
                optimizer.step()       # apply the update

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train():
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)

    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=LEARNING_RATE
    )

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, device, criterion, optimizer
        )
        val_loss, val_acc = run_epoch(model, val_loader, device, criterion)

        print(f"Epoch {epoch}/{EPOCHS} | "
              f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f}")

    MODELS_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nSaved model: {MODEL_PATH}")


if __name__ == "__main__":
    train()
