"""
MobileNetV2 backbone pretrained on ImageNet, adapted by transfer learning.

The convolutional stack already encodes edges, textures and shapes, so it stays frozen. Only the classifier head is replaced
(1000 ImageNet classes -> 2) and trained.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

NUM_CLASSES = 2      # 0 = not cat, 1 = cat
FEATURE_DIM = 1280   # output width of the MobileNetV2 feature extractor


def get_model(pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    # Freeze the feature extractor.
    for param in model.parameters():
        param.requires_grad = False

    # The new head defaults to requires_grad=True, so it is the only part trained.
    model.classifier[1] = nn.Linear(FEATURE_DIM, NUM_CLASSES)
    return model


def load_trained_model(model_path, device) -> nn.Module:
    """Load fine-tuned weights and return the model in eval mode."""
    model = get_model().to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()
    return model


def summary():
    """Run a random image through the model and print what comes out."""
    model = get_model()
    dummy = torch.randn(1, 3, 224, 224)
    logits = model(dummy)
    probs = F.softmax(logits, dim=1)

    print(f"Input shape:   {tuple(dummy.shape)}")
    print(f"Output shape:  {tuple(logits.shape)}")
    print(f"Logits:        {logits.tolist()[0]}")
    print(f"Probabilities: {probs.tolist()[0]}")
    print(f"Confidence (cat): {probs[0][1].item():.3f}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable:,} of {total:,} "
          f"({trainable / total:.2%})")


if __name__ == "__main__":
    summary()
