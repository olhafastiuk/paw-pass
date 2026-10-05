"""
Single-image inference: image in, (is_cat, confidence) out.
"""
import torch
import torch.nn.functional as F

from src.config import MOBILENET_THRESHOLD
from src.ml.preprocessing import val_transform


def predict(model, image_pil, device, threshold: float = MOBILENET_THRESHOLD):
    """
    image_pil: PIL.Image
    Returns (is_cat: bool, confidence: float) where confidence is P(cat).
    """
    # unsqueeze(0): [3, 224, 224] -> [1, 3, 224, 224]; the model expects a batch
    tensor = val_transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probability = F.softmax(logits, dim=1)[0, 1].item()

    return probability >= threshold, probability
