"""
Transform pipelines that turn a raw image into model input.
"""
from torchvision import transforms

from src.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

_normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),       # a mirrored cat is still a cat
    transforms.RandomRotation(10),           # slight camera tilt
    transforms.ColorJitter(brightness=0.2),  # day / evening lighting
    transforms.ToTensor(),
    _normalize,
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    _normalize,
])
