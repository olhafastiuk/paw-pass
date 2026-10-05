"""
Central configuration: paths, dataset definition, and hyperparameters.

Deliberately free of heavy imports (torch, cv2) so that light modules and
tests can read the settings without pulling in the whole stack.
"""
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"
LOGS_DIR = PROJECT_ROOT / "logs"
MODEL_PATH = MODELS_DIR / "cat_detector.pth"

# --- Dataset ---
# Oxford-IIIT Pet ships 37 breeds: 12 feline, 25 canine.
# Single source of truth for the binary cat / not-cat mapping.
CAT_BREEDS = frozenset({
    "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair",
    "Egyptian Mau", "Maine Coon", "Persian", "Ragdoll",
    "Russian Blue", "Siamese", "Sphynx",
})

# --- Preprocessing ---
IMAGE_SIZE = 224                       # input size expected by ImageNet backbones
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # per-channel RGB statistics of ImageNet
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Split ---
VAL_TEST_FRACTION = 0.30  # 30% goes to val + test, the remaining 70% to train
RANDOM_STATE = 42         # fixed so the split is reproducible across runs

# --- Training ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 5

# --- Inference / door ---
# The two backends score on different scales, so each gets its own threshold.
# MobileNet: softmax P(cat) over the whole frame.
# YOLO: confidence of a single bounding box; yolov8n rarely exceeds ~0.8 on a
# webcam frame, and ultralytics itself defaults to 0.25.
#
# Each backend also gets a lower release threshold (hysteresis): the score
# has to clear the upper bar to start believing in a cat, and fall below the
# lower one to stop. Confidence wobbling around a single threshold would
# otherwise flip the door open and shut every second.
MOBILENET_THRESHOLD = 0.85
MOBILENET_RELEASE_THRESHOLD = 0.60
YOLO_THRESHOLD = 0.50
YOLO_RELEASE_THRESHOLD = 0.30
FRAMES_TO_OPEN = 5      # consecutive cat frames required to open
FRAMES_TO_CLOSE = 5     # consecutive non-cat frames required to close
MIN_OPEN_SECONDS = 3.0  # once open, stay open at least this long
YOLO_CAT_CLASS_ID = 15  # "cat" class index in the COCO label set
