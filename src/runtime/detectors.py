"""
Two interchangeable ways to answer "is there a cat in this frame".

Both expose the same interface: detect(frame_bgr) -> (is_cat, confidence)
and annotate(frame_bgr), which draws whatever the backend can localise,
so camera.py stays unaware of which one is running and the backend can be
switched with a --model flag.
"""
import cv2
from PIL import Image

from src.config import (MOBILENET_RELEASE_THRESHOLD, MOBILENET_THRESHOLD,
                        MODEL_PATH, YOLO_CAT_CLASS_ID, YOLO_RELEASE_THRESHOLD,
                        YOLO_THRESHOLD)
from src.ml.inference import predict
from src.ml.model import load_trained_model


class MobileNetDetector:
    """Classification: looks at the whole frame and scores it cat / not cat."""

    name = "mobilenet"

    def __init__(self, device, threshold: float = MOBILENET_THRESHOLD,
                 release_threshold: float = MOBILENET_RELEASE_THRESHOLD):
        self.device = device
        self.threshold = threshold
        self.release_threshold = release_threshold
        self.model = load_trained_model(MODEL_PATH, device)

    def detect(self, frame_bgr):
        # OpenCV frames are BGR; PIL and the model expect RGB.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return predict(self.model, Image.fromarray(rgb), self.device, self.threshold)

    def annotate(self, frame_bgr):
        """Classification has no location to draw."""


class YoloDetector:
    """
    Detection: locates objects and their bounding boxes, and reports the
    highest-scoring cat box. COCO class 15 is "cat", so no training is needed.
    """

    name = "yolo"

    def __init__(self, threshold: float = YOLO_THRESHOLD,
                 release_threshold: float = YOLO_RELEASE_THRESHOLD,
                 weights: str = "yolov8n.pt"):
        from ultralytics import YOLO  # imported lazily: optional dependency

        self.threshold = threshold
        self.release_threshold = release_threshold
        self.model = YOLO(weights)
        self.cat_boxes = []  # [(x1, y1, x2, y2, confidence)] from the last frame

    def detect(self, frame_bgr):
        results = self.model(frame_bgr, verbose=False)
        self.cat_boxes = []

        for result in results:
            for box in result.boxes:
                if int(box.cls) == YOLO_CAT_CLASS_ID:
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                    self.cat_boxes.append((x1, y1, x2, y2, float(box.conf)))

        best = max((b[4] for b in self.cat_boxes), default=0.0)
        return best >= self.threshold, best

    def annotate(self, frame_bgr):
        """
        Draw every cat box from the last frame.
        Green: above threshold. Yellow: a cat candidate below threshold,
        useful for seeing what the threshold filters out.
        """
        for x1, y1, x2, y2, conf in self.cat_boxes:
            color = (0, 255, 0) if conf >= self.threshold else (0, 255, 255)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_bgr, f"cat {conf:.2f}", (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def build_detector(model_type: str, device):
    if model_type == "mobilenet":
        return MobileNetDetector(device)
    if model_type == "yolo":
        return YoloDetector()
    raise ValueError(f"Unknown model: {model_type}")
