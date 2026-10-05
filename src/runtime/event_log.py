"""
Recording what the system decided, and on which frame.

Every door state change is appended to a CSV and the frame that triggered it
is saved as a JPEG. Two reasons: the run can be reviewed without watching it
live, and the false positives it collects are exactly the hard negatives the
dataset is missing.
"""
import csv
from datetime import datetime
from pathlib import Path

from src.config import LOGS_DIR

CSV_FIELDS = ["timestamp", "event", "model", "confidence", "frame_file"]


class EventLogger:
    def __init__(self, log_dir: Path = LOGS_DIR, save_frames: bool = True):
        self.log_dir = Path(log_dir)
        self.frames_dir = self.log_dir / "frames"
        self.csv_path = self.log_dir / "events.csv"
        self.save_frames = save_frames

        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.save_frames:
            self.frames_dir.mkdir(exist_ok=True)

        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as f:
                csv.writer(f).writerow(CSV_FIELDS)

    def log(self, event: str, model: str, confidence: float, frame=None) -> dict:
        """Append one row, optionally storing the frame it was decided on."""
        stamp = datetime.now()
        frame_file = ""

        if frame is not None and self.save_frames:
            import cv2  # imported here so the logger stays usable without OpenCV

            frame_file = f"{stamp:%Y%m%d_%H%M%S_%f}_{event}.jpg"
            cv2.imwrite(str(self.frames_dir / frame_file), frame)

        row = {
            "timestamp": stamp.isoformat(timespec="seconds"),
            "event": event,
            "model": model,
            "confidence": round(float(confidence), 4),
            "frame_file": frame_file,
        }
        with self.csv_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)

        return row
