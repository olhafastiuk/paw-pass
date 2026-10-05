"""
Real-time loop: camera -> detector -> debouncer -> door.

The loop itself stays thin. Deciding whether a frame holds a cat is the
detector's job, smoothing the stream of verdicts is the debouncer's, and
recording what happened is the logger's.
"""
import cv2

from src.runtime.debounce import CLOSE, OPEN, FrameDebouncer, frame_is_cat
from src.runtime.detectors import build_detector
from src.device import get_device
from src.runtime.door_controller import DoorController
from src.runtime.event_log import EventLogger

GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (255, 255, 0)

WINDOW = "Cat Door System"
ESC = 27
QUIT_KEYS = {ord("q"), ord("Q"), ESC}


def should_quit(key: int) -> bool:
    """
    Q or Esc, pressed while the camera window has focus.
    Esc is there because Q stops working under a non-Latin keyboard layout.
    """
    return (key & 0xFF) in QUIT_KEYS


def window_closed() -> bool:
    """True once the preview window has been closed with the mouse."""
    return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1


def draw_overlay(frame, is_cat, confidence, door, model_name, streak):
    color = GREEN if is_cat else RED
    label = f"CAT {confidence:.2f}" if is_cat else f"no cat {confidence:.2f}"

    cv2.putText(frame, label, (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    cv2.putText(frame, f"DOOR: {door.state}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(frame, f"model: {model_name}   streak: {streak}", (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 1)


def run_camera(model_type: str = "mobilenet", camera_index: int = 0,
               headless: bool = False, log: bool = True):
    detector = build_detector(model_type, get_device())
    door = DoorController()
    debouncer = FrameDebouncer()
    logger = EventLogger() if log else None

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Could not open the camera. "
              "On macOS check System Settings > Privacy & Security > Camera.")
        return

    if headless:
        print(f"Model: {detector.name}. Press Ctrl+C to quit.")
    else:
        print(f"Model: {detector.name}. Click the camera window, then press "
              f"Q or Esc to quit (or Ctrl+C here).")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue  # skip a bad frame instead of ending the loop

            _, confidence = detector.detect(frame)
            is_cat = frame_is_cat(confidence, door.is_open,
                                  detector.threshold, detector.release_threshold)
            action = debouncer.update(is_cat)

            # Only a real state change is worth recording.
            if action == OPEN and not door.is_open:
                door.open()
                if logger:
                    logger.log(OPEN, detector.name, confidence, frame)
            elif action == CLOSE and door.is_open:
                door.close()
                if logger:
                    logger.log(CLOSE, detector.name, confidence, frame)

            if headless:
                continue

            detector.annotate(frame)
            draw_overlay(frame, is_cat, confidence, door,
                         detector.name, debouncer.streak)
            cv2.imshow(WINDOW, frame)

            if should_quit(cv2.waitKey(1)) or window_closed():
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera()  # defaults; use `python main.py camera --help` for options
