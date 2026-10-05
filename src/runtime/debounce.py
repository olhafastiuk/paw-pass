"""
Turning a stream of per-frame scores into a door decision.

Three layers of smoothing, each fixing a different failure:

  hysteresis    confidence wobbling around one threshold (0.49, 0.51, 0.48)
                would flip the verdict every frame, so the bar to start
                believing in a cat is higher than the bar to keep believing
  streaks       a single blurry or unlucky frame must not move the door, so
                it reacts only to several consecutive frames
  min open time even with the above, the door should not shut the instant
                the cat steps out of frame
"""
import time

from src.config import FRAMES_TO_CLOSE, FRAMES_TO_OPEN, MIN_OPEN_SECONDS

OPEN = "open"
CLOSE = "close"
HOLD = "hold"


def frame_is_cat(confidence: float, door_open: bool,
                 threshold: float, release_threshold: float) -> bool:
    """
    Per-frame verdict with hysteresis.

    While the door is closed a frame needs `threshold` to count as a cat;
    once it is open, anything above the lower `release_threshold` still does.
    """
    bar = release_threshold if door_open else threshold
    return confidence >= bar


class FrameDebouncer:
    def __init__(self, frames_to_open=FRAMES_TO_OPEN,
                 frames_to_close=FRAMES_TO_CLOSE,
                 min_open_seconds=MIN_OPEN_SECONDS,
                 clock=time.monotonic):
        self.frames_to_open = frames_to_open
        self.frames_to_close = frames_to_close
        self.min_open_seconds = min_open_seconds
        self.clock = clock

        self.cat_streak = 0
        self.no_cat_streak = 0
        self._opened_at = None

    def update(self, is_cat: bool) -> str:
        """Feed one frame verdict; return OPEN, CLOSE or HOLD."""
        if is_cat:
            self.cat_streak += 1
            self.no_cat_streak = 0
        else:
            self.no_cat_streak += 1
            self.cat_streak = 0

        if self.cat_streak >= self.frames_to_open:
            if self._opened_at is None:
                self._opened_at = self.clock()
            return OPEN

        if self.no_cat_streak >= self.frames_to_close:
            if self._opened_at is not None:
                if self.clock() - self._opened_at < self.min_open_seconds:
                    return HOLD
            self._opened_at = None
            return CLOSE

        return HOLD

    @property
    def streak(self) -> int:
        """Length of the current streak, whichever direction it runs in."""
        return max(self.cat_streak, self.no_cat_streak)
