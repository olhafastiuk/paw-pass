from src.runtime.debounce import CLOSE, HOLD, OPEN, FrameDebouncer, frame_is_cat


class FakeClock:
    """Lets a test move time forward by hand instead of sleeping."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make(frames_to_open=3, frames_to_close=3, min_open_seconds=0.0, clock=None):
    return FrameDebouncer(frames_to_open, frames_to_close,
                          min_open_seconds, clock or FakeClock())


def feed(debouncer, frames):
    """Feed a sequence of verdicts and return the list of actions."""
    return [debouncer.update(is_cat) for is_cat in frames]


# --- streaks ---------------------------------------------------------------

def test_single_cat_frame_does_not_open():
    assert make().update(True) == HOLD


def test_opens_after_enough_consecutive_cat_frames():
    assert feed(make(), [True, True, True]) == [HOLD, HOLD, OPEN]


def test_interrupted_streak_restarts():
    d = make(frames_to_open=3, frames_to_close=5)
    assert OPEN not in feed(d, [True, True, False, True, True])


def test_closes_only_after_enough_empty_frames():
    d = make(frames_to_open=2, frames_to_close=3)
    feed(d, [True, True])
    assert feed(d, [False, False]) == [HOLD, HOLD]
    assert d.update(False) == CLOSE


def test_brief_disappearance_does_not_close():
    """One lost frame must not slam the door on a cat that is still there."""
    d = make(frames_to_open=2, frames_to_close=5)
    feed(d, [True, True])
    assert CLOSE not in feed(d, [False, False, True, False, False, True])


def test_streak_reports_current_run():
    d = make(frames_to_open=10, frames_to_close=10)
    feed(d, [True, True, True])
    assert d.streak == 3
    d.update(False)
    assert d.streak == 1


# --- minimum open time -----------------------------------------------------

def test_stays_open_for_minimum_time():
    clock = FakeClock()
    d = make(frames_to_open=2, frames_to_close=2, min_open_seconds=3.0, clock=clock)
    feed(d, [True, True])                 # opens at t = 0

    clock.now = 1.0
    assert feed(d, [False, False]) == [HOLD, HOLD]   # enough frames, too early


def test_closes_once_minimum_time_has_passed():
    clock = FakeClock()
    d = make(frames_to_open=2, frames_to_close=2, min_open_seconds=3.0, clock=clock)
    feed(d, [True, True])

    clock.now = 3.5
    assert feed(d, [False, False]) == [HOLD, CLOSE]


def test_minimum_time_counts_from_opening_not_from_last_cat_frame():
    clock = FakeClock()
    d = make(frames_to_open=2, frames_to_close=2, min_open_seconds=3.0, clock=clock)
    feed(d, [True, True])                 # opens at t = 0
    clock.now = 2.0
    feed(d, [True, True])                 # cat still there, not a new opening
    clock.now = 3.1
    assert feed(d, [False, False]) == [HOLD, CLOSE]


# --- hysteresis ------------------------------------------------------------

def test_closed_door_needs_the_upper_threshold():
    assert frame_is_cat(0.45, door_open=False, threshold=0.5, release_threshold=0.3) is False
    assert frame_is_cat(0.55, door_open=False, threshold=0.5, release_threshold=0.3) is True


def test_open_door_keeps_believing_down_to_release_threshold():
    assert frame_is_cat(0.45, door_open=True, threshold=0.5, release_threshold=0.3) is True
    assert frame_is_cat(0.25, door_open=True, threshold=0.5, release_threshold=0.3) is False


# Confidence hovering around 0.5, the failure seen on a live YOLO run:
# three frames just under the bar, three just over, repeated.
HOVERING = [0.55, 0.56, 0.54] + [0.48, 0.47, 0.49, 0.52, 0.53, 0.51] * 3


def simulate(scores, threshold, release_threshold):
    """Run scores through hysteresis + debouncer; return the door changes."""
    d = make(frames_to_open=3, frames_to_close=3)
    door_open = False
    changes = []

    for score in scores:
        action = d.update(frame_is_cat(score, door_open, threshold, release_threshold))
        if action == OPEN and not door_open:
            door_open = True
            changes.append(OPEN)
        elif action == CLOSE and door_open:
            door_open = False
            changes.append(CLOSE)

    return changes


def test_single_threshold_flickers():
    """Control case: with no gap between the thresholds the door flips."""
    assert simulate(HOVERING, 0.5, 0.5).count(CLOSE) >= 2


def test_hysteresis_stops_the_flicker():
    assert simulate(HOVERING, 0.5, 0.3) == [OPEN]
