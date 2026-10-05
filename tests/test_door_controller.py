from src.runtime.door_controller import DoorController


def test_starts_closed():
    door = DoorController(verbose=False)
    assert door.is_open is False
    assert door.state == "CLOSED"


def test_open_and_close_flip_state():
    door = DoorController(verbose=False)
    door.open()
    assert door.state == "OPEN"
    door.close()
    assert door.state == "CLOSED"


def test_repeated_calls_are_idempotent():
    door = DoorController(verbose=False)
    door.open()
    door.open()
    assert door.is_open is True
    door.close()
    door.close()
    assert door.is_open is False
