"""
Lock abstraction.

Currently an in-memory state plus console output. Wiring a real lock (GPIO,
a relay, a smart-home HTTP call) only changes the bodies of open() and
close(); nothing else in the project needs to know.
"""


class DoorController:
    def __init__(self, verbose: bool = True):
        self._is_open = False
        self.verbose = verbose

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def state(self) -> str:
        return "OPEN" if self._is_open else "CLOSED"

    def open(self) -> None:
        if not self._is_open:
            self._is_open = True
            if self.verbose:
                print("DOOR: OPEN")

    def close(self) -> None:
        if self._is_open:
            self._is_open = False
            if self.verbose:
                print("DOOR: CLOSED")
