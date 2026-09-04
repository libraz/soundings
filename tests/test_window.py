"""Telling a store from a window, against machines that are one or the other.

The distinction cannot be made by writing to an address and reading it back: the
write points the window and the read follows it, so a window answers exactly as a
store would. That is not a hypothetical. It is how 25512 addresses of this unit
came to be recorded as storage.

So the fakes here are the two machines the verdict has to separate, and the third
case is the one that matters most: a run whose own stores read as windows has a
broken method and must say so rather than report what it found.

No hardware.
"""

from __future__ import annotations

from soundings import roland, window

ONE = (0x41, 0x04, 0x24)
TWO = (0x51, 0x04, 0x24)
WINDOW = (0x42, 0x04, 0x24)
OWN = (0x40, 0x01, 0x33)


class FakeUnit:
    """Two real stores, an address of its own, and a window onto the last one addressed."""

    def __init__(self, *, window_is_a_store: bool = False, stores_are_one: bool = False):
        self.memory = {ONE: 0x40, TWO: 0x40, OWN: 0x40, WINDOW: 0x40}
        self.facing = ONE
        self.window_is_a_store = window_is_a_store
        self.stores_are_one = stores_are_one

    def _resolve(self, address):
        if address in (ONE, TWO):
            # `stores_are_one` is the pathology that matters: two addresses that
            # are secretly the same storage agree with each other and read as
            # ordinary stores, while leaving no two values to tell anything apart.
            if self.stores_are_one:
                return ONE
            self.facing = address
            return address
        if address == WINDOW and not self.window_is_a_store:
            return self.facing
        return address

    def send(self, message: list[int]) -> None:
        if len(message) > 9 and message[4] == roland.CMD_DT1:
            address = (message[5], message[6], message[7])
            self.memory[self._resolve(address)] = message[8]

    def exchange(self, request: list[int], timeout: float = 0.0) -> list[int]:
        if len(request) < 11 or request[4] != roland.CMD_RQ1:
            return []
        address = (request[5], request[6], request[7])
        held = self.memory.get(self._resolve(address))
        return [] if held is None else roland.dt1(address, [held])

    def receive(self, timeout: float = 0.0) -> list[int]:
        return []


def run(unit: FakeUnit, candidates=(WINDOW, OWN)) -> window.Result:
    return window.Prober(unit, settle=0.0, read_timeout=0.0).run((ONE, TWO), list(candidates))


def verdict_for(result: window.Result, address: tuple[int, int, int]) -> str:
    text = f"{address[0]:02X} {address[1]:02X} {address[2]:02X}"
    return next(c.verdict for c in result.candidates if c.address == text)


def test_an_address_that_shows_the_last_one_addressed_is_named_a_window() -> None:
    result = run(FakeUnit())
    assert verdict_for(result, WINDOW) == window.IS_A_WINDOW


def test_an_address_that_answers_the_same_either_way_holds_its_own_value() -> None:
    result = run(FakeUnit())
    assert verdict_for(result, OWN) == window.HOLDS_ITS_OWN


def test_the_same_address_reads_as_a_store_on_a_machine_where_it_is_one() -> None:
    """The verdict has to come from the machine and not from the address, or it is
    a restatement of what was already believed."""
    result = run(FakeUnit(window_is_a_store=True))
    assert verdict_for(result, WINDOW) == window.HOLDS_ITS_OWN


def test_a_write_through_a_window_is_followed_to_where_it_landed() -> None:
    """Which store a write reaches is a separate question from which one a read
    reports, and the two need not agree."""
    result = run(FakeUnit())
    entry = next(c for c in result.candidates if c.address == "42 04 24")
    assert entry.write_reached == "41 04 24"


def test_two_stores_that_are_secretly_one_are_caught_before_anything_is_read() -> None:
    """The vacuity guard, against the pathology that hides best. Two addresses
    backed by one storage answer identically, so they read as perfectly ordinary
    stores -- and leave the run with no two values to tell any candidate apart,
    so every candidate reads as holding its own value. That is the same output as
    a run that genuinely found no windows."""
    result = run(FakeUnit(stores_are_one=True))
    assert not result.controls_held_their_own
    assert verdict_for(result, WINDOW) == window.HOLDS_ITS_OWN
    assert "never two values to tell anything apart" in window.summarise(result)


def test_the_stores_are_put_back() -> None:
    unit = FakeUnit()
    result = run(unit)
    assert result.restored
    assert unit.memory[ONE] == 0x40 and unit.memory[TWO] == 0x40
