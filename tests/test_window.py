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

import re

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


class SharedRegion:
    """A run of addresses backed by one storage: writing any of them writes it."""

    def __init__(self, start=(0x20, 0x00, 0x00), length=8, shared=True):
        self.start, self.length, self.shared = start, length, shared
        self.own = dict.fromkeys(range(length), 0x40)
        self.one = 0x40

    def _index(self, address):
        if address[:2] != self.start[:2]:
            return None
        i = address[2] - self.start[2]
        return i if 0 <= i < self.length else None

    def send(self, message: list[int]) -> None:
        if len(message) > 9 and message[4] == roland.CMD_DT1:
            i = self._index((message[5], message[6], message[7]))
            if i is None:
                return
            if self.shared:
                self.one = message[8]
            else:
                self.own[i] = message[8]

    def exchange(self, request: list[int], timeout: float = 0.0) -> list[int]:
        if len(request) < 11 or request[4] != roland.CMD_RQ1:
            return []
        address = (request[5], request[6], request[7])
        i = self._index(address)
        if i is None:
            return []
        return roland.dt1(address, [self.one if self.shared else self.own[i]])

    def receive(self, timeout: float = 0.0) -> list[int]:
        return []


def hold(unit) -> window.Held:
    prober = window.Prober(unit, settle=0.0, read_timeout=0.0)
    return window.hold_probe(prober, unit.start, unit.length)


def test_a_run_of_addresses_sharing_one_storage_answers_with_one_value() -> None:
    """The case a write-then-read probe cannot see. Every address here would pass
    that probe, because the write lands in the shared storage and the read that
    follows reports it."""
    result = hold(SharedRegion(shared=True))
    assert result.verdict == window.ONE_VALUE_BETWEEN_THEM
    assert result.distinct == 1


def test_addresses_that_each_hold_something_answer_with_different_values() -> None:
    result = hold(SharedRegion(shared=False))
    assert result.verdict == window.KEPT_ITS_OWN
    assert result.distinct == 2


def test_the_writes_all_happen_before_any_of_the_reads() -> None:
    """The order is the method. Reading each address straight after writing it
    would let a shared storage answer as though every address held its own."""
    unit = SharedRegion(shared=True)
    order = []
    send, exchange = unit.send, unit.exchange
    unit.send = lambda m: (order.append("w"), send(m))[1]
    unit.exchange = lambda r, timeout=0.0: (order.append("r"), exchange(r, timeout))[1]

    hold(unit)
    # Eight reads to snapshot, eight writes to give each a value, eight reads to
    # collect them, then eight writes to put them back and eight to check. No
    # write is followed by a read of the same address, which is the whole point.
    assert re.fullmatch(r"r{8}w{8}r{8}w{8}r{8}", "".join(order))


def test_the_originals_are_put_back() -> None:
    unit = SharedRegion(shared=False)
    result = hold(unit)
    assert result.restored
    assert set(unit.own.values()) == {0x40}


def test_a_clamped_neighbour_is_called_indistinguishable_and_not_shared() -> None:
    """An address bounded to one value answers with it whichever extreme it was
    given, so it cannot be told from its neighbour -- which is a fact about its
    range and not about whose storage it is. Calling that sharing would put most
    of a real address space in the wrong bucket."""

    class Clamped(SharedRegion):
        def __init__(self):
            super().__init__(length=4, shared=False)

        def send(self, message):
            if len(message) > 9 and message[4] == roland.CMD_DT1:
                i = self._index((message[5], message[6], message[7]))
                if i is None:
                    return
                # The first two are pinned at 40; the rest take what they are given.
                self.own[i] = 0x40 if i < 2 else message[8]

    result = hold(Clamped())
    assert result.verdict == window.SOME_INDISTINGUISHABLE
    assert result.indistinguishable_pairs == 1
    assert result.distinct > 1
