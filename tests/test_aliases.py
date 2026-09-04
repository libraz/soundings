"""The attribution rules, tested against a machine that does what the test says.

Every claim this project makes about where a parameter lives rests on the rule
in `Scanner.attribute`, and the rule is a filter: it is defined as much by what
it refuses as by what it keeps. A refusal that is never exercised is a comment.
So each case here builds a unit that behaves one specific way -- a byte that
follows, a byte that drifts, a byte that latches, a message that goes missing on
its first attempt -- and asserts what the scan says about it.

No hardware. The fake stands in for the link, not for the SC-8850: it is a
stated behaviour, which is what makes the assertion mean something.
"""

from __future__ import annotations

import pytest

from soundings import roland
from soundings.aliases import (
    Attribution,
    Scanner,
    Snapshotter,
    Stimulus,
    control_run,
    differences,
)

REGION = ((0x40, 0x11, 0x00), 4)


class FakeUnit:
    """A machine with four bytes, that answers RQ1 and reacts to what it is told.

    `react` is handed every message sent and may write into `memory`. That is the
    whole model: storage, and a rule about what reaches it.
    """

    def __init__(self, react, memory=(0, 0, 0, 0)):
        self.memory = list(memory)
        self.react = react
        self.sent: list[list[int]] = []

    def send(self, message: list[int]) -> None:
        self.sent.append(list(message))
        self.react(self, list(message))

    def exchange(self, request: list[int], timeout: float = 0.0) -> list[int]:
        parsed = _parse_rq1(request)
        if parsed is None:
            return []
        (top, mid, low), size = parsed
        data = self.memory[low : low + size]
        return roland.dt1((top, mid, low), data)

    def receive(self, timeout: float = 0.0) -> list[int]:
        return []


def _parse_rq1(raw: list[int]) -> tuple[tuple[int, int, int], int] | None:
    if len(raw) < 11 or raw[4] != roland.CMD_RQ1:
        return None
    address = (raw[5], raw[6], raw[7])
    size = (raw[8] << 14) | (raw[9] << 7) | raw[10]
    return address, size


def scanner_for(unit: FakeUnit, *, restless=frozenset()) -> Scanner:
    shot = Snapshotter(unit, [REGION], timeout=0.0)
    return Scanner(shot, restless=set(restless), settle=0.0)


def stimulus(**kw) -> Stimulus:
    return Stimulus(
        label=kw.get("label", "test"),
        kind=kw.get("kind", "cc"),
        build=lambda v: [[0xB0, 0x07, v]],
        values=kw.get("values", (0x20, 0x60)),
    )


def store_at(index: int):
    """A unit that keeps the value of a control change at one address."""

    def react(unit: FakeUnit, message: list[int]) -> None:
        if message[0] == 0xB0:
            unit.memory[index] = message[2]

    return react


def test_a_byte_that_follows_is_attributed_and_called_verbatim():
    unit = FakeUnit(store_at(2))
    hit = scanner_for(unit).attribute(stimulus())
    assert hit is not None
    assert hit.addresses == ["40 11 02"]
    assert hit.verbatim == ["40 11 02"]
    assert not hit.on_second_attempt


def test_a_byte_that_is_scaled_is_attributed_but_not_verbatim():
    def react(unit: FakeUnit, message: list[int]) -> None:
        if message[0] == 0xB0:
            unit.memory[1] = min(127, message[2] * 2)

    hit = scanner_for(FakeUnit(react)).attribute(stimulus())
    assert hit is not None
    assert hit.addresses == ["40 11 01"]
    assert hit.verbatim == []


def test_a_message_that_stores_nothing_is_not_attributed():
    assert scanner_for(FakeUnit(lambda unit, message: None)).attribute(stimulus()) is None


def test_a_byte_that_only_moves_out_is_not_attributed():
    """A latch is not storage. Something that takes a value and keeps it has
    changed once, which every rule here is built to refuse as evidence."""

    def react(unit: FakeUnit, message: list[int]) -> None:
        if message[0] == 0xB0 and unit.memory[3] == 0:
            unit.memory[3] = message[2]

    assert scanner_for(FakeUnit(react)).attribute(stimulus()) is None


def test_a_byte_that_moves_on_its_own_is_caught_by_the_control_run():
    """The exclusion list exists because a drifting byte follows every stimulus.

    Without it the drift is attributed to whichever message was in flight, and
    the attribution looks exactly like a real one.
    """

    class Drifting(FakeUnit):
        counter = 0

        def exchange(self, request, timeout=0.0):
            Drifting.counter += 1
            self.memory[0] = Drifting.counter % 128
            return super().exchange(request, timeout)

    unit = Drifting(store_at(2))
    shot = Snapshotter(unit, [REGION], timeout=0.0)
    restless = control_run(shot, rounds=2, settle=0.0)
    assert restless == {"40 11 00"}
    hit = Scanner(shot, restless=restless, settle=0.0).attribute(stimulus())
    assert hit is not None
    assert hit.addresses == ["40 11 02"], "the drifting byte must not be attributed"


def test_a_stimulus_missed_on_the_first_attempt_is_found_on_the_retry():
    """The failure the retry was added for: a prime that does not land, over a
    unit already holding the value the stimulus is about to move to."""
    state = {"dropped": False}

    def react(unit: FakeUnit, message: list[int]) -> None:
        if message[0] != 0xB0:
            return
        if not state["dropped"]:
            state["dropped"] = True
            return  # the prime never arrives
        unit.memory[2] = message[2]

    unit = FakeUnit(react, memory=(0, 0, 0x60, 0))
    scanner = scanner_for(unit)
    hit = scanner.attribute(stimulus())
    assert hit is not None, "a single pass reports this as absent"
    assert hit.on_second_attempt
    assert scanner.recovered == ["test"]
    assert hit.addresses == ["40 11 02"]


def test_a_region_that_cannot_be_read_is_left_out_rather_than_read_as_zero():
    unit = FakeUnit(store_at(2))
    unit.exchange = lambda request, timeout=0.0: []
    shot = Snapshotter(unit, [REGION], timeout=0.0)
    assert shot.take() == {}
    assert shot.unread == 1
    assert differences({}, {}) == []


def test_a_narrow_range_is_scanned_inside_itself():
    """A stimulus whose parameter clamps answers two out-of-range values
    identically, and reads as storing nothing. The range travels with it."""

    def react(unit: FakeUnit, message: list[int]) -> None:
        if message[0] == 0xB0:
            unit.memory[2] = min(7, message[2])

    unit = FakeUnit(react)
    assert scanner_for(unit).attribute(stimulus(values=(0x20, 0x60))) is None
    assert scanner_for(unit).attribute(stimulus(values=(0x01, 0x06))) is not None


def test_a_stimulus_that_does_not_put_the_unit_back_is_named():
    """Residue is the thing a control run cannot see: it is not drift, it is one
    message changing what the next one means."""

    def react(unit: FakeUnit, message: list[int]) -> None:
        if message[0] == 0xB0:
            unit.memory[2] = message[2]
            unit.memory[3] += 1  # never comes back

    scanner = scanner_for(FakeUnit(react))
    scanner.attribute(stimulus())
    assert scanner.residue == {"test": ["40 11 03"]}


class LatchedBank(FakeUnit):
    """The SC-8850's bank select, as measured.

    Bank select is held and changes nothing readable. A program change commits
    the triple, and the unit takes it only if that tone exists: otherwise the
    part keeps what it had and no byte moves at all. Byte 0 is the bank MSB,
    byte 1 the program, byte 2 the map that CC32 chose.
    """

    EXISTS = {(0, 0), (0, 48), (8, 0)}

    def __init__(self):
        super().__init__(self._react, memory=(0, 0, 0, 0))
        self.msb = 0
        self.lsb = 0

    @staticmethod
    def _react(unit: LatchedBank, message: list[int]) -> None:
        if message[0] == 0xB0 and message[1] == 0:
            unit.msb = message[2]
        elif message[0] == 0xB0 and message[1] == 32:
            unit.lsb = message[2]
        elif message[0] == 0xC0:
            if (unit.msb, message[1]) in unit.EXISTS:
                unit.memory[0] = unit.msb
                unit.memory[1] = message[1]
                unit.memory[2] = unit.lsb


def test_a_latch_left_by_one_stimulus_does_not_silence_the_next():
    """The regression this was written for. Driving only the half under test
    leaves the other half set, and every later program change is discarded --
    silently, because the latch is not in the address space being read."""
    from soundings.aliases import bank_then_program

    unit = LatchedBank()
    scanner = scanner_for(unit)

    # Sets the bank MSB to 3, which no program exists in.
    assert scanner.attribute(bank_then_program(0, 0)) is None

    hit = scanner.attribute(bank_then_program(0, 32))
    assert hit is not None, "the map select must survive the previous stimulus"
    assert hit.addresses == ["40 11 02"]

    only_one_half = Stimulus(
        label="one half",
        kind="channel",
        build=lambda v: [[0xB0, 32, v], [0xC0, 0]],
        values=(0x00, 0x03),
    )
    unit.msb = 3
    assert scanner_for(unit).attribute(only_one_half) is None, "and this is what the old form did"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_a_family_that_landed_somewhere_is_never_reported_as_unreached():
    """An attribution is itself proof the message arrived, so a run that finds
    something cannot also say nothing of its kind got through. It did: the drum
    NRPN scan landed 4 stimuli across 13 blocks and announced the opposite,
    because the name the stimuli were selected by was matched against the kind
    they are recorded as, and those are different words for different things."""
    from soundings.cli.scan import _kind_reached

    stimuli = [Stimulus(label="NRPN 1C 24", kind="nrpn", build=lambda v: [], values=(0x20, 0x60))]
    landed = [Attribution(label="NRPN 1C 24", kind="nrpn", values=(0x20, 0x60))]

    assert _kind_reached("drum-nrpn", stimuli, landed)
    assert _kind_reached("nrpn", stimuli, landed)


def test_a_family_that_landed_nothing_is_still_reported_as_unreached():
    """The guard has to keep working, or a family the unit never received reads
    as a family the unit does not store."""
    from soundings.cli.scan import _kind_reached

    stimuli = [Stimulus(label="NRPN 1C 24", kind="nrpn", build=lambda v: [], values=(0x20, 0x60))]

    assert not _kind_reached("drum-nrpn", stimuli, [])
    # A control change is its own proof, being what the positive control is.
    assert _kind_reached("cc", stimuli, [])


def test_something_landing_that_this_run_did_not_send_does_not_count():
    """Only the stimuli sent can show the path carried them."""
    from soundings.cli.scan import _kind_reached

    stimuli = [Stimulus(label="NRPN 1C 24", kind="nrpn", build=lambda v: [], values=(0x20, 0x60))]
    stray = [Attribution(label="CC7", kind="cc", values=(0x20, 0x60))]

    assert not _kind_reached("drum-nrpn", stimuli, stray)
