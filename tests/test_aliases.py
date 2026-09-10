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


def test_every_universal_message_is_a_well_formed_exclusive():
    """A byte over 7F inside a system exclusive ends the message early, so the
    unit receives something shorter than what was meant and the stimulus is a
    null about a message that was never sent."""
    from soundings.aliases import universal_stimuli

    for stimulus in universal_stimuli(9, 36):
        for value in stimulus.values:
            (message,) = stimulus.build(value)
            assert message[0] == 0xF0 and message[-1] == 0xF7, stimulus.label
            assert all(b < 0x80 for b in message[1:-1]), stimulus.label
            assert message[1] in (0x7E, 0x7F), stimulus.label
            assert message[2] == 0x7F, "addressed to every device"


def test_the_byte_that_varies_is_the_last_one_before_the_terminator():
    """Which byte moves is the whole stimulus. A global parameter control puts
    its slot path first for exactly this reason, and getting it wrong varies a
    slot index while reporting a parameter."""
    from soundings.aliases import universal_stimuli

    for stimulus in universal_stimuli(9, 36):
        low, high = (stimulus.build(v)[0] for v in stimulus.values)
        assert low[-2] != high[-2], stimulus.label
        assert low[:-2] == high[:-2], stimulus.label


def test_a_stimulus_whose_two_values_are_equal_could_detect_nothing():
    from soundings.aliases import universal_stimuli

    for stimulus in universal_stimuli(9, 36):
        assert stimulus.values[0] != stimulus.values[1], stimulus.label


def test_the_key_based_controls_carry_the_channel_and_note_they_were_given():
    """The only universal message that addresses one drum note, and so the only
    one whose target moves with the caller's arguments rather than being fixed."""
    from soundings.aliases import universal_stimuli

    keyed = [s for s in universal_stimuli(9, 0x31) if s.label.startswith("key-based")]
    assert len(keyed) == 4
    for stimulus in keyed:
        message = stimulus.build(0x20)[0]
        assert message[3:5] == [0x0A, 0x01], stimulus.label
        assert message[5] == 9 and message[6] == 0x31, stimulus.label
        assert "note 49" in stimulus.label


def _write_hit(label: str, *addresses: str) -> Attribution:
    return Attribution(
        label=label, kind="address", values=(0x20, 0x60), readings=dict.fromkeys(addresses, [])
    )


def test_a_write_seen_under_another_top_byte_is_named():
    """The control a mirror scan makes on itself. Twelve of this unit's blocks
    follow a write to `41`, so a run that includes one has shown it can report a
    landing outside the block written to, and its other negatives mean something."""
    from soundings.aliases import landed_outside_its_own_block

    found = [
        _write_hit("DT1 41 04 24", "41 04 24", "42 04 24", "4F 04 24"),
        _write_hit("DT1 21 04 24", "21 04 24"),
    ]
    assert landed_outside_its_own_block(found) == {"DT1 41 04 24": ["42 04 24", "4F 04 24"]}


def test_a_run_where_every_write_stayed_home_reports_no_control():
    """Empty is the finding, not the absence of one: nothing showed the scan could
    have reported a mirror, so 'nothing else moved' is about the scan."""
    from soundings.aliases import landed_outside_its_own_block

    assert landed_outside_its_own_block([_write_hit("DT1 40 11 32", "40 11 32")]) == {}


def test_a_stimulus_that_is_not_an_address_write_has_no_block_to_be_outside_of():
    from soundings.aliases import landed_outside_its_own_block

    hit = Attribution(label="CC7", kind="cc", values=(0x20, 0x60), readings={"40 11 19": []})
    assert landed_outside_its_own_block([hit]) == {}


def test_a_saved_scan_is_given_the_control_by_the_same_rule():
    from soundings.aliases import WHY_LANDED_OUTSIDE, read_outside_again

    payload = {
        "attributed": [
            {
                "stimulus": "DT1 41 04 24",
                "kind": "address",
                "bytes": {"41 04 24": [], "42 04 24": []},
            }
        ]
    }
    assert read_outside_again(payload) == {"DT1 41 04 24": ["42 04 24"]}
    assert payload["landed_outside_its_own_block"]["why"] == WHY_LANDED_OUTSIDE


def test_something_landing_that_this_run_did_not_send_does_not_count():
    """Only the stimuli sent can show the path carried them."""
    from soundings.cli.scan import _kind_reached

    stimuli = [Stimulus(label="NRPN 1C 24", kind="nrpn", build=lambda v: [], values=(0x20, 0x60))]
    stray = [Attribution(label="CC7", kind="cc", values=(0x20, 0x60))]

    assert not _kind_reached("drum-nrpn", stimuli, stray)


def test_addresses_named_on_the_command_line_are_not_topped_up_from_a_record():
    """The run narrows the list to what it could read and put back, and writes
    the remainder back into the same field. Adding a record's addresses every
    time would return the dropped ones after they had been dropped."""
    import argparse

    from soundings.cli.scan import _addresses

    args = argparse.Namespace(addresses=["40 11 19"], addresses_from=["never read"])
    assert _addresses(args) == [(0x40, 0x11, 0x19)]


def test_a_kind_address_run_with_nowhere_to_write_refuses():
    """It refuses rather than defaulting to a list of addresses. Which ones are
    worth asking a third way into is a finding about the unit, and a default
    would make one unit's finding the starting point for every other."""
    import argparse

    import pytest as _pytest

    from soundings.cli.scan import _addresses

    args = argparse.Namespace(addresses=[], addresses_from=[])
    with _pytest.raises(SystemExit):
        _addresses(args)


def test_a_mode_pair_sends_one_controller_at_one_value_and_another_at_the_other():
    """The state is named by two controllers, so what the value selects is which
    message goes out. Sent as two separate stimuli each would land nothing and
    read as two dead controllers rather than as one setting."""
    from soundings.aliases import mode_pair

    stimulus = mode_pair("omni on or off", 9, (125, 0), (124, 0))

    assert stimulus.values == (0, 1)
    assert stimulus.build(0) == [[0xB9, 125, 0]]
    assert stimulus.build(1) == [[0xB9, 124, 0]]


def test_the_normal_state_is_the_value_the_scan_primes_and_returns_to():
    """Omni off and mono mode change how every later channel message is received.
    Priming with the deviant member would put the run in that state for most of
    its length, and the return leg would leave it there."""
    from soundings.aliases import mode_stimuli

    low, high = {}, {}
    for stimulus in mode_stimuli(0):
        low[stimulus.label] = stimulus.build(stimulus.values[0])
        high[stimulus.label] = stimulus.build(stimulus.values[1])

    assert low["local control on or off"] == [[0xB0, 122, 0x7F]]
    assert low["omni on or off"] == [[0xB0, 125, 0]]
    assert low["poly or mono"] == [[0xB0, 127, 0]]
    assert high["local control on or off"] == [[0xB0, 122, 0x00]]
    assert high["omni on or off"] == [[0xB0, 124, 0]]
    assert high["poly or mono"] == [[0xB0, 126, 1]]


def test_the_three_action_mode_messages_are_not_sent_by_this_scan():
    """120, 121 and 123 have one value and no opposite, so the out-and-back
    reports them as landing nothing whatever the unit does with them. Sending
    them here would manufacture three nulls and publish them as findings."""
    from soundings.aliases import mode_stimuli

    sent = {
        message[1]
        for stimulus in mode_stimuli(0)
        for value in stimulus.values
        for message in stimulus.build(value)
    }
    assert sent == {122, 124, 125, 126, 127}
    assert not sent & {120, 121, 123}


def test_a_mode_run_carries_the_gap_it_left_and_not_the_one_it_closed():
    """The cc scan's caveat says the channel mode messages need a run of their
    own. Carried into that run it would say the work still had to be done."""
    from soundings.aliases import MODE_NOT_SCANNED, NOT_SCANNED, not_scanned

    assert not_scanned("mode") == MODE_NOT_SCANNED
    assert not_scanned("cc") == NOT_SCANNED
    assert not_scanned("universal") == NOT_SCANNED


def test_the_state_put_back_afterwards_is_the_low_value_of_every_pair():
    """Whatever the retry left, the unit ends where the stimuli found it. Held
    against the stimuli rather than written out twice, so a pair whose normal
    member changed cannot leave the restoring trio behind."""
    from soundings.aliases import MODE_NORMAL, mode_stimuli

    normal = [stimulus.build(stimulus.values[0])[0] for stimulus in mode_stimuli(0)]
    assert sorted((m[1], m[2]) for m in normal) == sorted(MODE_NORMAL)


class ShortReplyUnit(FakeUnit):
    """A unit that answers a region read with fewer bytes than it was asked for.

    Measured, not invented: one block answered a thirty-two byte request with
    thirty bytes and a checksum that verified, reproducibly, and the two it left
    out were not the two that go silent when the same block is asked an offset
    at a time.
    """

    def __init__(self, react, memory=(0, 0, 0, 0), *, drop=1):
        super().__init__(react, memory)
        self.drop = drop

    def exchange(self, request: list[int], timeout: float = 0.0) -> list[int]:
        parsed = _parse_rq1(request)
        if parsed is None:
            return []
        (top, mid, low), size = parsed
        data = self.memory[low : low + size]
        return roland.dt1((top, mid, low), data[: len(data) - self.drop])


def test_a_reply_shorter_than_the_request_is_not_laid_down_over_the_addresses():
    """Counting up from the start puts every byte after the gap one address low.

    Worse than a region nobody read: a missing byte is visibly missing, while a
    byte on the wrong address is a value a later stage compares and publishes.
    """
    unit = ShortReplyUnit(lambda u, m: None, memory=(0x11, 0x22, 0x33, 0x44))
    shot = Snapshotter(unit, [REGION], timeout=0.0)
    assert shot.take() == {}
    assert shot.unread == 1
    assert shot.short == [((0x40, 0x11, 0x00), 4, 3)]


def test_a_reply_of_the_length_asked_for_is_still_laid_down():
    """The refusal is about the length disagreeing, not about region reads."""
    unit = FakeUnit(lambda u, m: None, memory=(0x11, 0x22, 0x33, 0x44))
    shot = Snapshotter(unit, [REGION], timeout=0.0)
    assert shot.take() == {
        (0x40, 0x11, 0x00): 0x11,
        (0x40, 0x11, 0x01): 0x22,
        (0x40, 0x11, 0x02): 0x33,
        (0x40, 0x11, 0x03): 0x44,
    }
    assert shot.unread == 0
    assert shot.short == []
