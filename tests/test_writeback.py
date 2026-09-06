"""The guards a run long enough to be left alone needs.

The write probe is the one command that changes the unit, and over the whole
address space it runs for hours with nobody watching it. Three things then
matter that do not matter in a two minute run: that it never writes where it
was told not to, that a unit which stops talking stops the run instead of being
walked past, and that an interruption costs the last region rather than all of
them.

No hardware. The fake is a stated behaviour, not a model of the SC-8850.
"""

from __future__ import annotations

import argparse

from soundings import roland, writeback
from soundings.cli import contents


class FakeUnit:
    """A byte of memory per address, answering RQ1 until it is told to go deaf."""

    def __init__(
        self,
        memory: dict[tuple[int, int, int], int],
        *,
        deaf_after: int | None = None,
        drop_writes: set[int] | None = None,
        takes=None,
    ):
        self.memory = dict(memory)
        self.deaf_after = deaf_after
        self.drop_writes = drop_writes or set()
        # What the address does with a value it is given: None means it keeps
        # whatever it is sent, otherwise the value is kept only if `takes` says so
        # and the address is left as it was if not -- a refusal, not a clamp.
        self.takes = takes
        self.reads = 0
        self.writes = 0
        self.written: list[tuple[tuple[int, int, int], int]] = []

    def send(self, message: list[int]) -> None:
        if len(message) > 9 and message[4] == roland.CMD_DT1:
            self.writes += 1
            address = (message[5], message[6], message[7])
            self.written.append((address, message[8]))
            if self.writes in self.drop_writes:
                return
            if self.takes is not None and not self.takes(message[8]):
                return
            if address in self.memory:
                self.memory[address] = message[8]

    def exchange(self, request: list[int], timeout: float = 0.0) -> list[int]:
        self.reads += 1
        if self.deaf_after is not None and self.reads > self.deaf_after:
            return []
        if len(request) < 11 or request[4] != roland.CMD_RQ1:
            return []
        address = (request[5], request[6], request[7])
        if address not in self.memory:
            return []
        return roland.dt1(address, [self.memory[address]])

    def receive(self, timeout: float = 0.0) -> list[int]:
        return []


def writer_for(unit: FakeUnit) -> writeback.Writer:
    return writeback.Writer(unit, settle=0.0, read_timeout=0.0)


def test_the_never_write_address_is_never_sent_a_byte() -> None:
    """The list is the whole of the protection, so its effect is asserted rather
    than assumed: System Mode Set reinitialises the unit, which would discard the
    state the probe is in the middle of restoring."""
    mode_set = (0x40, 0x00, 0x7F)
    assert mode_set in writeback.NEVER_WRITE
    unit = FakeUnit({(0x40, 0x00, 0x7D): 0, (0x40, 0x00, 0x7E): 0, mode_set: 0})
    result = writer_for(unit).probe_region((0x40, 0x00, 0x7D), 3)

    assert not any(address == mode_set for address, _ in unit.written)
    assert "40 00 7F (never-write list)" in result.skipped


def test_a_byte_that_cannot_be_read_is_never_written_to() -> None:
    """There would be nothing to put back, so the address is left alone. This is
    what makes a run over addresses of unknown purpose safe to leave alone."""
    unit = FakeUnit({(0x40, 0x01, 0x30): 0x04})
    result = writer_for(unit).probe_region((0x40, 0x01, 0x30), 4)

    assert {a for a, _ in unit.written} == {(0x40, 0x01, 0x30)}
    assert result.skipped == [
        "40 01 31 (original could not be read)",
        "40 01 32 (original could not be read)",
        "40 01 33 (original could not be read)",
    ]


def test_a_unit_that_stops_talking_is_not_mistaken_for_silent_addresses() -> None:
    """A deaf unit answers nothing, every byte reads as unreadable, and unreadable
    bytes are skipped rather than written to -- so without a canary the probe
    walks the rest of the map producing empty regions and exits clean."""
    canary = (0x40, 0x01, 0x30)
    unit = FakeUnit({canary: 0x04}, deaf_after=0)
    writer = writer_for(unit)

    assert not writer.answering(canary)

    talking = FakeUnit({canary: 0x04})
    assert writer_for(talking).answering(canary)


def test_a_selector_resting_at_its_lowest_value_is_still_read_as_refusing() -> None:
    """The measured shape at 00 01 xx: 64 one-of-four selectors, all taking 00 to
    03 and refusing everything above. The 48 resting at 01, 02 or 03 were read as
    refusing, and the 16 resting at 00 were filed as indistinguishable -- with 01,
    02 and 03 recorded in their own rows going in and coming back."""
    address = (0x00, 0x01, 0x00)
    unit = FakeUnit({address: 0x00}, takes=lambda v: v <= 0x03)
    probe = writer_for(unit).probe_byte(address, 0x00)

    assert probe.classification == "refuses out of range"
    assert probe.accepted == [0x00, 0x01, 0x02, 0x03]
    assert probe.range == "00..03 of the values tried"


def test_a_sibling_resting_higher_is_read_the_same_way() -> None:
    """Same address kind, same verdict: the value it happened to be holding is
    not supposed to decide what it is."""
    address = (0x00, 0x01, 0x10)
    unit = FakeUnit({address: 0x01}, takes=lambda v: v <= 0x03)
    probe = writer_for(unit).probe_byte(address, 0x01)

    assert probe.classification == "refuses out of range"
    assert probe.accepted == [0x00, 0x01, 0x02, 0x03]


def test_an_address_that_only_ever_takes_one_value_stays_undecided() -> None:
    """The ladder overturns the verdict only when it finds something. An address
    that takes nothing but the value it rests at leaves a clamp to that value and
    a refusal of everything else genuinely indistinguishable, and neither is
    asserted."""
    address = (0x00, 0x01, 0x00)
    unit = FakeUnit({address: 0x00}, takes=lambda v: v == 0x00)
    probe = writer_for(unit).probe_byte(address, 0x00)

    assert probe.classification == writeback.UNDECIDABLE
    assert probe.accepted == [0x00]


def test_a_clamping_address_keeps_its_verdict_through_the_ladder() -> None:
    """Of 11544 addresses filed as clamping, every one the ladder pushed past its
    bound read that bound back, so the re-read must leave them alone."""
    address = (0x40, 0x01, 0x32)
    unit = FakeUnit({address: 0x04})
    unit.send = _clamping(unit, address, 0x07)
    probe = writer_for(unit).probe_byte(address, 0x04)

    assert probe.classification == "clamps"
    assert probe.range == "00..07"


def _clamping(unit: FakeUnit, address: tuple[int, int, int], bound: int):
    """A byte that holds the edge of its range rather than refusing what is past it."""

    def send(message: list[int]) -> None:
        if len(message) > 9 and message[4] == roland.CMD_DT1:
            unit.writes += 1
            unit.written.append(((message[5], message[6], message[7]), message[8]))
            unit.memory[address] = min(message[8], bound)

    return send


def test_a_record_read_back_carries_everything_the_verdict_needs() -> None:
    """The verdict is a reading of the rows, so a run already made can be read
    again -- but only if what comes back off disk is the probe that was written."""
    address = (0x00, 0x01, 0x10)
    unit = FakeUnit({address: 0x01}, takes=lambda v: v <= 0x03)
    probe = writer_for(unit).probe_byte(address, 0x01)
    again = writeback.ByteProbe.from_json(probe.to_json())

    assert again.to_json() == probe.to_json()
    assert again.accepted == probe.accepted
    assert again.range == probe.range
    assert (again.low, again.high) == (probe.low, probe.high)


def test_reading_a_saved_run_again_gives_what_measuring_it_again_would() -> None:
    """The claim the re-reading rests on: a probe writes the same sequence
    whatever it concludes, so the rows are the whole of the measurement and the
    verdict above them is derived. Applying the rule twice must change nothing."""
    address = (0x00, 0x01, 0x00)
    unit = FakeUnit({address: 0x00}, takes=lambda v: v <= 0x03)
    probe = writer_for(unit).probe_byte(address, 0x00)
    payload = {"regions": [{"bytes": [probe.to_json()], "classifications": {}}]}

    assert writeback.read_verdicts_again(payload) == 0
    assert payload["regions"][0]["bytes"][0] == probe.to_json()


def test_a_run_saved_under_the_old_rule_is_brought_up_to_the_new_one() -> None:
    """The 50 addresses filed as indistinguishable while their own rows told them
    apart. The rows are untouched; only the reading of them moves."""
    address = (0x00, 0x01, 0x00)
    unit = FakeUnit({address: 0x00}, takes=lambda v: v <= 0x03)
    stale = writer_for(unit).probe_byte(address, 0x00).to_json()
    rows = [list(r) for r in stale["wrote_read"]]
    stale["classification"] = writeback.UNDECIDABLE
    stale["range"] = "00..00"
    payload = {"regions": [{"bytes": [stale], "classifications": {}}]}

    assert writeback.read_verdicts_again(payload) == 1
    got = payload["regions"][0]["bytes"][0]
    assert got["classification"] == "refuses out of range"
    assert got["range"] == "00..03 of the values tried"
    assert got["wrote_read"] == rows
    assert payload["regions"][0]["classifications"] == {"refuses out of range": 1}


def test_a_lost_write_is_retried_rather_than_stopping_the_run() -> None:
    """Measured on the unit: one write in 213328 did not arrive and the address
    kept the last value that did, which stopped a whole run. A lost message is
    not a fact about the address, so the restore is given more than one go."""
    address = (0x46, 0x09, 0x1C)
    unit = FakeUnit({address: 0x00})
    writer = writer_for(unit)
    # The restore is the sixth write: 00, 7F, then three interior values, then it.
    unit.drop_writes = {6}
    probe = writer.probe_byte(address, 0x00)

    assert probe.restored
    assert probe.restore_tries == 2
    assert unit.memory[address] == 0x00
    assert probe.to_json()["restore_tries"] == 2


def test_a_restore_that_never_takes_still_stops_the_run() -> None:
    """The retry is for a lost message, not a licence to carry on regardless."""
    address = (0x46, 0x09, 0x1C)
    unit = FakeUnit({address: 0x00}, drop_writes=set(range(6, 40)))
    writer = writer_for(unit)
    probe = writer.probe_byte(address, 0x00)

    assert not probe.restored
    assert probe.restore_tries == 0


def test_a_run_that_stops_keeps_the_bytes_measured_before_the_failure() -> None:
    """They were measured and restored like any others, and they are the only
    record of what led up to the failure."""
    good, bad = (0x46, 0x09, 0x00), (0x46, 0x09, 0x01)
    unit = FakeUnit({good: 0x00, bad: 0x00})
    writer = writer_for(unit)
    # Everything from the second byte's restore onwards goes missing. Six writes
    # per byte: 00, 7F, three interior values, then the original back.
    unit.drop_writes = set(range(12, 60))

    try:
        writer.probe_region((0x46, 0x09, 0x00), 2)
    except writeback.RestoreFailed as exc:
        assert exc.region is not None
        assert [b.address for b in exc.region.bytes] == ["46 09 00", "46 09 01"]
        assert exc.region.bytes[0].restored
    else:
        raise AssertionError("the failed restore should have stopped the region")


def test_a_resumed_run_skips_only_the_regions_already_finished() -> None:
    regions = [((0x40, 0x01, 0x30), 24), ((0x40, 0x02, 0x00), 16), ((0x40, 0x03, 0x00), 32)]
    kept = [{"start": "40 02 00", "length": 16, "region_restored": True}]
    assert contents._still_to_do(regions, kept) == [
        ((0x40, 0x01, 0x30), 24),
        ((0x40, 0x03, 0x00), 32),
    ]


def test_a_region_probed_at_a_different_length_is_not_taken_as_done() -> None:
    """Matching on the address alone would leave the tail of the longer region
    unprobed while the file said the address had been covered."""
    regions = [((0x40, 0x01, 0x30), 24)]
    kept = [{"start": "40 01 30", "length": 8, "region_restored": True}]
    assert contents._still_to_do(regions, kept) == regions


def test_the_region_a_run_stopped_in_is_measured_again_and_not_duplicated() -> None:
    """It is written out for its evidence but it is not coverage: it stopped part
    way through, so a resume re-probes it whole and the partial entry gives way."""
    regions = [((0x40, 0x01, 0x30), 24), ((0x40, 0x02, 0x00), 16)]
    kept = [
        {"start": "40 01 30", "length": 24, "region_restored": True},
        {"start": "40 02 00", "length": 16, "region_restored": False},
    ]
    todo = contents._still_to_do(regions, kept)
    assert todo == [((0x40, 0x02, 0x00), 16)]
    assert contents._superseded(kept, todo) == [kept[0]]


def test_prefixes_order_the_regions_as_well_as_filtering_them(tmp_path) -> None:
    """What an interrupted run leaves behind is decided entirely by the order it
    met the regions in, so the order asked for is the order probed."""
    path = tmp_path / "map.json"
    path.write_text(
        '{"regions": ['
        '{"address": "41 00 00", "size": 4},'
        '{"address": "40 01 30", "size": 8},'
        '{"address": "50 00 00", "size": 2}]}'
    )
    args = argparse.Namespace(regions=[], map=str(path), prefix=["50 ", "40 "])
    assert contents._probe_regions(args) == [((0x50, 0x00, 0x00), 2), ((0x40, 0x01, 0x30), 8)]
