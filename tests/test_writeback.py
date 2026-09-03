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
from soundings.cli import space


class FakeUnit:
    """A byte of memory per address, answering RQ1 until it is told to go deaf."""

    def __init__(
        self,
        memory: dict[tuple[int, int, int], int],
        *,
        deaf_after: int | None = None,
        drop_writes: set[int] | None = None,
    ):
        self.memory = dict(memory)
        self.deaf_after = deaf_after
        self.drop_writes = drop_writes or set()
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
    assert space._still_to_do(regions, kept) == [
        ((0x40, 0x01, 0x30), 24),
        ((0x40, 0x03, 0x00), 32),
    ]


def test_a_region_probed_at_a_different_length_is_not_taken_as_done() -> None:
    """Matching on the address alone would leave the tail of the longer region
    unprobed while the file said the address had been covered."""
    regions = [((0x40, 0x01, 0x30), 24)]
    kept = [{"start": "40 01 30", "length": 8, "region_restored": True}]
    assert space._still_to_do(regions, kept) == regions


def test_the_region_a_run_stopped_in_is_measured_again_and_not_duplicated() -> None:
    """It is written out for its evidence but it is not coverage: it stopped part
    way through, so a resume re-probes it whole and the partial entry gives way."""
    regions = [((0x40, 0x01, 0x30), 24), ((0x40, 0x02, 0x00), 16)]
    kept = [
        {"start": "40 01 30", "length": 24, "region_restored": True},
        {"start": "40 02 00", "length": 16, "region_restored": False},
    ]
    todo = space._still_to_do(regions, kept)
    assert todo == [((0x40, 0x02, 0x00), 16)]
    assert space._superseded(kept, todo) == [kept[0]]


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
    assert space._probe_regions(args) == [((0x50, 0x00, 0x00), 2), ((0x40, 0x01, 0x30), 8)]
