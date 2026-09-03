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

    def __init__(self, memory: dict[tuple[int, int, int], int], *, deaf_after: int | None = None):
        self.memory = dict(memory)
        self.deaf_after = deaf_after
        self.reads = 0
        self.written: list[tuple[tuple[int, int, int], int]] = []

    def send(self, message: list[int]) -> None:
        if len(message) > 9 and message[4] == roland.CMD_DT1:
            address = (message[5], message[6], message[7])
            self.written.append((address, message[8]))
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


def test_a_resumed_run_skips_only_the_regions_already_finished() -> None:
    regions = [((0x40, 0x01, 0x30), 24), ((0x40, 0x02, 0x00), 16), ((0x40, 0x03, 0x00), 32)]
    kept = [{"start": "40 02 00", "length": 16}]
    assert space._still_to_do(regions, kept) == [
        ((0x40, 0x01, 0x30), 24),
        ((0x40, 0x03, 0x00), 32),
    ]


def test_a_region_probed_at_a_different_length_is_not_taken_as_done() -> None:
    """Matching on the address alone would leave the tail of the longer region
    unprobed while the file said the address had been covered."""
    regions = [((0x40, 0x01, 0x30), 24)]
    kept = [{"start": "40 01 30", "length": 8}]
    assert space._still_to_do(regions, kept) == regions


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
