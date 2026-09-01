"""Reading the archive back, against the archive this repository actually publishes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soundings import archive

UNIT = Path(__file__).parents[1] / "data" / "units" / "roland-sc8850-01"
MAP = UNIT / "address-map.json"
WRITE_PROBE = UNIT / "write-probe.json"


def test_every_region_in_the_map_is_read():
    published = [r for r in json.loads(MAP.read_text())["regions"] if r["size"]]
    assert len(archive.regions(MAP)) == len(published)


def test_a_prefix_selects_a_subset_and_nothing_else():
    everything = archive.regions(MAP)
    under_40 = archive.regions(MAP, "40 ")
    assert under_40, "the map has no 40 block, so this test is checking nothing"
    assert len(under_40) < len(everything)
    assert all(start[0] == 0x40 for start, _ in under_40)


def test_a_region_is_a_start_address_and_a_size():
    for start, size in archive.regions(MAP):
        assert len(start) == 3
        assert all(0 <= b <= 0xFF for b in start)
        assert size > 0


def test_zero_sized_regions_are_left_out():
    """A region with no size has no bytes to snapshot, and would read as an empty diff."""
    assert all(size for _, size in archive.regions(MAP))


def test_accepting_bytes_are_the_ones_a_mark_would_survive():
    found = archive.accepting_bytes(WRITE_PROBE)
    assert found
    published = {
        b["address"]: b
        for region in json.loads(WRITE_PROBE.read_text())["regions"]
        for b in region["bytes"]
    }
    for address in found:
        assert published[address]["classification"] == "accepts"
        assert published[address]["restored"] is True


def test_a_byte_that_was_not_restored_is_never_offered_as_a_target():
    found = set(archive.accepting_bytes(WRITE_PROBE))
    unrestored = {
        b["address"]
        for region in json.loads(WRITE_PROBE.read_text())["regions"]
        for b in region["bytes"]
        if not b["restored"]
    }
    assert not (found & unrestored)


@pytest.mark.parametrize("address", archive.ALIASED_BYTES)
def test_the_aliased_bytes_are_three_hex_bytes(address: str):
    assert len(address.split()) == 3
    assert all(0 <= int(b, 16) <= 0x7F for b in address.split())
