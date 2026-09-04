"""Reading the archive back, against the archive this repository actually publishes."""

from __future__ import annotations

import json
from pathlib import Path

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


def test_a_prefix_takes_a_whole_top_byte_and_not_the_ones_that_share_a_digit():
    """'4' would sweep away 40 and 41 with the twelve blocks that hold nothing.
    Prefixes are matched against the written form, where the space after the top
    byte is what keeps two hex digits from meaning sixteen."""
    kept, skipped = archive.split_off_prefixes(
        ["40 04 24", "41 04 24", "42 04 24", "4F 04 24"], ["42", "4F"]
    )
    assert kept == ["40 04 24", "41 04 24"]
    assert skipped == ["42 04 24", "4F 04 24"]


def test_what_a_prefix_removed_comes_back_with_it():
    """The caveat travels with the result. A caller handed only the remainder has
    nothing to say about what it stopped covering."""
    kept, skipped = archive.split_off_prefixes(["20 00 00", "21 00 00"], [])
    assert kept == ["20 00 00", "21 00 00"]
    assert skipped == []


def test_the_addresses_a_scan_reached_are_read_from_the_scan(tmp_path):
    """Which addresses are worth asking a third way into is a finding about the
    unit. Held as a list beside the code it drifts from the records it came
    from, and it did: the list this replaced missed three addresses the same
    unit's scans had attributed since."""
    first = tmp_path / "cc.json"
    first.write_text(
        json.dumps(
            {
                "attributed": [
                    {"stimulus": "CC7", "stores_verbatim": ["40 11 19"]},
                    {"stimulus": "CC10", "stores_verbatim": ["40 11 1C", "40 11 19"]},
                ]
            }
        )
    )
    second = tmp_path / "nrpn.json"
    second.write_text(
        json.dumps({"attributed": [{"stimulus": "NRPN 1 8", "stores_verbatim": ["40 21 04"]}]})
    )

    assert archive.stores_reached([first, second]) == ["40 11 19", "40 11 1C", "40 21 04"]


def test_a_stimulus_that_landed_nowhere_contributes_no_address(tmp_path):
    """An attribution with nothing under it is a message the unit did not store,
    which is a result rather than an address to write to."""
    scan = tmp_path / "cc.json"
    scan.write_text(
        json.dumps(
            {
                "attributed": [
                    {"stimulus": "CC7", "stores_verbatim": []},
                    {"stimulus": "CC10", "stores_verbatim": None},
                ]
            }
        )
    )

    assert archive.stores_reached([scan]) == []


def test_a_bounded_mark_keeps_its_blocks_and_reports_everything_else():
    """A run confined to a few blocks and a run that excluded a few are two
    different claims. Reading which one was made from the order of two return
    values is how they get mixed up, so the two calls are separate."""
    addresses = ["20 00 00", "21 00 00", "40 11 19", "41 04 24"]
    kept, outside = archive.keep_only_prefixes(addresses, ["20", "21"])
    assert kept == ["20 00 00", "21 00 00"]
    assert outside == ["40 11 19", "41 04 24"]


def test_no_prefix_bounds_nothing_and_leaves_nothing_out():
    kept, outside = archive.keep_only_prefixes(["20 00 00"], [])
    assert kept == ["20 00 00"]
    assert outside == []
