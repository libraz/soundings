"""Reading past a mapped region, to see whether it ends where the map says.

The result is a list of negatives, so what is under test is mostly the bound on
them: an address that answers nothing to a single-byte read has not been shown
not to exist, and a run that lost the unit has not shown anything at all.

No hardware.
"""

from __future__ import annotations

from soundings import boundary


def test_the_address_past_a_region_is_the_one_the_unit_would_answer_at() -> None:
    """The two blocks that turned out short are the worked example."""
    assert boundary.unpack(boundary.past_the_end("40 10 00", 26)) == "40 10 1A"
    assert boundary.unpack(boundary.past_the_end("40 42 00", 2)) == "40 42 02"


def test_an_address_carries_seven_bits_a_byte_not_eight() -> None:
    """A region ending at 7F rolls into the next middle byte, not into 80."""
    assert boundary.unpack(boundary.past_the_end("40 01 7E", 2)) == "40 02 00"
    assert boundary.unpack(boundary.past_the_end("40 7F 7F", 1)) == "41 00 00"


def test_a_region_that_answered_nothing_past_its_end_says_only_that() -> None:
    found = [boundary.Region(address="40 01 00", mapped_size=40)]
    result = boundary.summarise(found, "40 01 00", deaf=False)
    assert result["regions_reaching_past_their_mapped_end"] == 0
    assert "single-byte read" in result["note"]
    assert "first_address_past_the_end" not in result["regions"][0]


def test_a_region_that_answered_past_its_end_names_where_and_what() -> None:
    region = boundary.Region(address="40 10 00", mapped_size=26)
    region.stopped_at = "40 10 1A"
    region.answered_beyond = 2
    region.values = ["40", "40"]
    result = boundary.summarise([region], "40 01 00", deaf=False)
    assert result["addresses_found_that_way"] == 2
    assert result["regions"][0]["first_address_past_the_end"] == "40 10 1A"
    assert result["regions"][0]["values"] == ["40", "40"]


def test_a_run_that_lost_the_unit_says_so_rather_than_reporting_a_clean_map() -> None:
    """Every question here is a negative, and a silent unit answers all of them."""
    result = boundary.summarise([boundary.Region("40 01 00", 40)], "40 01 00", deaf=True)
    assert result["stopped"] == boundary.WENT_DEAF
    assert "stops talking" in result["positive_control"]["why"]


def test_a_resumed_scan_keeps_the_blocks_a_canary_answered_for() -> None:
    blocks = [boundary.Block(address=f"40 4{d} 00", asked=128) for d in range(3)]
    blocks[1].answered["40 41 20"] = "01"
    record = boundary.scanned(blocks, "40 01 30", deaf=False)
    assert [b.address for b in boundary.restore_blocks(record)] == [
        "40 40 00",
        "40 41 00",
        "40 42 00",
    ]
    assert boundary.restore_blocks(record)[1].answered == {"40 41 20": "01"}


def test_a_resumed_scan_asks_again_the_block_the_canary_stopped_on() -> None:
    """Its silences are what a unit that has stopped talking produces."""
    blocks = [boundary.Block(address=f"40 4{d} 00", asked=128) for d in range(3)]
    record = boundary.scanned(blocks, "40 01 30", deaf=True)
    assert [b.address for b in boundary.restore_blocks(record)] == ["40 40 00", "40 41 00"]


def test_a_scan_that_stopped_on_its_first_block_resumes_with_nothing() -> None:
    record = boundary.scanned([boundary.Block(address="40 40 00", asked=128)], "40 01 30", True)
    assert boundary.restore_blocks(record) == []
