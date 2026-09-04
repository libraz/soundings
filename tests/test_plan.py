"""Choosing the pair an address is asked at, from what the write probe measured.

The failure guarded here is a pair the unit clamps to one byte. Both settings
land on the same value, the two sets of takes are takes of one setting, and the
run reports a parameter that does nothing -- which is the same shape as a real
null and carries no sign of being anything else.

No hardware: what is under test is the reading of a write-probe record.
"""

from __future__ import annotations

from soundings import plan


def probe(*rows, skipped=()) -> dict:
    return {"regions": [{"start": "40 11 00", "bytes": list(rows), "skipped": list(skipped)}]}


def row(address: str, original: str, spec: str, classification: str = "accepts") -> dict:
    return {
        "address": address,
        "original": original,
        "range": spec,
        "classification": classification,
    }


def test_the_pair_is_the_two_ends_of_the_measured_range() -> None:
    """A value outside what the address accepts is clamped to one the other
    setting may already hold, and two settings that arrive as one byte read as a
    parameter that does nothing."""
    asks, _ = plan.plan_block(probe(row("40 11 02", "00", "00..10", "clamps")), "40 11")

    assert [a.values for a in asks] == [(0, 16)]


def test_the_setting_nearer_the_power_on_value_is_asked_first() -> None:
    """It fixes the reference every other take is aligned against, and the
    power-on value is the one setting known to make a sound. A reference with no
    signal in it leaves the alignment correlating noise against noise."""
    asks, _ = plan.plan_block(probe(row("40 11 19", "64", "00..7F")), "40 11")

    assert asks[0].values == (127, 0)


def test_a_power_on_value_at_the_low_end_leads_from_there() -> None:
    """The rule has to be able to choose either end, or it is the low end wearing
    a justification."""
    asks, _ = plan.plan_block(probe(row("40 11 1D", "00", "00..7F")), "40 11")

    assert asks[0].values == (0, 127)


def test_an_address_that_accepts_one_value_is_reported_rather_than_dropped() -> None:
    """It cannot be asked, and that is a result: an address with no second
    setting is not a parameter with a setting, whatever else it is. Dropping it
    would leave the block looking one address shorter than it is."""
    asks, skipped = plan.plan_block(probe(row("40 11 17", "08", "08..08", "unchanging")), "40 11")

    assert asks == []
    assert [(s.address, s.why) for s in skipped] == [("40 11 17", plan.ONE_VALUE)]


def test_a_range_the_probe_never_established_is_asked_over_the_whole_span() -> None:
    """The probe skips an address it cannot read back, having nothing to restore.
    Leaving it out would silently shorten the block; asking it needs the caveat,
    because a null there cannot separate an address that ignored the write from
    one that took it and reached nothing."""
    asks, _ = plan.plan_block(probe(skipped=["40 11 01 (original could not be read)"]), "40 11")

    assert [(a.address, a.values) for a in asks] == [("40 11 01", (0, 127))]
    assert not asks[0].range_established
    assert asks[0].caveat == plan.RANGE_UNREAD


def test_the_trailing_words_on_a_range_do_not_stop_it_being_read() -> None:
    """A row that refuses out-of-range values records "00..02 of the values
    tried", and the words are part of what it established."""
    asks, _ = plan.plan_block(
        probe(row("40 11 15", "00", "00..02 of the values tried", "refuses out of range")), "40 11"
    )

    assert asks[0].values == (0, 2)
    assert asks[0].accepted_range == "00..02 of the values tried"


def test_an_address_outside_the_block_is_not_planned() -> None:
    """The prefix is what says which part is being asked, and a part parameter
    written to the wrong part answers about a part nobody listened to."""
    asks, skipped = plan.plan_block(probe(row("40 12 19", "64", "00..7F")), "40 11")

    assert asks == [] and skipped == []


def test_every_address_in_the_block_is_either_asked_or_said_not_to_be() -> None:
    """The two lists have to account for the block, or an address can go missing
    between them and the plan reads as complete."""
    record = probe(
        row("40 11 03", "01", "00..01", "clamps"),
        row("40 11 17", "08", "08..08", "unchanging"),
        skipped=["40 11 18 (original could not be read)"],
    )

    asks, skipped = plan.plan_block(record, "40 11")

    assert sorted([a.address for a in asks] + [s.address for s in skipped]) == [
        "40 11 03",
        "40 11 17",
        "40 11 18",
    ]


def test_the_caveat_travels_in_the_json() -> None:
    """The record is the archive, so a caveat the JSON drops did not happen."""
    asks, _ = plan.plan_block(probe(skipped=["40 11 01 (original could not be read)"]), "40 11")

    assert plan.RANGE_UNREAD in asks[0].to_json()["caveat"]
