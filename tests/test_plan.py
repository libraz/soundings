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


def test_a_parameter_printed_with_values_is_asked_inside_them() -> None:
    """The failure the measured range cannot catch.

    A block that stores and returns every seven-bit value at every address never
    clamps, so the guard the range is trusted for never fires -- and a parameter
    printed with six values asked at nought against a hundred and twenty-seven is
    two writes the engine reaches one setting with. On this unit not one such
    parameter was ever heard that way, and two of them answered at once when asked
    at two of their own.
    """
    record = probe(row("40 11 02", "03", "00..7F"))

    asks, _ = plan.plan_block(record, "40 11", {"40 11 02": "00/01/02/03/04/05"})

    assert asks[0].values == (5, 0), "the value nearer the power-on setting leads"
    assert asks[0].printed_values == "00/01/02/03/04/05"
    assert asks[0].values_from == plan.FROM_THE_PRINTED_VALUES
    assert asks[0].accepted_range == "00..7F", "what it accepts is still what was measured"


def test_a_parameter_whose_printed_values_are_not_a_run_from_nought_is_asked_at_them() -> None:
    """Two states are not always nought and one.

    A rotor's speed switch is printed as the bytes nought and 127, so the pair the
    ends of its accepted range give it is exactly its two settings -- and a rule
    counting the names between the slashes and asking at nought against one would
    have taken a slot that was asked correctly and asked it at a value it has none
    of, which is the same defect in the other direction.
    """
    record = probe(row("40 11 0D", "00", "00..7F"))

    asks, _ = plan.plan_block(record, "40 11", {"40 11 0D": "00/7F"})

    assert asks[0].values == (0, 127)
    assert asks[0].values_from == plan.FROM_THE_PRINTED_VALUES


def test_a_parameter_whose_values_are_a_run_between_two_ends_is_asked_at_those() -> None:
    """A tone gain is printed 34-4C and its setting column names no states at all,
    so nothing that counts names would ever have looked at it."""
    record = probe(row("40 11 13", "40", "00..7F"))

    asks, _ = plan.plan_block(record, "40 11", {"40 11 13": "34–4C"})

    assert asks[0].values == (52, 76)


def test_a_parameter_pointed_at_a_conversion_table_keeps_the_measured_pair() -> None:
    """That column gives a setting at every one of the 128 values, so it narrows
    nothing and the row must not say a page decided its pair."""
    record = probe(row("40 11 04", "00", "00..7F"))

    asks, _ = plan.plan_block(record, "40 11", {"40 11 04": "*6"})

    assert asks[0].values == (0, 127)
    assert asks[0].values_from == plan.FROM_THE_RANGE
    assert "printed_values" not in asks[0].to_json()


#: One column of the conversion grid wraps: it reaches the same place from both
#: directions and the page says so in each of the two cells rather than leaving it
#: to be noticed. Trimmed to the values a pair is ever chosen from.
WRAPS = {0: "L180(=R180)", 64: "0", 127: "R180(=L180)"}


def test_a_column_that_comes_back_to_itself_is_not_asked_at_both_of_its_ends() -> None:
    """The two ends of that byte are one setting, and the takes of one setting
    differ by nothing -- which is exactly what the run reports for a parameter that
    does nothing. One such pair is published, conclusive, and wrong about the
    unit."""
    record = probe(row("40 11 04", "40", "00..7F"))

    asks, _ = plan.plan_block(record, "40 11", {"40 11 04": "*13"}, {"40 11 04": WRAPS})

    assert asks[0].values == (64, 127)
    assert asks[0].values_from == plan.FROM_THE_POWER_ON_VALUE


def test_a_column_whose_ends_are_two_settings_keeps_the_pair_its_range_gives() -> None:
    """The rule has to leave the other thirteen columns alone, or it is a rewrite of
    every referred parameter wearing one column's justification."""
    record = probe(row("40 11 04", "40", "00..7F"))
    rates = {0: "0.05", 64: "2.60", 127: "10.00"}

    asks, _ = plan.plan_block(record, "40 11", {"40 11 04": "*6"}, {"40 11 04": rates})

    assert asks[0].values == (127, 0)
    assert asks[0].values_from == plan.FROM_THE_RANGE


def test_a_wrapped_column_whose_power_on_is_also_that_setting_cannot_be_asked() -> None:
    """The replacement is the unit's own power-on value, so where that is printed as
    the setting the ends already give, there is no pair left to name and saying so
    beats asking it somewhere chosen for the sake of having a pair."""
    record = probe(row("40 11 04", "00", "00..7F"))

    _, skipped = plan.plan_block(record, "40 11", {"40 11 04": "*13"}, {"40 11 04": WRAPS})

    assert [s.why for s in skipped] == [plan.ENDS_ARE_ONE_SETTING]


def test_a_printed_value_the_address_will_not_take_is_not_written() -> None:
    """A page and a unit disagreeing is a finding, not a licence to write past the
    range this unit was measured to accept."""
    asks, _ = plan.plan_block(
        probe(row("40 11 02", "00", "00..02")), "40 11", {"40 11 02": "00/01/02/03/04/05"}
    )

    assert asks[0].values == (0, 2)


def test_an_address_with_no_printed_values_keeps_the_measured_pair() -> None:
    """The page's cell is handed in per address, so a block carrying both kinds has
    to put each row on its own source and say which."""
    record = probe(row("40 11 02", "00", "00..7F"), row("40 11 03", "00", "00..7F"))

    asks, _ = plan.plan_block(record, "40 11", {"40 11 03": "00/01"})

    assert [(a.address, a.values) for a in asks] == [("40 11 02", (0, 127)), ("40 11 03", (0, 1))]
    assert asks[0].values_from == plan.FROM_THE_RANGE
    assert asks[0].to_json()["values_from"] == plan.FROM_THE_RANGE
    assert "printed_values" not in asks[0].to_json()


def test_a_parameter_printed_with_one_value_cannot_be_asked() -> None:
    """However much its address accepts. Reported rather than dropped, for the
    same reason an address accepting one value is."""
    record = probe(row("40 11 02", "00", "00..7F"))

    asks, skipped = plan.plan_block(record, "40 11", {"40 11 02": "00–00"})

    assert asks == []
    assert [(s.address, s.why) for s in skipped] == [("40 11 02", plan.ONE_PRINTED_VALUE)]


def test_an_address_accepting_none_of_its_printed_values_is_not_asked() -> None:
    """Writing past the measured range to reach a printed value would be the page
    overruling the unit, on a stage whose whole subject is what the unit does."""
    record = probe(row("40 11 02", "00", "00..02"))

    asks, skipped = plan.plan_block(record, "40 11", {"40 11 02": "34–4C"})

    assert asks == []
    assert [s.why for s in skipped] == [plan.NONE_OF_ITS_PRINTED_VALUES]


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
