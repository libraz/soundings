"""Sorting a marked byte into what the reset did to it.

The rule is a partition, and a partition is only worth the byte it cannot lose.
Two of the outcomes are not verdicts at all -- a byte the snapshot never reached,
and a byte with nothing to be compared against -- and both were dropped silently
for as long as the probe existed, which is why the counts in three published runs
came out two short of what each said it had marked.

No hardware.
"""

from __future__ import annotations

from soundings.resets import ResetResult, compare, read_gap_again

A = (0x21, 0x04, 0x24)
B = (0x21, 0x04, 0x25)
C = (0x21, 0x04, 0x26)
D = (0x21, 0x04, 0x27)
E = (0x21, 0x04, 0x28)


def result() -> ResetResult:
    return ResetResult(label="GS Reset", message="F0", note="")


def test_every_marked_byte_lands_in_exactly_one_outcome():
    """The property the two new lists exist for. Anything else is a byte broken
    on purpose whose fate the record does not state."""
    marked = dict.fromkeys((A, B, C, D, E), 0x2A)
    after = {A: 0x40, B: 0x2A, C: 0x7F, E: 0x40}  # D was not read back
    baseline = {A: 0x40, B: 0x40, C: 0x40, D: 0x40}  # E has no power-on value

    out = result()
    compare(out, marked, after, baseline)

    assert out.restored == ["21 04 24"]
    assert out.left_marked == {"21 04 25": 0x2A}
    assert out.changed_to_something_else == {"21 04 26": [0x2A, 0x7F]}
    assert out.not_read_afterwards == ["21 04 27"]
    assert out.missing_from_the_baseline == ["21 04 28"]

    accounted = (
        len(out.restored)
        + len(out.left_marked)
        + len(out.changed_to_something_else)
        + len(out.not_read_afterwards)
        + len(out.missing_from_the_baseline)
    )
    assert accounted == len(marked)


def test_a_byte_that_was_never_marked_can_still_differ_from_power_on():
    """The second tally is over the whole readable map, so a reset is measured by
    what it changed as well as by what it put back."""
    out = result()
    compare(out, {}, {A: 0x01, B: 0x40}, {A: 0x40, B: 0x40})
    assert out.differs_from_power_on == {"21 04 24": [0x40, 0x01]}


def test_the_shortfall_in_a_saved_run_is_stated_from_its_own_counts():
    payload = {
        "resets": [
            {
                "reset": "GS Reset",
                "bytes_marked": 10,
                "restored_to_the_power_on_value": ["a", "b"],
                "left_holding_the_mark": {"c": "2A"},
                "changed_to_neither": {},
            }
        ]
    }
    assert read_gap_again(payload) == {"GS Reset": 7}
    assert payload["marked_without_an_outcome"]["by_reset"] == {"GS Reset": 7}


def test_a_run_that_accounted_for_everything_reports_no_shortfall():
    payload = {
        "resets": [
            {
                "reset": "GS Reset",
                "bytes_marked": 3,
                "restored_to_the_power_on_value": ["a"],
                "left_holding_the_mark": {},
                "changed_to_neither": {},
                "not_read_afterwards": ["b"],
                "missing_from_the_baseline": ["c"],
            }
        ]
    }
    assert read_gap_again(payload) == {"GS Reset": 0}


def test_a_byte_the_two_reads_disagreed_about_is_left_out_rather_than_picked():
    """Deciding between them needs a third read, and a wrong decision is
    invisible afterwards: the byte reads as having been changed by whichever
    reset is measured against this capture next."""
    from soundings.resets import agreed_bytes

    first = {(0x40, 0x00, 0x00): 0x10, (0x40, 0x00, 0x01): 0x20}
    second = {(0x40, 0x00, 0x00): 0x10, (0x40, 0x00, 0x01): 0x7F}

    agreed, disagreed, once = agreed_bytes(first, second)

    assert agreed == {(0x40, 0x00, 0x00): 0x10}
    assert disagreed == [(0x40, 0x00, 0x01)]
    assert once == []


def test_a_byte_only_one_read_answered_is_neither_agreed_nor_disagreed():
    """A region that failed to read once has not been shown to differ from
    itself. Counting it as a disagreement would report a lost reply as a unit
    that moved on its own."""
    from soundings.resets import agreed_bytes

    agreed, disagreed, once = agreed_bytes({(0x40, 0x00, 0x00): 0x10}, {})

    assert agreed == {}
    assert disagreed == []
    assert once == [(0x40, 0x00, 0x00)]


def test_a_capture_says_that_the_power_cycle_is_a_claim_and_not_a_measurement():
    """Nothing in the run can tell a unit fresh from the mains switch from one
    an earlier run wrote to, and a later reader has no way to recover which it
    was. The record carries the assertion as an assertion."""
    from soundings.resets import CAPTURED_IS_ASSERTED, power_on_record

    record = power_on_record(
        {(0x40, 0x00, 0x00): 0x10},
        {(0x40, 0x00, 0x00): 0x10},
        unit_id="roland-sc88pro-01",
        identity_reply="F0 7E 10 06 02 F7",
        captured="immediately after a power cycle",
        regions_read=2,
        regions_unread=0,
    )

    assert record["captured_is_asserted_not_measured"] == CAPTURED_IS_ASSERTED
    assert record["captured"] == "immediately after a power cycle"
    assert record["unit_id"] == "roland-sc88pro-01"
    assert record["values"] == {"40 00 00": "10"}
    assert record["read_disagreed_at"] == []


def test_a_channel_mode_message_is_addressed_to_a_part_and_says_so():
    """Every other subject is addressed to the unit. A label that did not carry
    the channel would put three runs on three different parts under one name."""
    from soundings.resets import channel_mode

    subject = channel_mode("Reset All Controllers", 9, 121)

    assert subject.message == [0xB9, 121, 0x00]
    assert subject.label == "Reset All Controllers on channel 10"


def test_the_channel_mode_subjects_are_the_three_with_no_opposite():
    """122, 124, 125, 126 and 127 name states, and a state is what the alias scan
    can measure. Sending them here as well would measure them twice by the
    weaker of the two methods."""
    from soundings.resets import channel_mode_catalogue

    controllers = {r.message[1] for r in channel_mode_catalogue(0)}
    assert controllers == {120, 121, 123}


def test_a_channel_mode_subject_carries_what_the_mark_does_not_cover():
    """The specification defines these over controller values and the mark is
    written by SysEx, so a message that restores an internal value never written
    back into the address space reads here as restoring nothing."""
    from soundings.resets import CHANNEL_MODE_NOTE, channel_mode_catalogue

    assert all(r.note == CHANNEL_MODE_NOTE for r in channel_mode_catalogue(0))
