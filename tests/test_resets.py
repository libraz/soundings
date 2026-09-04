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
