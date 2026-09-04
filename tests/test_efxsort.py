"""Sorting insertion effects by what they did to the unit's own repeatability.

The failure guarded here is the same one the delay-tracked route guards, arriving
by a different door. A type that reached nothing -- unrouted, or asked under a
note that could not hear it -- repeats exactly as well at both settings, which is
what a type standing still looks like. Filing it as static would be a verdict on
the routing wearing the effect's name.

No hardware and no audio: what is under test is the reading of a contrast record.
"""

from __future__ import annotations

import json

import pytest

from soundings import efxsort


def record(
    *,
    audible: bool = True,
    repeatability: bool = False,
    shape: bool = True,
    inconclusive: bool = False,
    unrepeatable: list[float] | None = None,
    stimulus: str = "struck_kit",
) -> dict:
    return {
        "by_stimulus": [
            {
                "stimulus_name": stimulus,
                "audible": audible,
                "inconclusive": inconclusive,
                "changed_the_shape": shape,
                "changed_the_level": False,
                "changed_the_repeatability": repeatability,
                "each_setting_unrepeatable_db": unrepeatable or [-60.0, -60.0],
            }
        ]
    }


def test_a_type_whose_takes_stopped_repeating_is_sorted_as_moving() -> None:
    """A free-running modulator is at a different phase every strike, so the takes
    made with it on disagree where the bypassed ones agree. That asymmetry is the
    evidence; no residual can show the change itself."""
    found = efxsort.sort_one(record(repeatability=True, unrepeatable=[-61.9, -3.8]), "02 01")

    assert found is not None
    assert found.verdict == "moves"
    assert found.heard_by == "repeatability"


def test_an_audible_type_that_repeats_alike_at_both_settings_stands_still() -> None:
    """The positive pile. Without it the sort has nowhere to put a filter."""
    found = efxsort.sort_one(record(unrepeatable=[-43.6, -47.6]), "01 00")

    assert found is not None and found.verdict == "static"


def test_a_type_that_changed_nothing_audible_is_sorted_as_neither() -> None:
    """It reached nothing, so it holds no evidence about a modulator. An effect
    that never touched the signal repeats as well as one standing still."""
    found = efxsort.sort_one(record(audible=False, shape=False), "03 00")

    assert found is not None and found.verdict == "could not say"


def test_a_type_whose_yardstick_swallowed_everything_is_sorted_as_neither() -> None:
    """A run nothing could have cleared is not a null. Reporting it as static
    would state a fact about the effect on a measurement with no power to find
    one."""
    found = efxsort.sort_one(record(audible=False, shape=False, inconclusive=True), "04 00")

    assert found is not None and found.verdict == "could not say"


def test_a_record_with_no_stimulus_is_passed_over() -> None:
    """A run that asked nothing produced no evidence, and inventing a verdict from
    an empty record would put a type in a pile it was never measured for."""
    assert efxsort.sort_one({"by_stimulus": []}, "05 00") is None


def test_a_type_asked_under_several_notes_moves_if_any_of_them_saw_it() -> None:
    """The same union rule that makes audible-under-any audible. A note that could
    not hear the modulator says nothing about the modulator."""
    two = record(repeatability=False, stimulus="struck")
    two["by_stimulus"].append(
        record(repeatability=True, unrepeatable=[-58.0, -5.0], stimulus="struck_kit")[
            "by_stimulus"
        ][0]
    )

    found = efxsort.sort_one(two, "06 00")

    assert found is not None and found.verdict == "moves"
    assert found.stimulus == "struck_kit"


def test_the_three_piles_account_for_every_type() -> None:
    found = [
        efxsort.sort_one(record(repeatability=True), "01 00"),
        efxsort.sort_one(record(), "02 00"),
        efxsort.sort_one(record(audible=False, shape=False), "03 00"),
    ]
    piles = efxsort.partition(found)

    assert sum(len(v) for v in piles.values()) == 3
    assert set(piles["moving"]) & set(piles["static"]) == set()


class Tracked:
    def __init__(self, type_id: str, verdict: str) -> None:
        self.type_id, self.verdict = type_id, verdict


def test_the_two_routes_disagreeing_is_reported_and_not_reconciled() -> None:
    """They rest on different properties: tracking can miss a motion it cannot
    follow, and this route would read anything else that stops a take repeating
    as a modulator. Averaging them would hide both failures."""
    here = [efxsort.sort_one(record(repeatability=True), "01 00")]

    assert efxsort.disagreements(here, [Tracked("01 00", "static")]) == [
        {"type": "01 00", "by_repeatability": "moves", "by_delay_track": "static"}
    ]


def test_one_route_declining_is_not_a_disagreement() -> None:
    """ "Could not say" is a route standing aside, which the other is free to
    answer. Counting it as a clash would fill the record with the delay track's
    known blind spots."""
    here = [efxsort.sort_one(record(repeatability=True), "01 00")]

    assert efxsort.disagreements(here, [Tracked("01 00", "could not say")]) == []
    assert efxsort.disagreements(here, []) == []


def test_a_directory_of_records_is_sorted_by_the_type_its_file_is_named_for(tmp_path) -> None:
    """The type comes from the filename the capture wrote, so a record cannot be
    attributed to a type it does not hold."""
    (tmp_path / "02-01.json").write_text(json.dumps(record(repeatability=True)))
    (tmp_path / "01-00.json").write_text(json.dumps(record()))

    found = efxsort.survey(tmp_path)

    assert [f.type_id for f in found] == ["01 00", "02 01"]
    assert [f.verdict for f in found] == ["static", "moves"]


@pytest.mark.parametrize("verdict", ["moves", "static", "could not say"])
def test_every_verdict_survives_the_json_round_trip(verdict: str) -> None:
    """The record is the archive, so a verdict the JSON drops would not exist."""
    made = {
        "moves": record(repeatability=True),
        "static": record(),
        "could not say": record(audible=False, shape=False),
    }[verdict]

    assert efxsort.sort_one(made, "07 00").to_json()["verdict"] == verdict
