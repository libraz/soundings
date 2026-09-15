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


def test_a_reading_inside_the_calibration_gap_is_not_filed_as_standing_still() -> None:
    """The modulator bar sits between a real chorus and the worst a pure level
    change reached on this chain. A setting landing between those two is evidence
    for neither, and filing it as static would state a fact about the effect on
    the strength of a number nobody can interpret."""
    found = efxsort.sort_one(record(unrepeatable=[-50.1, -21.0]), "01 25")

    assert found is not None
    assert found.inside_the_gap
    assert found.verdict == "could not say"


def test_a_reading_far_under_the_bar_is_still_static() -> None:
    """The gap has to have an outside, or every asymmetric reading becomes
    undecided and the static pile empties."""
    found = efxsort.sort_one(record(unrepeatable=[-50.1, -32.8]), "01 02")

    assert found is not None and not found.inside_the_gap
    assert found.verdict == "static"


def test_a_type_that_moved_is_not_reconsidered_against_the_gap() -> None:
    """It cleared the bar, so the band below the bar says nothing about it."""
    found = efxsort.sort_one(record(repeatability=True, unrepeatable=[-41.5, -1.7]), "01 20")

    assert found is not None and not found.inside_the_gap
    assert found.verdict == "moves"


def test_the_gap_is_a_reason_inside_could_not_say_and_not_a_fourth_pile() -> None:
    """A type in the gap has to be counted exactly once, or the piles stop
    accounting for the types."""
    found = [
        efxsort.sort_one(record(unrepeatable=[-50.1, -21.0]), "01 25"),
        efxsort.sort_one(record(audible=False, shape=False), "03 00"),
    ]
    piles = efxsort.partition(found)

    assert sum(len(v) for v in piles.values()) == len(found)
    assert efxsort.inside_the_gap(found) == ["01 25"]
    assert set(efxsort.inside_the_gap(found)) <= set(piles["could_not_say"])


def test_a_pile_with_nothing_inaudible_in_it_says_nothing_about_being_inaudible() -> None:
    """The defect this exists to keep out, which reached the archive once.

    One reason was published over the whole undecided pile, and the pile held a
    single type that was plainly audible and undecided for the calibration's
    sake. A reader taking the pile's sentence was told that type changed nothing
    the note could hear, beside a row of the same record reading `audible: true`.
    It is the shape a ladder gets when the caveat naming a recovered band is
    emitted over rows saying every rung failed, and it is fixed the same way:
    the reason belongs to the ground, not to the pile.
    """
    found = [efxsort.sort_one(record(unrepeatable=[-50.1, -21.0]), "01 25")]

    reasons = efxsort.grounds(found)

    assert [g["ground"] for g in reasons] == ["inside the calibration gap"]
    assert reasons[0]["types"] == ["01 25"]
    assert reasons[0]["why"] == efxsort.WHY_INSIDE_THE_GAP
    assert efxsort.WHY_NOT_AUDIBLE not in [g["why"] for g in reasons]


def test_every_undecided_type_is_counted_under_exactly_one_ground() -> None:
    """The grounds have to account for the pile and not overlap it. A type can
    satisfy two of them at once -- inaudible and inside the gap -- and counting
    it twice would make the reasons add up to more types than are undecided."""
    found = [
        efxsort.sort_one(record(unrepeatable=[-50.1, -21.0]), "01 25"),
        efxsort.sort_one(record(audible=False, shape=False, unrepeatable=[-50.1, -21.0]), "03 00"),
        efxsort.sort_one(record(inconclusive=True), "01 10"),
        efxsort.sort_one(record(repeatability=True, unrepeatable=[-61.9, -3.8]), "02 01"),
    ]
    undecided = efxsort.partition(found)["could_not_say"]

    named = [t for reason in efxsort.grounds(found) for t in reason["types"]]

    assert sorted(named) == sorted(undecided)
    assert len(named) == len(set(named))


def test_a_type_that_was_decided_is_given_no_ground_at_all() -> None:
    """The field says why a type is undecided, so a decided one carries nothing:
    a ground on a type with a verdict would read as a doubt nobody measured."""
    moving = efxsort.sort_one(record(repeatability=True, unrepeatable=[-61.9, -3.8]), "02 01")
    static = efxsort.sort_one(record(unrepeatable=[-60.0, -60.0]), "01 00")

    assert moving is not None and static is not None
    assert moving.ground is None and static.ground is None
    assert moving.to_json()["undecided_because"] is None


def test_an_effect_that_makes_the_takes_agree_better_is_not_a_modulator() -> None:
    """The asymmetry has to run the right way. The audible module's flag does not
    care which setting repeated worse, because for a parameter in general either
    value could switch something on; here the settings are known and only the
    routed one can carry a modulator.

    A guard rather than a correction: no type measured on this unit reaches the
    modulator bar in this direction, so nothing recorded would have been sorted
    wrongly without it.
    """
    found = efxsort.sort_one(record(repeatability=True, unrepeatable=[-15.0, -45.0]), "01 10")

    assert found is not None
    assert found.steadier_when_routed
    assert found.verdict == "static"


def test_a_modulator_in_the_expected_direction_is_untouched_by_the_guard() -> None:
    """The guard must not swallow the pile it is protecting."""
    found = efxsort.sort_one(record(repeatability=True, unrepeatable=[-41.5, -1.7]), "01 20")

    assert found is not None and not found.steadier_when_routed
    assert found.verdict == "moves"


def test_a_small_difference_in_either_direction_is_not_called_steadier() -> None:
    """Every pair of settings differs a little. Only a difference over the same
    bar the modulator claim uses is a fact about the effect."""
    found = efxsort.sort_one(record(unrepeatable=[-41.5, -48.1]), "01 03")

    assert found is not None and not found.steadier_when_routed


def test_a_route_that_declined_everywhere_is_reported_beside_the_disagreements() -> None:
    """An empty disagreement list reads as the two routes agreeing, and two routes
    agree only where both spoke. Measured on this unit the delay track stood aside
    on every type, which would otherwise have been recorded as unanimity."""
    here = [
        efxsort.sort_one(record(repeatability=True), "01 20"),
        efxsort.sort_one(record(), "01 00"),
    ]
    other = [Tracked("01 20", "could not say"), Tracked("01 00", "could not say")]

    assert efxsort.disagreements(here, other) == []
    assert efxsort.declined_elsewhere(here, other) == ["01 20", "01 00"]


def test_a_type_the_other_route_answered_is_not_counted_as_declined() -> None:
    """Otherwise the count would say a route stood aside where it agreed."""
    here = [efxsort.sort_one(record(repeatability=True), "01 20")]

    assert efxsort.declined_elsewhere(here, [Tracked("01 20", "moves")]) == []
