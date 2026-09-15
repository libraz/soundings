"""A rotor that has to accelerate to the entry its byte names, and stops short.

Every other rate slot steps to its entry. This one is aimed at it, and where it
ends up is a second thing the table does not say. What the arrival adds is one
number for the whole table rather than one per entry, which is what keeps it from
being a table with a free entry in it.
"""

from __future__ import annotations

import copy

import pytest

from soundings import reproduce

WIDE = "0.05 - 10.0"

TEN = {
    "tables": {
        WIDE: {
            "kind": "steps",
            "first_hz": 0.05,
            "runs": [
                {"through": 99, "step_hz": 0.05},
                {"through": 119, "step_hz": 0.10},
                {"through": 125, "step_hz": 0.50},
            ],
        }
    }
}

REST, SHORT = 0.595, 0.245
"""The low rotor's two, near enough: it sits at a shade under six tenths of a
hertz and stops a quarter of a hertz below whatever it is sent to."""


def stalling(rest=REST, short=SHORT):
    model = copy.deepcopy(TEN)
    model["arrival"] = {
        "kind": "stops-short",
        "rests_at_hz": rest,
        "short_by_hz": short,
    }
    return model


def test_a_model_with_no_arrival_is_the_table_and_nothing_else():
    """So that adding this cannot have moved any claim already resting on it."""
    for value in range(128):
        plain = reproduce.rate_of(TEN, WIDE, value)
        assert plain == reproduce.rate_of(stalling(rest=1e9), WIDE, value)


def test_under_where_it_rests_the_rotor_arrives_exactly():
    """Coming down there is no stall, and that asymmetry is the whole signature.

    A difference driven to zero by a right shift underflows on the way up while
    it is still wide; on the way down the same shift of a negative number floors
    instead, so the last step is always taken and the target is reached.
    """
    model = stalling()
    for value in range(0, 11):  # 0.05 through 0.55, all under where it rests
        assert reproduce.rate_of(model, WIDE, value) == pytest.approx(
            reproduce.rate_of(TEN, WIDE, value)
        )


def test_well_above_it_the_rotor_lands_one_fixed_distance_under_the_entry():
    model = stalling()
    for value in (21, 32, 64, 99, 110, 125):
        assert reproduce.rate_of(model, WIDE, value) == pytest.approx(
            reproduce.rate_of(TEN, WIDE, value) - SHORT
        )


def test_between_the_two_the_rotor_does_not_move_at_all():
    """Five entries of the table answered by one number, which is the reading.

    Where the gap to the entry is narrower than the stall, nothing happens: the
    byte moves, the target moves and the rotor stays. Any model that subtracts a
    constant everywhere has to make this stretch a staircase.
    """
    model = stalling()
    plateau = [reproduce.rate_of(model, WIDE, v) for v in range(11, 16)]
    assert plateau == [pytest.approx(REST)] * 5
    assert reproduce.rate_of(model, WIDE, 10) < REST


def test_where_the_plateau_ends_is_fixed_by_how_far_short_it_stops():
    """The two are one number, so the corner is a prediction and not a parameter.

    The first setting that moves the rotor is the first whose entry clears where
    it rests by more than the stall. Widen the stall and the corner moves up the
    byte; nothing else in the model may be touched to put it back.
    """
    for short in (0.10, 0.245, 0.40):
        model = stalling(short=short)
        moved = [
            v for v in range(128) if reproduce.rate_of(model, WIDE, v) != REST
            and reproduce.rate_of(TEN, WIDE, v) > REST
        ]
        first = min(moved)
        assert reproduce.rate_of(TEN, WIDE, first) > REST + short
        assert reproduce.rate_of(TEN, WIDE, first - 1) <= REST + short


def test_the_last_entry_is_still_held_past_the_end_of_the_runs():
    """The arrival is applied to the entry, not instead of the table's own end."""
    model = stalling()
    assert reproduce.rate_of(model, WIDE, 125) == pytest.approx(10.0 - SHORT)
    assert reproduce.rate_of(model, WIDE, 127) == pytest.approx(10.0 - SHORT)


def test_the_competing_readings_of_the_gap_are_here_to_be_scored_and_lose():
    """Each is written out rather than argued about, and each has its own defect.

    A constant taken off every entry has to take the bottom of the table to
    nothing; a scaling cannot hold a gap that does not grow; a lag caught still
    closing leaves a gap in proportion to the one it set out with; and a stall run
    on how long a turn takes is a shrinking number of hertz as the rotor speeds
    up. The renderer's job is to let each say that in its own numbers.
    """
    take_off = dict(TEN, arrival={"kind": "short-by-a-constant", "short_by_hz": SHORT})
    assert reproduce.rate_of(take_off, WIDE, 32) == pytest.approx(1.65 - SHORT)
    assert reproduce.rate_of(take_off, WIDE, 0) == pytest.approx(1e-6)

    scale = dict(TEN, arrival={"kind": "scaled", "by": 0.9})
    assert reproduce.rate_of(scale, WIDE, 32) == pytest.approx(1.65 * 0.9)
    assert reproduce.rate_of(scale, WIDE, 0) == pytest.approx(0.05 * 0.9)

    late = dict(TEN, arrival={"kind": "still-closing", "rests_at_hz": 0.5,
                              "of_the_gap_left": 0.25})
    assert reproduce.rate_of(late, WIDE, 32) == pytest.approx(1.65 - 0.25 * 1.15)
    assert reproduce.rate_of(late, WIDE, 9) == pytest.approx(0.5)

    slow = dict(TEN, arrival={"kind": "stops-short-on-the-period",
                              "rests_at_hz": REST, "long_by_s": 0.1})
    assert reproduce.rate_of(slow, WIDE, 10) == pytest.approx(0.55)
    # A tenth of a second is a quarter of a hertz at one and a bit and a couple of
    # hertz at ten, which is the whole of why the two stalls are separable.
    assert 1.65 - reproduce.rate_of(slow, WIDE, 32) == pytest.approx(0.2333, abs=1e-3)
    assert 10.0 - reproduce.rate_of(slow, WIDE, 127) == pytest.approx(5.0, abs=1e-3)


def test_steps_straight_to_it_is_the_table_written_out():
    """The closed reading, named rather than left as an absence.

    It is the strongest competitor here and it is already in the tree; saying so
    with a key rather than with nothing keeps a candidate that is not this one
    from being mistaken for a model nobody wrote.
    """
    straight = dict(TEN, arrival={"kind": "steps-straight-to-it"})
    for value in (0, 11, 16, 32, 127):
        assert reproduce.rate_of(straight, WIDE, value) == pytest.approx(
            reproduce.rate_of(TEN, WIDE, value)
        )


def test_an_arrival_this_renderer_does_not_know_is_refused():
    model = copy.deepcopy(TEN)
    model["arrival"] = {"kind": "coasts", "rests_at_hz": 0.5}
    with pytest.raises(ValueError, match="arrival"):
        reproduce.rate_of(model, WIDE, 32)


def test_the_arrival_is_reachable_through_the_rate_score_and_separates_there():
    """A sweep of the stalling rotor, answered by both readings of it.

    The record is built from the model rather than from the unit, so what this
    asserts is the wiring and the sign: the model that stalls answers its own
    curve and the one that does not leans, in octaves, by more than a run of this
    kind resolves.
    """
    model = stalling()
    values = [4, 8, 11, 13, 15, 16, 20, 32, 64, 110, 127]
    record = {
        "address": "40 03 03",
        "readings": [
            {
                "value": v,
                "rate_hz": reproduce.rate_of(model, WIDE, v),
                "rates": [
                    reproduce.rate_of(model, WIDE, v) * (1 - 0.0005),
                    reproduce.rate_of(model, WIDE, v) * (1 + 0.0005),
                ],
                "agreeing": 4,
                "of": 4,
                "slowest_measurable_hz": 0.034,
            }
            for v in values
        ],
    }
    stalled = reproduce.score_against_rates(model, record, printed_range=WIDE)
    straight = reproduce.score_against_rates(TEN, record, printed_range=WIDE)
    assert stalled["median_abs"] == pytest.approx(0.0, abs=1e-9)
    assert stalled["worst_abs"] == pytest.approx(0.0, abs=1e-9)
    assert straight["worst_abs"] > 0.4


def test_a_rotor_sent_down_from_a_faster_entry_arrives_on_it():
    """The asymmetry itself, asked of the renderer rather than of a curve.

    Where the rotor set off from is the reading's, not the model's: a record that
    carries it is a record of one byte written twice with no reset between, and the
    model has to answer both rows from the one arrival it holds.
    """
    model = stalling(rest=0.6, short=0.25)
    climbing = reproduce.rate_of(model, WIDE, 32)
    falling = reproduce.rate_of(model, WIDE, 32, came_from=127)
    assert climbing == pytest.approx(1.40, abs=1e-6)
    assert falling == pytest.approx(1.65, abs=1e-6)


def test_where_a_climbing_rotor_stops_does_not_depend_on_where_it_set_off():
    """The gap is a property of the step and not of the journey, so a rotor sent up
    from an entry it had already reached stops where one sent up from rest does."""
    model = stalling(rest=0.6, short=0.25)
    assert reproduce.rate_of(model, WIDE, 127, came_from=32) == pytest.approx(
        reproduce.rate_of(model, WIDE, 127), abs=1e-9
    )


def test_a_rotor_that_had_not_moved_yet_is_the_same_as_one_from_rest():
    """`rest` is what a run writes for a take that followed a reset and nothing
    else, and it has to render as the take every other record is made of."""
    model = stalling(rest=0.6, short=0.25)
    assert reproduce.rate_of(model, WIDE, 32, came_from="rest") == pytest.approx(
        reproduce.rate_of(model, WIDE, 32), abs=1e-9
    )


def test_a_table_with_no_arrival_ignores_where_the_byte_came_from():
    """Which is every other rate slot on this unit: a byte that names a rate outright
    returns it whatever it was set to before."""
    assert reproduce.rate_of(TEN, WIDE, 32, came_from=127) == pytest.approx(
        reproduce.rate_of(TEN, WIDE, 32), abs=1e-9
    )


def test_two_takes_of_one_byte_are_scored_apart():
    """A record of one byte written from two places, put through the class's own
    scoring. Both rows are answered from the same two numbers, which is what makes
    the second of them a prediction rather than a second fit."""
    record = {
        "address": "40 03 03",
        "readings": [
            {"value": 32, "came_from": "rest", "rate_hz": 1.4071, "agreeing": 4, "of": 4,
             "rates": [1.4071, 1.4071], "slowest_measurable_hz": 0.303},
            {"value": 32, "came_from": "127", "rate_hz": 1.6520, "agreeing": 4, "of": 4,
             "rates": [1.6520, 1.6520], "slowest_measurable_hz": 0.303},
        ],
    }
    scored = reproduce.score_against_rates(
        stalling(rest=0.5953, short=0.2445), record, printed_range=WIDE
    )
    by_where = {row["came_from"]: row for row in scored["rows"]}
    assert abs(by_where["rest"]["residual"][0]) < 0.005
    assert abs(by_where["127"]["residual"][0]) < 0.005
    apart = by_where["127"]["unit_reading"][0] - by_where["rest"]["unit_reading"][0]
    said = by_where["127"]["model_reading"][0] - by_where["rest"]["model_reading"][0]
    assert said == pytest.approx(apart, abs=0.005)
    assert apart > 0.2, "the two takes of one byte have to be far apart to mean anything"


POINTS = {
    "tables": {
        WIDE: {
            "kind": "points",
            "first_hz": 0.05,
            "made_from": "one constant, divided by the position of the entry plus one",
            "points": [[0, 8.17], [8, 9.48], [16, 9.68]],
        }
    }
}
"""A rate named per setting, which is the shape a record sweeping something other
than the rate byte needs -- here an acceleration, read with the rate held still."""


def test_a_table_of_points_returns_what_it_names():
    for value, hz in ((0, 8.17), (8, 9.48), (16, 9.68)):
        assert reproduce.rate_of(POINTS, WIDE, value) == pytest.approx(hz)


def test_a_table_of_points_that_does_not_say_what_made_them_refuses():
    """The guard against the candidate that is the measurement. A table free to
    hold any number reproduces any reading and says nothing, so the renderer will
    not read one that does not say how many numbers it really took."""
    loose = copy.deepcopy(POINTS)
    loose["tables"][WIDE].pop("made_from")
    with pytest.raises(ValueError, match="made_from"):
        reproduce.rate_of(loose, WIDE, 0)


def test_a_table_of_points_refuses_a_setting_it_does_not_name():
    """Rather than interpolating. What sits between two settings a run asked is a
    thing nobody measured, and returning a number for it would publish the
    interpolation as a reading."""
    with pytest.raises(ValueError, match="names no rate"):
        reproduce.rate_of(POINTS, WIDE, 4)


def test_a_table_of_points_is_given_no_floor_of_its_own():
    """The floor of one entry is the distance to the setting next door, which on a
    record like this is whatever the run happened to ask -- so a coarse sweep would
    buy itself a floor wide enough to hide any lean."""
    record = {
        "address": "40 03 05",
        "readings": [
            {"value": v, "rate_hz": hz, "agreeing": 4, "of": 4,
             "rates": [hz, hz * 1.0001], "slowest_measurable_hz": 0.303}
            for v, hz in ((0, 8.17), (8, 9.48), (16, 9.68))
        ],
    }
    scored = reproduce.score_against_rates(POINTS, record, printed_range=WIDE)
    assert scored["floor_of_one_entry"] == 0.0
    assert scored["floor"] == pytest.approx(scored["floor_the_run_resolved"], abs=5e-6)
