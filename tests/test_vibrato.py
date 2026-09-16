"""A periodic modulation of pitch, measured inside one take rather than between two.

The failure guarded here reports a number rather than an error. A struck note
settles in pitch as it decays, and what a straight line leaves of that curve is a
slow deep oscillation sitting at the bottom of the search -- measured on a real
piano note with the vibrato depth at zero, 356 cents at 1.23 Hz, deeper than
anything the unit's own vibrato produces.

Signals are synthesised: what is under test is the tracker.
"""

from __future__ import annotations

import numpy as np
import pytest

from soundings import vibrato

RATE = 22050
LEAD = 0.6


def tone(
    *,
    seconds: float = 2.0,
    f0: float = 261.6,
    hz: float = 0.0,
    cents: float = 0.0,
    drift_cents: float = 0.0,
    decay: float = 0.0,
) -> np.ndarray:
    """A note with an optional vibrato, an optional settling curve and an optional decay."""
    n = int(RATE * seconds)
    lead = int(RATE * LEAD)
    t = np.arange(n) / RATE
    swing = (cents / 2.0) * np.sin(2 * np.pi * hz * t) if hz else np.zeros(n)
    # A curve rather than a ramp, which is what a note settling in pitch does and
    # what a linear detrend cannot take out.
    settle = drift_cents * np.exp(-3.0 * t) if drift_cents else np.zeros(n)
    freq = f0 * 2 ** ((swing + settle) / 1200.0)
    note = np.sin(2 * np.pi * np.cumsum(freq) / RATE)
    if decay:
        note = note * np.exp(-decay * t)
    out = np.random.default_rng(7).normal(0, 1e-6, n)
    out[lead:] += note[: n - lead]
    return out


def test_a_known_vibrato_comes_back_at_its_own_rate() -> None:
    found = vibrato.measure(tone(hz=5.0, cents=100.0), RATE)

    assert found.found
    assert abs(found.rate_hz - 5.0) < 0.5
    assert 60.0 < found.depth_cents < 160.0


def test_a_steady_note_reports_no_modulation() -> None:
    """The null the whole thing rests on. Without it every note has a vibrato."""
    assert not vibrato.measure(tone(), RATE).found


def test_a_note_settling_in_pitch_is_not_reported_as_a_slow_vibrato() -> None:
    """The measured false positive: a linear fit leaves the curve behind, and what
    is left is a deep slow oscillation at the bottom of the search. A cubic trend
    and a refusal to read the band's own edge as a peak are what stop it."""
    found = vibrato.measure(tone(drift_cents=200.0), RATE)

    assert not found.found


def test_a_vibrato_survives_the_settling_curve_it_sits_on() -> None:
    """The guard must not swallow what it is protecting: a real note both settles
    and, if the unit says so, wobbles."""
    found = vibrato.measure(tone(hz=5.0, cents=100.0, drift_cents=200.0), RATE)

    assert found.found and abs(found.rate_hz - 5.0) < 0.6


def test_frames_with_no_signal_in_them_are_dropped() -> None:
    """The same defect the delay tracker had. A frequency estimated from silence
    is the noise's own strongest period reported as a note, and a struck note
    spends most of a take decaying towards it."""
    found = vibrato.measure(tone(decay=14.0), RATE)

    assert found.frames_below_floor > 0
    assert not found.found


def test_the_control_says_how_shallow_a_vibrato_this_material_could_carry() -> None:
    """A rate the tracker cannot recover from a vibrato put there by hand is not
    evidence the unit applied none, and nothing in the output separates the two."""
    vouched = vibrato.control(tone(), RATE)

    assert vouched["depths_recovered_cents"]
    assert vouched["shallowest_recovered_cents"] == min(vouched["depths_recovered_cents"])
    assert vouched["injected_rate_hz"] == vibrato.CONTROL_RATE_HZ


def test_a_modulation_too_slow_to_have_held_three_cycles_is_not_claimed() -> None:
    """One cycle of a slow oscillation and one bend of a trend are the same curve,
    and there is nothing in a single cycle to tell them apart."""
    found = vibrato.measure(tone(seconds=1.4, hz=0.8, cents=100.0), RATE)

    assert not found.found


def test_a_track_too_short_to_search_is_refused_rather_than_guessed() -> None:
    """A handful of frames will fit any rate asked of them."""
    assert not vibrato.measure(np.zeros(int(RATE * 0.7)), RATE).found


@pytest.mark.parametrize("found_one", [True, False])
def test_every_reading_survives_the_json_round_trip(found_one: bool) -> None:
    """The record is the archive, so a figure the JSON drops did not happen."""
    signal = tone(hz=5.0, cents=100.0) if found_one else tone()

    written = vibrato.measure(signal, RATE).to_json()

    assert (written["rate_hz"] is not None) is found_one
    assert written["searched_hz"] == list(vibrato.SEARCH_HZ)


def wobble(depth_cents: float | None) -> vibrato.Wobble:
    """A reading that found a modulation at this depth, or one that found none."""
    if depth_cents is None:
        return vibrato.Wobble()
    return vibrato.Wobble(rate_hz=5.3, depth_cents=depth_cents, f0_hz=261.6)


def assembled(by_setting, *, floor: float | None = 12.0) -> dict:
    vouched = None if floor is None else {"shallowest_recovered_cents": floor}
    return vibrato.record(
        by_setting,
        takes=".cache/takes/whatever",
        searched_hz=vibrato.SEARCH_HZ,
        control=vouched,
        control_taken_from=None if floor is None else "quiet at 0",
    )


def test_a_depth_under_what_the_control_reached_is_named_rather_than_left_plain() -> None:
    """The control bounds the run in both directions. A rate found at a depth the
    tracker was never shown able to reach sits in the rows in the same shape as one
    found well above it, and nothing else in the record separates them."""
    written = assembled({"vibrato at 0": [wobble(2.34), wobble(None)]})

    assert written["shallower_than_the_control_recovered"]["rows"] == [
        {"setting": "vibrato at 0", "take": 0, "depth_cents": 2.34}
    ]


def test_a_depth_the_control_reached_is_not_named() -> None:
    written = assembled({"vibrato at 127": [wobble(37.57), wobble(12.0)]})

    assert written["shallower_than_the_control_recovered"]["rows"] == []


def test_nothing_is_called_unsupported_when_no_control_bounds_the_run() -> None:
    """Without a control there is no floor, so no row is under one. Reporting them
    all would put the run's every reading under a caveat about a measurement that
    was never made."""
    written = assembled({"vibrato at 0": [wobble(2.34)]}, floor=None)

    assert written["shallower_than_the_control_recovered"]["rows"] == []
    assert "why_one_setting_carries_the_control" not in written


def test_the_control_says_which_setting_it_came_from_and_what_that_costs() -> None:
    """A take that already carries a modulation cannot carry an injected one too,
    so one setting's take bounds every row. The reason lives with the rows rather
    than in the code that assembled them."""
    written = assembled({"vibrato at 127": [wobble(37.57)]})

    assert written["control_taken_from"] == "quiet at 0"
    assert written["why_one_setting_carries_the_control"] == (
        vibrato.WHY_ONE_SETTING_CARRIES_THE_CONTROL
    )


def test_a_modulation_above_the_search_is_refused_rather_than_named_at_its_top() -> None:
    """The guard at the bottom of the band has a twin at the top, and without it a
    run returns one figure for every take.

    A track whose only periodicity is faster than the search puts its maximum in
    the highest bin, exactly as a trend puts its maximum in the lowest. Read as a
    peak it comes back at the top of the band on every take of a sweep -- the same
    rate at every setting, which reads as a byte the modulator does not follow
    rather than as a band that was searched in the wrong place.
    """
    fast = vibrato.measure(tone(hz=40.0, cents=60.0), RATE, search_hz=(0.5, 15.0))

    assert fast.rate_hz != pytest.approx(15.0, abs=0.2)
    assert not fast.found
    assert vibrato.WHY_AT_THE_EDGE in fast.notes


def test_a_modulation_inside_the_search_is_still_found() -> None:
    """The guard refuses an edge, not a band. A rate the search covers properly has
    to survive it, or the fix has bought a false negative for every run."""
    inside = vibrato.measure(tone(hz=5.0, cents=60.0), RATE, search_hz=(0.5, 15.0))

    assert inside.found
    assert inside.rate_hz == pytest.approx(5.0, abs=0.2)
    assert vibrato.WHY_AT_THE_EDGE not in inside.notes


def test_the_two_refusals_are_told_apart_in_the_record() -> None:
    """A row refused for sitting at the edge of the search and one refused for
    holding too little signal bound different things, and a reader who cannot tell
    them apart cannot tell a band chosen wrongly from a take that was too quiet."""
    trend = vibrato.measure(tone(drift_cents=300.0, decay=3.0), RATE, search_hz=(0.5, 15.0))

    assert not trend.found
    assert trend.notes == [vibrato.WHY_AT_THE_EDGE]
