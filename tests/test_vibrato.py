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
