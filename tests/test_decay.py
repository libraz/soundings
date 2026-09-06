"""Controls for the tail measurement, on decays whose times are already known.

The way this fails is by returning a number. A curve that ran into the noise
flattens into a shelf and fits as several seconds of reverb; a tail made of two
decays in sequence fits as one somewhere between them; a band with nothing in it
fits whatever the noise did. Every one of those would be published as a decay
time for a machine, so each is given a signal it must refuse.
"""

from __future__ import annotations

import numpy as np
import pytest

from soundings import decay

SR = 48000


def tail(
    seconds: float = 3.0, *, rt60: float = 1.2, seed: int = 2, damping: float = 1.0
) -> np.ndarray:
    """Noise decaying at a known rate, which is what a reverb tail is.

    `damping` above one makes the top decay faster than the bottom, by scaling
    each band's rate with frequency the way a lossy delay network does.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    if damping == 1.0:
        return rng.standard_normal(n) * 10 ** (-3.0 * t / rt60)
    out = np.zeros(n)
    for index, centre in enumerate(decay.OCTAVE_CENTRES):
        share = damping ** (index / (len(decay.OCTAVE_CENTRES) - 1))
        band = decay.band_limit(rng.standard_normal(n), SR, centre)
        out += band * 10 ** (-3.0 * t / (rt60 / share))
    return out


def hush(seconds: float = 0.6, *, level: float = 1e-5, seed: int = 8) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(int(seconds * SR)) * level


def test_a_known_decay_time_comes_back() -> None:
    found = decay.measure(tail(rt60=1.2), SR, noise=hush())
    assert len(found.measured) >= 6
    assert np.median([b.seconds for b in found.measured]) == pytest.approx(1.2, rel=0.05)
    for band in found.measured:
        # Twice the bound: the bound is on the spread, and one draw in eight bands
        # sitting outside a one-sigma figure is the ordinary case, not a defect.
        assert band.seconds == pytest.approx(1.2, rel=2 * band.scatter), band.centre_hz


def test_the_bound_on_a_band_covers_how_far_it_actually_wanders() -> None:
    """A narrow band over a short fit holds few independent samples, and its time
    scatters accordingly. A bound that did not cover that would make a noisy low
    band read as a design decision about damping."""
    seen = {}
    for seed in range(12):
        for band in decay.measure(tail(rt60=1.2, seed=seed), SR, noise=hush()).measured:
            seen.setdefault(band.centre_hz, []).append((band.seconds, band.scatter))
    for centre, rows in seen.items():
        times = np.array([t for t, _ in rows])
        bound = float(np.mean([s for _, s in rows]))
        assert times.std() / times.mean() < bound, centre
    assert seen[63.0][0][1] > seen[8000.0][0][1] * 5


def test_a_shorter_decay_is_told_from_a_longer_one() -> None:
    short = decay.measure(tail(rt60=0.4), SR, noise=hush())
    long = decay.measure(tail(rt60=2.0, seconds=5.0), SR, noise=hush())
    assert short.measured and long.measured
    assert np.median([b.seconds for b in short.measured]) == pytest.approx(0.4, rel=0.15)
    assert np.median([b.seconds for b in long.measured]) == pytest.approx(2.0, rel=0.15)


def test_damping_shows_up_as_the_top_decaying_faster() -> None:
    """The measurement that separates a flat delay network from a lossy one."""
    flat = decay.measure(tail(rt60=1.2), SR, noise=hush())
    lossy = decay.measure(tail(rt60=1.2, damping=3.0), SR, noise=hush())
    assert flat.damping == pytest.approx(1.0, abs=0.2)
    assert lossy.damping > 2.0


def test_a_tail_that_ran_into_the_noise_is_not_a_long_reverb() -> None:
    """Integrating past the floor flattens the curve into a shelf, which fits as
    a decay several times longer than the one that is actually there."""
    loud_floor = hush(level=2e-2)
    signal = tail(rt60=0.5) + loud_floor[: int(3.0 * SR)].mean() * 0  # keep the shape
    signal = signal + np.random.default_rng(12).standard_normal(signal.size) * 2e-2
    found = decay.measure(signal, SR, noise=loud_floor)
    for band in found.measured:
        assert band.seconds < 1.0, f"{band.centre_hz} Hz read {band.seconds:.2f} s"


def test_an_empty_band_is_refused_rather_than_fitted() -> None:
    """A band with nothing in it still has noise, and noise still fits a line."""
    n = int(2.0 * SR)
    t = np.arange(n) / SR
    only_low = decay.band_limit(np.random.default_rng(4).standard_normal(n), SR, 125.0)
    found = decay.measure(only_low * 10 ** (-3.0 * t / 1.0), SR, noise=hush())
    high = [b for b in found.bands if b.centre_hz >= 2000.0]
    assert high and not any(b.measured for b in high)
    assert all(b.reason for b in high)


def test_two_decays_in_sequence_are_reported_as_curved() -> None:
    """A number would still come out; what says it means nothing is the curvature."""
    n = int(3.0 * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(6)
    signal = rng.standard_normal(n) * (10 ** (-3.0 * t / 0.15) + 0.02 * 10 ** (-3.0 * t / 2.5))
    found = decay.measure(signal, SR, noise=hush())
    curved = [b for b in found.bands if not b.measured and not b.reason]
    assert curved, "a bent curve was fitted as if it were one decay"


def test_a_band_refused_for_curving_says_so_in_the_record(tmp_path) -> None:
    """The refusal has to reach the archive, not only the printed prose.

    Measured on a saved pair, all eight bands of one effect type came back
    `measured: false` with an empty reason and a 3.5 s time beside it, because
    the sentence naming the curvature was composed where the run is described
    and not where it is written down. A reader of the record sees a refusal with
    no reason and a number, and takes the number."""
    n = int(3.0 * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(6)
    signal = rng.standard_normal(n) * (10 ** (-3.0 * t / 0.15) + 0.02 * 10 ** (-3.0 * t / 2.5))
    found = decay.measure(signal, SR, noise=hush())

    curved = [b.to_json() for b in found.bands if not b.measured and not b.reason]

    assert curved, "a bent curve was fitted as if it were one decay"
    for band in curved:
        assert band["reason"].startswith("curved by")
        assert band["rt60_s"] is None, "a time beside `measured: false` is read as a time"
        assert band["curvature_db"] is not None, "the curvature is the refusal's own evidence"


def test_silence_yields_no_decay_at_all() -> None:
    found = decay.measure(hush(seconds=3.0), SR, noise=hush())
    assert not found.measured
    assert np.isnan(found.damping)


def played(decay_s: float, *, seconds: float = 3.0, seed: int = 15) -> np.ndarray:
    """A note with a lead-in, decaying at a known rate."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    note = np.random.default_rng(seed).standard_normal(n) * 10 ** (-3.0 * t / decay_s)
    note[: int(0.3 * SR)] = 0.0
    return note


def room(rt60: float, *, seconds: float = 1.5, seed: int = 3) -> np.ndarray:
    n = int(seconds * SR)
    return np.random.default_rng(seed).standard_normal(n) * 10 ** (-3.0 * np.arange(n) / SR / rt60)


def through(note: np.ndarray, impulse: np.ndarray, *, send: float = 0.3) -> np.ndarray:
    wet = np.convolve(note, impulse, mode="full")[: note.size] / np.abs(impulse).sum()
    return note + send * wet


def test_the_effect_is_isolated_from_the_note_rather_than_waited_out() -> None:
    """A reverb measured from the wet take alone is the note until the note stops."""
    note = played(0.25)
    isolated, noise = decay.isolate_tail(note, through(note, room(1.0)), SR, lead=0.25)
    found = decay.measure(isolated, SR, noise=noise)
    assert found.measured
    assert np.median([b.seconds for b in found.measured]) == pytest.approx(1.0, rel=0.2)


def test_a_reverb_shorter_than_the_note_measures_the_note() -> None:
    """The caveat the stimulus has to be chosen against: a convolution of two
    decays falls at the slower of them, so a short room under a long note is
    invisible however cleanly it is isolated."""
    note = played(1.8)
    isolated, noise = decay.isolate_tail(note, through(note, room(0.5)), SR, lead=0.25)
    found = decay.measure(isolated, SR, noise=noise)
    assert found.measured
    assert np.median([b.seconds for b in found.measured]) > 1.4


def test_a_return_that_does_not_clear_the_subtraction_floor_is_refused() -> None:
    """A return the same size as what subtracting two takes of one setting leaves
    is that subtraction's error, and it fits a decay as readily as anything else.

    Measured on this unit: one insertion effect type left a return standing 24.9
    dB over the lead-in noise at 63 Hz and 6.9 dB over the subtraction floor, and
    reported an 11 second decay from a two second tail. A second type's return sat
    3.2 dB *below* the floor in one band and still produced eight times."""
    floor = tail(rt60=1.2, seed=3)
    barely = tail(rt60=1.2, seed=4) * 10 ** (3.0 / 20.0)

    found = decay.measure(barely, SR, noise=hush(), floor=floor)

    assert not found.measured
    assert found.floor_taken
    for band in found.bands:
        assert band.why.startswith(decay.NOT_OVER_THE_FLOOR)
        assert band.to_json()["rt60_s"] is None


def test_a_return_well_over_the_floor_is_measured_as_before() -> None:
    """The guard has to have an outside, or it is a rule that refuses everything.
    On the same run the two refused types came from, a genuine return cleared the
    floor by thirty dB and kept all eight of its bands."""
    floor = tail(rt60=1.2, seed=3) * 10 ** (-30.0 / 20.0)
    real = tail(rt60=1.2, seed=4)

    found = decay.measure(real, SR, noise=hush(), floor=floor)

    assert len(found.measured) >= 6
    assert np.median([b.seconds for b in found.measured]) == pytest.approx(1.2, rel=0.05)
    assert all(b.over_the_floor_db > decay.CLEARS_THE_FLOOR_DB for b in found.measured)


def test_a_run_with_no_floor_pair_says_the_floor_was_not_measured() -> None:
    """Absent is not passed. A band fitted without the floor ever being taken is
    unjudged, and a record that does not say so is read as one that cleared it."""
    found = decay.measure(tail(rt60=1.2), SR, noise=hush())

    written = found.to_json()

    assert not found.floor_taken
    assert written["subtraction_floor_measured"] is False
    assert written["why_no_subtraction_floor"] == decay.NO_FLOOR_TAKEN
    assert all(b["over_the_subtraction_floor_db"] is None for b in written["bands"])
    assert found.measured, "a run without the floor is still fitted, only unjudged"
