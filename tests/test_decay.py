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
    return np.random.default_rng(seed).standard_normal(n) * 10 ** (
        -3.0 * np.arange(n) / SR / rt60
    )


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
