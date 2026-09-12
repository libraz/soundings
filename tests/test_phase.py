"""A phase between two takes is a reading only where the two takes are one signal.

The material here is built so that the answer is known in closed form before the
measurement runs, and so that it is known in a form this measurement can report:
a delay between two takes is a straight line in frequency and cannot be told from
a delay the effect put there, so what is asserted is the part that survives that
line being taken out. A shape with a constant phase and a shape with none are the
two cases where that part is the whole answer.

Both of the things that break the measurement in the room are in the material --
a noise floor that is different in each take, and two clocks that do not run at
the same rate -- because both of them return plausible numbers rather than
failures.
"""

from __future__ import annotations

import numpy as np
import pytest

from soundings import phase

RATE = 48000
SECONDS = 4.0


def noise(seed: int, seconds: float = SECONDS) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 0.1, int(seconds * RATE))


def turned(signal: np.ndarray, degrees: float) -> np.ndarray:
    """The same signal with every frequency turned by the same angle.

    A constant angle is what this measurement can assert against: it has no
    straight line in it, so the delay the method removes removes none of it.
    """
    spectrum = np.fft.rfft(signal)
    spectrum[1:] *= np.exp(1j * np.radians(degrees))
    return np.fft.irfft(spectrum, n=signal.size)


def shaped(signal: np.ndarray, gain_db: float, above: float) -> np.ndarray:
    """A magnitude change with no phase in it at all, which is the other known answer."""
    spectrum = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(signal.size, 1.0 / RATE)
    spectrum[freq >= above] *= 10.0 ** (gain_db / 20.0)
    return np.fft.irfft(spectrum, n=signal.size)


def stretched(signal: np.ndarray, ppm: float) -> np.ndarray:
    """The same signal as a clock a few parts per million out would have caught it."""
    at = np.arange(signal.size) * (1.0 + ppm * 1e-6)
    return np.interp(at, np.arange(signal.size), signal, left=0.0, right=0.0)


@pytest.fixture
def pair():
    """A dry take, the same take turned by a quarter turn, and a floor in each."""
    source = noise(1)
    return source + noise(11) * 0.02, turned(source, 90.0) + noise(12) * 0.02


def at(found: dict, hz: float) -> dict:
    return next(r for r in found["readings"] if r["hz"] == hz)


def readable(found: dict) -> list[dict]:
    return [r for r in found["readings"] if (r["coherence"] or 0.0) >= phase.READABLE]


def test_a_known_phase_is_read_as_that_phase(pair) -> None:
    found = phase.measure(*pair, RATE)
    seen = readable(found)
    assert len(seen) > 10
    for reading in seen:
        assert reading["phase_deg_less_delay"] == pytest.approx(90.0, abs=8.0), reading["hz"]


def test_a_shape_with_no_phase_in_it_reads_as_no_phase() -> None:
    """An effect that only changes how much comes out changes no angle."""
    source = noise(1)
    found = phase.measure(
        source + noise(11) * 0.02, shaped(source, -9.0, 2000.0) + noise(12) * 0.02, RATE
    )
    for reading in readable(found):
        assert abs(reading["phase_deg_less_delay"]) < 10.0, reading["hz"]


def test_two_takes_that_share_no_signal_say_so_rather_than_returning_an_angle() -> None:
    """The whole reason coherence is published beside every figure.

    Two unrelated takes return a phase in every band. It is a number, it is as
    steady as a real one, and the only thing that separates it from a reading is
    the figure beside it.
    """
    found = phase.measure(noise(2), noise(3), RATE)
    assert not readable(found)
    assert all(r["phase_deg"] is not None for r in found["readings"])


def repeats(*seeds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Every pair of takes of one setting, which is what a bound is drawn from."""
    from itertools import combinations

    source = noise(1)
    return [
        (source + noise(a) * 0.02, source + noise(b) * 0.02)
        for a, b in combinations(seeds, 2)
    ]


def test_takes_of_one_setting_return_no_phase() -> None:
    """The control. What the method gives back when the answer is known to be nothing."""
    vouched = phase.control(repeats(21, 22, 23), RATE)
    assert vouched["readable_bands"] > 10
    assert vouched["largest_deg"] < 10.0
    assert len(vouched["pairs"]) == 3


def test_a_bound_is_drawn_from_every_pair_and_not_from_one() -> None:
    """One pair is one draw, and two repeats that happened to agree draw a low bound."""
    many = phase.control(repeats(21, 22, 23, 24), RATE)
    each = [p["largest_deg"] for p in many["pairs"]]
    assert len(each) == 6
    assert many["largest_deg"] == max(each)
    assert min(each) < max(each)


def test_a_phase_standing_above_the_control_is_the_one_the_record_calls_conclusive(
    pair,
) -> None:
    source = noise(1)
    vouched = phase.control(repeats(21, 22, 23), RATE)
    assert phase.is_conclusive(phase.measure(*pair, RATE), vouched)
    # And a pair that is the same signal twice is not, measured the same way.
    flat = phase.measure(source + noise(31) * 0.02, source + noise(32) * 0.02, RATE)
    assert not phase.is_conclusive(flat, vouched)


def test_a_take_that_starts_later_is_lined_up_before_anything_is_measured(pair) -> None:
    """Two takes of one note do not begin at the same sample."""
    dry, wet = pair
    found = phase.measure(dry, np.roll(wet, 120), RATE)
    assert all(abs(lag + 120) <= 2 for lag in found["aligned_by"]["lag_samples"])
    for reading in readable(found):
        assert reading["phase_deg_less_delay"] == pytest.approx(90.0, abs=12.0)


def test_the_delay_taken_out_is_published_in_samples(pair) -> None:
    """Removing a line silently would make a delay look like an effect that has none."""
    dry, wet = pair
    found = phase.measure(dry, np.roll(wet, 120), RATE)
    assert found["less_a_delay_of"]["samples"] is not None
    assert found["less_a_delay_of"]["fitted_over_bins"] > 100


def test_two_clocks_apart_collapse_a_take_read_in_one_block_and_not_one_read_in_many(
    pair,
) -> None:
    """The finding the block length exists for.

    A pair a few tens of parts per million apart is the same signal throughout and
    returns almost none of it when the whole take is aligned once: the phase turns
    steadily across the take and the average of a turning thing is nothing. Read a
    block at a time it comes back, and the drift is reported rather than removed
    silently.
    """
    dry, wet = pair
    drifting = stretched(wet, 60.0)
    whole = phase.measure(dry, drifting, RATE, block_s=SECONDS * 2)
    blocks = phase.measure(dry, drifting, RATE, block_s=0.5)
    assert at(whole, 4000.0)["coherence"] < 0.5
    assert at(blocks, 4000.0)["coherence"] > 0.8
    assert blocks["aligned_by"]["drift_samples"] != 0


def test_a_block_longer_than_the_takes_is_one_block_and_not_no_block(pair) -> None:
    found = phase.measure(*pair, RATE, block_s=SECONDS * 10)
    assert found["transforms"] > 0
    assert len(found["aligned_by"]["lag_samples"]) == 1


def test_every_band_the_set_holds_is_in_the_record(pair) -> None:
    """Bands the stimulus did not reach are published, not dropped.

    Which bands a stimulus carries into a measurement is a fact about the stimulus
    and belongs in the record the stimulus produced.
    """
    from soundings import efxbands

    found = phase.measure(*pair, RATE)
    assert [r["hz"] for r in found["readings"]] == list(efxbands.THIRD_OCTAVES)
