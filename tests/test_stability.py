"""Controls for the take comparator, on signals whose answer is already known.

The comparator's failure mode is not an exception, it is a number: a residual
that reads as "the unit does not repeat" when what actually happened is that the
alignment missed, or the level fit misfired, or the fractional shift low-passed
the copy it subtracted. Every one of those looks like a finding about the
hardware. So each is given a signal it must get exactly right.
"""

from __future__ import annotations

import numpy as np
import pytest

from soundings import stability

SR = 48000


def burst(
    seconds: float = 1.0, *, silence: float = 0.3, seed: int = 1, fundamental: float = 220.0
) -> np.ndarray:
    """Silence, then a decaying broadband transient -- the shape of a struck note."""
    rng = np.random.default_rng(seed)
    lead = np.zeros(int(silence * SR))
    n = int(seconds * SR)
    t = np.arange(n) / SR
    tone = sum(np.sin(2 * np.pi * fundamental * k * t) / k for k in (1, 2, 3, 5, 8))
    body = (tone + 0.05 * rng.standard_normal(n)) * np.exp(-3.0 * t)
    return np.concatenate([lead, body])


def test_a_signal_against_itself_leaves_nothing() -> None:
    signal = burst()
    result = stability.compare(signal, signal, SR)
    assert result.delay_samples == pytest.approx(0.0, abs=1e-6)
    assert result.correlation == pytest.approx(1.0, abs=1e-9)
    assert result.gain_db == pytest.approx(0.0, abs=1e-9)
    assert result.residual_db < -200


def test_an_integer_delay_is_recovered_and_removed() -> None:
    signal = burst()
    delayed = np.concatenate([np.zeros(137), signal])[: signal.size]
    result = stability.compare(signal, delayed, SR)
    assert result.delay_samples == pytest.approx(137.0, abs=0.01)
    assert result.residual_db < -100


def test_a_fractional_delay_is_recovered_and_removed() -> None:
    """The case an integer-only aligner turns into a rising treble residual."""
    signal = burst()
    delayed = stability.shift(signal, 0.5)
    result = stability.compare(signal, delayed, SR)
    assert result.delay_samples == pytest.approx(0.5, abs=0.02)
    assert result.residual_db < -60


def test_a_level_difference_is_reported_rather_than_left_in_the_residual() -> None:
    signal = burst()
    quieter = signal * 10 ** (-0.5 / 20)
    result = stability.compare(signal, quieter, SR)
    assert result.gain_db == pytest.approx(-0.5, abs=0.01)
    assert result.residual_db < -100


def test_an_unrelated_signal_leaves_the_whole_signal() -> None:
    """The negative control: nothing about the method can make these agree.

    A different seed is not a different signal here -- it changes only the noise
    term, and two takes of the same note really are alike to within their noise.
    Being unrelated takes a different note.
    """
    result = stability.compare(burst(fundamental=220.0), burst(fundamental=277.2), SR)
    assert result.residual_db > -6


def test_added_noise_lands_where_it_was_put() -> None:
    """A residual has to measure the difference, not merely be small."""
    rng = np.random.default_rng(7)
    signal = burst()
    noisy = signal + 0.01 * np.abs(signal).max() * rng.standard_normal(signal.size)
    result = stability.compare(signal, noisy, SR)
    assert -50 < result.residual_db < -25


def test_the_floor_rises_when_the_take_is_noisier() -> None:
    rng = np.random.default_rng(3)
    quiet = burst()
    loud = burst() + 0.001 * rng.standard_normal(quiet.size)
    assert stability.compare(loud, loud, SR).floor_db > stability.compare(quiet, quiet, SR).floor_db


def test_the_floor_is_measured_from_the_silence_before_the_note() -> None:
    rng = np.random.default_rng(11)
    signal = burst()
    lead = int(0.3 * SR)
    signal[:lead] += 0.001 * rng.standard_normal(lead)
    floor = stability.noise_floor(signal, SR, before=0.3)
    assert -80 < floor < -40


def test_a_tone_frequency_is_recovered_to_a_part_per_million() -> None:
    t = np.arange(int(20 * SR)) / SR
    signal = np.sin(2 * np.pi * 440.137 * t)
    found = stability.tone_frequency(signal, SR, expected=440.0)
    assert found is not None
    assert abs(found.frequency - 440.137) / 440.137 < 1e-6
    assert found.steady


def test_a_tone_frequency_ignores_the_partials_above_it() -> None:
    """A phase slope taken on the raw signal would sit between the partials."""
    t = np.arange(int(20 * SR)) / SR
    signal = sum(np.sin(2 * np.pi * 220.05 * k * t) / k for k in (1, 2, 3, 4))
    found = stability.tone_frequency(signal, SR, expected=220.0)
    assert found is not None
    assert abs(found.frequency - 220.05) / 220.05 < 1e-5
    assert found.steady


def test_vibrato_is_reported_as_unsteady_rather_than_as_a_frequency() -> None:
    """The failure the clock reading has to survive: a confident slope on a tone
    that was never steady enough for a slope to mean anything."""
    t = np.arange(int(20 * SR)) / SR
    signal = np.sin(2 * np.pi * 440.0 * t + 0.9 * np.sin(2 * np.pi * 5.0 * t))
    found = stability.tone_frequency(signal, SR, expected=440.0)
    assert found is not None
    assert not found.steady


def test_silence_has_no_frequency() -> None:
    assert stability.tone_frequency(np.zeros(int(5 * SR)), SR, expected=440.0) is None


def test_too_short_to_measure_reports_nothing_rather_than_a_number() -> None:
    t = np.arange(int(0.4 * SR)) / SR
    assert stability.tone_frequency(np.sin(2 * np.pi * 440 * t), SR, expected=440.0) is None


def test_a_quiet_stimulus_is_not_mistaken_for_a_contaminated_lead_in() -> None:
    """Measured on hardware: a velocity 30 note was refused by a relative check.

    Contamination raises the lead-in in absolute terms. A quiet note lowers the
    note instead and leaves the lead-in on the converter's own floor.
    """
    rng = np.random.default_rng(31)
    floor = 3e-4
    quiet = burst() * 0.05
    quiet[: int(0.3 * SR)] = floor * rng.standard_normal(int(0.3 * SR))
    loud = burst()
    loud[: int(0.3 * SR)] = floor * rng.standard_normal(int(0.3 * SR))
    quiet_lead = stability.loudest_lead_in([quiet], SR, before=0.25)
    loud_lead = stability.loudest_lead_in([loud], SR, before=0.25)
    assert abs(quiet_lead - loud_lead) < 1.0
    assert quiet_lead < -60.0


def test_a_contaminated_lead_in_reads_loud_in_absolute_terms() -> None:
    signal = burst()
    signal[: int(0.3 * SR)] += 0.05 * np.sin(2 * np.pi * 300 * np.arange(int(0.3 * SR)) / SR)
    assert stability.loudest_lead_in([signal], SR, before=0.25) > -40.0
