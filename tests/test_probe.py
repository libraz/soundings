"""Controls for the swept-sine probe, on paths whose answer is already known.

The sweep is the one measurement here that can be checked against arithmetic
rather than against another measurement, so it is worth checking hard. Its
failure modes are all numbers: an inverse filter with the wrong tilt gives an
impulse response with 3 dB per octave of slope on it, which reads as the effect
having a treble lift; a harmonic window that catches its neighbour reports
distortion at an order that produced none; a missing loopback division reports
the converters' anti-aliasing filter as the machine rolling off.
"""

from __future__ import annotations

import numpy as np
import pytest

from soundings import probe

SR = 48000


def sent(**kwargs) -> probe.Sweep:
    return probe.make_sweep(SR, seconds=2.0, low_hz=40.0, high_hz=16000.0, pad=1.0, **kwargs)


def through(sweep: probe.Sweep, impulse: np.ndarray) -> np.ndarray:
    """What comes back when the sweep passes through a known impulse response."""
    return np.convolve(sweep.played, impulse)[: sweep.played.size]


def test_a_path_that_does_nothing_gives_an_impulse_at_zero() -> None:
    sweep = sent()
    found = probe.deconvolve(through(sweep, np.array([1.0])), sweep)
    # The peak sits at the sweep's own length, which is why the latency is read
    # against `origin` rather than off the peak.
    assert found.peak == pytest.approx(found.origin, abs=2)
    assert found.origin > sweep.seconds * SR * 0.9
    assert found.latency_ms == pytest.approx(0.0, abs=0.05)


def test_a_known_delay_is_recovered_as_the_latency() -> None:
    """The whole point of not needing a trigger: the sweep times itself."""
    sweep = sent()
    impulse = np.zeros(2000)
    impulse[1234] = 1.0
    found = probe.deconvolve(through(sweep, impulse), sweep)
    assert found.peak - found.origin == pytest.approx(1234, abs=2)
    assert found.latency_ms == pytest.approx(1234 / SR * 1000.0, abs=0.05)


def test_the_deconvolved_response_is_flat_for_a_flat_path() -> None:
    """An inverse filter with the sweep's own pink tilt left in it would give a
    response sloping 3 dB an octave, which reads as the path lifting the treble."""
    sweep = sent()
    found = probe.deconvolve(through(sweep, np.array([1.0])), sweep)
    window = found.around(before=0.05, after=0.05)
    spectrum = np.abs(np.fft.rfft(window))
    bins = np.fft.rfftfreq(window.size, 1 / SR)
    band = (bins > 200) & (bins < 8000)
    level = 20 * np.log10(spectrum[band] / spectrum[band].mean())
    assert np.abs(level).max() < 1.5


def test_a_filter_in_the_path_comes_back_as_that_filter() -> None:
    from scipy import signal as dsp

    sweep = sent()
    sos = dsp.butter(4, 2000.0 / (SR / 2), btype="low", output="sos")
    impulse = dsp.sosfilt(sos, np.concatenate([[1.0], np.zeros(4095)]))
    found = probe.deconvolve(through(sweep, impulse), sweep)

    window = found.around(before=0.05, after=0.05)
    recovered = np.abs(np.fft.rfft(window))
    truth = np.abs(np.fft.rfft(impulse[:4096], window.size))
    bins = np.fft.rfftfreq(window.size, 1 / SR)
    band = (bins > 100) & (bins < 12000)
    error = 20 * np.log10(recovered[band] / truth[band])
    assert np.abs(error - error.mean()).max() < 1.0


def test_a_nonlinear_path_puts_its_harmonics_where_the_sweep_says() -> None:
    """The reason for a sweep rather than noise: an overdrive's distortion is
    separated from its response instead of folded into it."""
    sweep = sent()
    played = sweep.played
    found = probe.deconvolve(played + 0.05 * played**2, sweep)

    second = next(h for h in found.harmonics if h.order == 2)
    assert second.arrived_ms == pytest.approx(
        sweep.seconds * np.log(2) / np.log(16000 / 40) * 1000.0, rel=1e-6
    )
    assert -40 < second.level_db < -15
    assert found.distortion_db > -40


def test_a_linear_path_reports_no_harmonics() -> None:
    """The negative control: every order has a window, and a window always has a
    largest value in it."""
    sweep = sent()
    found = probe.deconvolve(through(sweep, np.array([1.0])), sweep)
    assert all(h.level_db < -80 for h in found.harmonics), [h.level_db for h in found.harmonics]
    assert found.distortion_db < -70


def test_the_orders_do_not_land_in_each_others_windows() -> None:
    """A window wide enough to catch its neighbour reports distortion at an order
    that produced none, which is a claim about the circuit."""
    sweep = sent()
    played = sweep.played
    found = probe.deconvolve(played + 0.05 * played**3, sweep)
    by_order = {h.order: h.level_db for h in found.harmonics}
    assert by_order[3] > by_order[2] + 20
    assert by_order[3] > by_order[4] + 20


def test_the_chain_is_divided_out_rather_than_left_in_series() -> None:
    """Without this the converters' own filter reads as the machine rolling off."""
    from scipy import signal as dsp

    sweep = sent()
    chain = dsp.sosfilt(
        dsp.butter(4, 15000.0 / (SR / 2), btype="low", output="sos"),
        np.concatenate([[1.0], np.zeros(2047)]),
    )
    machine = dsp.sosfilt(
        dsp.butter(2, 800.0 / (SR / 2), btype="high", output="sos"),
        np.concatenate([[1.0], np.zeros(2047)]),
    )
    both = np.convolve(chain, machine)[:2048]

    measured = probe.deconvolve(through(sweep, both), sweep)
    reference = probe.deconvolve(through(sweep, chain), sweep)
    recovered = probe.divide_out(measured, reference, length=0.05)

    bins = np.fft.rfftfreq(recovered.size, 1 / SR)
    band = (bins > 100) & (bins < 10000)
    got = np.abs(np.fft.rfft(recovered))[band]
    want = np.abs(np.fft.rfft(machine, recovered.size))[band]
    error = 20 * np.log10(got / want)
    assert np.abs(error - error.mean()).max() < 1.5


def test_a_response_lost_in_the_noise_says_so_rather_than_returning_one() -> None:
    sweep = sent()
    rng = np.random.default_rng(9)
    found = probe.deconvolve(rng.standard_normal(sweep.played.size) * 0.5, sweep)
    assert not found.usable


def test_a_response_well_over_the_noise_is_usable() -> None:
    sweep = sent()
    rng = np.random.default_rng(9)
    noisy = through(sweep, np.array([1.0])) + rng.standard_normal(sweep.played.size) * 1e-4
    found = probe.deconvolve(noisy, sweep)
    assert found.usable
    assert found.noise_db < -50


def test_a_sweep_starts_and_ends_without_a_step() -> None:
    """A step at either end rings across the whole band and lands on the result."""
    sweep = sent()
    assert abs(sweep.signal[0]) < 1e-6
    assert abs(sweep.signal[-1]) < 1e-3


def test_the_sweep_stays_inside_nyquist_whatever_is_asked_for() -> None:
    sweep = probe.make_sweep(SR, seconds=1.0, low_hz=20.0, high_hz=40000.0)
    assert sweep.high_hz < SR / 2
