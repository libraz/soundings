"""The pitch shifter does what its docstring promises, across +-12 semitones and
windows of 10 to 100 ms.

A sine at `f` comes out as lines at `f + k/T` (`T` the window). The largest is read
off a Hann-windowed spectrum zero-padded to `PADDED` points, so the reading resolves
one padded bin: it must sit within that of `f * ratio` when `f (1 - ratio) T` is an
even integer, and within `1/T` of it for any other `f`.

The ripple is read exactly rather than estimated: the node is linear in its input, so
driving it with a sine and a cosine gives the real and imaginary parts of its answer
to `exp(i w n)`, whose magnitude is the envelope sample by sample. `f` is chosen with
`f (1 - ratio) T` two thirds past an even integer, where the closed form dips to 0.5
at every crossover. The pointers read through `lagrange3`, whose magnitude error at
the 1.1 to 2.2 kHz used here is under 0.2%, so `RIPPLE_TOLERANCE` of 0.01 is five
times the interpolator's own error and a fiftieth of the dip. A shifter whose
pointers are not half a window apart swings its envelope across the full range and
misses by 0.5 or more.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from soundings.render.linear import REACH

from .test_render_graph import FS, drawn, graph_of, spec

SEMITONES = (-12, -7, -1, 1, 7, 12)
WINDOWS_MS = (10.0, 30.0, 100.0)
CROSSFADES = ("linear", "hann", "s_curve")
SETTLE_S = 0.2
STEADY_S = 1.2
PADDED = 2**20
RIPPLE_TOLERANCE = 0.01
TARGET_HZ = 2000.0


def _shifter(ratio, window_ms, crossfade, interpolation="lagrange3") -> dict:
    return graph_of(
        [
            {
                "id": "shift",
                "kind": "pitch",
                "input": "x",
                "ratio": spec(ratio),
                "window_ms": spec(window_ms),
                "crossfade": crossfade,
                "interpolation": interpolation,
            }
        ]
    )


def _ratio(semitones: int) -> float:
    return 2.0 ** (semitones / 12.0)


def _tone_at(ratio: float, window_ms: float, past_even: float) -> float:
    """The `f` near TARGET_HZ whose `f (1 - ratio) T` is an even integer plus `past_even`."""
    per_cycle = 1.0 / (abs(1.0 - ratio) * window_ms / 1000.0)
    j = max(round((TARGET_HZ / per_cycle - past_even) / 2.0), 0 if past_even else 1)
    return (2 * j + past_even) * per_cycle


def _n() -> np.ndarray:
    return np.arange(int((SETTLE_S + STEADY_S) * FS))


def _steady(y: np.ndarray) -> np.ndarray:
    return y[int(SETTLE_S * FS) :]


def _peak_hz(y: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(y * np.hanning(y.size), PADDED))
    return float(np.argmax(spectrum)) * FS / PADDED


def _crossfade_gain(shape: str, t: np.ndarray) -> np.ndarray:
    if shape == "linear":
        return t
    if shape == "hann":
        return np.sin(np.pi * t / 2.0) ** 2
    return t * t * (3.0 - 2.0 * t)


def _envelope(n: np.ndarray, f: float, ratio: float, window_ms: float, shape: str) -> np.ndarray:
    """`sqrt(1 - 4 g_a g_b sin^2(pi f (1 - ratio) T / 2))`, as the docstring states it."""
    period = window_ms * FS / 1000.0
    t = 1.0 - np.abs(2.0 * ((n / period) % 1.0) - 1.0)
    both = _crossfade_gain(shape, t) * _crossfade_gain(shape, 1.0 - t)
    u = f * (1.0 - ratio) * window_ms / 1000.0
    return np.sqrt(1.0 - 4.0 * both * np.sin(np.pi * u / 2.0) ** 2)


# ---------------------------------------------------------------- frequency


@pytest.mark.parametrize("crossfade", CROSSFADES)
@pytest.mark.parametrize("window_ms", WINDOWS_MS)
@pytest.mark.parametrize("semitones", SEMITONES)
def test_a_sine_on_an_even_excursion_comes_out_at_f_times_the_ratio(
    semitones, window_ms, crossfade
):
    ratio = _ratio(semitones)
    f = _tone_at(ratio, window_ms, 0.0)
    y = drawn(_shifter(ratio, window_ms, crossfade), np.sin(2 * np.pi * f * _n() / FS))
    assert abs(_peak_hz(_steady(y)) - f * ratio) <= FS / PADDED


@pytest.mark.parametrize("crossfade", CROSSFADES)
@pytest.mark.parametrize("window_ms", WINDOWS_MS)
@pytest.mark.parametrize("semitones", SEMITONES)
def test_a_sine_anywhere_comes_out_within_a_window_line_of_f_times_the_ratio(
    semitones, window_ms, crossfade
):
    ratio = _ratio(semitones)
    f = 1000.0
    y = drawn(_shifter(ratio, window_ms, crossfade), np.sin(2 * np.pi * f * _n() / FS))
    assert abs(_peak_hz(_steady(y)) - f * ratio) <= 1000.0 / window_ms + FS / PADDED


# ---------------------------------------------------------------- ripple


@pytest.mark.parametrize("crossfade", CROSSFADES)
@pytest.mark.parametrize("window_ms", WINDOWS_MS)
@pytest.mark.parametrize("semitones", SEMITONES)
def test_the_envelope_on_a_sine_is_the_crossfade_s_closed_form(semitones, window_ms, crossfade):
    ratio = _ratio(semitones)
    f = _tone_at(ratio, window_ms, 2.0 / 3.0)
    n = _n()
    model = _shifter(ratio, window_ms, crossfade)
    w = 2 * np.pi * f * n / FS
    measured = np.abs(drawn(model, np.cos(w)) + 1j * drawn(model, np.sin(w)))
    expected = _envelope(n, f, ratio, window_ms, crossfade)
    assert np.min(_steady(expected)) == pytest.approx(0.5, abs=1e-6)
    assert np.max(np.abs(_steady(measured) - _steady(expected))) <= RIPPLE_TOLERANCE


# ---------------------------------------------------------------- identity


@pytest.mark.parametrize("crossfade", CROSSFADES)
@pytest.mark.parametrize("interpolation", sorted(REACH))
def test_a_ratio_of_one_is_the_input_delayed_by_the_interpolator_s_reach(interpolation, crossfade):
    x = np.random.default_rng(7).standard_normal(4096)
    y = drawn(_shifter(1.0, 30.0, crossfade, interpolation), x)
    late = REACH[interpolation]
    np.testing.assert_allclose(y[late:], x[: x.size - late], rtol=0, atol=1e-12)


# ---------------------------------------------------------------- phase
#
# A sweep started `phi` cycles in is the phase-0 sweep `phi T` later, so drawing `x`
# at `phi` is drawing `x` delayed by `phi T` at phase 0, read `phi T` later. Both
# sides compute the same delays from sweep positions that differ only in rounding,
# so 1e-9 bounds them; half a cycle swaps the pointers and changes nothing.

PHASE_WINDOW_MS = 30.0


def _noise() -> np.ndarray:
    return np.random.default_rng(3).standard_normal(20000)


@pytest.mark.parametrize("crossfade", CROSSFADES)
def test_a_quarter_cycle_of_sweep_phase_is_the_sweep_a_quarter_window_later(crossfade):
    x = _noise()
    model = _shifter(1.5, PHASE_WINDOW_MS, crossfade)
    shift = int(PHASE_WINDOW_MS * FS / 1000.0 / 4)
    at_quarter = drawn(model, x, lfo_phase=0.25)
    later = drawn(model, np.concatenate([np.zeros(shift), x[:-shift]]))[shift:]
    steady = int(SETTLE_S * FS)
    assert np.max(np.abs(at_quarter - drawn(model, x))) > 0.1
    np.testing.assert_allclose(at_quarter[steady:-shift], later[steady:], rtol=0, atol=1e-9)


@pytest.mark.parametrize("crossfade", CROSSFADES)
def test_half_a_cycle_of_sweep_phase_swaps_the_pointers_and_draws_the_same(crossfade):
    x = _noise()
    model = _shifter(1.5, PHASE_WINDOW_MS, crossfade)
    np.testing.assert_allclose(drawn(model, x, lfo_phase=0.5), drawn(model, x), rtol=0, atol=1e-9)


# ---------------------------------------------------------------- refusals


@pytest.mark.parametrize("key", ["input", "ratio", "window_ms", "crossfade", "interpolation"])
def test_a_pitch_node_missing_a_required_value_is_refused(key):
    model = _shifter(1.5, 30.0, "hann")
    del model["nodes"][0][key]
    with pytest.raises(ValueError, match=f"has no `{key}`"):
        drawn(model, np.zeros(16))


@pytest.mark.parametrize(
    ("key", "value", "reason"),
    [
        ("crossfade", "cosine", "crossfades by 'cosine'"),
        ("interpolation", "cubic", "interpolates by 'cubic'"),
    ],
)
def test_a_pitch_node_outside_its_vocabulary_is_refused(key, value, reason):
    model = _shifter(1.5, 30.0, "hann")
    model["nodes"][0][key] = value
    with pytest.raises(ValueError, match=reason):
        drawn(model, np.zeros(16))


@pytest.mark.parametrize("key", ["ratio", "window_ms"])
def test_a_pitch_node_driven_by_a_control_is_refused(key):
    model = _shifter(1.5, 30.0, "hann")
    node = copy.deepcopy(model["nodes"][0])
    node[key] = {"control": "lfo", "map": {"kind": "points", "points": [[-1, 10.0], [1, 20.0]]}}
    lfo = {
        "id": "lfo",
        "kind": "lfo",
        "shape": "sine",
        "rate_hz": spec(1.0),
        "phase_offset": spec(0.0),
    }
    model["nodes"] = [lfo, node]
    with pytest.raises(ValueError, match=f"`{key}` driven by a control"):
        drawn(model, np.zeros(16))
