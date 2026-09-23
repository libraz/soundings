"""Each dynamics node does what its docstring promises, at the values models use.

Drive runs 0 to 40 dB, time constants 1 to 1000 ms, ratios 1.5 to infinity, hold
rates 1 to 16 kHz and word lengths 4 to 24 bits.

The alias bar is only a claim about the oversampler where the curve itself puts less
than the bar above the oversampled Nyquist: at 40 dB of drive a 997 Hz sine folds
-54 dB back at 8x however perfect the filters are. The test computes that ideal fold
from the curve's own harmonic series and requires it 10 dB under the bar before it
holds the node to the bar, so the fundamental is a checked choice and not a tuned one.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from soundings import render

from .test_render_graph import FS, drawn, graph_of, loaded, spec

ALIAS_BAR_DB = -80.0


def _shaper(curve, drive_db, oversample, **extra) -> dict:
    return {
        "id": "shape",
        "kind": "shaper",
        "input": "x",
        "curve": curve,
        "drive": spec(drive_db),
        "oversample": oversample,
        **extra,
    }


def _envelope(detector, topology, domain, attack_ms, release_ms, **extra) -> dict:
    return {
        "id": "env",
        "kind": "envelope",
        "input": "x",
        "detector": detector,
        "topology": topology,
        "domain": domain,
        "attack_ms": spec(attack_ms),
        "release_ms": spec(release_ms),
        **extra,
    }


def _computer(threshold_db, ratio, knee_db) -> dict:
    return {
        "id": "gc",
        "kind": "gain_computer",
        "input": "level",
        "threshold_db": spec(threshold_db),
        "ratio": spec(ratio),
        "knee_db": spec(knee_db),
    }


# ---------------------------------------------------------------- shaper


def test_an_identity_curve_gives_back_its_input():
    x = np.random.default_rng(3).uniform(-1.0, 1.0, 4096)
    model = graph_of([_shaper("points", 0.0, 1, points=[[-1.0, -1.0], [1.0, 1.0]])])
    np.testing.assert_allclose(drawn(model, x), x, rtol=0, atol=1e-15)


@pytest.mark.parametrize(
    ("curve", "expected"),
    [
        ("tanh", np.tanh),
        ("hard", lambda g: np.clip(g, -1.0, 1.0)),
        ("cubic", lambda g: np.where(np.abs(g) < 1.0, 1.5 * g - 0.5 * g**3, np.sign(g))),
    ],
)
@pytest.mark.parametrize("drive_db", [0.0, 6.0, 40.0])
def test_each_curve_is_its_formula_after_the_drive(curve, expected, drive_db):
    x = np.random.default_rng(4).uniform(-1.0, 1.0, 4096)
    y = drawn(graph_of([_shaper(curve, drive_db, 1)]), x)
    np.testing.assert_allclose(y, expected(10 ** (drive_db / 20) * x), rtol=0, atol=1e-15)


def _ideal_fold_db(f0: float, drive_db: float, oversample: int) -> float:
    """What tanh folds into the base band at `oversample` with perfect filters."""
    points = 1 << 16
    theta = 2 * np.pi * np.arange(points) / points
    power = np.abs(np.fft.rfft(np.tanh(10 ** (drive_db / 20) * np.sin(theta))) / points) ** 2
    freq = np.arange(power.size) * f0
    high = oversample * FS
    folded = np.abs(freq - np.round(freq / high) * high)
    aliased = (freq > high / 2) & (folded < FS / 2)
    return 10 * math.log10(power[aliased].sum() / power[freq < FS / 2].sum() + 1e-300)


@pytest.mark.parametrize("drive_db", [0.0, 20.0, 40.0])
def test_tanh_oversampled_eight_times_leaves_its_aliases_under_the_bar(drive_db):
    f0, oversample = 440.0, 8
    assert _ideal_fold_db(f0, drive_db, oversample) <= ALIAS_BAR_DB - 10.0
    n = np.arange(FS)
    y = drawn(graph_of([_shaper("tanh", drive_db, oversample)]), np.sin(2 * np.pi * f0 * n / FS))
    # The ends are where the resampler's filters run off the take.
    keep = slice(4000, FS - 4000)
    t = n[keep] / FS
    harmonics = np.arange(1, int(FS / 2 / f0) + 1) * f0
    basis = np.column_stack(
        [np.ones_like(t)]
        + [np.sin(2 * np.pi * f * t) for f in harmonics]
        + [np.cos(2 * np.pi * f * t) for f in harmonics]
    )
    fitted = basis @ np.linalg.lstsq(basis, y[keep], rcond=None)[0]
    aliased = y[keep] - fitted
    level = 10 * math.log10(np.sum(aliased**2) / np.sum(fitted**2))
    print(f"drive {drive_db} dB: aliases at {level:.1f} dB")
    assert level <= ALIAS_BAR_DB


def test_an_oversampled_shaper_is_refused_inside_a_loop():
    model = graph_of(
        [
            {
                "id": "sum",
                "kind": "mix",
                "inputs": ["x", "back"],
                "weights": {"x": spec(1.0), "back": spec(0.5)},
            },
            {**_shaper("tanh", 0.0, 8), "input": "sum"},
            {
                "id": "back",
                "kind": "delay",
                "input": "shape",
                "time_ms": spec(1.0),
                "interpolation": "none",
            },
        ],
        outputs={"y": "shape"},
    )
    with pytest.raises(ValueError, match="reads ahead"):
        drawn(model, np.zeros(1024))
    model["nodes"][1]["oversample"] = 1
    assert drawn(model, np.zeros(1024)).shape == (1024,)


# ---------------------------------------------------------------- envelope

DETECTORS = [("peak", "linear"), ("rms", "linear"), ("peak", "log")]


def _smoothed(y_db: np.ndarray, detector: str, domain: str) -> np.ndarray:
    """The quantity the detector smooths, recovered from its output in dB."""
    if domain == "log":
        return y_db
    return 10 ** (y_db / (20.0 if detector == "peak" else 10.0))


def _step_run(detector, topology, domain, attack_ms, release_ms, low, high, settle, hold):
    """Output at `low` for `settle` samples, `high` for `hold`, then `low` again for `hold`."""
    x = np.concatenate([np.full(settle, low), np.full(hold, high), np.full(hold, low)])
    node = _envelope(detector, topology, domain, attack_ms, release_ms, floor_db=spec(-120.0))
    if domain != "log":
        del node["floor_db"]
    y = _smoothed(drawn(graph_of([node]), x), detector, domain)
    q_low = _smoothed(np.array(20 * math.log10(low)), detector, domain)
    q_high = _smoothed(np.array(20 * math.log10(high)), detector, domain)
    return y, float(q_low), float(q_high)


CASES = [
    (detector, domain, topology, attack_ms, release_ms)
    for topology in ("branching", "decoupled")
    for detector, domain, attack_ms, release_ms in (
        [("peak", "linear", 1.0, 1000.0), ("peak", "linear", 1000.0, 1.0)]
        + [(d, m, 10.0, 100.0) for d, m in DETECTORS]
    )
]


# A one-pole run a sample at a time crosses 1 - 1/e within one sample of tau * fs, and
# a decay's summed area is tau * fs less half a sample per pole: one sample either way.
@pytest.mark.parametrize(("detector", "domain", "topology", "attack_ms", "release_ms"), CASES)
def test_an_envelope_reaches_63_percent_of_a_step_in_its_attack(
    detector, domain, topology, attack_ms, release_ms
):
    settle = int(20 * attack_ms * FS / 1000)
    hold = int(20 * (attack_ms + release_ms) * FS / 1000)
    y, low, high = _step_run(
        detector, topology, domain, attack_ms, release_ms, 0.1, 1.0, settle, hold
    )
    risen = (y[settle:] - low) / (high - low)
    crossed = int(np.argmax(risen >= 1.0 - math.exp(-1.0)))
    assert crossed == pytest.approx(attack_ms * FS / 1000, abs=1.0)


@pytest.mark.parametrize(("detector", "domain", "topology", "attack_ms", "release_ms"), CASES)
def test_a_decoupled_release_takes_attack_plus_release_and_a_branching_one_release(
    detector, domain, topology, attack_ms, release_ms
):
    settle = int(20 * attack_ms * FS / 1000)
    hold = int(20 * (attack_ms + release_ms) * FS / 1000)
    y, low, high = _step_run(
        detector, topology, domain, attack_ms, release_ms, 0.1, 1.0, settle, hold
    )
    falling = (y[settle + hold :] - low) / (high - low)
    tau_ms = release_ms + (attack_ms if topology == "decoupled" else 0.0)
    assert float(np.sum(falling)) == pytest.approx(tau_ms * FS / 1000, abs=1.0)


# ---------------------------------------------------------------- gain computer, vca


@pytest.mark.parametrize("ratio", [1.5, 4.0, math.inf])
@pytest.mark.parametrize("knee_db", [0.0, 6.0, 12.0])
def test_the_static_curve_has_slope_one_over_ratio_and_a_quadratic_knee(ratio, knee_db):
    threshold = -20.0
    level = np.linspace(-60.0, 0.0, 6001)
    model = graph_of(
        [
            _computer(threshold, ratio, knee_db),
            {"id": "amp", "kind": "vca", "input": "x", "control": "gc"},
        ],
        inputs=("level", "x"),
        outputs={"y": "amp"},
    )
    y = render.run(loaded(model), {"level": level, "x": 10 ** (level / 20)}, {})["y"]
    out = 20 * np.log10(y)
    over = level - threshold
    below, above = 2 * over <= -knee_db, 2 * over >= knee_db
    inside = ~below & ~above
    np.testing.assert_allclose(out[below], level[below], rtol=0, atol=1e-9)
    slope = np.diff(out[above]) / np.diff(level[above])
    np.testing.assert_allclose(slope, 1.0 / ratio, rtol=0, atol=1e-9)
    np.testing.assert_allclose(out[above], threshold + over[above] / ratio, rtol=0, atol=1e-9)
    if knee_db:
        curve = np.polyfit(over[inside], out[inside] - level[inside], 2)
        expected = (1.0 / ratio - 1.0) / (2.0 * knee_db) * np.poly1d([1.0, knee_db / 2]) ** 2
        np.testing.assert_allclose(curve, expected.coeffs, rtol=0, atol=1e-9)


def test_a_vca_is_its_input_times_its_control():
    rng = np.random.default_rng(5)
    x, c = rng.standard_normal(256), rng.standard_normal(256)
    model = graph_of(
        [{"id": "amp", "kind": "vca", "input": "x", "control": "c"}], inputs=("x", "c")
    )
    y = render.run(loaded(model), {"x": x, "c": c}, {})["y"]
    np.testing.assert_allclose(y, x * c, rtol=0, atol=0)


# ---------------------------------------------------------------- hold, quantize


@pytest.mark.parametrize("rate_hz", [1000, 3000, 7000, 16000])
def test_a_hold_changes_only_every_one_over_its_rate(rate_hz):
    x = np.random.default_rng(6).standard_normal(FS // 4)
    y = drawn(graph_of([{"id": "sh", "kind": "hold", "input": "x", "rate_hz": spec(rate_hz)}]), x)
    n = np.arange(x.size)
    tick = n * rate_hz // FS
    taken = (tick * FS + rate_hz - 1) // rate_hz
    np.testing.assert_array_equal(y, x[taken])
    changes = np.flatnonzero(np.diff(y)) + 1
    np.testing.assert_array_equal(changes, np.unique(taken)[1:])


@pytest.mark.parametrize("bits", [4, 8, 16, 18, 20, 24])
def test_a_quantizer_gives_at_most_two_to_the_bits_values_on_its_grid(bits):
    step = 2.0 ** (1 - bits)
    near_zero = [0.0, step / 4, -step / 4]
    x = np.concatenate([np.random.default_rng(7).uniform(-1.2, 1.2, 1 << 16), near_zero])
    y = drawn(graph_of([{"id": "q", "kind": "quantize", "input": "x", "bits": bits}]), x)
    assert np.unique(y).size <= 2**bits
    np.testing.assert_array_equal(y / step, np.round(y / step))
    inside = (x >= -1.0) & (x < 1.0 - step)
    assert np.abs(y[inside] - x[inside]).max() <= step / 2
    assert y.min() == -1.0 and y.max() == 1.0 - step
    np.testing.assert_array_equal(y[-len(near_zero) :], 0.0)


# ---------------------------------------------------------------- refusals


VALID = {
    "shaper": _shaper("tanh", 0.0, 1),
    "envelope": _envelope("peak", "branching", "linear", 1.0, 10.0),
    "gain_computer": {**_computer(-20.0, 4.0, 6.0), "input": "x"},
    "vca": {"id": "amp", "kind": "vca", "input": "x", "control": "x"},
    "hold": {"id": "sh", "kind": "hold", "input": "x", "rate_hz": spec(1000.0)},
    "quantize": {"id": "q", "kind": "quantize", "input": "x", "bits": 18},
}

REQUIRED = {
    "shaper": ("input", "curve", "drive", "oversample"),
    "envelope": ("input", "detector", "topology", "domain", "attack_ms", "release_ms"),
    "gain_computer": ("input", "threshold_db", "ratio", "knee_db"),
    "vca": ("input", "control"),
    "hold": ("input", "rate_hz"),
    "quantize": ("input", "bits"),
}

MISSING = [(kind, key) for kind, keys in REQUIRED.items() for key in keys]


def test_every_kind_the_dynamics_module_draws_is_checked():
    from soundings.render import dynamics

    assert set(dynamics.NODES) == set(VALID)


@pytest.mark.parametrize("kind", sorted(VALID))
def test_each_valid_node_loads(kind):
    assert loaded(graph_of([VALID[kind]]))["model"]["kind"] == "graph"


@pytest.mark.parametrize(("kind", "key"), MISSING)
def test_a_node_missing_a_required_value_is_refused(kind, key):
    node = copy.deepcopy(VALID[kind])
    del node[key]
    with pytest.raises(ValueError, match=f"has no `{key}`"):
        loaded(graph_of([node]))


@pytest.mark.parametrize(
    ("kind", "change", "reason"),
    [
        ("shaper", {"curve": "sigmoid"}, "not one of"),
        ("shaper", {"curve": "points"}, "no `points`"),
        ("shaper", {"oversample": 0}, "oversample"),
        ("shaper", {"oversample": 2.0}, "oversample"),
        ("envelope", {"detector": "average"}, "not one of"),
        ("envelope", {"topology": "smooth"}, "not one of"),
        ("envelope", {"domain": "db"}, "not one of"),
        ("envelope", {"domain": "log"}, "no `floor_db`"),
        ("envelope", {"detector": "rms", "domain": "log", "floor_db": spec(-120.0)}, "rms"),
        ("quantize", {"bits": 0}, "bits"),
        ("quantize", {"bits": 18.0}, "bits"),
    ],
)
def test_a_node_with_a_value_it_cannot_draw_is_refused(kind, change, reason):
    node = {**copy.deepcopy(VALID[kind]), **change}
    with pytest.raises(ValueError, match=reason):
        loaded(graph_of([node]))
