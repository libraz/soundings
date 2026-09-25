"""Each linear node does what its docstring promises, at the values models use.

Delays, modulated delays and oscillators are checked against the closed form they
promise. Sections and whole `lti` chains are checked against `reproduce.response()`:
an impulse is drawn until what is left of it is 120 dB under its peak (at most 2**20
samples) and its magnitude held within 0.01 dB of the curve wherever the curve is
above -60 dB.

Settings for the `response()` comparisons: every address a stage or model reads is
set to 64, then each address alone to 0, 32, 64, 96 and 127, then all of them together
to each of those. That walks every addressed parameter across its range; the bytes a
map lists are already walked stage by stage in `test_reproduce_coefficients.py`.
Two `lti` models are excluded: their loops return with no delay, which a sample loop
has no order for, so they are drawn only as a response.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import signal

from soundings import render, reproduce

from .test_render_graph import FS, MODELS, drawn, graph_of, loaded, spec

INTERPOLATIONS = ("none", "linear", "allpass", "lagrange3")
GRID = (0, 32, 64, 96, 127)
UNDELAYED_LOOPS = (
    "0120-a-loop-with-nothing-in-it.json",
    "0120-a-loop-through-a-blocked-return.json",
)
FLOOR_DB = -60.0
TOLERANCE_DB = 0.01


def _delay(time_ms, interpolation, **extra) -> dict:
    return graph_of(
        [
            {
                "id": "line",
                "kind": "delay",
                "input": "x",
                "time_ms": spec(time_ms),
                "interpolation": interpolation,
                **extra,
            }
        ]
    )


# ---------------------------------------------------------------- delay


@pytest.mark.parametrize("interpolation", INTERPOLATIONS)
@pytest.mark.parametrize("samples", [1, 32, 20000])
def test_a_whole_sample_delay_moves_an_impulse_exactly(interpolation, samples):
    x = np.zeros(samples + 64)
    x[5] = 1.0
    y = drawn(_delay(samples * 1000.0 / FS, interpolation), x)
    expected = np.zeros_like(x)
    expected[5 + samples] = 1.0
    np.testing.assert_allclose(y, expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize(
    ("interpolation", "tolerance"), [("linear", 1e-9), ("lagrange3", 1e-9), ("allpass", 1e-6)]
)
@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75])
def test_a_fractional_delay_has_its_group_delay_at_dc(interpolation, tolerance, fraction):
    x = np.zeros(4096)
    x[5] = 1.0
    y = drawn(_delay((32 + fraction) * 1000.0 / FS, interpolation), x)
    n = np.arange(y.size)
    assert abs(np.sum(n * y) / np.sum(y) - 5 - 32 - fraction) <= tolerance


@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75])
def test_no_interpolation_truncates(fraction):
    x = np.zeros(128)
    x[5] = 1.0
    y = drawn(_delay((32 + fraction) * 1000.0 / FS, "none"), x)
    assert np.flatnonzero(y).tolist() == [37]
    assert y[37] == 1.0


@pytest.mark.parametrize(
    ("rate_hz", "depth_ms", "time_ms"), [(0.05, 0.1, 0.5), (1.0, 2.0, 3.0), (10.0, 10.0, 12.0)]
)
def test_a_modulated_delay_reads_between_time_plus_and_minus_depth(rate_hz, depth_ms, time_ms):
    """Read off the phase of a drawn sine: `y = sin(w (n - d(n)))` gives `d(n)` back."""
    periods = 1.5
    size = int(periods * FS / rate_hz)
    tone = 500.0
    w = 2 * np.pi * tone / FS
    n = np.arange(size)
    model = graph_of(
        [
            {
                "id": "lfo",
                "kind": "lfo",
                "shape": "sine",
                "rate_hz": spec(rate_hz),
                "phase_offset": spec(0.0),
            },
            {
                "id": "line",
                "kind": "delay",
                "input": "x",
                "time_ms": spec(time_ms),
                "interpolation": "linear",
                "modulated_by": {"control": "lfo", "depth_ms": spec(depth_ms)},
            },
        ]
    )
    first = int((time_ms + depth_ms) * FS / 1000.0) + 1  # before this the line reads silence
    y = drawn(model, np.sin(w * n))[first:]
    n = n[first:]
    phase = np.unwrap(np.angle(signal.hilbert(y))) + np.pi / 2
    read = n - phase / w
    centre = time_ms * FS / 1000.0
    wanted = centre + depth_ms * FS / 1000.0 * np.sin(2 * np.pi * rate_hz * n / FS)
    # A phase fixes the delay only to within a period of the tone; take the whole periods.
    read += np.round(np.mean(wanted - read) * w / (2 * np.pi)) * 2 * np.pi / w
    # The analytic signal wraps at the ends; a tenth off each keeps the exact curve within 0.03.
    inner = slice(y.size // 10, y.size - y.size // 10)
    assert np.abs(read[inner] - wanted[inner]).max() <= 0.05


# ---------------------------------------------------------------- lfo


def _lfo(shape, rate_hz, phase_offset=0.0, **extra) -> dict:
    return {
        "id": "lfo",
        "kind": "lfo",
        "shape": shape,
        "rate_hz": spec(rate_hz),
        "phase_offset": spec(phase_offset),
        **extra,
    }


@pytest.mark.parametrize("rate_hz", [0.05, 0.5, 6.7, 10.0])
def test_an_lfo_keeps_its_period_without_drift(rate_hz):
    size = 60 * FS
    y = drawn(graph_of([_lfo("sine", rate_hz)]), np.zeros(size))
    n = np.arange(size)
    assert np.abs(y - np.sin(2 * np.pi * rate_hz * n / FS)).max() <= 1e-9


def test_the_phase_offset_and_the_starting_phase_are_in_cycles():
    size = 4 * FS
    model = graph_of([_lfo("sine", 2.0, 0.25)])
    y = drawn(model, np.zeros(size), lfo_phase=0.1)
    n = np.arange(size)
    assert np.abs(y - np.sin(2 * np.pi * (2.0 * n / FS + 0.35))).max() <= 1e-9


@pytest.mark.parametrize(
    ("shape", "extra", "at"),
    [
        ("sine", {}, [0.0, 1.0, 0.0, -1.0]),
        ("triangle", {}, [0.0, 1.0, 0.0, -1.0]),
        ("square", {}, [1.0, 1.0, -1.0, -1.0]),
        ("saw", {}, [0.0, 0.5, -1.0, -0.5]),
        ("points", {"points": [[0.0, 0.0], [0.5, 1.0]]}, [0.0, 0.5, 1.0, 0.5]),
    ],
)
def test_each_shape_at_the_quarter_cycles(shape, extra, at):
    y = drawn(graph_of([_lfo(shape, FS / 64.0, **extra)]), np.zeros(64))
    np.testing.assert_allclose(y[[0, 16, 32, 48]], at, atol=1e-12)


def test_a_slewed_rate_reaches_63_percent_of_a_step_in_its_time_constant():
    slew_s = 0.1
    model = graph_of(
        [
            _lfo("square", 1.0),
            {
                "id": "saw",
                "kind": "lfo",
                "shape": "saw",
                "phase_offset": spec(0.1),
                "slew_s": spec(slew_s),
                "rate_hz": {
                    "control": "lfo",
                    "map": {"kind": "points", "points": [[-1.0, 1.0], [1.0, 5.0]]},
                    "source": "document",
                    "rests_on": [],
                    "fitted_on": [],
                },
            },
        ]
    )
    y = drawn(model, np.zeros(FS))
    rate = np.diff(y) * FS / 2.0
    step = FS // 2
    assert rate[step - 1] == pytest.approx(5.0, abs=1e-6)
    reached = (5.0 - rate[step - 1 + int(slew_s * FS)]) / (5.0 - 1.0)
    assert reached == pytest.approx(1.0 - np.exp(-1.0), abs=1e-3)


# ---------------------------------------------------------------- gain, mix, pan


def test_gain_in_decibels_and_as_a_ratio():
    x = np.random.default_rng(1).standard_normal(64)
    db = graph_of([{"id": "g", "kind": "gain", "input": "x", "gain": spec(-6.0), "unit": "db"}])
    ratio = graph_of(
        [{"id": "g", "kind": "gain", "input": "x", "gain": spec(0.5), "unit": "ratio"}]
    )
    np.testing.assert_allclose(drawn(db, x), x * 10 ** (-6.0 / 20), rtol=1e-15)
    np.testing.assert_allclose(drawn(ratio, x), x * 0.5, rtol=1e-15)


def test_a_mix_is_the_weighted_sum_of_its_inputs():
    x = np.random.default_rng(2).standard_normal(64)
    model = graph_of(
        [
            {"id": "g", "kind": "gain", "input": "x", "gain": spec(2.0), "unit": "ratio"},
            {
                "id": "m",
                "kind": "mix",
                "inputs": ["x", "g"],
                "weights": {"x": spec(0.25), "g": spec(-1.0)},
            },
        ]
    )
    np.testing.assert_allclose(drawn(model, x), 0.25 * x - 2.0 * x, rtol=1e-15)


@pytest.mark.parametrize("position", [0, 32, 64, 100, 127])
def test_a_pan_takes_its_two_sides_from_the_law_it_names(position):
    law = "pan-a-sine-cosine-pair.json"
    x = np.random.default_rng(3).standard_normal(64)
    model = graph_of(
        [{"id": "p", "kind": "pan", "input": "x", "position": spec(position), "law": law}],
        outputs={"y": "p.left", "z": "p.right"},
    )
    out = render.run(loaded(model), {"x": x}, {})
    left, right = reproduce._pan_sides(json.loads((MODELS / law).read_text()), position)
    np.testing.assert_allclose(out["y"], x * left, rtol=1e-15)
    np.testing.assert_allclose(out["z"], x * right, rtol=1e-15)


# ---------------------------------------------------------------- against response()


def _lti_models() -> list[tuple[str, dict]]:
    found = []
    for path in sorted(MODELS.glob("*.json")):
        model = json.loads(path.read_text())
        if model["model"]["kind"] == "lti":
            found.append((path.name, model))
    return found


def _addresses(node) -> list[str]:
    found = set()
    if isinstance(node, dict):
        if "byte" in node and "map" in node:
            found.add(node["byte"])
        for value in node.values():
            found.update(_addresses(value))
    elif isinstance(node, list):
        for value in node:
            found.update(_addresses(value))
    return sorted(found)


def _settings(addresses: list[str]) -> list[dict[str, int]]:
    settings = [dict.fromkeys(addresses, 64)]
    for address in addresses:
        for byte in GRID:
            settings.append({**dict.fromkeys(addresses, 64), address: byte})
    settings.extend(dict.fromkeys(addresses, byte) for byte in GRID)
    return settings


def impulse_response(model: dict, bytes_now: dict[str, int]) -> np.ndarray:
    """Drawn until the last eighth, summed, is 120 dB under the peak, or 2**20 samples.

    Summed, because a bin sees the part cut off as the sum of it: a pole this near
    one leaves samples each under -120 dB that add up to hundredths of a decibel.
    """
    size = 4096
    while True:
        x = np.zeros(size)
        x[0] = 1.0
        h = drawn(model, x, bytes_now)
        peak = np.abs(h).max()
        if np.abs(h[-size // 8 :]).sum() <= peak * 1e-6 or size >= 2**20:
            return h
        size *= 2


def worst_db(h: np.ndarray, reference: dict, bytes_now: dict[str, int]) -> float:
    """Largest magnitude difference from `response()`, where it is above -60 dB."""
    fs = float(reference["sample_rate_hz"])
    bins = np.unique(np.round(np.geomspace(10.0, 0.499 * fs, 1500) / fs * h.size).astype(int))
    freq = bins * fs / h.size
    drawn_h = np.fft.rfft(h)[bins]
    curve = reproduce.response(reference, bytes_now, freq)
    held = np.abs(curve) > 10 ** (FLOOR_DB / 20)
    if not held.any():
        return 0.0
    return float(np.abs(20 * np.log10(np.abs(drawn_h[held]) / np.abs(curve[held]))).max())


def _as_node_value(param):
    if isinstance(param, dict) and "fixed" in param:
        return spec(param["fixed"])
    if isinstance(param, dict) and "byte" in param:
        return {**param, "source": "law", "rests_on": [], "fitted_on": []}
    return param


def _feedback_free_stages() -> list:
    cases = []
    for name, model in _lti_models():
        for i, stage in enumerate(model["chain"]):
            if not (stage["kind"] == "allpass-chain" and "feedback" in stage):
                cases.append(pytest.param(model, stage, id=f"{name}[{i}]"))
    return cases


@pytest.mark.parametrize(("model", "stage"), _feedback_free_stages())
def test_a_section_is_the_response_of_its_stage(model, stage):
    addresses = _addresses(stage)
    node = {"id": "s", "kind": "section", "input": "x", "stage": stage["kind"]}
    node.update({k: _as_node_value(v) for k, v in stage.items() if k != "kind"})
    graph = graph_of([node], rate=model["sample_rate_hz"], bound=addresses)
    alone = {"sample_rate_hz": model["sample_rate_hz"], "chain": [stage]}
    for bytes_now in _settings(addresses):
        h = impulse_response(graph, bytes_now)
        assert worst_db(h, alone, bytes_now) <= TOLERANCE_DB, bytes_now


def test_a_section_driven_by_a_constant_control_is_the_fixed_section():
    x = np.random.default_rng(4).standard_normal(2000)
    shelf = {
        "id": "s",
        "kind": "section",
        "input": "x",
        "stage": "shelf",
        "side": "low",
        "order": 1,
        "gain_db": spec(9.0),
    }
    fixed = graph_of([{**shelf, "corner_hz": spec(2275.0)}])
    controlled = graph_of(
        [
            _lfo("points", 1.0, points=[[0.0, 0.5], [0.5, 0.5]]),
            {
                **shelf,
                "corner_hz": {
                    "control": "lfo",
                    "map": {"kind": "points", "points": [[-1.0, 100.0], [1.0, 3000.0]]},
                    "source": "document",
                    "rests_on": [],
                    "fitted_on": [],
                },
            },
        ]
    )
    assert np.abs(drawn(controlled, x) - drawn(fixed, x)).max() <= 1e-9


MOVED_STAGES = {
    "allpass-chain": {"sections": 8, "mix": 1.0},
    "allpass-chain-inverted": {"sections": 4, "mix": -0.7},
    "allpass-chain-resonant": {"sections": 4, "mix": 0.8, "q": 1.3},
    "peaking": {"q": 2.0, "gain_db": 12.0},
    "pole": {"side": "low", "sections": 2, "q": 3.0},
    "pole-one-multiply": {"side": "high", "sections": 1, "form": "one-multiply"},
    "shelf": {"side": "high", "order": 2, "gain_db": -6.0},
}
"""A stage per section kind, with the value a control moves left to the test."""

_MOVED = {"peaking": "centre_hz"}
_AS_WRITTEN = ("side", "form", "order")


def _kind(name: str) -> str:
    return "allpass-chain" if name.startswith("allpass-chain") else name.split("-")[0]


def _moved_stage(name: str, hz) -> dict:
    kind = _kind(name)
    stage = {"kind": kind, _MOVED.get(kind, "corner_hz"): {"fixed": hz}}
    for key, value in MOVED_STAGES[name].items():
        written = key in _AS_WRITTEN or (kind == "allpass-chain" and key == "sections")
        stage[key] = value if written else {"fixed": value}
    return stage


@pytest.mark.parametrize("name", sorted(MOVED_STAGES))
def test_the_sections_for_many_frequencies_are_the_sections_built_one_at_a_time(name):
    hz = np.geomspace(40.0, 12000.0, 257)
    many = reproduce.section_sos(_moved_stage(name, hz), {}, FS)
    one_at_a_time = np.stack([reproduce.section_sos(_moved_stage(name, f), {}, FS) for f in hz])
    assert many.shape == one_at_a_time.shape
    assert np.allclose(many, one_at_a_time, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("name", sorted(MOVED_STAGES))
def test_a_section_a_control_moves_is_rebuilt_at_every_sample(name):
    size = FS // 2
    x = np.random.default_rng(7).standard_normal(size)
    stage = _moved_stage(name, 0.0)
    node = {"id": "s", "kind": "section", "input": "x", "stage": stage.pop("kind")}
    node.update({k: spec(v["fixed"]) if isinstance(v, dict) else v for k, v in stage.items()})
    node[_MOVED.get(node["stage"], "corner_hz")] = {
        "control": "lfo",
        "map": {"kind": "points", "points": [[-1.0, 200.0], [1.0, 4000.0]]},
        "source": "document",
        "rests_on": [],
        "fitted_on": [],
    }
    y = drawn(graph_of([_lfo("sine", 3.0), node]), x)

    hz = 200.0 + (np.sin(2 * np.pi * 3.0 * np.arange(size) / FS) + 1.0) / 2.0 * 3800.0
    expected = np.empty(size)
    zi = None
    for i in range(size):
        sos = reproduce.section_sos(_moved_stage(name, hz[i]), {}, FS)
        zi = np.zeros((len(sos), 2)) if zi is None else zi
        expected[i : i + 1], zi = signal.sosfilt(sos, x[i : i + 1], zi=zi)
    assert np.abs(y - expected).max() <= 1e-7 * np.abs(expected).max()


def test_a_control_may_not_decide_how_many_sections_a_stage_has():
    stage = {"kind": "pole", "side": "low", "corner_hz": {"fixed": 1000.0}}
    stage["sections"] = {"fixed": np.array([1.0, 2.0])}
    with pytest.raises(ValueError, match="sections"):
        reproduce.section_sos(stage, {}, FS)


def _lti_graph(name: str, model: dict) -> dict:
    return graph_of(
        [{"id": "eq", "kind": "lti", "input": "x", "model": name}],
        rate=model["sample_rate_hz"],
        bound=_addresses(model["chain"]),
    )


def _feedback_models() -> list:
    return [
        pytest.param(name, model, id=name)
        for name, model in _lti_models()
        if name not in UNDELAYED_LOOPS
        and any(s["kind"] == "allpass-chain" and "feedback" in s for s in model["chain"])
    ]


def test_four_models_close_their_loop_through_a_delay():
    assert len(_feedback_models()) == 4


@pytest.mark.parametrize(("name", "model"), _feedback_models())
def test_a_loop_expanded_in_time_is_the_loop_the_formula_closes(name, model):
    stage = next(s for s in model["chain"] if "feedback" in s)
    others = [a for a in _addresses(model["chain"]) if a != stage["feedback"]["byte"]]
    graph = _lti_graph(name, model)
    for byte in range(0, 128, 16):
        bytes_now = {**dict.fromkeys(others, 64), stage["feedback"]["byte"]: byte}
        h = impulse_response(graph, bytes_now)
        assert worst_db(h, model, bytes_now) <= TOLERANCE_DB, bytes_now


def _drawable_lti_models() -> list:
    return [
        pytest.param(name, model, id=name)
        for name, model in _lti_models()
        if name not in UNDELAYED_LOOPS
    ]


def test_sixty_six_lti_models_are_drawn_in_time():
    assert len(_drawable_lti_models()) == 66


@pytest.mark.parametrize(("name", "model"), _drawable_lti_models())
def test_an_lti_model_drawn_in_time_is_its_response(name, model):
    graph = _lti_graph(name, model)
    for bytes_now in _settings(_addresses(model["chain"])):
        h = impulse_response(graph, bytes_now)
        assert worst_db(h, model, bytes_now) <= TOLERANCE_DB, bytes_now
