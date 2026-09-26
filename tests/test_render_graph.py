"""A graph is refused at the door for each thing that would make it unreadable, and a
loop is drawn the same whether it is computed a sample or a block at a time.

Each refusal is one defect on an otherwise loadable model, so each is refused for
its own reason. Printed rows come from the fixture unit under `render-fixtures/root`,
whose `01 24` prints `40 03 03`, `04` and `05`.

The round trip through the model's rate is checked on a bypassed take of the unit
when one is under `.cache/takes/`. A take is ROM output and is never copied into the
tree, so without one the test skips.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from soundings import render, reproduce, takes

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODELS = ROOT / "inferences" / "models"
FIXTURE_ROOT = HERE / "data" / "render-fixtures" / "root"
PRINTED = ("40 03 03", "40 03 04", "40 03 05")
FS = 32000
ROUND_TRIP_TAKE = (
    ROOT / ".cache" / "takes" / "book-d2-1" / "01-24-40-03-05-held_organ" / "held_organ-out-00.wav"
)
ROUND_TRIP_CHANNELS = (2, 3)


def spec(value) -> dict:
    """A constant value, as a node carries one."""
    return {"value": value, "source": "document", "rests_on": [], "fitted_on": []}


def graph_of(nodes, *, inputs=("x",), outputs=None, rate=FS, bound=()) -> dict:
    """A graph model on the fixture unit, every printed row accounted for."""
    rows = dict.fromkeys(PRINTED, "fixed_at_power_on")
    rows.update(dict.fromkeys(bound, "bound"))
    return {
        "model": {
            "schema_version": 1,
            "id": "test-graph",
            "class": "whole-0124",
            "candidate": "test",
            "unit_id": "fixture-unit",
            "type": "01 24",
            "kind": "graph",
            "made_by": "tests",
        },
        "sample_rate_hz": rate,
        "inputs": list(inputs),
        "outputs": outputs if outputs is not None else {"y": nodes[-1]["id"]},
        "nodes": nodes,
        "rows": rows,
    }


def loaded(model: dict) -> dict:
    return render.load_graph(model, models_dir=MODELS, root=FIXTURE_ROOT)


def drawn(model: dict, x: np.ndarray, bytes_now=None, **kwargs) -> np.ndarray:
    """The graph's `y` for one input `x`, at the model's own rate."""
    return render.run(loaded(model), {"x": x}, bytes_now or {}, **kwargs)["y"]


def delay_node(id_, input_, time_ms, interpolation, **extra) -> dict:
    return {
        "id": id_,
        "kind": "delay",
        "input": input_,
        "time_ms": spec(time_ms),
        "interpolation": interpolation,
        **extra,
    }


# ---------------------------------------------------------------- refusals


def _valid() -> dict:
    return graph_of([delay_node("line", "x", 1.0, "linear")])


def _unknown_kind(m):
    m["nodes"][0]["kind"] = "reverb"


def _input_names_nothing(m):
    m["nodes"][0]["input"] = "nowhere"


def _missing_required_value(m):
    del m["nodes"][0]["interpolation"]


def _delay_free_cycle(m):
    m["nodes"] = [
        {
            "id": "a",
            "kind": "mix",
            "inputs": ["x", "b"],
            "weights": {"x": spec(1.0), "b": spec(0.5)},
        },
        {"id": "b", "kind": "gain", "input": "a", "gain": spec(0.5), "unit": "ratio"},
    ]
    m["outputs"] = {"y": "a"}


def _row_missing(m):
    del m["rows"]["40 03 04"]


def _bound_with_no_reader(m):
    m["rows"]["40 03 04"] = "bound"


def _read_but_not_bound(m):
    m["nodes"][0]["time_ms"] = {
        "byte": "40 03 04",
        "map": {"kind": "points", "points": [[0, 1.0], [127, 2.0]]},
        "source": "law",
        "rests_on": [],
        "fitted_on": [],
    }


def _lti_at_another_rate(m):
    m["sample_rate_hz"] = 44100
    m["nodes"] = [
        {"id": "eq", "kind": "lti", "input": "x", "model": "0120-a-loop-a-sample-late.json"}
    ]


def _lti_with_an_undelayed_loop(m):
    m["nodes"] = [
        {"id": "eq", "kind": "lti", "input": "x", "model": "0120-a-loop-with-nothing-in-it.json"}
    ]


def _lti_that_is_not_lti(m):
    m["nodes"] = [{"id": "eq", "kind": "lti", "input": "x", "model": "pan-a-sine-cosine-pair.json"}]


REFUSALS = {
    "unknown kind": (_unknown_kind, "not a node kind"),
    "input naming no node": (_input_names_nothing, "names nothing"),
    "missing required value": (_missing_required_value, "has no `interpolation`"),
    "delay-free cycle": (_delay_free_cycle, "loop with no delay"),
    "printed row missing": (_row_missing, "rows omits"),
    "bound row with no reader": (_bound_with_no_reader, "no node reads"),
    "read row not bound": (_read_but_not_bound, "is not `bound`"),
    "lti at another rate": (_lti_at_another_rate, "sample_rate_hz"),
    "lti with an undelayed loop": (_lti_with_an_undelayed_loop, "no delay"),
    "lti that is not lti": (_lti_that_is_not_lti, "not a `lti`"),
}


def test_the_valid_base_loads():
    assert loaded(_valid())["model"]["kind"] == "graph"


@pytest.mark.parametrize("name", sorted(REFUSALS))
def test_a_graph_is_refused_for_its_own_reason(name):
    breaks, reason = REFUSALS[name]
    model = copy.deepcopy(_valid())
    breaks(model)
    with pytest.raises(ValueError, match=reason):
        loaded(model)


def test_reproduce_hands_a_graph_file_to_the_render_package(tmp_path):
    import shutil

    for part in ("data", "documents"):
        shutil.copytree(FIXTURE_ROOT / part, tmp_path / part)
    models = tmp_path / "inferences" / "models"
    models.mkdir(parents=True)
    path = models / "g.json"
    path.write_text(json.dumps(_valid()))
    assert reproduce.load(path)["model"]["kind"] == "graph"
    broken = _valid()
    del broken["rows"]["40 03 04"]
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="rows omits"):
        reproduce.load(path)


# ---------------------------------------------------------------- loops


def comb(g: float, time_ms: float, interpolation: str, modulated=None) -> dict:
    """`y = x + g * y` delayed: a mix, a delay and a gain closed into one loop."""
    extra = {}
    nodes = []
    if modulated is not None:
        rate_hz, depth_ms = modulated
        nodes.append(
            {
                "id": "lfo",
                "kind": "lfo",
                "shape": "sine",
                "rate_hz": spec(rate_hz),
                "phase_offset": spec(0.0),
            }
        )
        extra["modulated_by"] = {"control": "lfo", "depth_ms": spec(depth_ms)}
    nodes += [
        {
            "id": "sum",
            "kind": "mix",
            "inputs": ["x", "fb"],
            "weights": {"x": spec(1.0), "fb": spec(1.0)},
        },
        delay_node("line", "sum", time_ms, interpolation, **extra),
        {"id": "fb", "kind": "gain", "input": "line", "gain": spec(g), "unit": "ratio"},
    ]
    return graph_of(nodes, outputs={"y": "sum"})


@pytest.mark.parametrize("g", [0.3, 0.9, -0.9])
@pytest.mark.parametrize("delay", [2, 31, 32, 1000])
def test_a_fixed_loop_falls_by_its_gain_every_trip(g, delay):
    x = np.zeros(12 * delay + 1)
    x[0] = 1.0
    y = drawn(comb(g, delay * 1000.0 / FS, "none"), x)
    trips = np.arange(13)
    np.testing.assert_allclose(y[trips * delay], g**trips, rtol=1e-12, atol=0)
    others = np.delete(y, trips * delay)
    assert np.abs(others).max() == 0.0
    fall_db = 20 * np.log10(np.abs(y[trips[1:] * delay] / y[trips[:-1] * delay]))
    np.testing.assert_allclose(fall_db, 20 * np.log10(abs(g)), atol=1e-9)


@pytest.mark.parametrize("interpolation", ["none", "linear", "allpass", "lagrange3"])
@pytest.mark.parametrize("g", [0.3, 0.9, -0.9])
@pytest.mark.parametrize("delay", [2, 31, 32, 1000])
def test_a_fixed_loop_in_blocks_is_the_loop_a_sample_at_a_time(interpolation, g, delay):
    reach = {"none": 0, "linear": 1, "allpass": 1, "lagrange3": 2}[interpolation]
    if delay - reach < 1:
        pytest.skip("refused as unrenderable, tested below")
    x = np.random.default_rng(delay).standard_normal(3 * delay + 400)
    model = comb(g, (delay + 0.37) * 1000.0 / FS, interpolation)
    blocks = drawn(model, x)
    samples = drawn(model, x, max_block=1)
    assert np.abs(blocks - samples).max() <= 1e-12


MODULATED = [
    (0.5, 0.0, "linear"),
    (0.5, 0.25, "lagrange3"),
    (2.0, 1.0, "linear"),
    (2.0, 1.9, "none"),
    (5.0, 2.5, "allpass"),
    (5.0, 4.9, "linear"),
]


@pytest.mark.parametrize(("time_ms", "depth_ms", "interpolation"), MODULATED)
def test_a_modulated_loop_in_blocks_is_the_loop_a_sample_at_a_time(
    time_ms, depth_ms, interpolation
):
    x = np.random.default_rng(7).standard_normal(6000)
    model = comb(0.7, time_ms, interpolation, modulated=(9.0, depth_ms))
    blocks = drawn(model, x)
    samples = drawn(model, x, max_block=1)
    assert np.abs(blocks - samples).max() <= 1e-12


@pytest.mark.parametrize(
    ("time_ms", "depth_ms", "interpolation"),
    [(0.5, 0.5, "linear"), (2.0, 1.99, "lagrange3"), (1.0, 0.97, "none")],
)
def test_a_loop_whose_delay_swings_under_a_sample_is_unrenderable(time_ms, depth_ms, interpolation):
    model = comb(0.7, time_ms, interpolation, modulated=(9.0, depth_ms))
    with pytest.raises(render.Unrenderable, match="line"):
        drawn(model, np.zeros(4000))


# ---------------------------------------------------------------- the heaviest graph

FDN_SAMPLES = (1031, 1327, 1523, 1709, 1913, 2111, 2311, 2503)


def fdn_graph() -> dict:
    """Eight delays in one loop through a Householder matrix, each damped by a pole.

    Two of the lines are modulated. It is the largest strongly connected component
    among the test graphs, so it is the one a take's wall time is measured on.
    """
    lines = len(FDN_SAMPLES)
    matrix = 0.8 * (np.eye(lines) - 2.0 / lines)
    nodes = [
        {
            "id": "lfo",
            "kind": "lfo",
            "shape": "sine",
            "rate_hz": spec(0.7),
            "phase_offset": spec(0.0),
        }
    ]
    for i, samples in enumerate(FDN_SAMPLES):
        source = "in_l" if i % 2 == 0 else "in_r"
        weights = {source: spec(1.0)}
        weights.update({f"damp{j}": spec(float(matrix[i, j])) for j in range(lines)})
        nodes.append({"id": f"sum{i}", "kind": "mix", "inputs": list(weights), "weights": weights})
        extra = {}
        if i < 2:
            extra["modulated_by"] = {"control": "lfo", "depth_ms": spec(0.3)}
        nodes.append(delay_node(f"line{i}", f"sum{i}", samples * 1000.0 / FS, "linear", **extra))
        nodes.append(
            {
                "id": f"damp{i}",
                "kind": "section",
                "input": f"line{i}",
                "stage": "pole",
                "side": "low",
                "corner_hz": spec(8000.0),
                "sections": spec(1),
            }
        )
    for side, first in (("l", 0), ("r", 1)):
        taps = [f"damp{j}" for j in range(first, lines, 2)]
        nodes.append(
            {
                "id": f"out_{side}",
                "kind": "mix",
                "inputs": taps,
                "weights": {t: spec(0.25) for t in taps},
            }
        )
    return graph_of(nodes, inputs=("in_l", "in_r"), outputs={"out_l": "out_l", "out_r": "out_r"})


def test_the_heaviest_graph_in_blocks_is_the_graph_a_sample_at_a_time():
    rng = np.random.default_rng(3)
    inputs = {"in_l": rng.standard_normal(3000), "in_r": rng.standard_normal(3000)}
    model = loaded(fdn_graph())
    blocks = render.run(model, inputs, {})
    samples = render.run(model, inputs, {}, max_block=1)
    for name in ("out_l", "out_r"):
        assert np.abs(blocks[name] - samples[name]).max() <= 1e-12
        assert np.abs(blocks[name][FDN_SAMPLES[0] :]).max() > 0


def test_a_take_comes_back_in_its_own_shape_with_only_the_pair_drawn():
    rate = 48000
    take = np.random.default_rng(5).standard_normal((4800, 6))
    out = render.render_take(loaded(fdn_graph()), take, rate, {}, channels=(2, 3))
    assert out.shape == take.shape
    assert not out[:, [0, 1, 4, 5]].any()
    assert out[:, 2:4].any()


# ---------------------------------------------------------------- the round trip


def identity() -> dict:
    return graph_of([], inputs=("in_l", "in_r"), outputs={"out_l": "in_l", "out_r": "in_r"})


@pytest.mark.skipif(not ROUND_TRIP_TAKE.is_file(), reason="no bypassed take in .cache/takes")
def test_the_round_trip_through_the_model_rate_is_below_the_repeat_floor():
    """48 kHz to 32 and back, both sides cut at `CUT_HZ`, 10 dB under the -55 dB floor."""
    take, rate = takes.read(ROUND_TRIP_TAKE)
    out = render.render_take(loaded(identity()), take, rate, {}, channels=ROUND_TRIP_CHANNELS)
    left = right = 0.0
    for c in ROUND_TRIP_CHANNELS:
        was = reproduce.band_limited(take[:, c], rate, render.CUT_HZ)
        now = reproduce.band_limited(out[:, c], rate, render.CUT_HZ)
        left += float(np.sum((was - now) ** 2))
        right += float(np.sum(was**2))
    assert 10 * np.log10(left / right) <= -65.0


def test_the_graph_the_layout_check_calls_good_is_one_the_renderer_loads():
    """The layout check and the renderer read one shape, so they agree on its example."""
    good = json.loads((HERE / "data" / "render-fixtures" / "good-graph.json").read_text())
    render.load_graph(good, models_dir=MODELS, root=FIXTURE_ROOT)


@pytest.mark.parametrize(("states", "refused"), [({"0": 0.5, "1": 1.0}, True),
                                                 ({"0": 0.5, "1": 1.0, "*": 1.0}, False)])
def test_a_byte_value_the_model_names_no_state_for_is_unrenderable(states, refused):
    by_byte = {"byte": "40 03 04", "map": {"kind": "states", "values": states},
               "source": "law", "rests_on": [], "fitted_on": []}
    model = graph_of([{"id": "g", "kind": "gain", "input": "x", "gain": by_byte,
                       "unit": "ratio"}], bound=("40 03 04",))
    x = np.ones(64)
    if refused:
        with pytest.raises(render.Unrenderable, match="40 03 04"):
            drawn(model, x, {"40 03 04": 127})
    else:
        assert np.allclose(drawn(model, x, {"40 03 04": 127}), 1.0)


def _by_mode(values: dict) -> dict:
    return {"byte": "40 03 04", "map": {"kind": "states", "values": values},
            "source": "law", "rests_on": [], "fitted_on": []}


def _two_branches(quiet: dict) -> dict:
    """`y` mixes a branch the mode byte keeps with one it weights at zero."""
    return graph_of([
        quiet,
        {"id": "loud", "kind": "gain", "input": "x", "gain": spec(0.5), "unit": "ratio"},
        {"id": "y", "kind": "mix", "inputs": ["loud", "quiet"],
         "weights": {"loud": spec(1.0), "quiet": _by_mode({"0": 0.0, "1": 1.0})}},
    ], bound=("40 03 04",))


def _quiet_gain(gain: dict) -> dict:
    return {"id": "quiet", "kind": "gain", "input": "x", "gain": gain, "unit": "ratio"}


@pytest.mark.parametrize(("mode", "drawn_quiet"), [(0, False), (1, True)])
def test_a_node_heard_only_through_a_weight_of_zero_is_not_drawn(monkeypatch, mode, drawn_quiet):
    seen = []
    kind = render.graph.NODES["gain"]

    def counted(node, drawing, a, b):
        seen.append(node["id"])
        kind.render(node, drawing, a, b)

    monkeypatch.setitem(render.graph.NODES, "gain", dataclasses.replace(kind, render=counted))
    x = np.random.default_rng(3).standard_normal(256)
    y = drawn(_two_branches(_quiet_gain(spec(2.0))), x, {"40 03 04": mode})
    assert ("quiet" in seen) is drawn_quiet
    np.testing.assert_array_equal(y, 0.5 * x + (2.0 * x if drawn_quiet else 0.0))


def test_a_loop_with_a_silenced_branch_draws_what_the_loop_without_it_draws():
    silenced = comb(0.7, 2.0, "linear")
    silenced["nodes"].append(
        {"id": "aside", "kind": "gain", "input": "line", "gain": spec(-3.0), "unit": "ratio"})
    silenced["nodes"][0]["inputs"].append("aside")
    silenced["nodes"][0]["weights"]["aside"] = _by_mode({"0": 0.0, "1": 1.0})
    silenced["rows"]["40 03 04"] = "bound"
    x = np.random.default_rng(5).standard_normal(4000)
    np.testing.assert_array_equal(
        drawn(silenced, x, {"40 03 04": 0}), drawn(comb(0.7, 2.0, "linear"), x))


def test_a_silenced_node_with_no_state_named_still_makes_the_setting_unrenderable():
    model = _two_branches(_quiet_gain(_by_mode({"1": 2.0})))
    with pytest.raises(render.Unrenderable, match="40 03 04"):
        drawn(model, np.ones(64), {"40 03 04": 0})
