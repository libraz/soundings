"""Carrying a type through stages 7-9, on takes made here from sines and noise.

Nothing below is a recording of a unit. A "unit" is a known graph drawn on a
synthetic bypass take with fresh noise added per take, so what a comparison
should conclude is known before it runs: the graph that made the takes has to
pass, a candidate differing from it only in how it interpolates has to lose on
power, and the identity model has to be caught out whenever the takes carry an
effect at all. Takes are 48 kHz and the graphs 32 kHz, so every drawing goes
through the same round trip a real one does.

`readings` is exercised with `efx-rate`, the cheapest reading stage that reads
synthetic takes: an amplitude-modulated tone is what it reads a rate from.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from soundings import stages, takes
from soundings.cli import report
from soundings.render import graph

from .test_inferences_layout import FIXTURE_ROOT, _stage_errors

ROOT = Path(__file__).resolve().parent.parent
FS = 48000
UNIT = "fixture-unit"
TYPE = "01 24"
HELD = {"40 03 00": "01 24", "40 42 22": "01"}
SETTINGS = (0, 32, 64, 96, 127)
TONE = {
    "name": "tone", "program": 16, "note": 81, "velocity": 100, "channel": 1,
    "writes": [], "volume": 100, "hold_s": 1.6, "lead_s": 0.2,
}
# Stage 8 takes the class the effect stands in most often, a tie going to the lower floor,
# so with one directory each `tone` is stage 8 and `voice` stage 9.
VOICE = {**TONE, "name": "voice", "program": 17, "note": 76}
PARTIALS = {"tone": ((440, 0.1), (880, 0.05), (1320, 0.03), (1760, 0.02)),
            "voice": ((330, 0.08), (660, 0.05), (990, 0.04), (1650, 0.02)),
            "applause": ()}
WAVES_CLAIM = "inferences/fixture-unit/mix-is-a-crossfade.json"
WAVES_RECORD = "data/units/fixture-unit/efx-bands/mix.json"
RATE_CLAIM = "inferences/fixture-unit/rate-is-a-log-table.json"
RATE_RECORD = "data/units/fixture-unit/efx-rate/rate.json"


@pytest.fixture(autouse=True)
def _no_rendered_mark_left(monkeypatch):
    """The mark is process-wide; a test must neither inherit nor leave one."""
    monkeypatch.setattr(takes, "_RENDERED_READ", set())


# ---------------------------------------------------------------- building a tree


def _value(**spec) -> dict:
    return {"source": "law", "rests_on": [], "fitted_on": [], **spec}


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))


def _tree(tmp: Path, *, static: bool) -> Path:
    root = tmp / "root"
    shutil.copytree(FIXTURE_ROOT, root)
    unit = root / "data" / "units" / UNIT
    _write(unit / "efx-map" / "types.json",
           {"effects": [{"type": TYPE, "msb": 1, "lsb": 36, "parameters": [40] * 20,
                         "sends_to_reverb_chorus_delay": [40, 0, 0]}]})
    _write(unit / "efx-sort" / "by-repeatability.json",
           {"static": [TYPE] if static else [], "moving": [] if static else [TYPE]})
    (root / "inferences" / "models").mkdir(parents=True)
    return root


def _comb(model_id: str, interpolation: str, *, cls="whole-0124", fitted_on=()) -> dict:
    """A static comb: one fractional delay crossfaded with the dry path by 40 03 05."""
    time = _value(byte="40 03 03", map={"kind": "points", "points": [[0, 0.3], [127, 1.3]]},
                  source="document", rests_on=["documents/fixture-doc/effect-list.json"])
    dry = _value(byte="40 03 05", map={"kind": "points", "points": [[0, 1.0], [127, 0.5]]},
                 rests_on=[WAVES_CLAIM], fitted_on=[list(f) for f in fitted_on])
    wet = _value(byte="40 03 05", map={"kind": "points", "points": [[0, 0.0], [127, 0.5]]},
                 rests_on=[WAVES_CLAIM], fitted_on=[list(f) for f in fitted_on])
    nodes = []
    for side in ("l", "r"):
        nodes += [
            {"id": f"tap_{side}", "kind": "delay", "input": f"in_{side}",
             "interpolation": interpolation, "time_ms": time},
            {"id": f"mix_{side}", "kind": "mix", "inputs": [f"in_{side}", f"tap_{side}"],
             "weights": {f"in_{side}": dry, f"tap_{side}": wet}},
        ]
    return _graph(model_id, cls, nodes, {"out_l": "mix_l", "out_r": "mix_r"},
                  {"40 03 03": "bound", "40 03 04": "fixed_at_power_on", "40 03 05": "bound"})


def _tremolo(model_id: str, *, log: bool, cls="whole-0124") -> dict:
    """A level swung by a sine whose rate 40 03 03 picks between two printed ends."""
    rate = _value(byte="40 03 03",
                  map={"kind": "points", "points": [[0, 1.0], [127, 8.0]], "log": log},
                  rests_on=[RATE_CLAIM])
    swing = _value(control="lfo", map={"kind": "points", "points": [[-1, 0.4], [1, 1.0]]},
                   source="document", rests_on=["documents/fixture-doc/effect-list.json"])
    nodes = [{"id": "lfo", "kind": "lfo", "shape": "sine", "rate_hz": rate,
              "phase_offset": _value(value=0.0, source="document",
                                     rests_on=["documents/fixture-doc/effect-list.json"])}]
    nodes += [{"id": f"vca_{s}", "kind": "gain", "input": f"in_{s}", "unit": "ratio",
               "gain": swing} for s in ("l", "r")]
    return _graph(model_id, cls, nodes, {"out_l": "vca_l", "out_r": "vca_r"},
                  {"40 03 03": "bound", "40 03 04": "fixed_at_power_on",
                   "40 03 05": "fixed_at_power_on"})


def _graph(model_id, cls, nodes, outputs, rows) -> dict:
    return {
        "model": {"schema_version": 1, "id": model_id, "class": cls, "candidate": model_id,
                  "unit_id": UNIT, "type": TYPE, "kind": "graph", "made_by": "by hand"},
        "sample_rate_hz": 32000, "why_this_rate": "a test graph",
        "inputs": ["in_l", "in_r"], "outputs": outputs, "nodes": nodes, "rows": rows,
    }


def _put_model(root: Path, model: dict) -> Path:
    path = root / "inferences" / "models" / f"{model['model']['id']}.json"
    _write(path, model)
    return path


def _loaded(root: Path, model: dict) -> dict:
    return graph.load_graph(model, models_dir=root / "inferences" / "models", root=root)


def _clean(stimulus: dict, seconds: float) -> np.ndarray:
    n = int(seconds * FS)
    t = np.arange(n) / FS
    lead = int(stimulus["lead_s"] * FS)
    held = int(stimulus["hold_s"] * FS)
    out = np.zeros((n, 6))
    for f, a in PARTIALS[stimulus["name"]]:
        out[lead : lead + held, 2] += a * np.sin(2 * np.pi * f * t[lead : lead + held])
        out[lead : lead + held, 3] += a * np.sin(2 * np.pi * f * t[lead : lead + held] + 0.3)
    return out


def _noisy(take: np.ndarray, rng) -> np.ndarray:
    out = take.copy()
    out[:, :4] += 1e-4 * rng.standard_normal((take.shape[0], 4))
    return out


def _walked(value: int) -> str:
    return f"v{value:03d}"


def _directory(root, rel, *, stimulus, address, seconds, draw, seed, named=_walked) -> None:
    """A walk-shaped directory: two bypass takes, one silence, two takes per setting."""
    rng = np.random.default_rng(seed)
    where = root / ".cache" / "takes" / rel
    clean = _clean(stimulus, seconds)
    entries = []

    def keep(name, setting, take, samples):
        takes.write(where / name, samples, FS)
        entries.append({"file": name, "stimulus": stimulus["name"], "setting": setting,
                        "take": take, "sample_rate": FS, "channels": 6, "seconds": seconds})

    for k in range(2):
        keep(f"{stimulus['name']}-out-{k:02d}.wav", "out", k, _noisy(clean, rng))
    keep(f"{stimulus['name']}-silence-00.wav", "silence", 0,
         _noisy(np.zeros_like(clean), rng))
    for value in SETTINGS:
        for k in range(2):
            keep(f"{stimulus['name']}-{named(value)}-{k:02d}.wav", named(value), k,
                 _noisy(draw(value, k, 2, clean), rng))
    _write(where / "takes-manifest.json", {
        "type": TYPE, "prepared": [{"address": a, "bytes": b} for a, b in HELD.items()],
        "settle_s": 0.4, "address": address, "settings": list(SETTINGS),
        "stimuli": [stimulus], "label": rel, "takes": entries,
    })


def _drawn_by(root: Path, model: dict | None, address: str):
    """The unit's side: a graph drawn the way the renderer draws, or nothing drawn."""
    loaded = stages.IDENTITY_LOADED if model is None else _loaded(root, model)
    on = stages.power_on(root, UNIT, TYPE)

    def draw(value, k, n, clean):
        now = stages.bytes_now(on, stages.writes_of(HELD), address, value)
        return graph.render_take(loaded, clean, FS, now, channels=(2, 3), lfo_phase=k / n)

    return draw


def _ledger(root: Path, directories: dict[str, dict]) -> None:
    _write(root / ".cache" / "takes-ledger.json", {"unit": UNIT, "directories": {
        rel: {
            "type": TYPE, "address": spec["address"],
            "settings": {**{spec.get("named", _walked)(v): 2 for v in SETTINGS}, "out": 2,
                         "silence": 1},
            "stimuli": [{**spec["stimulus"], "class": spec["stimulus"]["name"]}],
            "held": HELD,
            "consumed_by": [{"record": r, "via": "direct"} for r in spec.get("read_by", [])],
        }
        for rel, spec in directories.items()
    }})


def _claim(root: Path, path: str, record: str, values) -> None:
    _write(root / path, {"rests_on": {"measurements": [
        {"file": record, "keys": [f"readings[value={v}].x" for v in values]}]}})


WAVES_DIR = "syn/01-24-40-03-05-tone"
WAVES_P2 = "syn/01-24-40-03-05-voice"


def _waves_tree(tmp: Path, *, unit_is_identity: bool = False) -> Path:
    root = _tree(tmp, static=True)
    a = _comb("whole-0124-linear", "linear")
    everywhere = [(d, v) for d in (WAVES_DIR, WAVES_P2) for v in SETTINGS]
    for model in (a, _comb("whole-0124-truncated", "none"),
                  _comb("whole-0124-fitted-everywhere", "linear", fitted_on=everywhere),
                  _comb("fitted-a", "linear", cls="fitted-only", fitted_on=everywhere),
                  _comb("fitted-b", "none", cls="fitted-only", fitted_on=everywhere),
                  _comb("worse-first-a", "none", cls="worse-first", fitted_on=everywhere),
                  _comb("worse-first-b", "linear", cls="worse-first", fitted_on=everywhere)):
        _put_model(root, model)
    draw = _drawn_by(root, None if unit_is_identity else a, "40 03 05")
    for seed, (rel, stim) in enumerate(((WAVES_DIR, TONE), (WAVES_P2, VOICE))):
        _directory(root, rel, stimulus=stim, address="40 03 05", seconds=2.0,
                   draw=draw, seed=seed)
    _ledger(root, {WAVES_DIR: {"address": "40 03 05", "stimulus": TONE,
                               "read_by": [WAVES_RECORD]},
                   WAVES_P2: {"address": "40 03 05", "stimulus": VOICE}})
    _claim(root, WAVES_CLAIM, WAVES_RECORD, (0, 127))
    return root


@pytest.fixture(scope="module")
def waves_root(tmp_path_factory) -> Path:
    return _waves_tree(tmp_path_factory.mktemp("waves"))


RATE_DIR = "syn/01-24-40-03-03-tone"
RATE_TONE = {**TONE, "hold_s": 4.0, "lead_s": 0.6}


@pytest.fixture(scope="module")
def readings_root(tmp_path_factory) -> Path:
    root = _tree(tmp_path_factory.mktemp("readings"), static=False)
    a = _tremolo("whole-0124-log", log=True)
    _put_model(root, a)
    _put_model(root, _tremolo("whole-0124-linear-map", log=False))
    _directory(root, RATE_DIR, stimulus=RATE_TONE, address="40 03 03", seconds=5.0,
               draw=_drawn_by(root, a, "40 03 03"), seed=7)
    _write(root / RATE_RECORD, {
        "record": {"invocation": [
            "efx-rate", f".cache/takes/{RATE_DIR}", "--type", TYPE, "--slot", "40 03 03",
            "--setting", r"tone-v(?P<value>\d{3})-\d+", "--channel", "2",
            "--out", RATE_RECORD]},
        "type": TYPE, "channel": {"read": 2}, "readings": [],
    })
    _ledger(root, {RATE_DIR: {"address": "40 03 03", "stimulus": RATE_TONE,
                              "read_by": [RATE_RECORD]}})
    _claim(root, RATE_CLAIM, RATE_RECORD, (0, 127))
    return root


def _all_passed(block: dict) -> bool:
    return all(block["gates"][g]["passed"] for g in ("gross", "qualitative", "breakdown", "power"))


# ---------------------------------------------------------------- the comparisons


def test_waves_the_graph_that_made_the_takes_passes_and_its_interpolation_loses(waves_root):
    found = stages.stage(waves_root, UNIT, TYPE, class_name="whole-0124")
    assert _stage_errors(found) == []
    p1 = found["p1"]
    assert p1["comparison"] == {"used": "waves", "chosen_by": "by_repeatability_class"}
    assert p1["control"]["separated"] is True
    assert p1["verdict"] == "passed"
    assert _all_passed(p1)
    assert p1["gates"]["power"]["ranking"][0]["candidate"] == "whole-0124-linear"
    assert p1["gates"]["power"]["decoy"] == "whole-0124-truncated"
    assert p1["held_out_settings"] == [[WAVES_DIR, 32], [WAVES_DIR, 64], [WAVES_DIR, 96]]
    assert p1["excluded_from_ranking"] == ["whole-0124-fitted-everywhere"]
    assert {row["dir"] for row in p1["compared"]} == {WAVES_DIR}
    assert found["p2"]["verdict"] == "passed"
    assert {row["dir"] for row in found["p2"]["compared"]} == {WAVES_P2}
    assert "stopped" not in found


def test_readings_the_graph_that_made_the_takes_passes_and_its_map_interpolation_loses(
    readings_root,
):
    found = stages.stage(readings_root, UNIT, TYPE, class_name="whole-0124")
    assert _stage_errors(found) == []
    p1 = found["p1"]
    assert p1["comparison"] == {"used": "readings", "chosen_by": "by_repeatability_class",
                                "cut_hz": graph.CUT_HZ}
    assert p1["control"]["separated"] is True
    assert p1["gates"]["gross"]["passed"] and p1["gates"]["breakdown"]["passed"]
    assert p1["gates"]["qualitative"]["passed"] and p1["gates"]["power"]["passed"]
    assert p1["gates"]["measured_in"] == "octaves"
    assert p1["gates"]["power"]["decoy"] == "whole-0124-linear-map"
    assert p1["verdict"] == "passed"
    assert p1["held_out_settings"] == [[RATE_DIR, 32], [RATE_DIR, 64], [RATE_DIR, 96]]
    assert {row["channel"] for row in p1["compared"]} == {2}
    # One stimulus class only, so nothing is left for the stage after it.
    assert found["stopped"]["at"] == 9 and found["stopped"]["gate"] == "no_input_take"


def test_a_unit_drawn_by_the_identity_leaves_the_control_nothing_to_fail(tmp_path):
    root = _waves_tree(tmp_path, unit_is_identity=True)
    found = stages.stage(root, UNIT, TYPE, class_name="whole-0124")
    assert _stage_errors(found) == []
    assert found["p1"]["control"]["separated"] is False
    assert found["p1"]["verdict"] == "stopped"
    assert found["stopped"]["at"] == 8
    assert found["stopped"]["gate"] == "control_could_not_fail"


def test_a_setting_only_the_drawn_side_refused_costs_the_whole_span():
    reading = stages.READINGS["efx-time:ms"]
    unit = {("d", 0): [[1.0]], ("d", 1): [[2.0]], ("d", 2): [[3.0]], ("d", 3): [[None]]}
    drawn = {("d", 0): [[1.0]], ("d", 1): [[None]], ("d", 2): [[3.0]], ("d", 3): [[None]]}
    scored = stages.scored_readings("efx-time:ms", reading, unit, drawn, floor=0.1)
    assert scored["span"] == 2.0
    residuals = {row["value"]: row["residual"] for row in scored["rows"]}
    assert residuals == {0: [0.0], 1: [2.0], 2: [0.0]}
    assert scored["drawn_only_refused"] == [["d", 1]]
    assert stages.separated([scored]) is True


def _rate_model(slot: str, *maps: dict) -> dict:
    nodes = [{"id": f"n{i}", "kind": "gain", "gain": {"byte": slot, "map": m}}
             for i, m in enumerate(maps)]
    return {"model": {"id": "m"}, "nodes": nodes}


LINEAR = {"kind": "table", "entries": [0.05 * (i + 1) for i in range(126)], "out_of_range": 125}


def test_one_entry_of_a_rate_table_is_its_step_in_octaves_at_the_settings_compared():
    raw = _rate_model("40 03 03", LINEAR, dict(LINEAR))
    worth = stages.one_entry_octaves(raw, "40 03 03", [0, 6, 40, 72])
    steps = [math.log2(LINEAR["entries"][v] / LINEAR["entries"][v - 1]) for v in (6, 40, 72)]
    assert worth == pytest.approx(float(np.median(steps)))


@pytest.mark.parametrize("maps", [
    (),
    ({"kind": "points", "points": [[0, 0.05], [127, 10.0]]},),
    (LINEAR, {**LINEAR, "entries": [0.1 * (i + 1) for i in range(126)]}),
])
def test_a_rate_byte_whose_entries_are_not_one_measured_table_has_no_entry_floor(maps):
    assert stages.one_entry_octaves(_rate_model("40 03 03", *maps), "40 03 03", [6, 40]) == 0.0


def test_where_a_band_is_largest_is_no_reading_where_the_profile_has_no_width():
    rows = [
        {"value": 0, "largest_db": -9.9, "largest_at_hz": 100,
         "half_below_hz": None, "half_above_hz": None},
        {"value": 1, "largest_db": -9.8, "largest_at_hz": 10000,
         "half_below_hz": None, "half_above_hz": 12500},
        {"value": 2, "largest_db": 9.8, "largest_at_hz": 1000,
         "half_below_hz": 500, "half_above_hz": 2000},
    ]
    found = stages.collected("efx-bands", {"readings": rows}, "d")
    at = found["efx-bands:largest_at_hz"]["takes"]
    assert at[("d", 0)] == [[None]] and at[("d", 1)] == [[None]]
    assert at[("d", 2)] == [[math.log2(1000)]]
    assert found["efx-bands:largest_db"]["takes"][("d", 0)] == [[-9.9]]


def test_a_largest_band_nothing_cleared_is_read_as_no_deviation():
    reading = stages.READINGS["efx-bands:largest_db"]
    unit = {("d", 0): [[-9.0]], ("d", 1): [[None]], ("d", 2): [[6.0]]}
    drawn = {("d", 0): [[-9.0]], ("d", 1): [[0.3]], ("d", 2): [[None]]}
    scored = stages.scored_readings("efx-bands:largest_db", reading, unit, drawn, floor=0.2)
    residuals = {row["value"]: row["residual"] for row in scored["rows"]}
    assert residuals == {0: [0.0], 1: [0.3], 2: [-6.0]}
    assert scored["drawn_only_refused"] == []


@pytest.mark.parametrize(
    "name", sorted(k for k, r in stages.READINGS.items() if r.null_is is None))
def test_a_setting_the_unit_refused_is_not_compared(name):
    reading = stages.READINGS[name]
    unit = {("d", 0): [[None]], ("d", 1): [[10.0]], ("d", 2): [[11.0]]}
    drawn = {("d", 0): [[12.0]], ("d", 1): [[None]], ("d", 2): [[11.0]]}
    scored = stages.scored_readings(name, reading, unit, drawn, floor=0.1)
    residuals = {row["value"]: row["residual"] for row in scored["rows"]}
    assert residuals == {1: [scored["span"]], 2: [0.0]}
    assert scored["drawn_only_refused"] == [["d", 1]]


def test_a_position_is_held_against_the_width_of_the_band_it_was_read_in():
    found = {"band_width_octaves": 0.333333, "reference": {"floor_db": [0.1, 0.2]},
             "readings": []}
    assert stages.floor_of(found, stages.READINGS["efx-bands:fitted_at_hz"]) == 0.333333
    assert stages.floor_of(found, stages.READINGS["efx-bands:largest_at_hz"]) == 0.333333
    assert stages.floor_of(found, stages.READINGS["efx-bands:largest_db"]) == 0.2


def test_a_delay_is_held_against_the_step_it_was_read_in_where_its_floor_is_finer():
    reading = stages.READINGS["efx-time:ms"]
    step = {"quefrency_step_ms": 0.020833, "readings": []}
    assert stages.floor_of({**step, "floor_ms": 0.0}, reading) == 0.020833
    assert stages.floor_of({**step, "floor_ms": 4.7083}, reading) == 4.7083


def test_a_span_inside_twice_the_floor_is_a_null_the_model_must_show_nothing_on():
    reading = stages.READINGS["efx-time:ms"]
    unit = {("d", 0): [[300.0]], ("d", 1): [[300.0208]]}
    still = stages.scored_readings("efx-time:ms", reading, unit, unit, floor=0.0208)
    moved = stages.scored_readings(
        "efx-time:ms", reading, unit, {("d", 0): [[300.0]], ("d", 1): [[0.06]]}, floor=0.0208)
    assert still["is_null_record"] and still["model_stays_inside_the_floor"]
    assert moved["is_null_record"] and not moved["model_stays_inside_the_floor"]
    assert not stages.scored_readings(
        "efx-time:ms", reading, unit, unit, floor=0.01)["is_null_record"]


def test_a_swing_nothing_cleared_is_no_swing_and_a_take_without_signal_is_no_reading():
    def row(value, depth, frames):
        return {"value": value, "tracked_frames": frames,
                "level_in_db": {"depth": depth}, "balance": {"depth": depth}}

    rows = [row(0, None, 1670), row(1, 6.2, 1670), row(2, None, 12)]
    found = stages.collected("efx-sway", {"readings": rows}, "d")
    for name in ("efx-sway:level_in_db.depth", "efx-sway:balance.depth"):
        unit = found[name]["takes"]
        assert set(unit) == {("d", 0), ("d", 1)}
        drawn = {("d", 0): [[None]], ("d", 1): [[None]]}
        scored = stages.scored_readings(name, stages.READINGS[name], unit, drawn, floor=0.5)
        residuals = {row["value"]: row["residual"] for row in scored["rows"]}
        assert scored["span"] == 6.2
        assert residuals == {0: [0.0], 1: [-6.2]}
        assert scored["drawn_only_refused"] == []


def test_a_model_that_does_not_move_where_the_unit_does_is_said_to_hold():
    unit = {("d", 0): [[0.0]], ("d", 1): [[-3.0]], ("d", 2): [[3.0]]}
    drawn = {("d", 0): [[1.0]], ("d", 1): [[1.0]], ("d", 2): [[2.0]]}
    found = stages._directions(stages._medians(unit), stages._medians(drawn), 0.5)
    assert [(p["unit"], p["model"], p["same"]) for p in found] == [
        ("falls", "holds", False), ("rises", "rises", True)]


def test_a_record_name_spells_what_it_differs_by_as_it_was_passed():
    argv = ["efx-bands", ".cache/takes/d", "--type", "01 02", "--slot", "40 03 03",
            "--setting", "a-v(?P<value>\\d{3})", "--reference", "a-v000",
            "--reference-held", "40 42 22=00", "--out", "x.json"]
    assert stages._scored_name("efx-bands:largest_db", "efx-bands", argv) == (
        "efx-bands:largest_db [--reference-held 40 42 22=00]")


def test_a_setting_that_returns_the_room_is_no_reading_of_the_effect():
    rows = [
        {"value": 0, "largest_db": -64.4, "largest_at_hz": 2500, "half_below_hz": 100,
         "half_above_hz": 8000, "above_the_silence_db": 1.1},
        {"value": 1, "largest_db": -42.1, "largest_at_hz": 800, "half_below_hz": 100,
         "half_above_hz": 8000, "above_the_silence_db": 15.3},
    ]
    found = stages.collected("efx-bands", {"readings": rows}, "d")
    assert set(found["efx-bands:largest_db"]["takes"]) == {("d", 1)}
    assert set(found["efx-bands:largest_at_hz"]["takes"]) == {("d", 1)}


def test_a_stage_publishing_two_quantities_is_scored_as_two_records():
    def excursion(fit, phase):
        return {
            "readings": [{"value": v, "excursion_ms": f, "off_the_phase_ms": p}
                         for v, f, p in zip((0, 64, 127), fit, phase, strict=True)],
            "floor": {"ms": None}, "floor_off_the_phase": {"ms": 0.05},
        }

    unit = stages.collected("efx-excursion", excursion([None] * 3, [0.2, 0.9, 1.6]), "d")
    drawn = stages.collected("efx-excursion", excursion([None] * 3, [None] * 3), "d")
    assert sorted(unit) == ["efx-excursion:excursion_ms", "efx-excursion:off_the_phase_ms"]
    assert unit["efx-excursion:off_the_phase_ms"]["floor"] == 0.05
    scored = [
        stages.scored_readings(name, stages.READINGS[name], unit[name]["takes"],
                               drawn[name]["takes"], floor=unit[name]["floor"] or 0.0)
        for name in sorted(unit)
    ]
    assert [s["record"] for s in scored] == sorted(unit)
    assert scored[0]["is_null_record"] and scored[0]["rows"] == []
    assert scored[1]["drawn_only_refused"] == [["d", 0], ["d", 64], ["d", 127]]
    assert stages.separated(scored) is True


def test_two_runs_with_different_reading_arguments_score_as_two_records(tmp_path):
    """A run of one stage differing in a reading argument is not pooled.

    `--lines` is a convenient stand-in for `--band-set`: a flag `SIGNATURE_EXCLUDED`
    does not name, so it belongs to the signature and splits the record.
    """
    root = _tree(tmp_path, static=False)
    a = _tremolo("whole-0124-log", log=True)
    _put_model(root, a)
    _directory(root, RATE_DIR, stimulus=RATE_TONE, address="40 03 03", seconds=5.0,
               draw=_drawn_by(root, a, "40 03 03"), seed=7)
    plain = "data/units/fixture-unit/efx-rate/plain.json"
    lined = "data/units/fixture-unit/efx-rate/lined.json"
    base_argv = ["efx-rate", f".cache/takes/{RATE_DIR}", "--type", TYPE, "--slot", "40 03 03",
                "--setting", r"tone-v(?P<value>\d{3})-\d+", "--channel", "2"]
    _write(root / plain, {"record": {"invocation": [*base_argv, "--out", plain]},
                          "type": TYPE, "channel": {"read": 2}, "readings": []})
    _write(root / lined, {"record": {"invocation": [*base_argv, "--lines", "--out", lined]},
                          "type": TYPE, "channel": {"read": 2}, "readings": []})
    _ledger(root, {RATE_DIR: {"address": "40 03 03", "stimulus": RATE_TONE,
                              "read_by": [plain, lined]}})
    found = json.loads((root / ".cache/takes-ledger.json").read_text())
    made = stages.directory(root, found, RATE_DIR)
    cls = stages._Class(None, [made], len(SETTINGS), None, "readings", "test")
    phase = stages._Phase(root, found, TYPE, [(cls, [made])], {}, {})
    scored, _ = phase.scored(stages.identity())
    names = sorted(s["record"] for s in scored if s["record"].startswith("efx-rate:rate_hz"))
    assert names == ["efx-rate:rate_hz", "efx-rate:rate_hz [--lines]"]


def test_the_units_side_of_a_reading_is_kept_until_its_takes_change(readings_root, tmp_path,
                                                                    monkeypatch):
    from soundings import efxrate

    record = json.loads((readings_root / RATE_RECORD).read_text())
    where = readings_root / ".cache" / "takes" / RATE_DIR
    kept = tmp_path / "stage-readings"

    def read():
        return stages.read_with(record["record"]["invocation"], root=readings_root,
                                unit_dir=where, drawn_dir=None, channel=2,
                                writes=stages.writes_of(HELD), cache=kept)

    first = read()
    assert first["readings"] and len(list(kept.glob("*.json"))) == 1

    def refused(*args, **kwargs):
        raise AssertionError("the stage ran although its reading was kept")

    monkeypatch.setattr(efxrate, "read_directory", refused)
    assert read() == first
    take = where / "tone-v064-00.wav"
    later = take.stat().st_mtime_ns + 5_000_000_000
    os.utime(take, ns=(later, later))
    with pytest.raises(AssertionError, match="although its reading was kept"):
        read()


def test_a_drawn_reading_is_kept_until_the_takes_are_drawn_again(readings_root, tmp_path,
                                                                  monkeypatch):
    from soundings import efxrate

    record = json.loads((readings_root / RATE_RECORD).read_text())
    where = readings_root / ".cache" / "takes" / RATE_DIR
    drawn = tmp_path / "drawn"
    shutil.copytree(where, drawn)
    kept = tmp_path / "stage-readings"

    def read():
        return stages.read_with(record["record"]["invocation"], root=readings_root,
                                unit_dir=where, drawn_dir=drawn, channel=2,
                                writes=stages.writes_of(HELD), cache=kept)

    first = read()
    assert first["readings"]

    def refused(*args, **kwargs):
        raise AssertionError("the stage ran although its reading was kept")

    monkeypatch.setattr(efxrate, "read_directory", refused)
    assert read() == first
    take = drawn / "tone-v064-00.wav"
    later = take.stat().st_mtime_ns + 5_000_000_000
    os.utime(take, ns=(later, later))
    with pytest.raises(AssertionError, match="although its reading was kept"):
        read()


def test_a_kept_reading_is_not_read_back_once_the_code_reading_it_changes(monkeypatch):
    args = argparse.Namespace(directory="d", channel=2)
    before = stages._cache_key("efx-rate", args, Path("."))
    monkeypatch.setattr(stages, "_reader_source", lambda command: "another source")
    assert stages._cache_key("efx-rate", args, Path(".")) != before


def test_the_code_a_reading_runs_is_followed_through_its_imports():
    followed = stages._reader_modules("efx-rate")
    assert "efxrate" in followed and "rates" in followed
    assert "stages" not in followed and not any(m.startswith("cli") for m in followed)


def test_the_unit_side_of_a_reading_is_read_from_its_band_limited_copy(readings_root,
                                                                       monkeypatch):
    """`read_with`'s unit-side call is aimed at `cut_dir`, not `unit_dir`."""
    from soundings import efxrate

    record = json.loads((readings_root / RATE_RECORD).read_text())
    where = readings_root / ".cache" / "takes" / RATE_DIR
    cut = stages.cut_directory(readings_root, RATE_DIR)
    assert cut != where and cut.is_dir()
    seen = {}
    original = efxrate.read_directory

    def spy(takes_dir, **kwargs):
        seen["takes"] = Path(takes_dir)
        return original(takes_dir, **kwargs)

    monkeypatch.setattr(efxrate, "read_directory", spy)
    stages.read_with(record["record"]["invocation"], root=readings_root, unit_dir=where,
                     drawn_dir=None, channel=2, writes=stages.writes_of(HELD), cut_dir=cut)
    assert seen["takes"] == cut


def test_cut_directory_removes_energy_above_cut_hz_and_rebuilds_on_change(tmp_path):
    """The unit's own band-limited copy, same names, manifest copied."""
    root = tmp_path / "root"
    rel = "syn/probe"
    where = root / ".cache" / "takes" / rel
    where.mkdir(parents=True)
    n = FS * 1
    t = np.arange(n) / FS
    samples = np.zeros((n, 2))
    samples[:, 0] = 0.1 * np.sin(2 * np.pi * 15000.0 * t)
    takes.write(where / "a.wav", samples, FS)
    _write(where / "takes-manifest.json", {"takes": [{"file": "a.wav"}]})

    cut = stages.cut_directory(root, rel)
    assert cut == root / ".cache" / "cut" / rel
    cut_samples, rate = takes.read(cut / "a.wav")
    spectrum = np.abs(np.fft.rfft(cut_samples[:, 0]))
    freq = np.fft.rfftfreq(cut_samples.shape[0], 1.0 / rate)
    original_peak = 0.1 * n / 2
    assert spectrum[freq > graph.CUT_HZ].max() < 1e-3 * original_peak
    assert json.loads((cut / "takes-manifest.json").read_text()) == {"takes": [{"file": "a.wav"}]}

    first = (cut / "a.wav").stat().st_mtime_ns
    stages.cut_directory(root, rel)
    assert (cut / "a.wav").stat().st_mtime_ns == first

    later = (where / "a.wav").stat().st_mtime_ns + 5_000_000_000
    os.utime(where / "a.wav", ns=(later, later))
    stages.cut_directory(root, rel)
    assert (cut / "a.wav").stat().st_mtime_ns != first


def test_a_cut_copy_left_half_written_is_rebuilt_rather_than_read(tmp_path, monkeypatch):
    """A build stopped after its first take leaves a copy newer than its sources."""
    root = tmp_path / "root"
    rel = "syn/probe"
    where = root / ".cache" / "takes" / rel
    where.mkdir(parents=True)
    for name in ("a.wav", "b.wav"):
        takes.write(where / name, np.zeros((FS // 10, 2)), FS)
    _write(where / "takes-manifest.json", {"takes": [{"file": "a.wav"}, {"file": "b.wav"}]})

    written = takes.write

    def stops_after_one(path, *args, **kwargs):
        written(path, *args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(takes, "write", stops_after_one)
    with pytest.raises(KeyboardInterrupt):
        stages.cut_directory(root, rel)
    monkeypatch.setattr(takes, "write", written)

    cut = stages.cut_directory(root, rel)
    assert sorted(p.name for p in cut.iterdir()) == ["a.wav", "b.wav", "takes-manifest.json"]


def test_a_candidate_fitted_on_every_setting_is_stopped_and_ranked_nowhere(waves_root):
    found = stages.stage(waves_root, UNIT, TYPE, class_name="fitted-only")
    assert _stage_errors(found) == []
    assert found["p1"]["held_out_settings"] == []
    assert sorted(found["p1"]["excluded_from_ranking"]) == ["fitted-a", "fitted-b"]
    assert found["stopped"] == {**found["stopped"], "at": 8, "gate": "no_held_out_setting"}


def test_a_law_values_fitted_on_is_derived_from_its_claim_and_the_ledger(waves_root):
    model = json.loads((waves_root / "inferences/models/whole-0124-linear.json").read_text())
    ledger = json.loads((waves_root / ".cache/takes-ledger.json").read_text())
    assert stages.fitted_on(model, root=waves_root, ledger=ledger) == {
        (WAVES_DIR, 0), (WAVES_DIR, 127)}


def test_the_bytes_a_take_was_drawn_at_are_power_on_then_held_then_setting(waves_root):
    on = stages.power_on(waves_root, UNIT, TYPE)
    assert on["40 03 03"] == 40 and on["40 03 16"] == 40 and len(on) == 23
    assert (on["40 03 17"], on["40 03 18"], on["40 03 19"]) == (40, 0, 0)
    writes = stages.writes_of({"40 03 00": "01 24", "40 03 05": "7f", "40 03 06": "11"})
    assert writes == [("40 03 00", (0x01, 0x24)), ("40 03 05", (0x7F,)),
                      ("40 03 06", (0x11,))]
    now = stages.bytes_now(on, writes, "40 03 06", 3)
    assert (now["40 03 00"], now["40 03 01"]) == (0x01, 0x24)
    assert now["40 03 05"] == 0x7F and now["40 03 06"] == 3 and now["40 03 07"] == 40


# ---------------------------------------------------------------- what a drawing leaves


def test_a_process_that_read_drawn_takes_cannot_write_into_the_archive(waves_root, tmp_path):
    drawn = stages.render_directory(stages.candidate(
        waves_root / "inferences/models/whole-0124-linear.json", waves_root),
        waves_root, WAVES_DIR)
    report.write_json(str(tmp_path / "data" / "units" / "u" / "before.json"), {})
    takes.listing(drawn)
    with pytest.raises(SystemExit):
        report.write_json(str(tmp_path / "data" / "units" / "u" / "after.json"), {})
    report.write_json(str(tmp_path / ".cache" / "after.json"), {})
    assert (tmp_path / ".cache" / "after.json").is_file()
    kept = json.loads((drawn / "takes-manifest.json").read_text())["rendered"]
    assert {"model", "model_sha256", "from", "numpy", "scipy"} <= set(kept)


def test_a_changed_model_is_drawn_again_and_an_unchanged_one_is_not(tmp_path):
    root = _waves_tree(tmp_path)
    path = root / "inferences/models/whole-0124-linear.json"
    drawn = stages.render_directory(stages.candidate(path, root), root, WAVES_DIR)
    take = drawn / "tone-v127-00.wav"
    first_sha = json.loads((drawn / "takes-manifest.json").read_text())["rendered"]
    first = take.stat().st_mtime_ns
    stages.render_directory(stages.candidate(path, root), root, WAVES_DIR)
    assert take.stat().st_mtime_ns == first
    model = json.loads(path.read_text())
    model["nodes"][0]["time_ms"]["map"]["points"][1][1] = 1.9
    path.write_text(json.dumps(model))
    before = takes.read(take)[0]
    stages.render_directory(stages.candidate(path, root), root, WAVES_DIR)
    after = json.loads((drawn / "takes-manifest.json").read_text())["rendered"]
    assert after["model_sha256"] != first_sha["model_sha256"]
    assert not np.allclose(takes.read(take)[0], before)


def _halved(model_id: str, unit: str = UNIT) -> dict:
    """A send path standing in for the real one: both channels at half."""
    halve = _value(value=0.5, source="document",
                   rests_on=["documents/fixture-doc/effect-list.json"])
    nodes = [{"id": f"h_{s}", "kind": "gain", "input": f"in_{s}", "unit": "ratio", "gain": halve}
             for s in ("l", "r")]
    model = _graph(model_id, "sends", nodes, {"out_l": "h_l", "out_r": "h_r"},
                   dict.fromkeys(("40 03 03", "40 03 04", "40 03 05"), "fixed_at_power_on"))
    model["model"]["unit_id"] = unit
    return model


def test_every_drawing_passes_through_the_send_path_the_unit_names(tmp_path):
    root = _waves_tree(tmp_path)
    path = root / "inferences/models/whole-0124-linear.json"
    alone = stages.render_directory(stages.candidate(path, root), root, WAVES_DIR)
    before = takes.read(alone / "tone-v127-00.wav")[0]
    sends = _put_model(root, _halved("sends-half"))
    _write(root / stages.SENDS, {"drawn_after_every_type": sends.relative_to(root).as_posix()})
    drawn = stages.render_directory(stages.candidate(path, root), root, WAVES_DIR)
    # The send path is drawn at its own rate too, so one more round trip is in the take.
    np.testing.assert_allclose(takes.read(drawn / "tone-v127-00.wav")[0], 0.5 * before,
                               atol=5e-5)
    kept = json.loads((drawn / "takes-manifest.json").read_text())["rendered"]
    assert kept["after"] == "inferences/models/sends-half.json"
    assert kept["after_sha256"] == stages.candidate(sends, root).sha256


def test_a_send_path_named_for_another_unit_is_not_drawn(tmp_path):
    root = _waves_tree(tmp_path)
    sends = _put_model(root, _halved("sends-elsewhere", unit="another-unit"))
    _write(root / stages.SENDS, {"drawn_after_every_type": sends.relative_to(root).as_posix()})
    assert stages.drawn_after(root, UNIT) is None


def test_a_directory_newer_than_the_ledger_stops_the_stage(tmp_path):
    root = _waves_tree(tmp_path)
    shutil.copytree(root / ".cache/takes" / WAVES_DIR, root / ".cache/takes/syn/late")
    with pytest.raises(stages.LedgerBehind, match="soundings takes build"):
        stages.stage(root, UNIT, TYPE, class_name="whole-0124")


# ---------------------------------------------------------------- the readings table


@pytest.mark.parametrize("name", sorted(stages.READINGS))
def test_every_key_the_readings_table_names_is_in_that_stages_records(name):
    reading = stages.READINGS[name]
    records = [
        json.loads(p.read_text())
        for p in sorted(ROOT.glob(f"data/units/*/{reading.command}/**/*.json"))
    ]
    records = [r for r in records if isinstance(r, dict) and isinstance(r.get(reading.rows), list)]
    assert records, f"no published {name} record carries `{reading.rows}`"
    for record in records:
        for row in record[reading.rows]:
            assert stages.MISSING is not stages.at(row, reading.quantity), (
                f"a {name} row has no {reading.quantity}")
        if reading.floor is not None:
            assert stages.MISSING is not stages.at(record, reading.floor), (
                f"a {name} record has no {reading.floor}")
    module, _, function = reading.entry.rpartition(".")
    assert callable(getattr(__import__(f"soundings.{module}", fromlist=[function]), function))


def test_a_type_out_of_scope_is_written_as_excluded_and_nothing_is_compared(tmp_path):
    from soundings.cli import readings

    parser = __import__("argparse").ArgumentParser()
    readings.register(parser.add_subparsers(dest="command"))

    def run(*argv):
        args = parser.parse_args(["inferences", "stage", "fixture-unit", "00-00", *argv,
                                  "--root", str(tmp_path)])
        return args.func(args)

    assert run("--exclude", "no_effect") == 2
    assert run("--exclude", "no_effect", "--class", "whole-0000", "--write") == 2
    assert run("--exclude", "no_effect", "--write") == 0
    written = json.loads((tmp_path / "inferences/fixture-unit/stages/00-00.json").read_text())
    assert written["excluded"] == {"reason": "no_effect"}
    assert "p1" not in written
    assert _stage_errors(written) == []


def test_efx_bands_is_cut_to_explicit_bands_at_or_below_cut_hz():
    """Only band centres whose upper edge clears `graph.CUT_HZ` survive."""
    from soundings import efxbands

    base = ["efx-bands", ".cache/takes/x", "--type", TYPE, "--slot", "40 03 05",
           "--setting", r"v(?P<value>\d{3})", "--reference", "flat"]

    def bands(argv):
        cut = stages._band_cut_argv(argv, graph.CUT_HZ)
        return [float(cut[i + 1]) for i, token in enumerate(cut) if token == "--band"], cut

    default_bands, default_cut = bands(base)
    assert "--band-set" not in default_cut
    centres, width = efxbands.BAND_SETS["third-octave"]
    edge = 2 ** (width / 2)
    assert default_bands == [c for c in centres if c * edge <= graph.CUT_HZ]
    assert default_bands and max(default_bands) * edge <= graph.CUT_HZ

    twelfth_bands, twelfth_cut = bands([*base, "--band-set", "twelfth-octave"])
    assert "--band-set" not in twelfth_cut
    centres12, width12 = efxbands.BAND_SETS["twelfth-octave"]
    edge12 = 2 ** (width12 / 2)
    assert twelfth_bands == [c for c in centres12 if c * edge12 <= graph.CUT_HZ]
    # a finer set reaches a higher band than the third-octave default, still within reach
    assert max(twelfth_bands) > max(default_bands)

    explicit_bands, _ = bands([*base, "--band", "100", "--band", "13000"])
    assert explicit_bands == [100.0]


# ---------------------------------------------------------------- which takes are settings


def test_a_take_is_a_setting_when_its_name_is_a_byte_behind_one_prefix():
    assert stages.swept_bytes({"a": "v064", "b": "v127", "c": "out", "d": "silence",
                               "e": "flat-00", "f": "bypassed-01", "g": "silence-00"}) == {
        "a": 64, "b": 127}
    assert stages.swept_bytes({"a": "000", "b": "016"}) == {"a": 0, "b": 16}
    assert stages.swept_bytes({"a": "rate-009", "b": "rate-125", "c": "both-parked"}) == {
        "a": 9, "b": 125}
    # Two things swept in one directory: which byte a take was at is not said.
    assert stages.swept_bytes({"a": "rate-009", "b": "step-009"}) == {}
    assert stages.swept_bytes({"a": "200"}) == {}


# ---------------------------------------------------------------- the bypass check


INERT_RECORD = "data/units/fixture-unit/efx-bands/inert.json"


def _inert_record(root: Path, *, inert_at: int, address: str = "40 03 05",
                  held: tuple[str, ...] = ("40 42 22=01",)) -> None:
    """A band record of the type reading one setting inside its own floor and one far out."""
    argv = ["efx-bands", ".cache/takes/elsewhere", "--type", TYPE, "--slot", address]
    for pair in held:
        argv += ["--held", pair]
    loud = 127 if inert_at != 127 else 0
    _write(root / INERT_RECORD, {
        "record": {"invocation": [*argv, "--out", INERT_RECORD]},
        "type": TYPE, "address": address, "reference": {"floor_db": [0.1, 0.3]},
        "readings": [{"value": inert_at, "largest_db": -0.2}, {"value": loud, "largest_db": 6.0}],
    })


@pytest.fixture(scope="module")
def bypass_root(tmp_path_factory) -> Path:
    return _waves_tree(tmp_path_factory.mktemp("bypass"))


def test_the_bypass_check_passes_at_a_setting_a_band_record_reads_as_inert(bypass_root):
    # At 0 the comb crossfades to the dry path alone, so the unit is doing nothing there.
    _inert_record(bypass_root, inert_at=0)
    found = stages.stage(bypass_root, UNIT, TYPE)
    check = found["control"]["bypass_check"]
    assert check["result"] == "passed"
    assert [(a["dir"], a["setting"], a["record"]) for a in check["asked"]] == [
        (WAVES_DIR, 0, INERT_RECORD)]
    assert found["comparison"] == {"used": "waves", "chosen_by": "by_repeatability_class"}


def test_the_bypass_check_fails_where_the_unit_is_not_doing_nothing(bypass_root):
    # A record calling 127 inert is contradicted by the unit's own takes there.
    _inert_record(bypass_root, inert_at=127)
    found = stages.stage(bypass_root, UNIT, TYPE)
    check = found["control"]["bypass_check"]
    assert check["result"] == "failed"
    assert [(a["dir"], a["setting"]) for a in check["asked"]] == [(WAVES_DIR, 127)]
    assert check["asked"][0]["lifted_db"] > stages.WITHIN_THE_FLOOR_DB
    assert found["comparison"] == {"used": "readings", "chosen_by": "bypass_check_failed",
                                   "cut_hz": graph.CUT_HZ}


@pytest.mark.parametrize("record", [
    {"inert_at": 50},
    {"inert_at": 0, "address": "40 03 03"},
    {"inert_at": 0, "held": ("40 42 22=01", "40 03 03=7F")},
], ids=["a-setting-not-compared", "another-address", "another-held-state"])
def test_the_bypass_check_is_not_asked_without_an_inert_setting_of_this_state(bypass_root,
                                                                               record):
    _inert_record(bypass_root, **record)
    found = stages.stage(bypass_root, UNIT, TYPE)
    assert found["control"]["bypass_check"]["result"] == "not_asked"
    assert found["comparison"]["used"] == "waves"


# ---------------------------------------------------------------- which class is stage 8's


def _cls(name, used, floor, settings=5):
    return stages._Class(name, [], settings, floor, used, "by_repeatability_class")


def test_stage_8_takes_the_class_the_effect_stands_in_most_often():
    quiet = _cls("struck", "waves", -50.0)
    walked = _cls("held", "readings", None, settings=136)
    first, rest = stages.p1_class([quiet, walked], {"struck": 0, "held": 40})
    assert first is walked and rest == [quiet]


def test_a_tie_on_standing_settings_goes_to_waves_then_to_the_lower_floor():
    waves_high = _cls("a", "waves", -30.0)
    waves_low = _cls("b", "waves", -45.0)
    readings = _cls("c", "readings", -60.0)
    first, rest = stages.p1_class([readings, waves_high, waves_low],
                                  {"a": 3, "b": 3, "c": 3})
    assert first is waves_low
    # The rest keep the order they were given in, which is the order stage 9 takes them.
    assert [c.name for c in rest] == ["c", "a"]


def test_stage_8_is_not_the_quietest_class_when_the_effect_does_nothing_there(tmp_path):
    root = _tree(tmp_path, static=True)
    a = _comb("whole-0124-linear", "linear")
    _put_model(root, a)
    _directory(root, WAVES_DIR, stimulus=TONE, address="40 03 05", seconds=2.0,
               draw=_drawn_by(root, None, "40 03 05"), seed=0)
    _directory(root, WAVES_P2, stimulus=VOICE, address="40 03 05", seconds=2.0,
               draw=_drawn_by(root, a, "40 03 05"), seed=1)
    _ledger(root, {WAVES_DIR: {"address": "40 03 05", "stimulus": TONE},
                   WAVES_P2: {"address": "40 03 05", "stimulus": VOICE}})
    found = stages.stage(root, UNIT, TYPE)
    assert {row["dir"] for row in found["compared"]} == {WAVES_P2}
    assert [c["class"] for c in found["classes"]] == ["voice"]
    assert found["classes"][0]["standing"] == len(SETTINGS) - 1


# ---------------------------------------------------------------- one comparison per class


MIXED_TONE = "syn/01-24-40-03-05-tone-bare"
MIXED_APPLAUSE = "syn/01-24-40-03-05-applause"
APPLAUSE = {**TONE, "name": "applause", "program": 126, "note": 60}


def _bare(value: int) -> str:
    return f"{value:03d}"


@pytest.fixture(scope="module")
def mixed_stage(tmp_path_factory) -> dict:
    """A static type swept under a repeating tone, named without `v`, and under applause.

    Applause is first by name and holds as many settings; each of its takes is fresh
    noise, so two takes of one setting subtract to nothing smaller than themselves.
    """
    root = _tree(tmp_path_factory.mktemp("mixed"), static=True)
    a = _comb("whole-0124-linear", "linear")
    _put_model(root, a)
    _put_model(root, _comb("whole-0124-truncated", "none"))
    _directory(root, MIXED_TONE, stimulus=TONE, address="40 03 05", seconds=2.0,
               draw=_drawn_by(root, a, "40 03 05"), seed=11, named=_bare)
    rng = np.random.default_rng(12)

    def applause(value, k, n, clean):
        out = clean.copy()
        out[:, 2:4] = 0.05 * rng.standard_normal((clean.shape[0], 2))
        return out

    _directory(root, MIXED_APPLAUSE, stimulus=APPLAUSE, address="40 03 05", seconds=2.0,
               draw=applause, seed=13)
    _ledger(root, {MIXED_TONE: {"address": "40 03 05", "stimulus": TONE, "named": _bare},
                   MIXED_APPLAUSE: {"address": "40 03 05", "stimulus": APPLAUSE}})
    return stages.stage(root, UNIT, TYPE, class_name="whole-0124")


def test_a_setting_named_without_a_v_is_compared(mixed_stage):
    assert _stage_errors(mixed_stage) == []
    assert {(r["dir"], r["setting"]) for r in mixed_stage["p1"]["compared"]} == {
        (MIXED_TONE, v) for v in SETTINGS}


def test_each_stimulus_class_is_compared_its_own_way_and_waves_goes_first(mixed_stage):
    p1, p2 = mixed_stage["p1"], mixed_stage["p2"]
    assert p1["comparison"] == {"used": "waves", "chosen_by": "by_repeatability_class"}
    assert [c["class"] for c in p1["classes"]] == ["tone"]
    assert p1["classes"][0]["floor_db"] < stages.REPEATS_BELOW_DB
    assert [c["class"] for c in p2["classes"]] == ["applause"]
    assert p2["classes"][0]["used"] == "readings"
    assert p2["classes"][0]["chosen_by"] == "by_stimulus_floor"
    assert p2["classes"][0]["floor_db"] > stages.REPEATS_BELOW_DB
    assert p2["comparison"]["used"] == "readings"


def test_nothing_compared_stops_on_the_input_and_ranks_nobody_out(mixed_stage):
    p2 = mixed_stage["p2"]
    assert p2["compared"] == []
    assert p2["excluded_from_ranking"] == []
    assert p2["gates"] == {}
    assert mixed_stage["stopped"]["at"] == 9
    assert mixed_stage["stopped"]["gate"] == "no_input_take"
    assert MIXED_APPLAUSE in mixed_stage["stopped"]["needs"]


def test_what_a_stop_needs_names_the_quantity_the_directory_and_the_settings(tmp_path):
    root = _waves_tree(tmp_path, unit_is_identity=True)
    found = stages.stage(root, UNIT, TYPE, class_name="whole-0124")
    needs = found["stopped"]["needs"]
    assert found["stopped"]["gate"] == "control_could_not_fail"
    assert "waves:residual_db" in needs
    assert WAVES_DIR in needs
    assert all(str(v) in needs for v in SETTINGS)


def test_with_nobody_ranked_p0_is_the_candidate_nearest_the_unit(waves_root):
    found = stages.stage(waves_root, UNIT, TYPE, class_name="worse-first")
    assert sorted(found["p1"]["excluded_from_ranking"]) == ["worse-first-a", "worse-first-b"]
    assert found["p0"]["model"].endswith("worse-first-b.json")


# ---------------------------------------------------------------- a template for an unread walk


RATE_P2 = "syn/01-24-40-03-03-voice"
RATE_VOICE = {**VOICE, "hold_s": 4.0, "lead_s": 0.6}


def test_a_directory_no_record_read_is_read_from_the_types_own_invocation(tmp_path):
    root = _tree(tmp_path, static=False)
    a = _tremolo("whole-0124-log", log=True)
    _put_model(root, a)
    _put_model(root, _tremolo("whole-0124-linear-map", log=False))
    draw = _drawn_by(root, a, "40 03 03")
    _directory(root, RATE_DIR, stimulus=RATE_TONE, address="40 03 03", seconds=5.0,
               draw=draw, seed=7)
    _directory(root, RATE_P2, stimulus=RATE_VOICE, address="40 03 03", seconds=5.0,
               draw=draw, seed=8)
    _write(root / RATE_RECORD, {
        "record": {"invocation": [
            "efx-rate", f".cache/takes/{RATE_DIR}", "--type", TYPE, "--slot", "40 03 03",
            "--setting", r"tone-v(?P<value>\d{3})-\d+", "--channel", "2",
            "--out", RATE_RECORD]},
        "type": TYPE, "channel": {"read": 2}, "readings": [],
    })
    _ledger(root, {RATE_DIR: {"address": "40 03 03", "stimulus": RATE_TONE,
                              "read_by": [RATE_RECORD]},
                   RATE_P2: {"address": "40 03 03", "stimulus": RATE_VOICE}})
    _claim(root, RATE_CLAIM, RATE_RECORD, (0, 127))
    found = stages.stage(root, UNIT, TYPE, class_name="whole-0124")
    assert _stage_errors(found) == []
    p2 = found["p2"]
    assert p2["comparison"]["used"] == "readings"
    assert {(r["dir"], r["setting"]) for r in p2["compared"]} == {
        (RATE_P2, v) for v in SETTINGS}
    assert p2["control"]["separated"] is True


def test_a_template_reads_the_length_of_the_stimulus_it_is_aimed_at(tmp_path):
    root = _tree(tmp_path, static=False)
    short = {**RATE_VOICE, "hold_s": 2.5, "lead_s": 0.3}
    unstated = {**RATE_VOICE, "hold_s": None}
    for rel, stim in ((RATE_DIR, RATE_TONE), (RATE_P2, short), ("syn/unstated", unstated)):
        _directory(root, rel, stimulus={**stim, "hold_s": stim["hold_s"] or 1.0},
                   address="40 03 03", seconds=3.0, draw=_drawn_by(root, None, "40 03 03"),
                   seed=3)
    _ledger(root, {RATE_DIR: {"address": "40 03 03", "stimulus": RATE_TONE},
                   RATE_P2: {"address": "40 03 03", "stimulus": short},
                   "syn/unstated": {"address": "40 03 03", "stimulus": unstated}})
    found = json.loads((root / ".cache/takes-ledger.json").read_text())
    base = ["efx-rate", f".cache/takes/{RATE_DIR}", "--type", TYPE, "--slot", "40 03 03",
            "--setting", r"tone-v(?P<value>\d{3})-\d+"]

    def aimed(argv, rel=RATE_P2):
        return stages.templated(root, found, argv, RATE_DIR, stages.directory(root, found, rel))

    def flag(argv, name):
        return [argv[i + 1] for i, a in enumerate(argv) if a == name]

    held_long = aimed([*base, "--hold", "4.0", "--lead", "0.6"])
    assert flag(held_long, "--hold") == ["2.5"] and flag(held_long, "--lead") == ["0.3"]
    # A flag the template left to the stage's default is still the target's own length.
    defaulted = aimed(base)
    assert flag(defaulted, "--hold") == ["2.5"] and flag(defaulted, "--lead") == ["0.3"]
    # A length the target does not state is not borrowed from the template's directory.
    assert aimed([*base, "--hold", "4.0"], rel="syn/unstated") is None
    assert stages.LENGTH_FLAGS == {"--hold": "hold_s", "--lead": "lead_s"}


def test_records_in_two_units_are_gated_once_per_unit():
    unit = {("d", 0): [[1.0]], ("d", 1): [[2.0]], ("d", 2): [[3.0]]}
    far = {("d", 0): [[3.0]], ("d", 1): [[1.0]], ("d", 2): [[1.0]]}
    ms = stages.scored_readings("efx-time:ms", stages.READINGS["efx-time:ms"], unit, unit,
                                floor=0.1)
    db = stages.scored_readings("efx-bands:largest_db", stages.READINGS["efx-bands:largest_db"],
                                unit, far, floor=0.1)
    assert stages.separated([ms, db]) is True
    assert stages.separated([ms]) is False


def test_a_part_routed_past_the_effect_in_another_directory_is_the_input(tmp_path):
    # A stimulus leaves out the volume where it is the one it always sends.
    unwritten = {k: v for k, v in TONE.items() if k != "volume"}
    stimuli = {"params/40-03-03": {**unwritten, "class": TONE["name"]},
               "params/control": {**TONE, "volume": 127, "class": TONE["name"]}}
    clean = _clean(TONE, 1.0)
    dirs = {"params/40-03-03": ("40 03 03", ("0", "127")),
            "params/control": ("40 42 22", ("0", "1"))}
    for rel, (address, values) in dirs.items():
        entries = []
        for value in values:
            name = f"{TONE['name']}-{value}-00.wav"
            takes.write(tmp_path / ".cache/takes" / rel / name, clean, FS)
            entries.append({"file": name, "stimulus": TONE["name"], "setting": value,
                            "take": 0, "sample_rate": FS, "channels": 6, "seconds": 1.0})
        _write(tmp_path / ".cache/takes" / rel / "takes-manifest.json",
               {"type": TYPE, "address": address, "stimuli": [TONE], "takes": entries})
    _write(tmp_path / ".cache" / "takes-ledger.json", {"unit": UNIT, "directories": {
        rel: {"type": TYPE, "address": address, "settings": {v: 1 for v in values},
              "stimuli": [stimuli[rel]], "held": HELD, "consumed_by": []}
        for rel, (address, values) in dirs.items()}})
    found = stages._ledger(tmp_path, TYPE)
    made = stages.directory(tmp_path, found, "params/40-03-03")
    assert made.input_from == {TONE["name"]: "params/control"}
    assert sorted(made.by_setting) == [0, 127]


def _input_tree(tmp_path: Path, extra: dict) -> dict:
    """A swept directory, its own bypass takes, and `extra`: rel -> (address, values, manifest)."""
    clean = _clean(TONE, 1.0)
    dirs = {"params/40-03-03": ("40 03 03", ("0", "127"), {}),
            "params/control": ("40 42 22", ("0", "1"), {}), **extra}
    for rel, (address, values, more) in dirs.items():
        entries = []
        for value in values:
            name = f"{TONE['name']}-{value}-00.wav"
            takes.write(tmp_path / ".cache/takes" / rel / name, clean, FS)
            entries.append({"file": name, "stimulus": TONE["name"], "setting": value,
                            "take": 0, "sample_rate": FS, "channels": 6, "seconds": 1.0})
        _write(tmp_path / ".cache/takes" / rel / "takes-manifest.json",
               {"type": TYPE if address else None, "address": address, "stimuli": [TONE],
                "takes": entries, **more})
    _write(tmp_path / ".cache" / "takes-ledger.json", {"unit": UNIT, "directories": {
        rel: {"type": TYPE if address else None, "address": address,
              "settings": {v: 1 for v in values},
              "stimuli": [{**TONE, "class": TONE["name"]}], "held": HELD if address else None,
              "consumed_by": []}
        for rel, (address, values, _) in dirs.items()}})
    return stages._ledger(tmp_path, TYPE)


def test_the_part_with_its_reverb_send_at_0_is_the_input_before_any_bypass_take(tmp_path):
    found = _input_tree(tmp_path, {"cc91": (None, ("0", "100"), {"controller": 91})})
    made = stages.directory(tmp_path, found, "params/40-03-03")
    assert made.input_from == {TONE["name"]: "cc91"}
    assert made.input_is == {TONE["name"]: "dry"}
    assert [p.name for p in made.bypass[TONE["name"]]] == [f"{TONE['name']}-0-00.wav"]


def test_without_a_dry_take_the_bypass_take_is_the_input_and_is_said_to_be(tmp_path):
    found = _input_tree(tmp_path, {"cc93": (None, ("0", "100"), {"controller": 93})})
    made = stages.directory(tmp_path, found, "params/40-03-03")
    assert made.input_from == {TONE["name"]: "params/control"}
    assert made.input_is == {TONE["name"]: "bypass"}


def test_an_input_from_another_directory_is_the_one_taken_nearest_in_time(tmp_path):
    clean = _clean(TONE, 1.0)
    dirs = {"a-other-evening/control": ("40 42 22", ("0", "1"), 0),
            "params/40-03-03": ("40 03 03", ("0", "127"), 9 * 86400),
            "params/control": ("40 42 22", ("0", "1"), 9 * 86400 - 300)}
    for rel, (address, values, at) in dirs.items():
        entries = []
        for value in values:
            name = f"{TONE['name']}-{value}-00.wav"
            takes.write(tmp_path / ".cache/takes" / rel / name, clean, FS)
            os.utime(tmp_path / ".cache/takes" / rel / name, (1e9 + at, 1e9 + at))
            entries.append({"file": name, "stimulus": TONE["name"], "setting": value,
                            "take": 0, "sample_rate": FS, "channels": 6, "seconds": 1.0})
        _write(tmp_path / ".cache/takes" / rel / "takes-manifest.json",
               {"type": TYPE, "address": address, "stimuli": [TONE], "takes": entries})
    _write(tmp_path / ".cache" / "takes-ledger.json", {"unit": UNIT, "directories": {
        rel: {"type": TYPE, "address": address, "settings": {v: 1 for v in values},
              "stimuli": [{**TONE, "class": TONE["name"]}], "held": HELD, "consumed_by": []}
        for rel, (address, values, _) in dirs.items()}})
    made = stages.directory(tmp_path, stages._ledger(tmp_path, TYPE), "params/40-03-03")
    assert made.input_from == {TONE["name"]: "params/control"}
