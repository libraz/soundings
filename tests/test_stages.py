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

import json
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
# Stage 8 takes the commonest stimulus class, a tie going to the name first in order,
# so with one directory each `tone` is stage 8 and `voice` stage 9.
VOICE = {**TONE, "name": "voice", "program": 17, "note": 76}
PARTIALS = {"tone": ((440, 0.1), (880, 0.05), (1320, 0.03), (1760, 0.02)),
            "voice": ((330, 0.08), (660, 0.05), (990, 0.04), (1650, 0.02))}
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
           {"effects": [{"type": TYPE, "msb": 1, "lsb": 36, "parameters": [40] * 20}]})
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


def _directory(root, rel, *, stimulus, address, seconds, draw, seed) -> None:
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
            keep(f"{stimulus['name']}-v{value:03d}-{k:02d}.wav", f"v{value:03d}", k,
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
            "settings": {**{f"v{v:03d}": 2 for v in SETTINGS}, "out": 2, "silence": 1},
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
                  _comb("fitted-b", "none", cls="fitted-only", fitted_on=everywhere)):
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
    assert p1["comparison"] == {"used": "readings", "chosen_by": "by_repeatability_class"}
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
    assert on["40 03 03"] == 40 and on["40 03 16"] == 40 and len(on) == 20
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
