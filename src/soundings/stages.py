"""Carrying one effect type through stages 7-9: draw its candidates, compare, decide.

A candidate `graph` model is drawn on the unit's own bypass takes -- the effect's
input -- and what it drew is compared with the unit's takes in one of two ways,
chosen mechanically. `waves` subtracts takes, for a type the repeatability sort
calls static; `readings` reads both sides with the stages that published the type's
records, for one that moves. Either comparison is put into the shape
`reproduce.gates` scores, and the gates are not changed.

Every comparison is run twice: for the candidate, and for `IDENTITY`, which draws
the bypass take back unchanged. A comparison the identity passes could not have
found any candidate wrong, so it passes nothing. A setting a candidate's values
were fitted on -- declared, or derived through a claim's records and the ledger --
is not evidence for it, and a candidate with none left is kept out of the ranking.

A take is drawn at the type's power-on values, overlaid by the ledger's held writes,
overlaid by its setting. `writes_of` is the one conversion of the held block, used
for drawing and for a reading's `held` alike.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy

from . import ledger, reproduce, takes
from .inferences import EFFECT_BLOCK
from .render import graph

RENDERED_ROOT = Path(".cache/rendered")

SETTING = re.compile(r"^v(\d{3})$")
"""A take of one setting, as `walk` names it: the byte, zero-padded."""

BYPASS = re.compile(r"^(out|bypassed|bypassed-\d+)$")
"""A take with the effect routed out of the path, which is the effect's input."""

SILENCE = "silence"

STIMULUS_FIELDS = ("name", "program", "note", "velocity", "channel", "writes", "volume", "hold_s")
"""What two directories' stimuli must share for one's bypass take to feed the other."""

WITHIN_THE_FLOOR_DB = 3.0
"""How far over a setting's own repeat floor a subtraction may land and still be nothing."""

LEVEL_RESOLUTION_DB = 0.001
"""The step `reproduce.subtracted` rounds a fitted level to; no bound is finer than it."""

GATES = ("gross", "qualitative", "breakdown", "power")

IDENTITY = {
    "model": {
        "schema_version": 1, "id": "identity", "class": None, "candidate": "identity",
        "kind": "graph", "made_by": "soundings.stages",
    },
    "sample_rate_hz": 32000,
    "why_this_rate": "The rate the candidates are drawn at, so the control takes the same "
    "round trip they do.",
    "inputs": ["in_l", "in_r"],
    "outputs": {"out_l": "in_l", "out_r": "in_r"},
    "nodes": [],
    "rows": {},
}
"""The control: the bypass take drawn back as it came. Not a model file, so no class
counts it as a candidate and no ranking can make it a decoy."""

IDENTITY_LOADED = {**IDENTITY, "referenced": {}}


@dataclass(frozen=True)
class Reading:
    """One quantity a reading stage publishes, as a comparison scores it.

    `command` is the stage, `entry` the function its command calls, `rows` the list in
    its record each reading sits in, `quantity` the key read off each row, `floor` the
    record's own floor for that quantity (None where it publishes none) and
    `measured_in` what a residual is in. `octaves` compares the logarithm, as a rate
    or a frequency is compared.
    """

    command: str
    entry: str
    rows: str
    quantity: str
    floor: str | None
    measured_in: str
    octaves: bool = False


_READINGS = (
    Reading("efx-rate", "efxrate.read_directory", "readings", "rate_hz", None, "octaves",
            octaves=True),
    Reading("efx-excursion", "efxexcursion.read_directory", "readings", "excursion_ms",
            "floor.ms", "ms"),
    Reading("efx-excursion", "efxexcursion.read_directory", "readings", "off_the_phase_ms",
            "floor_off_the_phase.ms", "ms"),
    Reading("efx-time", "efxtime.read_directory", "readings", "ms", "floor_ms", "ms"),
    Reading("efx-bands", "efxbands.read_directory", "readings", "largest_db",
            "reference.floor_db", "dB"),
    Reading("efx-bands", "efxbands.read_directory", "readings", "largest_at_hz", None,
            "octaves", octaves=True),
    Reading("efx-bands", "efxbands.read_directory", "readings", "fitted_at_hz", None,
            "octaves", octaves=True),
    Reading("efx-sway", "efxsway.sweep", "readings", "level_in_db.depth", None, "dB"),
    Reading("efx-sway", "efxsway.sweep", "readings", "balance.depth", None, "dB"),
    Reading("efx-orders", "efxorders.read_directory", "readings", "all_of_them_db",
            "reference.floor_db", "dB"),
    Reading("decay", "decay.measure", "bands", "rt60_s", None, "s"),
)

READINGS = {f"{r.command}:{r.quantity}": r for r in _READINGS}
"""Each quantity a `readings` comparison scores, keyed `<command>:<quantity>`: one row
per quantity a stage's own record names as a reading, each its own record at the gates."""

COMMANDS = sorted({r.command for r in _READINGS})

STAGE_READINGS = Path(".cache/stage-readings")
"""Where the unit's side of a reading is kept, keyed by command, arguments and mtime."""

MISSING = object()


class LedgerBehind(RuntimeError):
    """The ledger does not hold a directory the stage would have to compare."""


def at(item, dotted: str):
    """The value under a dotted key, or `MISSING`."""
    for key in dotted.split("."):
        if not isinstance(item, dict) or key not in item:
            return MISSING
        item = item[key]
    return item


def type_of(text: str) -> str:
    """`01-24` or `01 24` as `01 24`."""
    return " ".join(text.replace("-", " ").upper().split())


# ---------------------------------------------------------------- the write state


def _shifted(address: str, by: int) -> str:
    *head, last = address.split()
    return " ".join([*head, f"{int(last, 16) + by:02X}"])


def writes_of(held: dict[str, str] | None) -> list[tuple[str, tuple[int, ...]]]:
    """A ledger's held block as the writes `--held` takes: address, then its bytes."""
    return [
        (ledger.norm_addr(address), tuple(int(b, 16) for b in values.split()))
        for address, values in sorted((held or {}).items())
        if values
    ]


def power_on(root: Path, unit: str, type_: str) -> dict[str, int]:
    """What a type powers up holding, `parameters[i]` at `40 03 (03 + i)`."""
    path = Path(root) / "data" / "units" / unit / "efx-map" / "types.json"
    effects = json.loads(path.read_text())["effects"]
    entry = next((e for e in effects if e["type"] == type_), None)
    if entry is None:
        raise ValueError(f"{path} has no power-on values for {type_}")
    first = f"{EFFECT_BLOCK} 03"
    return {_shifted(first, i): int(v) for i, v in enumerate(entry["parameters"])}


def bytes_now(on: dict[str, int], writes, address: str | None, value: int | None) -> dict:
    """Power-on, then each held write over consecutive addresses, then the setting."""
    now = dict(on)
    for where, values in writes:
        for i, byte in enumerate(values):
            now[_shifted(where, i)] = int(byte)
    if address is not None and value is not None:
        now[ledger.norm_addr(address)] = int(value)
    return now


def setting_value(name) -> int | None:
    found = SETTING.match(str(name))
    return int(found.group(1)) if found else None


# ---------------------------------------------------------------- the ledger


def _ledger(root: Path, type_: str) -> dict:
    """The ledger, refused when a directory of this type on disk is missing from it."""
    found = ledger.load(root)
    if found is None:
        raise LedgerBehind(f"no {ledger.LEDGER_PATH} under {root}: run `soundings takes build`")
    base = root / ledger.TAKES_ROOT
    missing = []
    for manifest in sorted(base.rglob("takes-manifest.json")):
        rel = manifest.parent.relative_to(base).as_posix()
        if rel in found["directories"]:
            continue
        try:
            kept = json.loads(manifest.read_text())
        except ValueError:
            continue
        if ledger.norm_bytes(kept.get("type")) == type_:
            missing.append(rel)
    if missing:
        raise LedgerBehind(
            f"the ledger is older than {len(missing)} directories of {type_} "
            f"({', '.join(missing[:3])}): run `soundings takes build` and ask again"
        )
    return found


def directories(found: dict, type_: str) -> list[str]:
    """The ledger's directories of a type holding at least one setting take."""
    return sorted(
        rel for rel, entry in found["directories"].items()
        if entry.get("type") == type_
        and any(setting_value(s) is not None for s in (entry.get("settings") or {}))
    )


def _stimulus_key(stimulus: dict) -> str | None:
    if any(stimulus.get(f) is None for f in STIMULUS_FIELDS):
        return None
    return json.dumps([stimulus.get(f) for f in STIMULUS_FIELDS], sort_keys=True)


@dataclass
class Directory:
    """A directory of the unit's takes, with the input each of its stimuli is drawn from."""

    rel: str
    where: Path
    address: str | None
    entry: dict
    by_setting: dict[int, list[dict]]
    bypass: dict[str, list[Path]] = field(default_factory=dict)
    input_from: dict[str, str] = field(default_factory=dict)

    @property
    def classes(self) -> set[str]:
        return {s.get("class") for s in self.entry.get("stimuli") or [] if s.get("class")}


def _bypasses(where: Path) -> dict[str, list[Path]]:
    listed, _ = takes.listing(where)
    found: dict[str, list[Path]] = {}
    for name, entry in sorted(listed.items(), key=lambda kv: (kv[1].get("take", 0), kv[0])):
        if BYPASS.match(str(entry.get("setting"))):
            found.setdefault(str(entry.get("stimulus")), []).append(where / name)
    return found


def directory(root: Path, found: dict, rel: str) -> Directory:
    """One ledger directory, its setting takes grouped, its input found or not."""
    where = root / ledger.TAKES_ROOT / rel
    entry = found["directories"][rel]
    listed, _ = takes.listing(where)
    by_setting: dict[int, list[dict]] = {}
    for name, item in sorted(listed.items(), key=lambda kv: (kv[1].get("take", 0), kv[0])):
        value = setting_value(item.get("setting"))
        if value is not None and (where / name).is_file():
            by_setting.setdefault(value, []).append({**item, "file": name})
    made = Directory(rel, where, entry.get("address"), entry, dict(sorted(by_setting.items())))
    own = _bypasses(where)
    stimuli = {str(e["stimulus"]) for items in by_setting.values() for e in items}
    for name in sorted(stimuli):
        if own.get(name):
            made.bypass[name], made.input_from[name] = own[name], rel
            continue
        mine = next((s for s in entry.get("stimuli") or [] if s.get("name") == name), None)
        key = None if mine is None else _stimulus_key(mine)
        if key is None:
            continue
        for other, theirs in sorted(found["directories"].items()):
            match = next(
                (s for s in theirs.get("stimuli") or [] if _stimulus_key(s) == key), None)
            if other == rel or match is None:
                continue
            there = _bypasses(root / ledger.TAKES_ROOT / other).get(match["name"])
            if there:
                made.bypass[name], made.input_from[name] = there, other
                break
    made.by_setting = {
        v: [e for e in items if str(e["stimulus"]) in made.bypass]
        for v, items in made.by_setting.items()
    }
    made.by_setting = {v: items for v, items in made.by_setting.items() if items}
    return made


# ---------------------------------------------------------------- candidates


@dataclass(frozen=True)
class Candidate:
    id: str
    shown_as: str
    raw: dict
    loaded: dict
    sha256: str


def candidate(path: str | Path, root: Path) -> Candidate:
    """A graph model file, loaded and checked, with the SHA-256 of its bytes."""
    path, root = Path(path), Path(root)
    data = path.read_bytes()
    raw = json.loads(data)
    loaded = graph.load_graph(raw, models_dir=root / "inferences" / "models", root=root)
    try:
        shown = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        shown = str(path)
    return Candidate(raw["model"]["id"], shown, raw, loaded, hashlib.sha256(data).hexdigest())


def identity() -> Candidate:
    data = json.dumps(IDENTITY, sort_keys=True).encode()
    return Candidate("identity", "identity", IDENTITY, IDENTITY_LOADED,
                     hashlib.sha256(data).hexdigest())


def members(root: Path, class_name: str) -> list[Candidate]:
    """Every model file naming the class, in file order."""
    found = []
    for path in sorted((root / "inferences" / "models").glob("*.json")):
        if json.loads(path.read_text())["model"].get("class") == class_name:
            found.append(candidate(path, root))
    if not found:
        raise ValueError(f"no model under inferences/models names the class {class_name}")
    return found


def _values(item):
    """Every value spec in a model: a dict carrying a `source`."""
    if isinstance(item, dict):
        if "source" in item:
            yield item
        for value in item.values():
            yield from _values(value)
    elif isinstance(item, list):
        for value in item:
            yield from _values(value)


def _setting(value):
    if isinstance(value, int):
        return value
    text = str(value)
    if text.isdigit():
        return int(text)
    found = setting_value(text)
    return text if found is None else found


def _rel(text: str) -> str:
    prefix = ledger.TAKES_ROOT.as_posix() + "/"
    return text[len(prefix):] if text.startswith(prefix) else text


def fitted_on(model: dict, *, root: Path, ledger: dict) -> set[tuple[str, object]]:
    """The `[directory, setting]` pairs any value was fitted on, declared or derived.

    Derived through each claim a `measured` or `law` value rests on: the claim's
    records and the settings its keys cite, and the directories whose ledger entry
    says those records read them.
    """
    readers: dict[str, set[str]] = {}
    for rel, entry in ledger["directories"].items():
        for hit in entry.get("consumed_by") or []:
            readers.setdefault(hit["record"], set()).add(rel)
    found: set[tuple[str, object]] = set()
    for spec in _values(model.get("nodes", [])):
        for pair in spec.get("fitted_on") or []:
            found.add((_rel(str(pair[0])), _setting(pair[1])))
        if spec.get("source") == "document":
            continue
        for cited in spec.get("rests_on") or []:
            path = Path(root) / str(cited)
            if not (str(cited).startswith("inferences/") and path.is_file()):
                continue
            claim = json.loads(path.read_text())
            for measurement in (claim.get("rests_on") or {}).get("measurements") or []:
                values = {
                    int(m.group(1))
                    for key in measurement.get("keys") or []
                    if (m := re.match(r"readings\[value=(\d+)\]", key))
                }
                for rel in readers.get(measurement["file"], ()):
                    found |= {(rel, v) for v in values}
    return found


# ---------------------------------------------------------------- drawing


def _pair(channel: int) -> tuple[int, int]:
    return channel - channel % 2, channel - channel % 2 + 1


def loudest(made: Directory) -> int:
    """The channel highest across the unit's setting takes, which `waves` reads."""
    names = [e["file"] for items in made.by_setting.values() for e in items]
    return takes.channel_reaching(made.where, names)[0]


def render_directory(
    cand: Candidate,
    root: Path,
    rel: str,
    *,
    channels: tuple[int, int] | None = None,
    found: dict | None = None,
    made: Directory | None = None,
) -> Path:
    """Draw a candidate on one directory, into `.cache/rendered/<model-id>/<rel>/`.

    Each setting take is drawn from a bypass take of its stimulus, its oscillators
    started `k / n` of a cycle apart across the setting's `n` takes; bypass and
    silence takes are copied. A directory already drawn by the same model bytes on
    the same channels is left as it is.
    """
    root = Path(root)
    found = found if found is not None else ledger.load(root)
    if found is None or rel not in found["directories"]:
        raise LedgerBehind(f"{rel} is not in the ledger: run `soundings takes build`")
    made = made if made is not None else directory(root, found, rel)
    channels = channels if channels is not None else _pair(loudest(made))
    out = root / RENDERED_ROOT / cand.id / rel
    if (out / "takes-manifest.json").is_file():
        # Read as a file and not through `takes`: checking what drew a directory is
        # not reading what it drew, and must not mark this process as having done so.
        kept = json.loads((out / "takes-manifest.json").read_text()).get("rendered", {})
        if kept.get("model_sha256") == cand.sha256 and kept.get("channels") == list(channels):
            return out
        shutil.rmtree(out)
    out.mkdir(parents=True)

    entry = made.entry
    on = power_on(root, found["unit"], entry["type"])
    writes = writes_of(entry.get("held"))
    listed, _ = takes.listing(made.where)
    kept_entries, unrenderable = [], []
    for name, item in sorted(listed.items()):
        setting = str(item.get("setting"))
        if BYPASS.match(setting) or setting == SILENCE:
            shutil.copy2(made.where / name, out / name)
            kept_entries.append(item)
    for value, items in made.by_setting.items():
        now = bytes_now(on, writes, made.address, value)
        for k, item in enumerate(items):
            inputs = made.bypass[str(item["stimulus"])]
            take, rate = takes.read(inputs[k % len(inputs)])
            try:
                drawn = graph.render_take(cand.loaded, take, rate, now, channels=channels,
                                          lfo_phase=k / len(items))
            except graph.Unrenderable:
                unrenderable.append(value)
                break
            takes.write(out / item["file"], drawn, rate)
            kept_entries.append({**item, "seconds": round(drawn.shape[0] / rate, 4)})
    method = takes.method_of(made.where)
    manifest = {**method, "takes": kept_entries, "rendered": {
        "model": cand.shown_as,
        "model_sha256": cand.sha256,
        "from": rel,
        "input_from": made.input_from,
        "channels": list(channels),
        "unrenderable": sorted(set(unrenderable)),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }}
    (out / "takes-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def _unrenderable(drawn: Path, rel: str) -> set[tuple[str, int]]:
    return {(rel, v) for v in takes.method_of(drawn)["rendered"].get("unrenderable", [])}


# ---------------------------------------------------------------- waves


def _cut(samples: np.ndarray, rate: int) -> np.ndarray:
    return reproduce.band_limited(samples, rate, graph.CUT_HZ)


def _wave(path: Path, channel: int) -> np.ndarray:
    samples, rate = takes.read(path)
    return _cut(samples[:, channel], rate)


def _unit_waves(made: Directory, channel: int) -> dict:
    """What the unit's own takes say at each setting: floor, doing nothing, level spread."""
    per: dict[int, dict] = {}
    for value, items in made.by_setting.items():
        waves = [_wave(made.where / e["file"], channel) for e in items]
        inputs = [made.bypass[str(e["stimulus"])] for e in items]
        plain = [_wave(src[k % len(src)], channel) for k, src in enumerate(inputs)]
        pairs = [reproduce.subtracted(a, b) for a, b in zip(waves, waves[1:], strict=False)]
        per[value] = {
            "waves": waves,
            "floor_pairs": [p["residual_db"] for p in pairs],
            "level_pairs": [p["level_fit_db"] for p in pairs],
            "doing_nothing": float(np.mean(
                [reproduce.subtracted(w, p)["residual_db"] for w, p in zip(waves, plain,
                                                                             strict=True)])),
        }
    return {v: s for v, s in per.items() if s["floor_pairs"]}


def _scored_waves(rel: str, unit: dict, drawn: Path, channel: int, lost: set) -> dict:
    """One directory in the shape `reproduce.gates` scores, per the `waves` table."""
    all_pairs = [f for s in unit.values() for f in s["floor_pairs"]]
    floor = float(np.std(all_pairs)) if len(all_pairs) > 1 else 0.0
    levels = [x for s in unit.values() for x in s["level_pairs"]]
    level_bound = max(3.0 * float(np.std(levels)) if len(levels) > 1 else 0.0,
                      LEVEL_RESOLUTION_DB)
    floors = {v: float(np.mean(s["floor_pairs"])) for v, s in unit.items()}
    lifted = {v: unit[v]["doing_nothing"] - floors[v] for v in unit}
    span = max(lifted.values()) if lifted else 0.0
    model, fitted = {}, {}
    for value, side in unit.items():
        if (rel, value) in lost:
            continue
        names = [drawn / p for p in _setting_files(drawn, value)]
        subtractions = [reproduce.subtracted(u, _wave(n, channel))
                        for u, n in zip(side["waves"], names, strict=False)]
        model[value] = float(np.mean([s["residual_db"] for s in subtractions]))
        fitted[value] = float(np.mean([s["level_fit_db"] for s in subtractions]))
    residual = {
        v: (span if v not in model else max(model[v] - floors[v], 0.0)) for v in unit
    }
    rows = [{"value": v, "dir": rel, "residual": [round(r, 3)]} for v, r in residual.items()]
    structured, said = reproduce._structured(rows, floor)
    worst = max(residual, key=residual.get) if residual else None
    return {
        "record": rel,
        "measured_in": "dB",
        "is_null_record": span <= WITHIN_THE_FLOOR_DB,
        "span": round(span, 3),
        "median_abs": float(np.median(list(residual.values()))) if residual else 0.0,
        "worst_abs": residual[worst] if worst is not None else 0.0,
        "worst_at": [rel, worst],
        "structured": structured,
        "structured_why": said,
        "reading_is_a_value_not_a_bound": any(
            model[v] > floors[v] + floor for v in model),
        "settled_by": None,
        "floor": round(floor, 4),
        "rows": rows,
        "properties": [
            {"property": "level", "value": v,
             "unit": f"within {level_bound:.4g} dB of the takes' own level",
             "model": f"{fitted[v]:.4g} dB", "same": abs(fitted[v]) <= level_bound}
            for v in fitted
        ],
        "model_stays_inside_the_floor": all(
            model.get(v, math.inf) <= floors[v] + WITHIN_THE_FLOOR_DB for v in unit),
        "model_largest": max((model.get(v, math.inf) - floors[v] for v in unit), default=0.0),
        "lost": sorted([rel, v] for r, v in lost if r == rel),
    }


def _setting_files(drawn: Path, value: int) -> list[str]:
    listed, _ = takes.listing(drawn)
    items = [(e.get("take", 0), n) for n, e in listed.items()
             if setting_value(e.get("setting")) == value and (drawn / n).is_file()]
    return [n for _, n in sorted(items)]


def bypass_check(dirs: list[Directory], unit: dict[str, dict]) -> dict:
    """Whether the bypass take stands in for the input where the effect does least."""
    weakest = None
    for made in dirs:
        for value, side in unit[made.rel].items():
            floor = float(np.mean(side["floor_pairs"]))
            lifted = side["doing_nothing"] - floor
            if weakest is None or lifted < weakest[0]:
                weakest = (lifted, made.rel, value, side["doing_nothing"], floor)
    if weakest is None:
        return {"passed": False, "why": "no setting was taken twice, so no floor was read"}
    lifted, rel, value, nothing, floor = weakest
    return {
        "dir": rel, "setting": value, "doing_nothing_db": round(nothing, 2),
        "floor_db": round(floor, 2), "within_db": WITHIN_THE_FLOOR_DB,
        "passed": lifted <= WITHIN_THE_FLOOR_DB,
    }


# ---------------------------------------------------------------- readings


def _subparser(parser, name: str):
    import argparse

    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return sub.choices[name]


def _rerooted(value, root: Path, unit_dir: Path, drawn_dir: Path | None):
    if not isinstance(value, str):
        return value
    path = Path(value) if Path(value).is_absolute() else root / value
    path = path.resolve()
    if drawn_dir is not None:
        try:
            return str(drawn_dir / path.relative_to(unit_dir.resolve()))
        except ValueError:
            pass
    return str(path)


def _channel_read(found: dict) -> int | None:
    read = (found.get("channel") or {}).get("read")
    if isinstance(read, list):
        return int(read[0]) if read else None
    return None if read is None else int(read)


def read_with(
    argv: list[str],
    *,
    root: Path,
    unit_dir: Path,
    drawn_dir: Path | None,
    channel: int | None,
    writes,
    cache: Path | None = None,
) -> dict:
    """A stage's own command, parsed from a record's invocation and run on one side.

    Positionals naming the unit's directory are re-rooted to the drawn one, the
    channel is set where one is given and the held block is `writes`. With `cache`,
    the result is kept there under the command, the parsed arguments and the newest
    mtime in the unit's directory, and a later call with all three unchanged reads it
    back instead of running the stage.
    """
    from .cli import build_parser

    parser = build_parser()
    args = parser.parse_args(argv)
    for action in _subparser(parser, argv[0])._actions:
        if not action.option_strings and action.dest != "help":
            setattr(args, action.dest,
                    _rerooted(getattr(args, action.dest), root, unit_dir, drawn_dir))
    if channel is not None:
        if hasattr(args, "channels"):
            args.channels = list(_pair(channel))
        elif hasattr(args, "channel"):
            args.channel = channel
    if hasattr(args, "held"):
        args.held = list(writes)
    kept = None if cache is None else Path(cache) / f"{_cache_key(argv[0], args, unit_dir)}.json"
    if kept is not None and kept.is_file():
        return json.loads(kept.read_text())
    with tempfile.TemporaryDirectory() as scratch:
        args.out = str(Path(scratch) / "reading.json")
        with contextlib.redirect_stdout(io.StringIO()):
            args.func(args)
        out = Path(args.out)
        found = json.loads(out.read_text()) if out.is_file() else {}
    if drawn_dir is not None and channel is not None:
        read = _channel_read(found)
        if found and read is not None and read != channel:
            raise ValueError(
                f"{argv[0]} read the drawn takes on channel {read} and the unit's on "
                f"{channel}; a comparison across two channels compares nothing"
            )
    if kept is not None and found:
        kept.parent.mkdir(parents=True, exist_ok=True)
        kept.write_text(json.dumps(found) + "\n")
    return found


def _cache_key(command: str, args, unit_dir: Path) -> str:
    """The command, its parsed arguments and the newest mtime among the directory's files."""
    said = {k: v for k, v in sorted(vars(args).items()) if k not in ("func", "out")}
    newest = max((f.stat().st_mtime_ns for f in Path(unit_dir).iterdir() if f.is_file()),
                 default=0)
    text = json.dumps([command, said, newest], default=str, sort_keys=True)
    return f"{command}-{hashlib.sha256(text.encode()).hexdigest()[:24]}"


def _quantity(row: dict, reading: Reading):
    value = at(row, reading.quantity)
    if value is MISSING or value is None:
        return None
    if reading.octaves:
        return math.log2(value) if value > 0 else None
    return float(value)


def takes_of(found: dict, reading: Reading, *, setting: int | None = None) -> dict:
    """A reading's output as `{setting: [per-take vector]}`, a refusal as None."""
    rows = found.get(reading.rows) or []
    if reading.rows != "readings":
        return {} if setting is None else {setting: [[_quantity(r, reading) for r in rows]]}
    by: dict[int, list[list]] = {}
    for row in rows:
        if isinstance(row.get("value"), int):
            by.setdefault(row["value"], []).append([_quantity(row, reading)])
    return by


def floor_of(found: dict, reading: Reading) -> float | None:
    """The floor a stage published beside its readings, the widest where it is a list."""
    if reading.floor is None:
        return None
    value = at(found, reading.floor)
    if isinstance(value, list):
        value = max((v for v in value if isinstance(v, (int, float))), default=None)
    return float(value) if isinstance(value, (int, float)) else None


def collected(command: str, found: dict, rel: str, *, setting: int | None = None) -> dict:
    """One run of a stage split into its quantities: `{name: {"takes", "floor"}}`.

    `takes` maps `(directory, setting)` to per-take vectors, as `scored_readings` reads.
    """
    out = {}
    for name, reading in READINGS.items():
        if reading.command != command:
            continue
        out[name] = {
            "takes": {(rel, v): vectors
                      for v, vectors in takes_of(found, reading, setting=setting).items()},
            "floor": floor_of(found, reading),
        }
    return out


def _medians(takes_by: dict) -> dict:
    out = {}
    for key, vectors in takes_by.items():
        width = max(len(v) for v in vectors)
        out[key] = [
            (float(np.median(kept)) if (kept := [v[i] for v in vectors
                                                  if i < len(v) and v[i] is not None])
             else None)
            for i in range(width)
        ]
    return out


def repeat_floor(unit: dict) -> float:
    """The widest spread one setting's takes showed, element by element, as a deviation."""
    spreads = [0.0]
    for vectors in unit.values():
        for i in range(max(len(v) for v in vectors)):
            kept = [v[i] for v in vectors if i < len(v) and v[i] is not None]
            if len(kept) > 1:
                spreads.append(float(np.std(kept)))
    return max(spreads)


def scored_readings(
    name: str,
    reading: Reading,
    unit: dict,
    drawn: dict,
    *,
    floor: float,
    span: float | None = None,
    properties_from: list | None = None,
    unrenderable=(),
) -> dict:
    """One stage's readings in the shape `reproduce.gates` scores, per the `readings` table.

    `unit` and `drawn` map `(directory, setting)` to per-take vectors. A setting only
    one side refused costs the whole span; one both refused is not a row.
    """
    u, d = _medians(unit), _medians(drawn)
    values = [x for vec in u.values() for x in vec if x is not None]
    own = (max(values) - min(values)) if values else 0.0
    span = own if (span is None or own > 0) else span
    rows, refused = [], []
    for key in sorted(set(u) | set(d) | set(unrenderable)):
        if key in unrenderable:
            rows.append({"dir": key[0], "value": key[1], "residual": [span]})
            continue
        uv, dv = u.get(key, []), d.get(key, [])
        residual = []
        for i in range(max(len(uv), len(dv))):
            a = uv[i] if i < len(uv) else None
            b = dv[i] if i < len(dv) else None
            if a is None and b is None:
                continue
            residual.append(span if a is None or b is None else b - a)
            if a is not None and b is None and list(key) not in refused:
                refused.append(list(key))
        if residual:
            rows.append({"dir": key[0], "value": key[1], "residual": residual})
    magnitudes = [(abs(x), [r["dir"], r["value"]]) for r in rows for x in r["residual"]]
    worst = max(magnitudes, key=lambda m: m[0]) if magnitudes else (0.0, None)
    structured, said = reproduce._structured(rows, floor, unit=reading.measured_in)
    properties = _directions(u, d, floor) or list(properties_from or [])
    null = span <= 0 or not rows
    return {
        "record": name,
        "measured_in": reading.measured_in,
        "is_null_record": null,
        "span": span,
        "median_abs": float(np.median([m[0] for m in magnitudes])) if magnitudes else 0.0,
        "worst_abs": worst[0],
        "worst_at": worst[1],
        "structured": structured,
        "structured_why": said,
        "reading_is_a_value_not_a_bound": True,
        "settled_by": None,
        "floor": floor,
        "rows": rows,
        "properties": properties,
        "drawn_only_refused": refused,
        "model_stays_inside_the_floor": all(m[0] <= floor for m in magnitudes),
        "model_largest": worst[0],
    }


def _directions(u: dict, d: dict, floor: float) -> list[dict]:
    """Neighbouring settings the unit moved between by more than the floor, and which way."""
    found = []
    for rel in sorted({k[0] for k in u}):
        keys = sorted(k for k in u if k[0] == rel)
        for a, b in zip(keys, keys[1:], strict=False):
            ua, ub = _middle(u.get(a)), _middle(u.get(b))
            da, db = _middle(d.get(a)), _middle(d.get(b))
            if None in (ua, ub) or abs(ub - ua) <= floor:
                continue
            same = da is not None and db is not None and np.sign(db - da) == np.sign(ub - ua)
            found.append({
                "property": "direction", "value": [rel, a[1], b[1]],
                "unit": "rises" if ub > ua else "falls",
                "model": "refused" if None in (da, db) else "rises" if db > da else "falls",
                "same": bool(same),
            })
    return found


def _middle(vector) -> float | None:
    kept = [x for x in (vector or []) if x is not None]
    return float(np.median(kept)) if kept else None


# ---------------------------------------------------------------- one phase


def separated(scored: list[dict]) -> bool:
    """Whether a comparison caught the identity out, so it could have caught anything."""
    if any(s.get("drawn_only_refused") for s in scored):
        return True
    carrying = [s for s in scored if not s["is_null_record"] and s["span"] > 0]
    if not carrying:
        return False
    found = reproduce.gates(scored, ranking=[])
    return not found["gross"]["passed"] or not found["breakdown"]["passed"]


def _consumers(root: Path, found: dict, made: Directory, type_: str) -> list[tuple[str, dict]]:
    """The published records of this type in `READINGS` stages that read a directory."""
    out = []
    for hit in made.entry.get("consumed_by") or []:
        path = root / hit["record"]
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        argv = (data.get("record") or {}).get("invocation") or []
        if not argv or argv[0] not in COMMANDS:
            continue
        said = data.get("type") or next(
            (argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--type"), None)
        if said is not None and ledger.norm_bytes(said) == type_:
            out.append((hit["record"], data))
    return sorted(out, key=lambda item: item[0])


class _Phase:
    """One comparison over a set of directories, shared by every model it is run on."""

    def __init__(self, root, found, type_, dirs, mode, inherited=None):
        self.root, self.found, self.type, self.dirs, self.mode = root, found, type_, dirs, mode
        self.inherited = inherited or {}
        self.channel: dict[str, int] = {}
        self.unit: dict = {}
        self.stages: dict[str, dict] = {}
        self.runs: dict[str, list] = {}
        if mode == "waves":
            for made in dirs:
                self.channel[made.rel] = loudest(made)
                self.unit[made.rel] = _unit_waves(made, self.channel[made.rel])
        else:
            self._read_units()

    def _read_units(self) -> None:
        self.runs: dict[str, list] = {}
        for made in self.dirs:
            writes = writes_of(made.entry.get("held"))
            for _, data in _consumers(self.root, self.found, made, self.type):
                argv = data["record"]["invocation"]
                command = argv[0]
                read = read_with(argv, root=self.root, unit_dir=made.where, drawn_dir=None,
                                 channel=self.channel.get(made.rel), writes=writes,
                                 cache=self.root / STAGE_READINGS)
                if not read:
                    continue
                self.channel.setdefault(made.rel, _channel_read(read))
                setting = self._decay_setting(made, argv) if command == "decay" else None
                if command == "decay" and setting is None:
                    continue
                for name, part in collected(command, read, made.rel, setting=setting).items():
                    stage = self.stages.setdefault(name, {"unit": {}, "floors": []})
                    for key, vectors in part["takes"].items():
                        if key[1] in made.by_setting:
                            stage["unit"].setdefault(key, []).extend(vectors)
                    if part["floor"] is not None:
                        stage["floors"].append(part["floor"])
                self.runs.setdefault(command, []).append((made, argv, setting))

    def _decay_setting(self, made: Directory, argv: list[str]) -> int | None:
        listed, _ = takes.listing(made.where)
        wet = Path(argv[2]).name
        return setting_value(listed.get(wet, {}).get("setting"))

    def compared(self) -> list[tuple[str, int]]:
        if self.mode == "waves":
            return [(rel, v) for rel, per in self.unit.items() for v in per]
        keys = {k for stage in self.stages.values() for k in stage["unit"]}
        return sorted(keys)

    def rows(self) -> list[dict]:
        by_rel = {m.rel: m for m in self.dirs}
        out = []
        for rel, value in self.compared():
            made = by_rel[rel]
            stimulus = str(made.by_setting[value][0]["stimulus"])
            out.append({"dir": rel, "setting": value, "input_from": made.input_from[stimulus],
                        "channel": int(self.channel[rel])})
        return out

    def scored(self, cand: Candidate) -> tuple[list[dict], list[str]]:
        """The model's scored records, and the directories it was drawn into."""
        drawn = {}
        for made in self.dirs:
            if made.rel not in self.channel or self.channel[made.rel] is None:
                continue
            drawn[made.rel] = render_directory(
                cand, self.root, made.rel, channels=_pair(self.channel[made.rel]),
                found=self.found, made=made)
        lost = set().union(*[_unrenderable(p, rel) for rel, p in drawn.items()])
        shown = [p.relative_to(self.root).as_posix() for p in drawn.values()]
        if self.mode == "waves":
            return [
                _scored_waves(rel, self.unit[rel], drawn[rel], self.channel[rel], lost)
                for rel in self.unit
            ], shown
        mine: dict[str, dict] = {}
        for command, runs in self.runs.items():
            for made, argv, setting in runs:
                read = read_with(argv, root=self.root, unit_dir=made.where,
                                 drawn_dir=drawn[made.rel], channel=self.channel[made.rel],
                                 writes=writes_of(made.entry.get("held")))
                for name, part in collected(command, read, made.rel, setting=setting).items():
                    for key, vectors in part["takes"].items():
                        mine.setdefault(name, {}).setdefault(key, []).extend(vectors)
        out = []
        for name, stage in sorted(self.stages.items()):
            ours = {k: v for k, v in mine.get(name, {}).items() if k in stage["unit"]}
            floor = max(stage["floors"]) if stage["floors"] else repeat_floor(stage["unit"])
            before = self.inherited.get(name, {})
            out.append(scored_readings(
                name, READINGS[name], stage["unit"], ours, floor=floor,
                span=before.get("span"), properties_from=before.get("properties"),
                unrenderable={k for k in lost if k in stage["unit"]}))
        return out, shown


def _ranked(cand: Candidate, scored: list[dict]) -> dict:
    carrying = [s for s in scored if s["span"] > 0 and not s["is_null_record"]]
    shares = [s["median_abs"] / s["span"] for s in carrying]
    return {
        "candidate": cand.id,
        "model": cand.shown_as,
        "mean_share_of_span": round(float(np.mean(shares)), 4) if shares else 1.0,
        "worst_share_of_span": round(max(shares), 4) if shares else 1.0,
        "leaning_records": sum(
            1 for s in carrying if s["structured"] and s["reading_is_a_value_not_a_bound"]),
        "contradicts_a_property": any(
            not p["same"] for s in scored for p in s.get("properties", [])),
    }


def _needs(gate: str, type_: str, detail: str) -> str:
    said = {
        "no_input_take": f"takes of {type_} with a bypass take of the same stimulus, "
        "beside them or in a directory the ledger matches on every stimulus field",
        "no_held_out_setting": f"a setting of {type_} that no value of any candidate "
        "was fitted on, declared or through a claim's records",
        "control_could_not_fail": f"a comparison of {type_} the identity model can fail: "
        "a setting that lifts the effect clear of the takes' own floor, read by a stage "
        "that publishes a quantity for this type from these directories",
        "unrenderable": f"a candidate for {type_} whose loops keep a sample of delay at "
        "every compared setting",
    }.get(gate, f"a candidate for {type_} that clears the {gate} gate, or a measurement "
                "that separates the ones standing")
    return f"{said}. {detail}".strip()


def _run(phase: _Phase, candidates: list[Candidate], comparison: dict, check=None):
    """The block a phase writes, the winner if there was one, and the gate it stopped at."""
    compared = phase.compared()
    control_scored, _ = phase.scored(identity())
    control = {"model": "identity", "separated": separated(control_scored),
               "gates": reproduce.gates(control_scored, ranking=[])}
    if check is not None:
        control["bypass_check"] = check
    block = {"verdict": "stopped", "gates": {}, "control": control,
             "compared": phase.rows(), "held_out_settings": [],
             "excluded_from_ranking": [], "comparison": comparison}
    per, drawn_on, ranking, winner = {}, [], [], None
    for cand in candidates:
        fitted = fitted_on(cand.raw, root=phase.root, ledger=phase.found)
        held = [[rel, v] for rel, v in compared if (rel, v) not in fitted]
        if not held:
            block["excluded_from_ranking"].append(cand.id)
            continue
        per[cand.id] = (cand, held, *phase.scored(cand))
        drawn_on += per[cand.id][3]
        ranking.append(_ranked(cand, per[cand.id][2]))
    ranking.sort(key=lambda r: r["mean_share_of_span"])
    if not candidates:
        return block, None, None, drawn_on
    if not compared:
        return block, None, "no_input_take", drawn_on
    if not ranking:
        return block, None, "no_held_out_setting", drawn_on
    winner, held, scored, _ = per[ranking[0]["candidate"]]
    block["_scored"] = scored
    block["held_out_settings"] = held
    block["gates"] = reproduce.gates(scored, ranking=ranking)
    lost = [s for s in scored if s.get("lost")]
    if lost and all(len(s["lost"]) == len(s["rows"]) for s in lost):
        return block, winner, "unrenderable", drawn_on
    if not control["separated"]:
        return block, winner, "control_could_not_fail", drawn_on
    failed = next((g for g in GATES if not block["gates"][g]["passed"]), None)
    if failed is None:
        block["verdict"] = "passed"
    return block, winner, failed, drawn_on


def _inherited(phase: _Phase, block: dict) -> dict:
    """What a later phase of one setting takes from this one: each stage's span and properties."""
    if phase.mode != "readings":
        return {}
    return {s["record"]: {"span": s["span"], "properties": s["properties"]}
            for s in block.get("_scored", [])}


def _split(dirs: list[Directory]) -> tuple[list[Directory], list[Directory]]:
    """The stimulus class holding the most settings, and up to two directories of any other.

    Counted in settings rather than directories: stage 9 takes one setting of a type,
    and a class holding only that is not the one the type was swept under.
    """
    counts: dict[str, int] = {}
    for made in dirs:
        for cls in made.classes:
            counts[cls] = counts.get(cls, 0) + len(made.by_setting)
    if not counts:
        return dirs, []
    first = min(counts, key=lambda c: (-counts[c], c))
    return ([m for m in dirs if first in m.classes],
            [m for m in dirs if first not in m.classes][:2])


def _static(root: Path, unit: str) -> set[str]:
    path = root / "data" / "units" / unit / "efx-sort" / "by-repeatability.json"
    return set(json.loads(path.read_text()).get("static") or [])


def _delay_rows(model: dict) -> list[str]:
    return sorted({
        spec["byte"] for node in model.get("nodes", []) if node.get("kind") == "delay"
        for spec in _values(node.get("time_ms")) if "byte" in spec
    })


def stage(root: str | Path, unit: str, type_: str, *, class_name: str | None = None) -> dict:
    """Stages 7-9 for one type, or the identity control alone when no class is named."""
    root = Path(root)
    type_ = type_of(type_)
    found = _ledger(root, type_)
    candidates = members(root, class_name) if class_name else []
    made = [directory(root, found, rel) for rel in directories(found, type_)]
    made = [m for m in made if m.by_setting]
    result: dict = {"type": type_}
    tail = {
        "invocation": {"unit": unit, "type": type_, "class": class_name},
        "rendered_with": {"numpy": np.__version__, "scipy": scipy.__version__},
    }
    if not made:
        result["stopped"] = {"at": 7, "gate": "no_input_take",
                             "needs": _needs("no_input_take", type_, "")}
        return {**result, "equivalent_under_this_test": [], **tail}

    first, rest = _split(made)
    mode, chosen_by, check = "readings", "by_repeatability_class", None
    if type_ in _static(root, unit):
        mode = "waves"
        phase = _Phase(root, found, type_, first, mode)
        check = bypass_check(first, phase.unit)
        if not check["passed"]:
            mode, chosen_by = "readings", "bypass_check_failed"
    if mode == "readings":
        phase = _Phase(root, found, type_, first, mode)
    comparison = {"used": mode, "chosen_by": chosen_by}
    block, winner, gate, drawn_on = _run(phase, candidates, comparison, check)
    inherited = _inherited(phase, block)
    block.pop("_scored", None)
    if not candidates:
        return {**result, "control": block["control"], "compared": block["compared"],
                "comparison": comparison, **tail}

    shown = winner or candidates[0]
    result["p0"] = {"model": shown.shown_as, "model_sha256": shown.sha256,
                    "rendered_on": sorted({p for p in drawn_on if f"/{shown.id}/" in f"/{p}"})}
    result["p1"] = block
    if gate is not None:
        result["stopped"] = {"at": 8, "gate": gate, "needs": _needs(gate, type_, "")}
    elif not rest:
        used = sorted(first[0].classes)
        result["stopped"] = {"at": 9, "gate": "no_input_take", "needs": _needs(
            "no_input_take", type_,
            f"Stage 9 needs directories under a stimulus class other than {used}.")}
    else:
        later = _Phase(root, found, type_, rest, mode, inherited)
        p2, _, gate2, drawn2 = _run(later, candidates, comparison)
        p2.pop("_scored", None)
        result["p2"] = p2
        result["p0"]["rendered_on"] = sorted(set(result["p0"]["rendered_on"]) | {
            p for p in drawn2 if f"/{shown.id}/" in f"/{p}"})
        if gate2 is not None:
            result["stopped"] = {"at": 9, "gate": gate2, "needs": _needs(gate2, type_, "")}
    equivalent = _delay_rows(shown.raw) if mode == "waves" else []
    return {**result, "equivalent_under_this_test": equivalent, **tail}


__all__ = [
    "IDENTITY",
    "READINGS",
    "LedgerBehind",
    "Reading",
    "bytes_now",
    "candidate",
    "fitted_on",
    "power_on",
    "render_directory",
    "scored_readings",
    "separated",
    "stage",
    "writes_of",
]
