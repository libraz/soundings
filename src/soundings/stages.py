"""Carrying one effect type through stages 7-9: draw its candidates, compare, decide.

A candidate `graph` model is drawn on the unit's own bypass takes -- the effect's
input -- and what it drew is compared with the unit's takes in one of two ways,
chosen mechanically per stimulus class. `waves` subtracts takes, for a type the
repeatability sort calls static under a stimulus whose takes repeat; `readings`
reads both sides with the stages that published the type's records, and a directory
none of them read is read by the type's own invocation of that stage re-aimed at it.
Either comparison is put into the shape `reproduce.gates` scores, once per unit a
record is measured in, and the gates are not changed.

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

from . import efxbands, ledger, reproduce, sway, takes
from .inferences import EFFECT_BLOCK
from .render import graph
from .stimuli import Stimulus

RENDERED_ROOT = Path(".cache/rendered")

SWEPT = re.compile(r"^(?P<prefix>(?:.*\D)?)(?P<value>\d{1,3})$")
"""A take of one setting: the byte, behind whatever prefix the run named its sweep with."""

BYPASS = re.compile(r"^(out|bypassed|bypassed-\d+)$")
"""A take with the effect routed out of the path, which is the effect's input."""

PART_ROUTING = re.compile(r"^40 4[0-9A-F] 22$")
"""GS: a part's insertion-effect switch. A directory sweeping it holds, at 0, the part
routed past the effect -- the input of every directory taken with the same stimulus."""

SILENCE = re.compile(r"^silence(-\d+)?$")
REFERENCE = re.compile(r"^flat(-\d+)?$")
"""The repeats of the setting a run held flat, which a band reading is taken against."""

ONE_ADDRESS = re.compile(rf"^{EFFECT_BLOCK} [0-9A-F]{{2}}$")
"""A directory a drawing can be made for sweeps one address of the effect's own block."""

REPEATS_BELOW_DB = -20.0
"""A stimulus class whose same-setting subtraction lands above this is compared by readings.

Measured: struck notes subtract to -28 to -55 dB, applause and noise to -0.3 and -0.1.
"""

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
    or a frequency is compared. `admitted_if` names row keys that must not be null for
    the quantity to be read at all, where the record's own rule says so. `null_is` is
    the value a null stands for where the record defines one (nothing cleared the
    floor); any other null on the unit's side is the record refusing that setting.
    `room` names the row key saying how far a take stood above the chain's silence; a
    row within `WITHIN_THE_FLOOR_DB` of it is the room and not a reading. `tracked`
    names a row key counting the frames that held signal and the count below which the
    take is likewise the room. `step` names the record key holding the reading's own
    resolution, which the floor is never finer than.
    """

    command: str
    entry: str
    rows: str
    quantity: str
    floor: str | None
    measured_in: str
    octaves: bool = False
    admitted_if: tuple[str, ...] = ()
    null_is: float | None = None
    room: str | None = None
    tracked: tuple[str, int] | None = None
    step: str | None = None


_READINGS = (
    Reading("efx-rate", "efxrate.read_directory", "readings", "rate_hz", None, "octaves",
            octaves=True),
    Reading("efx-excursion", "efxexcursion.read_directory", "readings", "excursion_ms",
            "floor.ms", "ms"),
    Reading("efx-excursion", "efxexcursion.read_directory", "readings", "off_the_phase_ms",
            "floor_off_the_phase.ms", "ms"),
    Reading("efx-time", "efxtime.read_directory", "readings", "ms", "floor_ms", "ms",
            step="quefrency_step_ms"),
    # Null is no band outside its floor, which is an answer and not a refusal.
    Reading("efx-bands", "efxbands.read_directory", "readings", "largest_db",
            "reference.floor_db", "dB", null_is=0.0, room="above_the_silence_db"),
    # A profile that never falls to half on both sides has no measured width, so the
    # band it is largest in is wherever the scatter put it.
    Reading("efx-bands", "efxbands.read_directory", "readings", "largest_at_hz",
            "band_width_octaves", "octaves", octaves=True,
            admitted_if=("half_below_hz", "half_above_hz"), room="above_the_silence_db"),
    # Where the unit's profile has no feature there is nothing to place.
    # The record holds a position against the band width, having no measured floor.
    Reading("efx-bands", "efxbands.read_directory", "readings", "fitted_at_hz",
            "band_width_octaves", "octaves", octaves=True, room="above_the_silence_db"),
    # Null is no swing above the reader's least depth, or no rate to fold it at.
    Reading("efx-sway", "efxsway.sweep", "readings", "level_in_db.depth", None, "dB",
            null_is=0.0, tracked=("tracked_frames", sway.LEAST_TRACKED_FRAMES)),
    Reading("efx-sway", "efxsway.sweep", "readings", "balance.depth", None, "dB",
            null_is=0.0, tracked=("tracked_frames", sway.LEAST_TRACKED_FRAMES)),
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


def role(setting) -> str | None:
    """What a take is to a comparison: `bypass`, `silence`, `reference`, `swept` or nothing."""
    text = str(setting)
    for name, pattern in (("bypass", BYPASS), ("silence", SILENCE), ("reference", REFERENCE)):
        if pattern.match(text):
            return name
    found = SWEPT.match(text)
    return "swept" if found and int(found.group("value")) <= 127 else None


def swept_bytes(settings: dict) -> dict:
    """Each swept take's byte, or nothing where the takes sweep under more than one prefix.

    Two prefixes are two things swept in one directory, and which address a take's
    byte was written to is then not said by the take.
    """
    found, prefixes = {}, set()
    for key, setting in settings.items():
        if role(setting) == "swept":
            match = SWEPT.match(str(setting))
            prefixes.add(match.group("prefix"))
            found[key] = int(match.group("value"))
    return found if len(prefixes) == 1 else {}


def _prefix(settings: dict) -> str | None:
    kept = {SWEPT.match(str(s)).group("prefix") for s in settings.values() if role(s) == "swept"}
    return kept.pop() if len(kept) == 1 else None


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
    """The ledger's directories of a type sweeping one address of the effect's block."""
    return sorted(
        rel for rel, entry in found["directories"].items()
        if entry.get("type") == type_
        and ONE_ADDRESS.match(entry.get("address") or "")
        and swept_bytes({s: s for s in entry.get("settings") or {}})
    )


def _stimulus_key(stimulus: dict) -> str | None:
    # A stimulus records its volume only where it is not the one always sent.
    stimulus = {"volume": Stimulus.volume, **stimulus}
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
    items: list[dict] = field(default_factory=list)
    bypass: dict[str, list[Path]] = field(default_factory=dict)
    input_from: dict[str, str] = field(default_factory=dict)

    def of_role(self, name: str) -> list[dict]:
        return [e for e in self.items if role(e["setting"]) == name]

    @property
    def prefix(self) -> str | None:
        return _prefix({e["file"]: e["setting"] for e in self.items})

    @property
    def classes(self) -> set[str]:
        return {s.get("class") for s in self.entry.get("stimuli") or [] if s.get("class")}


def _takes(root: Path, found: dict, rel: str) -> list[dict]:
    """A directory's takes on disk, each with the setting the ledger's resolution gives it.

    The stimulus is the manifest's, or the directory's one stimulus where it names one.
    """
    where = root / ledger.TAKES_ROOT / rel
    entry = found["directories"][rel]
    settings = ledger.take_settings(root, rel, entry)
    listed, _ = takes.listing(where)
    named = [s.get("name") for s in entry.get("stimuli") or []]
    only = named[0] if len(named) == 1 else None
    routing = bool(PART_ROUTING.match(entry.get("address") or ""))
    out = [
        {**listed.get(name, {}), "file": name,
         "setting": "bypassed" if routing and str(setting) == "0" else setting,
         "stimulus": listed.get(name, {}).get("stimulus", only)}
        for name, setting in settings.items() if (where / name).is_file()
    ]
    return sorted(out, key=lambda e: (e.get("take", 0), e["file"]))


def _bypasses(items: list[dict], where: Path) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    for item in items:
        if role(item["setting"]) == "bypass":
            found.setdefault(str(item["stimulus"]), []).append(where / item["file"])
    return found


def _taken_at(paths: list[Path]) -> float:
    """When takes were recorded, as their files' median modification time."""
    return float(np.median([p.stat().st_mtime for p in paths]))


def directory(root: Path, found: dict, rel: str) -> Directory:
    """One ledger directory, its setting takes grouped, its input found or not."""
    where = root / ledger.TAKES_ROOT / rel
    entry = found["directories"][rel]
    items = _takes(root, found, rel)
    bytes_ = swept_bytes({e["file"]: e["setting"] for e in items})
    if not ONE_ADDRESS.match(entry.get("address") or ""):
        bytes_ = {}
    by_setting: dict[int, list[dict]] = {}
    for item in items:
        if item["file"] in bytes_:
            by_setting.setdefault(bytes_[item["file"]], []).append(item)
    made = Directory(rel, where, entry.get("address"), entry, dict(sorted(by_setting.items())),
                     items)
    own = _bypasses(items, where)
    stimuli = {str(e["stimulus"]) for items in by_setting.values() for e in items}
    for name in sorted(stimuli):
        if own.get(name):
            made.bypass[name], made.input_from[name] = own[name], rel
            continue
        mine = next((s for s in entry.get("stimuli") or [] if s.get("name") == name), None)
        key = None if mine is None else _stimulus_key(mine)
        if key is None:
            continue
        taken_at = _taken_at([where / e["file"] for e in items])
        nearest = None
        for other, theirs in sorted(found["directories"].items()):
            match = next(
                (s for s in theirs.get("stimuli") or [] if _stimulus_key(s) == key), None)
            if other == rel or match is None:
                continue
            there = _bypasses(_takes(root, found, other),
                              root / ledger.TAKES_ROOT / other).get(match["name"])
            if there:
                apart = abs(_taken_at(there) - taken_at)
                if nearest is None or apart < nearest[0]:
                    nearest = (apart, there, other)
        if nearest is not None:
            made.bypass[name], made.input_from[name] = nearest[1], nearest[2]
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
    return swept_bytes({0: value}).get(0, str(value))


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
    the same channels from the same input is left as it is.
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
        if (kept.get("model_sha256") == cand.sha256 and kept.get("channels") == list(channels)
                and kept.get("input_from") == made.input_from):
            return out
        shutil.rmtree(out)
    elif out.exists():
        # What a run left behind when it stopped before writing the manifest.
        shutil.rmtree(out)
    out.mkdir(parents=True)

    entry = made.entry
    on = power_on(root, found["unit"], entry["type"])
    writes = writes_of(entry.get("held"))
    kept_entries, unrenderable = [], []
    for item in made.items:
        if role(item["setting"]) in ("bypass", "silence", "reference"):
            shutil.copy2(made.where / item["file"], out / item["file"])
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


CUT_ROOT = Path(".cache/cut")
"""Where the unit's own takes are kept band-limited to `graph.CUT_HZ`, for `readings`.
The drawn side is free of energy above it by construction; only the
unit's own recording needs cutting to compare the two on the same band."""


def cut_directory(root: Path, rel: str) -> Path:
    """A band-limited copy of `rel`'s takes, under `.cache/cut/<rel>/`.

    Same file names as the source, its manifest copied unchanged, filtered with the
    same FIR `_cut` uses. Rebuilt whenever a source file is newer than the copy.
    Built beside `out` and moved into place whole, so a build stopped part-way
    leaves no copy that looks finished.
    """
    root = Path(root)
    where = root / ledger.TAKES_ROOT / rel
    out = root / CUT_ROOT / rel
    sources = sorted(where.glob("*.wav"))
    manifest = where / "takes-manifest.json"
    newest = max(
        [p.stat().st_mtime_ns for p in sources]
        + ([manifest.stat().st_mtime_ns] if manifest.is_file() else []),
        default=0,
    )
    if out.is_dir():
        built = max((p.stat().st_mtime_ns for p in out.glob("*.wav")), default=-1)
        if built >= newest:
            return out
        shutil.rmtree(out)
    building = out.with_name(out.name + ".building")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    for path in sources:
        samples, rate = takes.read(path)
        cut = np.stack([_cut(samples[:, c], rate) for c in range(samples.shape[1])], axis=1)
        takes.write(building / path.name, cut, rate)
    if manifest.is_file():
        shutil.copy2(manifest, building / "takes-manifest.json")
    building.rename(out)
    return out


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


def _scored_waves(made: Directory, unit: dict, drawn: Path, channel: int, lost: set) -> dict:
    """One directory in the shape `reproduce.gates` scores, per the `waves` table."""
    rel = made.rel
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
        names = [drawn / e["file"] for e in made.by_setting[value]
                 if (drawn / e["file"]).is_file()]
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


def _band_records(root: Path, unit: str, type_: str) -> list[tuple[str, dict]]:
    """The unit's published `efx-bands` records of one type."""
    out = []
    for path in sorted((root / "data" / "units" / unit / "efx-bands").rglob("*.json")):
        data = json.loads(path.read_text())
        argv = (data.get("record") or {}).get("invocation") or []
        said = data.get("type") or next(
            (argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--type"), None)
        if said is not None and ledger.norm_bytes(said) == type_:
            out.append((path.relative_to(root).as_posix(), data))
    return out


def inert_settings(records: list[tuple[str, dict]], address: str, state: dict,
                   on: dict[str, int]) -> dict[int, str]:
    """Settings a band record at `address` reads inside its own floor, and which record did.

    Only a record made in `state` -- the same bytes at every other parameter address --
    speaks for it: a setting inert with one stage switched off is not inert with it on.
    """
    reading = READINGS["efx-bands:largest_db"]
    found: dict[int, str] = {}
    for rel, data in records:
        argv = data["record"]["invocation"]
        said = data.get("address") or next(
            (argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--slot"), None)
        if ledger.norm_addr(said) != address:
            continue
        theirs = bytes_now(on, writes_of(ledger.record_held(argv)), None, None)
        if any(theirs[a] != state[a] for a in on if a != address):
            continue
        floor = floor_of(data, reading)
        if floor is None:
            continue
        for row in data.get("readings") or []:
            value, largest = row.get("value"), row.get("largest_db")
            if isinstance(value, int) and isinstance(largest, (int, float)) and (
                    abs(largest) <= floor):
                found.setdefault(value, rel)
    return found


def bypass_check(root: Path, unit: str, type_: str, dirs: list[Directory],
                 waves: dict[str, dict]) -> dict:
    """Whether the bypass take stands in for the input at settings known to do nothing.

    Asked only at a compared setting a published band record of the type, at the same
    address and in the same held state, reads inside its own floor; there the unit's
    take less the bypass take has to land within `WITHIN_THE_FLOOR_DB` of the setting's
    floor. Which settings do nothing comes from that record, never from this subtraction.
    """
    records = _band_records(root, unit, type_)
    on = power_on(root, unit, type_)
    asked = []
    for made in dirs:
        state = bytes_now(on, writes_of(made.entry.get("held")), None, None)
        inert = inert_settings(records, made.address, state, on)
        for value, side in waves[made.rel].items():
            if value not in inert:
                continue
            floor = float(np.mean(side["floor_pairs"]))
            lifted = side["doing_nothing"] - floor
            asked.append({
                "dir": made.rel, "setting": value, "record": inert[value],
                "doing_nothing_db": round(side["doing_nothing"], 2),
                "floor_db": round(floor, 2), "lifted_db": round(lifted, 2),
                "result": "passed" if lifted <= WITHIN_THE_FLOOR_DB else "failed",
            })
    if not asked:
        return {"result": "not_asked", "why": "no compared setting is one a published band "
                "record of this type, address and held state reads inside its own floor"}
    failed = any(a["result"] == "failed" for a in asked)
    return {"result": "failed" if failed else "passed", "within_db": WITHIN_THE_FLOOR_DB,
            "asked": asked}


def _gates(scored: list[dict], ranking: list[dict]) -> dict:
    """`reproduce.gates` once per unit a record is measured in; a gate passes where all do."""
    by: dict[str, list[dict]] = {}
    for item in scored:
        by.setdefault(item.get("measured_in", "dB"), []).append(item)
    if len(by) <= 1:
        return reproduce.gates(scored, ranking=ranking)
    per = {unit: reproduce.gates(items, ranking=ranking) for unit, items in sorted(by.items())}
    combined = {g: {"passed": all(p[g]["passed"] for p in per.values())} for g in GATES}
    # A share of a span has no unit, so the worst of them is one number; a span is not.
    combined["gross"]["residual_over_span"] = max(
        p["gross"]["residual_over_span"] for p in per.values())
    combined["gross"]["span"] = None
    return {"measured_in": sorted(per), "by_measured_in": per, **combined}


def _each_unit(gates: dict) -> list[dict]:
    return list(gates["by_measured_in"].values()) if "by_measured_in" in gates else [gates]


# ---------------------------------------------------------------- readings


def _subparser(parser, name: str):
    import argparse

    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return sub.choices[name]


def _rerooted(value, root: Path, unit_dir: Path, drawn_dir: Path | None,
             cut_dir: Path | None = None):
    if not isinstance(value, str):
        return value
    path = Path(value) if Path(value).is_absolute() else root / value
    path = path.resolve()
    for base in (drawn_dir, cut_dir):
        if base is None:
            continue
        try:
            return str(base / path.relative_to(unit_dir.resolve()))
        except ValueError:
            continue
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
    cut_dir: Path | None = None,
) -> dict:
    """A stage's own command, parsed from a record's invocation and run on one side.

    Positionals naming the unit's directory are re-rooted to the drawn one, or to
    `cut_dir` (the unit's own band-limited copy) where no drawn one is
    given; the channel is set where one is given and the held block is `writes`. With
    `cache`, the result is kept there under the command, the parsed arguments and the
    newest mtime in `cut_dir` or `unit_dir`, and a later call with all three unchanged
    reads it back instead of running the stage.
    """
    from .cli import build_parser

    parser = build_parser()
    args = parser.parse_args(argv)
    for action in _subparser(parser, argv[0])._actions:
        if not action.option_strings and action.dest != "help":
            setattr(args, action.dest,
                    _rerooted(getattr(args, action.dest), root, unit_dir, drawn_dir, cut_dir))
    if channel is not None:
        if hasattr(args, "channels"):
            args.channels = list(_pair(channel))
        elif hasattr(args, "channel"):
            args.channel = channel
    if hasattr(args, "held"):
        args.held = list(writes)
    keyed_on = cut_dir if cut_dir is not None else unit_dir
    kept = None if cache is None else Path(cache) / f"{_cache_key(argv[0], args, keyed_on)}.json"
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
    if any(at(row, key) in (None, MISSING) for key in reading.admitted_if):
        return None
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
        if reading.room is not None and _in_the_room(row.get(reading.room)):
            continue
        if reading.tracked is not None and not _tracked(row, *reading.tracked):
            continue
        if isinstance(row.get("value"), int):
            by.setdefault(row["value"], []).append([_quantity(row, reading)])
    return by


def _in_the_room(above_the_silence) -> bool:
    return isinstance(above_the_silence, (int, float)) and above_the_silence <= WITHIN_THE_FLOOR_DB


def _tracked(row: dict, key: str, least: int) -> bool:
    count = at(row, key)
    return isinstance(count, (int, float)) and count >= least


def floor_of(found: dict, reading: Reading) -> float | None:
    """The floor a stage published beside its readings, the widest where it is a list.

    Never finer than the reading's own step: two settings a step apart are the nearest
    the reading can place them, so a spread of one step is its resolution.
    """
    if reading.floor is None:
        return None
    value = at(found, reading.floor)
    if isinstance(value, list):
        value = max((v for v in value if isinstance(v, (int, float))), default=None)
    if not isinstance(value, (int, float)):
        return None
    step = at(found, reading.step) if reading.step else None
    return max(float(value), float(step)) if isinstance(step, (int, float)) else float(value)


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


def _maps_of(node, address: str, out: list[dict]) -> list[dict]:
    if isinstance(node, dict):
        if node.get("byte") == address and isinstance(node.get("map"), dict):
            out.append(node["map"])
        for value in node.values():
            _maps_of(value, address, out)
    elif isinstance(node, list):
        for value in node:
            _maps_of(value, address, out)
    return out


def one_entry_octaves(raw: dict, address: str | None, values) -> float:
    """What one entry of a model's table for `address` is worth in octaves, at `values`.

    The second floor `reproduce.score_against_rates` judges a rate's lean against: a
    residual smaller than one entry is one no candidate built on the table could differ
    over. Only a table with an entry per setting has one; points, or two tables that
    disagree about the byte, return 0 and leave the run's own floor to govern alone.
    """
    maps = _maps_of(raw, address, []) if address else []
    if not maps or any(m.get("kind") != "table" for m in maps):
        return 0.0
    entries = maps[0].get("entries") or []
    if any(m.get("entries") != entries for m in maps[1:]):
        return 0.0
    last = len(entries) - 1
    steps = [
        float(np.log2(entries[v] / entries[v - 1]))
        for v in (int(v) for v in values)
        if 0 < v <= last and entries[v] > entries[v - 1] > 0
    ]
    return float(np.median(steps)) if steps else 0.0


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

    `unit` and `drawn` map `(directory, setting)` to per-take vectors. A setting the
    unit refused is not a row; one only the drawn side refused costs the whole span.
    """
    u, d = _medians(unit), _medians(drawn)
    if reading.null_is is not None:
        # A null the record defines is a value, and the span runs to it too.
        u, d = ({k: [reading.null_is if x is None else x for x in vec] for k, vec in side.items()}
                for side in (u, d))
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
            if reading.null_is is not None:
                a = reading.null_is if a is None else a
                b = reading.null_is if b is None else b
            if a is None:
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
    # A span that does not clear the floor by a doubling is a byte that did nothing, as
    # `reproduce` reads it: what a model owes it is to show nothing, not a share of it.
    null = not rows or span <= 2.0 * floor
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
                "model": ("refused" if None in (da, db) else "rises" if db > da
                          else "falls" if db < da else "holds"),
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
    found = _gates(scored, [])
    return not found["gross"]["passed"] or not found["breakdown"]["passed"]


def _consumers(root: Path, entry: dict, type_: str) -> list[tuple[str, dict]]:
    """The published records of this type in `READINGS` stages that read a directory."""
    out = []
    for hit in entry.get("consumed_by") or []:
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


def _published(root: Path, found: dict, type_: str) -> dict[str, tuple[str, list[str]]]:
    """Each reading stage with a published record of this type: its first record's directory
    and invocation, which a directory no record of that stage read is read from."""
    firsts: dict[str, tuple[str, str, list[str]]] = {}
    for rel, entry in sorted(found["directories"].items()):
        if entry.get("type") != type_:
            continue
        for record, data in _consumers(root, entry, type_):
            argv = data["record"]["invocation"]
            if argv[0] not in firsts or record < firsts[argv[0]][0]:
                firsts[argv[0]] = (record, rel, argv)
    return {command: (rel, argv) for command, (_, rel, argv) in sorted(firsts.items())}


TAKE_PATTERNS = ("--bypassed", "--control", "--still", "--reference", "--silence")
"""The flags a reading stage names takes other than its settings by."""

LENGTH_FLAGS = {"--hold": "hold_s", "--lead": "lead_s"}
"""A stage's flags tied to how long its stimulus was, and the stimulus field each is read from.

`--window` is a stretch the question chose and `--settled` a wait, so neither is here.
"""

SIGNATURE_EXCLUDED = (
    set(TAKE_PATTERNS)
    | {"--setting", "--held", "--out", "--slot", "--type", "--stimulus", "--channel",
       "--channels"}
    | set(LENGTH_FLAGS)
)
"""Flags a reading's signature ignores: they pick a take, a directory, or a
channel, not how the stage read one. Two runs pool as repeats of one setting only
when every other argument matches; a run that differs scores as its own record."""


def _signature_suffix(command: str, argv: list[str]) -> str:
    """What in `argv` differs from the stage's own defaults, spelled as it was passed.

    Read off the subcommand's own actions rather than the whole namespace, so a
    parent option unrelated to the reading (`--root`, `--device`, `command`) never
    shows up here. Empty where every reading argument is at its default, so a run
    made the ordinary way keeps the plain record name and only a run that asked for
    something else earns the bracket. Two runs whose non-default
    arguments are equal share a record; any difference -- including one being at the
    default and the other not -- gives each its own, since this string doubles as
    the two runs' grouping key.
    """
    from .cli import build_parser

    parser = build_parser()
    args = parser.parse_args(argv)
    sub = _subparser(parser, command)
    parts = []
    for action in sub._actions:
        if not action.option_strings or set(action.option_strings) & SIGNATURE_EXCLUDED:
            continue
        value = getattr(args, action.dest, action.default)
        if value == action.default:
            continue
        parts += _as_passed(argv, action)
    return ", ".join(sorted(parts))


def _as_passed(argv: list[str], action) -> list[str]:
    """Each occurrence of `action`'s flag in `argv`, with the tokens it took."""
    width = 0 if action.nargs == 0 else (action.nargs if isinstance(action.nargs, int) else 1)
    return [
        " ".join(argv[i : i + 1 + width])
        for i, token in enumerate(argv)
        if token in action.option_strings
    ]


def _scored_name(base: str, command: str, argv: list[str]) -> str:
    """`base` (a `READINGS` key), suffixed with what this run's own arguments differ by."""
    suffix = _signature_suffix(command, argv)
    return f"{base} [{suffix}]" if suffix else base


def _base_name(record: str) -> str:
    """The `READINGS` key a possibly suffixed record name was scored as."""
    return record.split(" [", 1)[0]


def _band_cut_argv(argv: list[str], cut_hz: float) -> list[str]:
    """`efx-bands`'s band list, replaced by explicit centres whose upper edge clears `cut_hz`.

    Reuses `efxbands.BAND_SETS` for the centres and each set's own width, and the edge
    `efxbands.energies` reads a band over (`centre * 2 ** (width_octaves / 2)`) rather
    than a bound worked out separately. A run already naming explicit `--band` centres
    keeps them, dropping only the ones over the limit; `cmd_efx_bands` reads such a
    list a third of an octave wide regardless of `--band-set`, so that is the width
    this filters it by.
    """
    from .cli import build_parser

    rest, bands, band_set = [], [], None
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--band" and i + 1 < len(argv):
            bands.append(float(argv[i + 1]))
            i += 2
            continue
        if token == "--band-set" and i + 1 < len(argv):
            band_set = argv[i + 1]
            i += 2
            continue
        rest.append(token)
        i += 1
    if bands:
        centres, width = bands, 1 / 3
    else:
        if band_set is None:
            sub = _subparser(build_parser(), "efx-bands")
            band_set = next(a.default for a in sub._actions if "--band-set" in a.option_strings)
        centres, width = efxbands.BAND_SETS[band_set]
    edge = 2 ** (width / 2)
    for centre in centres:
        if centre * edge <= cut_hz:
            rest += ["--band", str(centre)]
    return rest


def _role_pattern(pattern: str, source: list[dict], made: Directory) -> str | None:
    """The takes of `made` holding the role `pattern`'s own takes held where it was written."""
    compiled = re.compile(pattern)
    roles = {role(e["setting"]) for e in source if takes.named_by(compiled, e, e["file"])[1]}
    if len(roles) != 1 or None in roles:
        return None
    mine = sorted({e["setting"] for e in made.of_role(roles.pop())})
    if not mine:
        return None
    return "^(?:" + "|".join(re.escape(s) for s in mine) + ")$"


def templated(root: Path, found: dict, argv: list[str], source_rel: str,
              made: Directory) -> list[str] | None:
    """A stage's published invocation, aimed at a directory no record of that stage read.

    The takes directory becomes `made`'s, `--slot` its address, `--setting` its byte
    behind its one prefix, each other take pattern the takes of `made` holding the
    role the pattern's own takes held, and each of `LENGTH_FLAGS` the stage takes the
    length `made`'s swept stimulus states. `--held` is left for `read_with`. None where
    a role has no takes in `made`, the swept stimuli do not state one length, or the
    invocation names single takes.
    """
    from .cli import build_parser

    lengths = _lengths(argv[0], made, build_parser())
    if lengths is None:
        return None
    base = ledger.TAKES_ROOT.as_posix() + "/"
    source_dir = f"{base}{source_rel}"
    if source_dir not in argv or made.prefix is None:
        return None
    source = _takes(root, found, source_rel)
    out, i = [], 0
    while i < len(argv):
        token = argv[i]
        value = argv[i + 1] if i + 1 < len(argv) else None
        if token == source_dir:
            out.append(f"{base}{made.rel}")
        elif token.startswith(base):
            return None
        elif token == "--setting" and value is not None:
            out += [token, rf"^{re.escape(made.prefix)}(?P<value>\d{{1,3}})$"]
        elif token == "--slot" and value is not None:
            out += [token, made.address]
        elif token in TAKE_PATTERNS and value is not None:
            rebuilt = _role_pattern(value, source, made)
            if rebuilt is None:
                return None
            out += [token, rebuilt]
        elif token in LENGTH_FLAGS and value is not None:
            pass
        else:
            out.append(token)
            i += 1
            continue
        i += 1 if token == source_dir else 2
    for flag, seconds in lengths.items():
        out += [flag, str(seconds)]
    return out


def _lengths(command: str, made: Directory, parser) -> dict[str, float] | None:
    """Each length flag the stage takes, as the stimulus of `made`'s swept takes states it."""
    takes_flags = {s for a in _subparser(parser, command)._actions for s in a.option_strings}
    names = {str(e["stimulus"]) for items in made.by_setting.values() for e in items}
    stimuli = [s for s in made.entry.get("stimuli") or [] if s.get("name") in names]
    out = {}
    for flag, key in LENGTH_FLAGS.items():
        if flag not in takes_flags:
            continue
        stated = {s.get(key) for s in stimuli}
        if len(stated) != 1 or None in stated:
            return None
        out[flag] = stated.pop()
    return out


@dataclass
class _Class:
    """One stimulus class of a type's directories, and how it is compared."""

    name: str | None
    dirs: list[Directory]
    settings: int
    floor_db: float | None
    used: str
    chosen_by: str
    standing: int | None = None

    def shown(self, dirs: list[Directory]) -> dict:
        return {"class": self.name, "used": self.used, "chosen_by": self.chosen_by,
                "floor_db": None if self.floor_db is None else round(self.floor_db, 2),
                "settings": self.settings, "standing": self.standing,
                "dirs": [m.rel for m in dirs]}


def _classes(made: list[Directory], waves: dict, static: bool) -> list[_Class]:
    """The type's stimulus classes, the one stage 8 takes first.

    A static type's class compares waves where its takes repeat -- the median
    same-setting subtraction under `REPEATS_BELOW_DB` -- and readings elsewhere. Waves
    classes come first, lowest floor first; then the rest, most settings first.
    """
    groups: dict[str | None, list[Directory]] = {}
    for m in made:
        for name in sorted(m.classes) or [None]:
            groups.setdefault(name, []).append(m)
    found = []
    for name, dirs in groups.items():
        pairs = [f for m in dirs if m.rel in waves for s in waves[m.rel][1].values()
                 for f in s["floor_pairs"]]
        floor = float(np.median(pairs)) if pairs else None
        if not static:
            used, chosen = "readings", "by_repeatability_class"
        elif floor is None or floor > REPEATS_BELOW_DB:
            used, chosen = "readings", "by_stimulus_floor"
        else:
            used, chosen = "waves", "by_repeatability_class"
        found.append(_Class(name, dirs, sum(len(m.by_setting) for m in dirs), floor, used,
                            chosen))

    def order(c: _Class):
        floor = math.inf if c.floor_db is None else c.floor_db
        if c.used == "waves":
            return (0, floor, -c.settings, str(c.name))
        return (1, -c.settings, floor, str(c.name))

    return sorted(found, key=order)


def p1_class(classes: list[_Class], standing: dict) -> tuple[_Class, list[_Class]]:
    """Stage 8's class, and the rest in their order.

    The class with the most settings where the unit's own side shows the effect
    standing; a tie goes to `waves`, then to the lower floor.
    """

    def order(c: _Class):
        floor = math.inf if c.floor_db is None else c.floor_db
        return (-standing.get(c.name, 0), c.used != "waves", floor, -c.settings, str(c.name))

    first = min(classes, key=order)
    return first, [c for c in classes if c is not first]


def _waves_standing(dirs: list[Directory], waves: dict) -> int:
    """Settings where the unit less its bypass take lands more than the floor margin over it."""
    return sum(
        1 for m in dirs for side in waves[m.rel][1].values()
        if side["doing_nothing"] - float(np.mean(side["floor_pairs"])) > WITHIN_THE_FLOOR_DB
    )


class _Phase:
    """One comparison over a set of directories, shared by every model it is run on.

    Each directory is compared the way its stimulus class is: `waves` or `readings`.
    """

    def __init__(self, root, found, type_, parts, waves, published, inherited=None):
        self.root, self.found, self.type = root, found, type_
        self.dirs = [m for _, dirs in parts for m in dirs]
        self.used = {m.rel: cls.used for cls, dirs in parts for m in dirs}
        self.published = published
        self.inherited = inherited or {}
        self.channel: dict[str, int] = {}
        self.unit: dict = {}
        self.stages: dict[str, dict] = {}
        self.runs: dict[str, list] = {}
        self.unread: list[tuple[str, str]] = []
        for made in self.dirs:
            if self.used[made.rel] == "waves":
                channel, unit = waves.get(made.rel) or (loudest(made), None)
                self.channel[made.rel] = channel
                self.unit[made.rel] = unit if unit is not None else _unit_waves(made, channel)
        self._read_units([m for m in self.dirs if self.used[m.rel] == "readings"])

    @property
    def modes(self) -> set[str]:
        return set(self.used.values())

    def quantities(self) -> list[str]:
        names = ["waves:residual_db"] if "waves" in self.modes else []
        if "readings" in self.modes:
            names += sorted(n for n, r in READINGS.items() if r.command in self.published) or [
                "a reading of any stage in READINGS, none of which has a published record "
                f"of {self.type}"]
        return names

    def _read_units(self, dirs: list[Directory]) -> None:
        for made in dirs:
            writes = writes_of(made.entry.get("held"))
            cut_dir = cut_directory(self.root, made.rel)
            runs = [(data["record"]["invocation"], False)
                    for _, data in _consumers(self.root, made.entry, self.type)]
            have = {argv[0] for argv, _ in runs}
            for command, (source_rel, argv) in self.published.items():
                if command in have:
                    continue
                aimed = templated(self.root, self.found, argv, source_rel, made)
                if aimed is None:
                    self.unread.append((made.rel, command))
                    continue
                runs.append((aimed, True))
            for argv, from_template in runs:
                command = argv[0]
                reading_argv = _band_cut_argv(argv, graph.CUT_HZ) if command == "efx-bands" \
                    else argv
                try:
                    read = read_with(reading_argv, root=self.root, unit_dir=made.where,
                                     drawn_dir=None, channel=self.channel.get(made.rel),
                                     writes=writes, cache=self.root / STAGE_READINGS,
                                     cut_dir=cut_dir)
                except (ValueError, FileNotFoundError, SystemExit):
                    if not from_template:
                        raise
                    self.unread.append((made.rel, command))
                    continue
                if not read:
                    continue
                self.channel.setdefault(made.rel, _channel_read(read))
                setting = self._decay_setting(made, argv) if command == "decay" else None
                if command == "decay" and setting is None:
                    continue
                for name, part in collected(command, read, made.rel, setting=setting).items():
                    record_name = _scored_name(name, command, argv)
                    stage = self.stages.setdefault(
                        record_name, {"unit": {}, "floors": [], "base": name})
                    for key, vectors in part["takes"].items():
                        if key[1] in made.by_setting:
                            stage["unit"].setdefault(key, []).extend(vectors)
                    if part["floor"] is not None:
                        stage["floors"].append(part["floor"])
                self.runs.setdefault(command, []).append((made, argv, setting))

    def _decay_setting(self, made: Directory, argv: list[str]) -> int | None:
        wet = Path(argv[2]).name
        return next((v for v, items in made.by_setting.items()
                     if any(e["file"] == wet for e in items)), None)

    def standing(self) -> int:
        """Settings at which some stage admitted a reading of the unit's own takes."""
        return len({
            key for stage in self.stages.values() for key, vectors in stage["unit"].items()
            if any(x is not None for vector in vectors for x in vector)
        })

    def compared(self) -> list[tuple[str, int]]:
        keys = {(rel, v) for rel, per in self.unit.items() for v in per}
        keys |= {k for stage in self.stages.values() for k in stage["unit"]}
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
            if self.channel.get(made.rel) is None:
                continue
            drawn[made.rel] = render_directory(
                cand, self.root, made.rel, channels=_pair(self.channel[made.rel]),
                found=self.found, made=made)
        lost = set().union(*[_unrenderable(p, rel) for rel, p in drawn.items()])
        shown = [p.relative_to(self.root).as_posix() for p in drawn.values()]
        by_rel = {m.rel: m for m in self.dirs}
        out = [
            _scored_waves(by_rel[rel], self.unit[rel], drawn[rel], self.channel[rel], lost)
            for rel in self.unit
        ]
        mine: dict[str, dict] = {}
        for command, runs in self.runs.items():
            for made, argv, setting in runs:
                if made.rel not in drawn:
                    continue
                reading_argv = _band_cut_argv(argv, graph.CUT_HZ) if command == "efx-bands" \
                    else argv
                read = read_with(reading_argv, root=self.root, unit_dir=made.where,
                                 drawn_dir=drawn[made.rel], channel=self.channel[made.rel],
                                 writes=writes_of(made.entry.get("held")))
                for name, part in collected(command, read, made.rel, setting=setting).items():
                    record_name = _scored_name(name, command, argv)
                    for key, vectors in part["takes"].items():
                        mine.setdefault(record_name, {}).setdefault(key, []).extend(vectors)
        for name, stage in sorted(self.stages.items()):
            ours = {k: v for k, v in mine.get(name, {}).items() if k in stage["unit"]}
            floor = max(stage["floors"]) if stage["floors"] else repeat_floor(stage["unit"])
            if READINGS[stage["base"]].octaves:
                # A rate's lean is judged against the coarser of the run's floor and one
                # entry of the candidate's own table, as `score_against_rates` does.
                steps = [one_entry_octaves(cand.raw, by_rel[rel].address, [v])
                         for rel, v in stage["unit"] if rel in by_rel and v > 0]
                if steps and all(s > 0 for s in steps):
                    floor = max(floor, float(np.median(steps)))
            before = self.inherited.get(name, {})
            out.append(scored_readings(
                name, READINGS[stage["base"]], stage["unit"], ours, floor=floor,
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


_MEASUREMENT = {
    "no_input_take": "takes of {type} with a bypass take of the same stimulus, beside them or "
    "in a directory the ledger matches on every stimulus field, swept at one address of the "
    "effect's block and read by a stage that publishes a quantity for this type",
    "no_held_out_setting": "a setting of {type} that no value of any candidate was fitted on, "
    "declared or through a claim's records",
    "control_could_not_fail": "a comparison of {type} the identity model can fail: a setting "
    "that lifts the effect clear of the takes' own floor, read by a stage that publishes a "
    "quantity for this type from these directories",
    "unrenderable": "a candidate for {type} whose loops keep a sample of delay at every "
    "compared setting",
    "power": "a setting of {type} where the runner-up's reading departs from the winner's by "
    "more than the floor, or a runner-up that leans where the winner does not",
}


def _quantity_of(record: str) -> str:
    base = _base_name(record)
    return base if base in READINGS else "waves:residual_db"


def _undecided(gate: str, phase: _Phase, block: dict, scored: list[dict],
               control: list[dict]) -> tuple[list[str], list[tuple[str, object]]]:
    """The quantities a gate stopped on, and the `(directory, setting)` it stopped at."""
    compared = [tuple(k) for k in phase.compared()]
    if gate == "no_input_take":
        pairs = [(m.rel, v) for m in phase.dirs for v in m.by_setting]
        return phase.quantities(), pairs
    if gate == "control_could_not_fail":
        return sorted({_quantity_of(s["record"]) for s in control}), compared
    if gate == "no_held_out_setting":
        return phase.quantities(), compared
    if gate == "unrenderable":
        lost = sorted({tuple(x) for s in scored for x in s.get("lost", [])})
        return sorted({_quantity_of(s["record"]) for s in scored if s.get("lost")}), lost
    quantities, pairs = set(), []
    for gates in _each_unit(block["gates"]):
        if gate == "gross":
            for s in scored:
                if s["span"] > 0 and not s["is_null_record"] and (
                        s["median_abs"] / s["span"] > reproduce.GROSS_CEILING):
                    quantities.add(_quantity_of(s["record"]))
                    pairs += [(r["dir"], r["value"]) for r in s["rows"]
                              if max(abs(x) for x in r["residual"])
                              > reproduce.GROSS_CEILING * s["span"]]
        elif gate == "breakdown":
            for broke in gates.get("breakdown", {}).get("broke", []):
                quantities.add(_quantity_of(broke["record"]))
                if broke.get("at"):
                    pairs.append(tuple(broke["at"]))
            for record in gates.get("breakdown", {}).get("leaning", []):
                quantities.add(_quantity_of(record))
        elif gate == "qualitative":
            for item in gates.get("qualitative", {}).get("checked", []):
                if item["same"]:
                    continue
                quantities.add(_quantity_of(item["record"]))
                value = item.get("value")
                if isinstance(value, list) and len(value) == 3:
                    pairs += [(value[0], value[1]), (value[0], value[2])]
                elif value is not None:
                    pairs.append((item["record"], value))
        else:
            quantities |= {_quantity_of(s["record"]) for s in scored}
            pairs = compared
    return sorted(quantities) or phase.quantities(), sorted(set(pairs), key=str) or compared


def _where(pairs) -> str:
    by: dict[str, list] = {}
    for rel, value in pairs:
        by.setdefault(str(rel), []).append(value)
    return "; ".join(f"{rel} at {', '.join(str(v) for v in sorted(set(vs), key=str))}"
                     for rel, vs in sorted(by.items())) or "no directory"


def _needs(gate: str, type_: str, quantities: list[str], pairs) -> str:
    """What a stop needs: the gate, the quantity, where it went undecided, and the measurement."""
    measurement = _MEASUREMENT.get(
        gate, "a candidate for {type} that clears the " + gate + " gate, or a measurement that "
        "separates the ones standing").format(type=type_)
    return (f"{gate} on {', '.join(quantities)}, undecided at {_where(pairs)}. "
            f"Needs {measurement}.")


def _run(phase: _Phase, candidates: list[Candidate], comparison: dict, classes: list[dict],
         check=None):
    """The block a phase writes, the model p0 shows, the gate it stopped at, and its `needs`."""
    compared = phase.compared()
    control_scored, _ = phase.scored(identity())
    control = {"model": "identity", "separated": separated(control_scored),
               "gates": _gates(control_scored, [])}
    if check is not None:
        control["bypass_check"] = check
    block = {"verdict": "stopped", "gates": {}, "control": control,
             "compared": phase.rows(), "held_out_settings": [],
             "excluded_from_ranking": [], "comparison": comparison, "classes": classes}

    def stop(gate, scored=()):
        quantities, pairs = _undecided(gate, phase, block, list(scored), control_scored)
        return _needs(gate, phase.type, quantities, pairs)

    if not candidates:
        return block, None, None, [], None
    if not compared:
        return block, None, "no_input_take", [], stop("no_input_take")
    held_by = {}
    for cand in candidates:
        fitted = fitted_on(cand.raw, root=phase.root, ledger=phase.found)
        held_by[cand.id] = [[rel, v] for rel, v in compared if (rel, v) not in fitted]
    block["excluded_from_ranking"] = [c.id for c in candidates if not held_by[c.id]]
    ranked = [c for c in candidates if held_by[c.id]] or candidates
    per, drawn_on, ranking = {}, [], []
    for cand in ranked:
        scored, shown = phase.scored(cand)
        per[cand.id] = (cand, held_by[cand.id], scored)
        drawn_on += shown
        ranking.append(_ranked(cand, scored))
    if block["excluded_from_ranking"] == [c.id for c in candidates]:
        nearest = min(ranking, key=lambda r: r["worst_share_of_span"])["candidate"]
        return block, per[nearest][0], "no_held_out_setting", drawn_on, stop(
            "no_held_out_setting")
    ranking.sort(key=lambda r: r["mean_share_of_span"])
    winner, held, scored = per[ranking[0]["candidate"]]
    block["_scored"] = scored
    block["held_out_settings"] = held
    block["gates"] = _gates(scored, ranking)
    lost = [s for s in scored if s.get("lost")]
    if lost and all(len(s["lost"]) == len(s["rows"]) for s in lost):
        return block, winner, "unrenderable", drawn_on, stop("unrenderable", scored)
    if not control["separated"]:
        return block, winner, "control_could_not_fail", drawn_on, stop(
            "control_could_not_fail", scored)
    failed = next((g for g in GATES if not block["gates"][g]["passed"]), None)
    if failed is None:
        block["verdict"] = "passed"
        return block, winner, None, drawn_on, None
    return block, winner, failed, drawn_on, stop(failed, scored)


def _inherited(block: dict) -> dict:
    """What a later phase takes from this one: each reading's span and properties."""
    return {s["record"]: {"span": s["span"], "properties": s["properties"]}
            for s in block.get("_scored", []) if _base_name(s["record"]) in READINGS}


def _static(root: Path, unit: str) -> set[str]:
    path = root / "data" / "units" / unit / "efx-sort" / "by-repeatability.json"
    return set(json.loads(path.read_text()).get("static") or [])


def _delay_rows(model: dict) -> list[str]:
    return sorted({
        spec["byte"] for node in model.get("nodes", []) if node.get("kind") == "delay"
        for spec in _values(node.get("time_ms")) if "byte" in spec
    })


def _comparison(cls: _Class) -> dict:
    """The `used`/`chosen_by` a stage records, with `cut_hz` where it read below it."""
    out = {"used": cls.used, "chosen_by": cls.chosen_by}
    if cls.used == "readings":
        out["cut_hz"] = graph.CUT_HZ
    return out


def stage(root: str | Path, unit: str, type_: str, *, class_name: str | None = None) -> dict:
    """Stages 7-9 for one type, or the identity control alone when no class is named.

    Stage 8 takes the stimulus class `p1_class` picks from the unit's side alone; a
    waves class whose bypass check fails is compared by readings instead. Stage 9 takes
    up to two directories of the classes after it, each compared the way its class is.
    """
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
        needs = _needs("no_input_take", type_, ["any quantity"],
                       [(rel, "no swept setting with an input")
                        for rel in sorted(r for r, e in found["directories"].items()
                                          if e.get("type") == type_)])
        result["stopped"] = {"at": 7, "gate": "no_input_take", "needs": needs}
        return {**result, "equivalent_under_this_test": [], **tail}

    static = type_ in _static(root, unit)
    waves = {}
    if static:
        for m in made:
            channel = loudest(m)
            waves[m.rel] = (channel, _unit_waves(m, channel))
    classes = _classes(made, waves, static)
    published = _published(root, found, type_)
    phases: dict = {}
    for cls in classes:
        if cls.used == "waves":
            cls.standing = _waves_standing(cls.dirs, waves)
        else:
            phases[cls.name] = _Phase(root, found, type_, [(cls, cls.dirs)], waves, published)
            cls.standing = phases[cls.name].standing()
    first, rest = p1_class(classes, {c.name: c.standing for c in classes})
    check = None
    if first.used == "waves":
        check = bypass_check(root, unit, type_, first.dirs,
                             {m.rel: waves[m.rel][1] for m in first.dirs})
        if check["result"] == "failed":
            first.used, first.chosen_by = "readings", "bypass_check_failed"
    phase = phases.get(first.name) if first.name in phases else None
    if phase is None:
        phase = _Phase(root, found, type_, [(first, first.dirs)], waves, published)
    comparison = _comparison(first)
    block, shown, gate, drawn_on, needs = _run(
        phase, candidates, comparison, [first.shown(first.dirs)], check)
    inherited = _inherited(block)
    block.pop("_scored", None)
    if not candidates:
        return {**result, "control": block["control"], "compared": block["compared"],
                "comparison": comparison, "classes": block["classes"], **tail}

    shown = shown or candidates[0]
    result["p0"] = {"model": shown.shown_as, "model_sha256": shown.sha256,
                    "rendered_on": sorted({p for p in drawn_on if f"/{shown.id}/" in f"/{p}"})}
    result["p1"] = block
    used = {m.rel for m in first.dirs}
    later: list[tuple[_Class, list[Directory]]] = []
    for cls in rest:
        room = 2 - sum(len(dirs) for _, dirs in later)
        mine = [m for m in cls.dirs if m.rel not in used][:max(room, 0)]
        used |= {m.rel for m in mine}
        if mine:
            later.append((cls, mine))
    modes = {first.used}
    if gate is not None:
        result["stopped"] = {"at": 8, "gate": gate, "needs": needs}
    elif not later:
        result["stopped"] = {"at": 9, "gate": "no_input_take", "needs": _needs(
            "no_input_take", type_, phase.quantities(),
            [(f"a directory under a stimulus class other than {first.name!r}", "any setting")])}
    else:
        p2_phase = _Phase(root, found, type_, later, waves, published, inherited)
        head = later[0][0]
        p2, _, gate2, drawn2, needs2 = _run(
            p2_phase, candidates, _comparison(head),
            [cls.shown(dirs) for cls, dirs in later])
        p2.pop("_scored", None)
        result["p2"] = p2
        modes |= p2_phase.modes
        result["p0"]["rendered_on"] = sorted(set(result["p0"]["rendered_on"]) | {
            p for p in drawn2 if f"/{shown.id}/" in f"/{p}"})
        if gate2 is not None:
            result["stopped"] = {"at": 9, "gate": gate2, "needs": needs2}
    equivalent = _delay_rows(shown.raw) if "waves" in modes else []
    return {**result, "equivalent_under_this_test": equivalent, **tail}


__all__ = [
    "IDENTITY",
    "READINGS",
    "LedgerBehind",
    "Reading",
    "bytes_now",
    "candidate",
    "cut_directory",
    "fitted_on",
    "inert_settings",
    "p1_class",
    "power_on",
    "render_directory",
    "scored_readings",
    "separated",
    "stage",
    "swept_bytes",
    "templated",
    "writes_of",
]
