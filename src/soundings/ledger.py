"""A working index over `.cache/takes`, built from the takes and the records that read them.

It answers, per directory of takes, which type, address and settings it holds,
what state the unit was held in, and which published records read it -- the
questions that decide what is left to record and what can be read off disk.

The ledger is one entry per directory, resolved from whatever says what a
directory is: its own manifest first, and where that is silent, the published
records that cite it. **It is not published and not a measurement** -- it can be
rebuilt from the takes and the archive at any time, so it lives at
`.cache/takes-ledger.json`, gitignored, beside the takes it describes.

**A citation is followed rather than trusted.** A published record's
`invocation` may point straight at a directory under `.cache/takes`, or at
another place under `.cache` that itself points onward -- an intermediate record
`efx-params` writes one `--slot` at a time, say. This module walks that chain up
to three hops and gives up with a named reason rather than guessing, because a
guess here is a measurement wearing the wrong type's held state.
"""

from __future__ import annotations

import json
import re
import struct
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

TAKES_ROOT = Path(".cache/takes")
LEDGER_PATH = Path(".cache/takes-ledger.json")

MAX_HOPS = 3
"""How many `.cache` locations beyond a record's own citation may be opened.

A citation that is itself under `.cache/takes` costs no hop at all. One that
points elsewhere costs one hop to open, and each hop past it costs one more; a
chain still not at a takes directory after the third is `depth_exceeded` rather
than opened a fourth time.
"""

REASON_CODES = ("path_missing", "depth_exceeded", "chain_ends_without_takes")
"""The closed vocabulary a record that never reaches a takes directory is given.

Closed because a free-text reason is satisfied by anything a writer cares to
type, and a summary line built from it could not be checked from the outside
(`AGENTS.md`, decision `理由は閉じた語彙のコードにする`).
"""

TYPE_SOURCES = ("manifest_type", "manifest_prepared", "citing_record", "dirname")
HELD_SOURCES = ("manifest_prepared", "citing_invocation", "citing_intermediate")


# --------------------------------------------------------------------------
# Small, shared normalisers. Every address and byte string in this module
# passes through these once, so "40 03 00" and "40 42 22" always compare equal
# to themselves regardless of which flag or field they were read from.
# --------------------------------------------------------------------------


def norm_addr(text: str | None) -> str | None:
    """An address as `'40 03 00'` -- upper-case hex pairs, single-spaced."""
    if not text:
        return None
    return " ".join(part.upper() for part in text.split())


def norm_bytes(text: str | None) -> str | None:
    """A byte string the same way an address is: upper-case, single-spaced."""
    if not text:
        return None
    return " ".join(part.upper() for part in text.split())


_TYPE_ADDRESS = "40 03 00"


def _parse_address_bytes(token: str) -> tuple[str, str] | None:
    """`'40 03 00=01 50'` as `('40 03 00', '01 50')`, or None if it is not one."""
    if "=" not in token:
        return None
    addr, _, value = token.partition("=")
    addr = norm_addr(addr)
    value = norm_bytes(value)
    if not addr or not value:
        return None
    return addr, value


def _flag_values(argv: list[str], flag: str) -> list[str]:
    """Every value that followed `flag` in an argv list, in the order they appear."""
    return [argv[i + 1] for i, tok in enumerate(argv) if tok == flag and i + 1 < len(argv)]


def _leading_positionals(argv: list[str]) -> list[str]:
    """The tokens between the stage name and the first flag.

    Every command this repository writes places its positional arguments first
    and its flags after, so the boundary is the first token spelled with a
    leading `-`. Used only to find `.cache` paths, so a flag's own value being
    swept up here by mistake is harmless unless it happens to start with
    `.cache/`, which none of the flags that are not themselves followed here do.
    """
    out = []
    for tok in argv[1:]:
        if tok.startswith("-"):
            break
        out.append(tok)
    return out


def _cache_tokens_all(argv: list[str]) -> list[str]:
    """Every element of a published record's own invocation that names a `.cache` path."""
    return [tok for tok in argv if isinstance(tok, str) and tok.startswith(".cache/")]


def _cache_tokens_chain(argv: list[str]) -> list[str]:
    """What an intermediate record's invocation offers to keep following: `--save` and positionals.

    Deliberately narrower than `_cache_tokens_all`: an intermediate record's
    other flags -- `--control`, `--types-from` -- cite paths that are not what it
    read, and following them would bind a directory to a record that never
    touched it.
    """
    tokens = _flag_values(argv, "--save") + _leading_positionals(argv)
    return [tok for tok in tokens if isinstance(tok, str) and tok.startswith(".cache/")]


def _payload_cache_tokens(payload: dict) -> list[str]:
    """`dry`, `wet` and `floor_from`, the three payload fields that also cite `.cache`."""
    tokens: list[str] = []
    for key in ("dry", "wet"):
        value = payload.get(key)
        if isinstance(value, str):
            tokens.append(value)
    floor_from = payload.get("floor_from")
    if isinstance(floor_from, list):
        tokens.extend(v for v in floor_from if isinstance(v, str))
    elif isinstance(floor_from, str):
        tokens.append(floor_from)
    return [t for t in tokens if t.startswith(".cache/")]


def _record_type(argv: list[str], payload: dict) -> str | None:
    """What a record's own invocation or payload asserts its type was, if anything."""
    values = _flag_values(argv, "--type")
    if values:
        return norm_bytes(values[-1])
    for value in _flag_values(argv, "--prepare"):
        parsed = _parse_address_bytes(value)
        if parsed and parsed[0] == _TYPE_ADDRESS:
            return parsed[1]
    for entry in payload.get("prepared") or []:
        if isinstance(entry, dict) and norm_addr(entry.get("address")) == _TYPE_ADDRESS:
            return norm_bytes(entry.get("bytes"))
    return None


def _record_held(argv: list[str]) -> dict[str, str]:
    """The address=bytes pairs a record's own `--held` and `--prepare` flags carry."""
    held: dict[str, str] = {}
    for flag in ("--held", "--prepare"):
        for value in _flag_values(argv, flag):
            parsed = _parse_address_bytes(value)
            if parsed:
                held[parsed[0]] = parsed[1]
    return held


def _payload_prepared(payload: dict) -> dict[str, str]:
    """The address=bytes pairs an intermediate record's own `prepared` field carries."""
    held: dict[str, str] = {}
    for entry in payload.get("prepared") or []:
        if isinstance(entry, dict):
            addr = norm_addr(entry.get("address"))
            value = norm_bytes(entry.get("bytes"))
            if addr:
                held[addr] = value
    return held


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


# --------------------------------------------------------------------------
# Following a citation to the directory it names.
# --------------------------------------------------------------------------


@dataclass
class Intermediate:
    """One `.cache` record opened while chasing a citation to a takes directory."""

    record_id: str
    record_type: str | None
    prepared: dict[str, str]


@dataclass
class Hit:
    """One published record's account of one directory it was traced to."""

    record_id: str
    via: str | list[str]
    measured_at: str | None
    record_type: str | None
    record_held: dict[str, str]
    setting_pattern: str | None
    slot_flag: str | None
    stimulus_flag: str | None
    intermediates: list[Intermediate] = field(default_factory=list)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_token(
    root: Path,
    takes_root: Path,
    token: str,
    hops: int,
    visited: frozenset[str],
    trail: list[str],
    intermediates: list[Intermediate],
    cited: set[str],
) -> tuple[list[tuple[Path, list[str], list[Intermediate]]], str | None]:
    """Follow one `.cache` path to the takes directories it ultimately names.

    Returns the directories found, each with the trail of intermediate records
    that led there, and -- only meaningful when nothing was found -- the one
    reason code that best describes why.
    """
    path = (root / token).resolve()
    if _is_under(path, takes_root.resolve()):
        directory = path if path.is_dir() else path.parent
        return [(directory, list(trail), list(intermediates))], None
    key = str(path)
    if key in visited:
        return [], "chain_ends_without_takes"
    if not path.exists():
        return [], "path_missing"
    cited.add(Path(token).as_posix())
    if hops >= MAX_HOPS:
        return [], "depth_exceeded"
    jsons = [path] if path.is_file() else sorted(path.glob("*.json"))
    if not jsons:
        return [], "chain_ends_without_takes"
    next_visited = visited | {key}
    found: list[tuple[Path, list[str], list[Intermediate]]] = []
    reasons: list[str] = []
    for one in jsons:
        try:
            data = json.loads(one.read_text())
        except (OSError, ValueError):
            reasons.append("chain_ends_without_takes")
            continue
        argv = ((data.get("record") or {}).get("invocation")) or []
        display = _display_path(one, root)
        next_intermediates = [
            *intermediates,
            Intermediate(display, _record_type(argv, data), _payload_prepared(data)),
        ]
        next_tokens = _cache_tokens_chain(argv)
        if not next_tokens:
            reasons.append("chain_ends_without_takes")
            continue
        for nt in next_tokens:
            sub_found, sub_reason = _resolve_token(
                root, takes_root, nt, hops + 1, next_visited, [*trail, display], next_intermediates, cited
            )
            found.extend(sub_found)
            if sub_reason:
                reasons.append(sub_reason)
    if found:
        return found, None
    if "path_missing" in reasons:
        return [], "path_missing"
    if "depth_exceeded" in reasons:
        return [], "depth_exceeded"
    return [], "chain_ends_without_takes"


def _pick_reason(reasons: list[str]) -> str:
    for code in REASON_CODES:
        if code in reasons:
            return code
    return "chain_ends_without_takes"


def _scan_records(
    unit_dir: Path, root: Path, takes_root: Path
) -> tuple[dict[Path, list[Hit]], list[dict], set[str]]:
    """Every published record under `unit_dir`, traced to the directories it reads."""
    hits: dict[Path, list[Hit]] = {}
    unbound: list[dict] = []
    cited: set[str] = set()
    if not unit_dir.is_dir():
        return hits, unbound, cited
    for path in sorted(unit_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        record = data.get("record")
        if not isinstance(record, dict):
            continue
        argv = record.get("invocation") or []
        tokens = sorted(set(_cache_tokens_all(argv)) | set(_payload_cache_tokens(data)))
        if not tokens:
            continue
        record_id = _display_path(path, root)
        record_type = _record_type(argv, data)
        record_held = _record_held(argv)
        setting_pattern = (_flag_values(argv, "--setting") or [None])[-1]
        slot_flag = (_flag_values(argv, "--slot") or [None])[-1]
        stimulus_flag = (_flag_values(argv, "--stimulus") or [None])[-1]
        measured_at = record.get("measured_at")

        found: dict[Path, tuple[list[str], list[Intermediate]]] = {}
        reasons: list[str] = []
        for token in tokens:
            token_hits, reason = _resolve_token(root, takes_root, token, 0, frozenset(), [], [], cited)
            for directory, trail, intermeds in token_hits:
                found.setdefault(directory, (trail, intermeds))
            if reason:
                reasons.append(reason)

        if found:
            for directory, (trail, intermeds) in found.items():
                hits.setdefault(directory, []).append(
                    Hit(
                        record_id=record_id,
                        via="direct" if not trail else trail,
                        measured_at=measured_at,
                        record_type=record_type,
                        record_held=record_held,
                        setting_pattern=setting_pattern,
                        slot_flag=slot_flag,
                        stimulus_flag=stimulus_flag,
                        intermediates=intermeds,
                    )
                )
        else:
            unbound.append({"record": record_id, "reason": _pick_reason(reasons)})
    return hits, unbound, cited


# --------------------------------------------------------------------------
# Reading a directory itself: WAV headers only, never the sample data.
# --------------------------------------------------------------------------


def _wav_header(path: Path) -> dict:
    """`channels`, `sample_rate`, `frames` and `seconds`, from the chunk headers alone.

    Never the sample data -- the archive this reads is 230 GB and a take can be
    tens of megabytes, while every header needed here is a few hundred bytes at
    the front of the file.
    """
    channels = sample_rate = bits = frames = None
    with open(path, "rb") as f:
        riff = f.read(12)
        if len(riff) < 12 or riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
            raise ValueError(f"{path} is not a RIFF/WAVE file")
        while True:
            head = f.read(8)
            if len(head) < 8:
                break
            chunk_id, chunk_size = struct.unpack("<4sI", head)
            pad = chunk_size % 2
            if chunk_id == b"fmt ":
                core = f.read(min(chunk_size, 16))
                if len(core) >= 16:
                    _, channels, sample_rate, _, _, bits = struct.unpack("<HHIIHH", core[:16])
                f.seek(chunk_size - len(core) + pad, 1)
            elif chunk_id == b"data":
                if channels and bits:
                    frames = chunk_size // (channels * (bits // 8))
                break
            else:
                f.seek(chunk_size + pad, 1)
    seconds = round(frames / sample_rate, 4) if frames and sample_rate else None
    return {"channels": channels, "sample_rate": sample_rate, "frames": frames, "seconds": seconds}


def _take_directories(takes_root: Path) -> list[Path]:
    """Every directory directly holding a WAV, once each.

    Matches `find .cache/takes -name '*.wav' -exec dirname {} + | sort -u`.
    """
    found: set[Path] = set()
    if takes_root.is_dir():
        for wav in takes_root.rglob("*.wav"):
            found.add(wav.parent)
    return sorted(found)


_EFX_PARAMS_DIRNAME = re.compile(r"(?:^|/)efx-params-([0-9A-Fa-f]{2})-([0-9A-Fa-f]{2})(?:-|/|$)")
_ENUM_STATES_DIRNAME = re.compile(r"(?:^|/)enum-states/([0-9A-Fa-f]{2})-([0-9A-Fa-f]{2})(?:-|/|$)")
_ADDRESS_DIRNAME = re.compile(r"^([0-9A-Fa-f]{2})-([0-9A-Fa-f]{2})-([0-9A-Fa-f]{2})$")


def _type_from_dirname(relpath: Path) -> str | None:
    text = relpath.as_posix() + "/"
    for pattern in (_EFX_PARAMS_DIRNAME, _ENUM_STATES_DIRNAME):
        found = pattern.search(text)
        if found:
            return f"{found.group(1).upper()} {found.group(2).upper()}"
    return None


def _address_from_dirname(name: str) -> str | None:
    found = _ADDRESS_DIRNAME.match(name)
    if not found:
        return None
    return " ".join(g.upper() for g in found.groups())


def _resolve_type(
    manifest: dict, relpath: Path, consumed: list[Hit]
) -> tuple[str | None, str | None, list[dict] | None]:
    candidates: list[tuple[str, str]] = []
    mtype = manifest.get("type")
    if isinstance(mtype, str) and mtype.strip():
        candidates.append(("manifest_type", norm_bytes(mtype)))
    for entry in manifest.get("prepared") or []:
        if isinstance(entry, dict) and norm_addr(entry.get("address")) == _TYPE_ADDRESS:
            candidates.append(("manifest_prepared", norm_bytes(entry.get("bytes"))))
            break
    seen_citing: set[str] = set()
    for hit in consumed:
        if hit.record_type and hit.record_type not in seen_citing:
            candidates.append(("citing_record", hit.record_type))
            seen_citing.add(hit.record_type)
    dirname_type = _type_from_dirname(relpath)
    if dirname_type:
        candidates.append(("dirname", dirname_type))

    if not candidates:
        return None, None, None
    values = {value for _, value in candidates}
    if len(values) == 1:
        return candidates[0][1], candidates[0][0], None
    return None, None, [{"source": source, "value": value} for source, value in candidates]


def _resolve_address(manifest: dict, dirpath: Path, consumed: list[Hit]) -> str | None:
    addr = manifest.get("address")
    if isinstance(addr, str) and addr.strip():
        return norm_addr(addr)
    for hit in consumed:
        if hit.slot_flag:
            return norm_addr(hit.slot_flag)
    return _address_from_dirname(dirpath.name)


def _resolve_settings(
    manifest: dict, wavs: list[str], consumed: list[Hit]
) -> tuple[dict[str, int] | None, str]:
    if manifest.get("takes"):
        counts: dict[str, int] = {}
        for entry in manifest["takes"]:
            setting = str(entry.get("setting"))
            counts[setting] = counts.get(setting, 0) + 1
        return counts, "manifest"
    for hit in consumed:
        if not hit.setting_pattern:
            continue
        try:
            compiled = re.compile(hit.setting_pattern)
            if "value" not in (compiled.groupindex or {}):
                continue
        except re.error:
            continue
        counts = {}
        for name in wavs:
            found = compiled.search(Path(name).stem)
            if found:
                value = found.group("value")
                counts[value] = counts.get(value, 0) + 1
        if counts:
            return counts, f"citing_regex:{hit.record_id}"
    return None, "absent"


def _resolve_stimuli(manifest: dict, consumed: list[Hit]) -> list[dict] | None:
    raw = manifest.get("stimuli")
    if isinstance(raw, list) and raw:
        return [_with_class(entry) for entry in raw if isinstance(entry, dict)]
    for hit in consumed:
        if hit.stimulus_flag:
            return [_with_class({"name": hit.stimulus_flag})]
    return None


def _with_class(entry: dict) -> dict:
    name = entry.get("name")
    cls = "held" if isinstance(name, str) and name.startswith("held-") else name
    return {**entry, "class": cls}


def _resolve_held(
    manifest: dict, dir_type: str | None, consumed: list[Hit]
) -> tuple[dict[str, str] | None, str | None, list[dict] | None]:
    conflicts: list[dict] = []

    manifest_prepared = _payload_prepared(manifest)
    if manifest_prepared:
        return manifest_prepared, "manifest_prepared", None

    for hit in consumed:
        if not hit.record_held:
            continue
        if hit.record_type is None or hit.record_type != dir_type:
            conflicts.append(
                {
                    "record": hit.record_id,
                    "source": "citing_invocation",
                    "record_type": hit.record_type,
                    "dir_type": dir_type,
                }
            )
            continue
        return hit.record_held, "citing_invocation", (conflicts or None)

    for hit in consumed:
        for intermediate in hit.intermediates:
            if not intermediate.prepared:
                continue
            if intermediate.record_type is None or intermediate.record_type != dir_type:
                conflicts.append(
                    {
                        "record": intermediate.record_id,
                        "source": "citing_intermediate",
                        "record_type": intermediate.record_type,
                        "dir_type": dir_type,
                    }
                )
                continue
            return intermediate.prepared, "citing_intermediate", (conflicts or None)

    return None, None, (conflicts or None)


def _av_fields(manifest: dict, wavs: list[str], dirpath: Path) -> dict:
    if manifest.get("takes"):
        rows = [
            (entry.get("sample_rate"), entry.get("channels"), entry.get("seconds"))
            for entry in manifest["takes"]
        ]
    else:
        rows = []
        for name in wavs:
            header = _wav_header(dirpath / name)
            rows.append((header["sample_rate"], header["channels"], header["seconds"]))

    def pick(index: int):
        values = [row[index] for row in rows if row[index] is not None]
        uniq = sorted(set(values))
        if not uniq:
            return None
        return uniq[0] if len(uniq) == 1 else uniq

    return {"sample_rate": pick(0), "channels": pick(1), "seconds": pick(2)}


def _epoch(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _entry(relpath: Path, dirpath: Path, consumed: list[Hit], started: float) -> dict:
    manifest_path = dirpath / "takes-manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            manifest = {}

    wavs = sorted(p.name for p in dirpath.glob("*.wav"))
    listed = {
        entry.get("file") for entry in manifest.get("takes", ()) if isinstance(entry, dict)
    }

    dir_type, type_from, type_conflict = _resolve_type(manifest, relpath, consumed)
    address = _resolve_address(manifest, dirpath, consumed)
    settings, settings_from = _resolve_settings(manifest, wavs, consumed)
    stimuli = _resolve_stimuli(manifest, consumed)
    held, held_from, held_conflict = _resolve_held(manifest, dir_type, consumed)
    av = _av_fields(manifest, wavs, dirpath)

    stats = [(dirpath / name).stat() for name in wavs]
    total_bytes = sum(s.st_size for s in stats)
    mtimes = [s.st_mtime for s in stats]
    newest_mtime = max(mtimes) if mtimes else None
    seconds_per_take_observed = None
    if len(mtimes) > 1:
        seconds_per_take_observed = round((max(mtimes) - min(mtimes)) / (len(mtimes) - 1), 3)

    consumed_by = [{"record": hit.record_id, "via": hit.via} for hit in consumed]
    rewritten_after: list[str] = []
    unchecked: list[str] = []
    if newest_mtime is not None:
        for hit in consumed:
            if hit.measured_at is None:
                unchecked.append(hit.record_id)
                continue
            record_epoch = _epoch(hit.measured_at)
            if record_epoch is not None and newest_mtime > record_epoch:
                rewritten_after.append(hit.record_id)

    return {
        "type": dir_type,
        "type_from": type_from,
        "type_conflict": type_conflict,
        "address": address,
        "settings": settings,
        "settings_from": settings_from,
        "stimuli": stimuli,
        "held": held,
        "held_from": held_from,
        "held_conflict": held_conflict,
        "sample_rate": av["sample_rate"],
        "channels": av["channels"],
        "seconds": av["seconds"],
        "bytes": total_bytes,
        "newest_mtime": newest_mtime,
        "seconds_per_take_observed": seconds_per_take_observed,
        "consumed_by": consumed_by,
        "not_listed": sorted(set(wavs) - listed),
        "rewritten_after": rewritten_after,
        "unchecked": unchecked,
        "in_flight": bool(newest_mtime is not None and newest_mtime > started),
    }


def build(unit: str, root: str | Path = ".", *, started: float | None = None) -> dict:
    """The ledger: one entry per directory under `.cache/takes`, and who has read each.

    Reads the unit's whole published archive and every take's manifest (or, where
    one is missing, the WAV headers of what is in the directory), but never a
    take's sample data and never the hardware -- this is `needs_unit=False`.
    """
    # Resolved once, here: `_resolve_token` below always resolves a citation
    # to an absolute path before comparing it against `takes_root`, and a
    # directory found that way has to be the same Path object (by value) as
    # the one `_take_directories` walked to, or `hits.get(directory)` below
    # never matches and every entry reads as unread.
    root = Path(root).resolve()
    takes_root = root / TAKES_ROOT
    unit_dir = root / "data" / "units" / unit
    started = time.time() if started is None else started

    directories = _take_directories(takes_root)
    hits, unbound, cited = _scan_records(unit_dir, root, takes_root)

    entries = {}
    for directory in directories:
        relpath = directory.relative_to(takes_root)
        entries[relpath.as_posix()] = _entry(relpath, directory, hits.get(directory, []), started)

    return {
        "unit": unit,
        "built_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "directories": entries,
        "unbound_records": sorted(unbound, key=lambda u: u["record"]),
        "cited_cache_paths": sorted(cited),
    }


# --------------------------------------------------------------------------
# Queries over an already-built ledger. None of these touch `.cache/takes`
# again -- they read the one file `build` wrote.
# --------------------------------------------------------------------------


def load(root: str | Path = ".") -> dict | None:
    path = Path(root) / LEDGER_PATH
    if not path.exists():
        return None
    return json.loads(path.read_text())


def row(found: dict, type_: str, slot: str) -> dict:
    """Every directory entry matching a type and address."""
    wanted_type = norm_bytes(type_)
    wanted_addr = norm_addr(slot)
    return {
        relpath: entry
        for relpath, entry in found["directories"].items()
        if entry.get("type") == wanted_type and entry.get("address") == wanted_addr
    }


def unread(found: dict) -> list[tuple[str, dict]]:
    """Directories no published record's `consumed_by` reaches, largest first."""
    rows = [
        (relpath, entry)
        for relpath, entry in found["directories"].items()
        if not entry.get("consumed_by")
    ]
    rows.sort(key=lambda item: item[1].get("bytes", 0), reverse=True)
    return rows


def rewritten(found: dict) -> list[tuple[str, dict]]:
    """Directories whose newest take is younger than a record that read them."""
    return [
        (relpath, entry)
        for relpath, entry in found["directories"].items()
        if entry.get("rewritten_after")
    ]


__all__ = [
    "HELD_SOURCES",
    "LEDGER_PATH",
    "REASON_CODES",
    "TAKES_ROOT",
    "TYPE_SOURCES",
    "build",
    "load",
    "norm_addr",
    "norm_bytes",
    "row",
    "rewritten",
    "unread",
]
