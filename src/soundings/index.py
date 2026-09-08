"""One listing of everything a unit's directory holds, generated from the records.

A directory of two hundred files answers "is it here" and not "what is here".
This answers the second, and it is derived rather than kept: every line comes from
the record it names, so an index that has gone stale is a failing test rather than
a paragraph somebody has to notice is wrong.

What it says about a record is only what the record says about itself -- the stage
that wrote it, and whichever of block, address, controller, channel or effect type
it states it is about. It does not summarise a finding, and it does not repeat a
method statement: a reader who wants either has the file named beside it, and a
summary that drifted from the record it summarised would be worse than none.

The stage is the directory, and the directory is the command that wrote the
record. That is the whole of the naming convention, and it is checkable: the
envelope inside a record carries the same word.
"""

from __future__ import annotations

import json
from pathlib import Path

#: How the listing was arrived at. In the file because a reader who finds it
#: disagreeing with the directory needs to know which of the two is derived.
NOTE = (
    "Generated from the records in this directory, one entry per file. The stage is "
    "the directory the record sits in, which is the command that wrote it; what the "
    "record is about is read from the record's own fields. Nothing here is a finding "
    "or a summary of one -- the file named in each entry is the record."
)

#: Fields a record uses to say what it is about, in the order they are looked for.
#: Read from the record rather than parsed out of its file name, so an entry
#: cannot say one thing while the record says another.
#:
#: A record's own summary line is deliberately not among them. It restates fields
#: that are already here and adds the values the run asked at, which is the
#: record's content rather than its subject -- and an index that begins carrying
#: content is an index that can be wrong about it.
SUBJECT = (
    "block",
    "address",
    "controller",
    "channel",
    "type",
    "type_id",
    "map_select",
    "region_prefix",
)

#: Records that are the unit rather than a measurement of it, so they sit at the
#: top of the directory instead of under a stage.
AT_THE_ROOT = ("meta.json", "measurements.json", "index.json")


def about(found: dict) -> dict:
    """What one record states it is about, in the record's own terms.

    A field left empty is dropped rather than shown blank. A scan that watched the
    whole map carries an empty region prefix, which says the run was not narrowed
    -- true, and not a thing the run is about.
    """
    return {key: found[key] for key in SUBJECT if found.get(key) not in (None, "")}


def entry(unit: Path, path: Path) -> dict:
    """One line of the listing, read from the record it names."""
    found = json.loads(path.read_text())
    stamp = found.get("record") if isinstance(found, dict) else None
    out: dict = {"file": str(path.relative_to(unit))}
    if isinstance(found, dict):
        if subject := about(found):
            out["about"] = subject
        if isinstance(stamp, dict):
            if stamp.get("measured_at"):
                out["measured_at"] = stamp["measured_at"]
            if stamp.get("stage"):
                out["written_by"] = stamp["stage"]
    return out


def survey(unit: str | Path) -> dict:
    """Every record the unit holds, grouped by the stage that wrote it."""
    unit = Path(unit)
    meta = json.loads((unit / "meta.json").read_text()) if (unit / "meta.json").exists() else {}
    stages: dict[str, list[dict]] = {}
    kept_by_hand = []
    for path in sorted(unit.rglob("*.json")):
        rel = path.relative_to(unit)
        if rel.parent == Path("."):
            if path.name not in AT_THE_ROOT:
                # Named rather than filed under a stage, because a record at the
                # top of a unit that is not one of the hand-kept ones has no
                # stage to be read against, and dropping it here would hide that.
                kept_by_hand.append(str(rel))
            continue
        stages.setdefault(rel.parts[0], []).append(entry(unit, path))
    return {
        "unit_id": meta.get("unit_id", unit.name),
        "note": NOTE,
        "records": sum(len(v) for v in stages.values()),
        "stages": {stage: stages[stage] for stage in sorted(stages)},
        **({"filed_under_no_stage": sorted(kept_by_hand)} if kept_by_hand else {}),
    }


def render(found: dict) -> str:
    """The listing as a person reads it, one stage at a time."""
    lines = [f"{found['unit_id']}  --  {found['records']} records"]
    for stage, entries in found["stages"].items():
        lines.append(f"\n{stage}/  ({len(entries)})")
        for row in entries:
            subject = ", ".join(f"{k} {v}" for k, v in row.get("about", {}).items())
            name = Path(row["file"]).name
            lines.append(f"  {name:<34}  {subject}" if subject else f"  {name}")
    if stray := found.get("filed_under_no_stage"):
        lines.append("\nfiled under no stage")
        lines.extend(f"  {name}" for name in stray)
    return "\n".join(lines)
