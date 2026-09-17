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


#: Where a record that holds several runs puts each one. A run of one of these
#: carries its own `about`, so a record whose whole subject is a list still says
#: what each part of it was -- three types at one address, or one type under three
#: voices.
WITHIN = ("runs", "findings", "readings", "results")


def also_about(found: dict) -> list[dict]:
    """The subjects a record states one run at a time, where it states none itself.

    Read only when the top level names nothing, because that is the case this is
    for: a record about a list of parameters has no one subject to put at its top
    and would otherwise read as a record about nothing at all. It is still the
    record speaking about itself -- the same fields, one level down -- and a
    coverage figure that skipped these would count three parameters as none.
    """
    seen: list[dict] = []
    for key in WITHIN:
        for row in found.get(key) or []:
            if not isinstance(row, dict) or not isinstance(row.get("about"), dict):
                continue
            if (subject := about(row["about"])) and subject not in seen:
                seen.append(subject)
    return seen


def entry(unit: Path, path: Path) -> dict:
    """One line of the listing, read from the record it names."""
    found = json.loads(path.read_text())
    stamp = found.get("record") if isinstance(found, dict) else None
    out: dict = {"file": str(path.relative_to(unit))}
    if isinstance(found, dict):
        if subject := about(found):
            out["about"] = subject
        elif nested := also_about(found):
            out["also_about"] = nested
        if isinstance(stamp, dict):
            if stamp.get("measured_at"):
                out["measured_at"] = stamp["measured_at"]
            if stamp.get("stage"):
                out["written_by"] = stamp["stage"]
    return out


def names_it(value: object) -> bool:
    """Whether a subject field names the record's subject or describes its reading.

    `channel` is both, depending on the stage that wrote it. On a scan it is the
    MIDI channel a message was sent on, which is what the record is about; on a
    stage that reads audio it is an object saying which interface input every
    figure was taken from and how loud the others were, which is a fact about the
    reading. A name is a name; anything with structure inside it is not one.
    """
    return isinstance(value, str | int) and not isinstance(value, bool)


def as_a_subject(about: dict) -> str:
    """One record's subject as a single key, so two records about one thing meet.

    Empty where the record names no subject, which is not the same as a record
    whose subject happens to be rare -- so the caller keeps those apart rather
    than filing them all under one blank.

    **One field, one name.** `decay` and `phase` once spelled the effect type
    `type_id` where the other stages spelled it `type`, and a fold from one to the
    other stood here so that a phase reading and a band profile of one parameter
    met rather than reading as two parameters -- a coverage figure counting the
    same work twice and the same gap not at all. Both stages now write `type` and
    their records have been re-published from the command lines they carry, so
    there is one spelling and nothing to fold. A second name kept alive here would
    be a second convention with nothing on the other end of it.
    """
    said = {key: about[key] for key in SUBJECT if key in about and names_it(about[key])}
    return ", ".join(f"{key} {said[key]}" for key in SUBJECT if key in said)


def subjects(stages: dict[str, list[dict]]) -> dict:
    """The same records grouped by what they are about rather than by who wrote them.

    The listing is by stage because that is how the directory is laid out. A
    question about coverage is not a question about the directory: it asks which
    parameter of which type has been read, and one subject is reached from several
    stages -- a rate from one, a band profile from another, a balance from a third.
    So the entries are turned the other way round here and the count falls out of
    the grouping rather than being kept beside it.

    **The records that name no subject are listed and not counted.** A coverage
    figure whose denominator quietly drops what it could not read is the failure
    this section exists to show, and the archive already has thirteen of them: a
    stage that takes only a directory of takes has nothing on its command line
    that says what the takes were of, so the subject survives in the file name and
    in prose and nowhere a query can reach.
    """
    grouped: dict[str, dict] = {}
    silent: list[str] = []
    for stage, entries in stages.items():
        for entry in entries:
            said = [entry["about"]] if entry.get("about") else entry.get("also_about") or []
            keys = [key for key in (as_a_subject(one) for one in said) if key]
            if not keys:
                silent.append(entry["file"])
                continue
            for key in keys:
                row = grouped.setdefault(key, {"stages": [], "records": []})
                if entry["file"] not in row["records"]:
                    row["records"].append(entry["file"])
                if stage not in row["stages"]:
                    row["stages"].append(stage)
    for row in grouped.values():
        row["records"].sort()
        row["stages"].sort()
    named = sum(len(row["records"]) for row in grouped.values())
    return {
        "note": (
            "Generated from the same entries the listing above holds, grouped by the "
            "subject each record states. A record naming no subject is in "
            "`naming_no_subject` and in no group: it is an absence and not a refusal, "
            "and a count that absorbed it would be a count of what was easy to read."
        ),
        "subjects": len(grouped),
        "records_naming_one": named,
        "naming_no_subject": sorted(silent),
        "by_subject": {key: grouped[key] for key in sorted(grouped)},
    }


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
    ordered = {stage: stages[stage] for stage in sorted(stages)}
    return {
        "unit_id": meta.get("unit_id", unit.name),
        "note": NOTE,
        "records": sum(len(v) for v in stages.values()),
        "stages": ordered,
        **({"filed_under_no_stage": sorted(kept_by_hand)} if kept_by_hand else {}),
        "what_they_are_about": subjects(ordered),
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
    if about := found.get("what_they_are_about"):
        both = [
            key
            for key in about["by_subject"]
            if key.startswith("address ") and ", type " in key
        ]
        lines.append(
            f"\n{about['subjects']} subjects over {about['records_naming_one']} records, "
            f"{len(both)} of them one address of one effect type"
        )
        if silent := about["naming_no_subject"]:
            # Printed rather than counted. Each of these is a record whose subject
            # survives in its file name and in prose and nowhere a query reaches,
            # and a number would not tell a reader which stage to go and fix.
            lines.append(f"\n{len(silent)} records name no subject a query can read")
            lines.extend(f"  {name}" for name in silent)
    return "\n".join(lines)
