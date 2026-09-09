"""Counting a unit's directory against the bar for a finished one.

The bar is in the completing-a-unit page. This says where one unit stands against
it, so that what is left is read out of the records rather than remembered, and
so that a stage nobody ran is not mistaken for one that did not apply.

A unit's records sit in a directory named after the command that wrote them, so
what is looked for here is a stage's directory rather than a file whose name
happens to start with the right word. The difference matters: a glob over names
finds whatever a later run chose to call itself, and misses a record that answers
the stage under another name.

**Every check here is structural**: a file is present, a flag is set, one count
agrees with another. A structural pass is not a reading of the record's prose.
Where a criterion turns on something only a reader can settle -- whether a
control was sufficient, whether a null carried its bound -- this says it cannot
decide, rather than passing it. A false pass retires a stage that was never run,
which is the one failure a progress report must not have.

What is left is reported as counts, never as hours. A rate belongs to a chain and
a stimulus, and multiplying by one here would make a figure measured on one unit
look like a fact about the next.

Nothing below names an address, a block or a size. Every number comes from the
directory being counted.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

MET = "met"
UNMET = "not met"
UNDECIDED = "not decidable from the files alone"
EXCLUDED = "does not apply"

#: Where a unit says a stage has nothing to answer with, keyed by the stage names
#: below. Read rather than assumed: a stage missing from the directory and a
#: stage the machine cannot answer look identical in a listing.
EXCLUSIONS_KEY = "stages_that_do_not_apply"


@dataclass
class Stage:
    """One line of the bar, and where this unit stands against it."""

    name: str
    verdict: str
    evidence: str
    remaining: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        out = {"stage": self.name, "verdict": self.verdict, "evidence": self.evidence}
        if self.remaining:
            out["remaining"] = self.remaining
        return out


def _load(unit: Path, name: str) -> dict | None:
    try:
        return json.loads((unit / name).read_text())
    except (FileNotFoundError, ValueError):
        return None


def _missing(name: str, file: str) -> Stage:
    return Stage(name, UNMET, f"no {file} in this directory")


#: What a stage calls the record that covers the whole of the address map, as
#: against one covering a block or an address. Several stages have one each.
WHOLE_MAP = "whole-map.json"


def _records(unit: Path, stage: str) -> list[Path]:
    """Every record one stage left behind, which is its directory's contents.

    A stage that never ran and one whose records were named something else are
    the same absence here, and that is the intended reading: a record answers for
    a stage by being filed under it, not by being called after it.
    """
    return sorted((unit / stage).glob("*.json"))


def identity(unit: Path) -> Stage:
    """The plate, the chain and the reply, or a statement that one is unresolved."""
    meta = _load(unit, "meta.json")
    if meta is None:
        return _missing("identity", "meta.json")
    wanted = [
        "identity_reply",
        "specifications_claimed",
        "rear_selector",
        "measurement_chain",
    ]
    absent = [k for k in wanted if not meta.get(k)]
    if absent:
        return Stage("identity", UNMET, f"meta.json carries nothing under {', '.join(absent)}")
    return Stage("identity", MET, f"meta.json carries {', '.join(wanted)}")


def address_map(unit: Path) -> Stage:
    """The sweep's own account of itself, which is the only one that can be checked."""
    found = _load(unit, f"sweep/{WHOLE_MAP}")
    if found is None:
        return _missing("address map", f"sweep/{WHOLE_MAP}")
    if not found.get("complete") or not found.get("trustworthy") or found.get("aborted"):
        return Stage(
            "address map",
            UNMET,
            "the sweep reports complete={}, trustworthy={}, aborted={}".format(
                found.get("complete"), found.get("trustworthy"), found.get("aborted")
            ),
        )
    tops = sorted({r["address"].split()[0] for r in found.get("regions", [])})
    return Stage(
        "address map",
        MET,
        f"{len(found['regions'])} regions over {len(tops)} top bytes, "
        "reported complete and trustworthy",
    )


def power_on(unit: Path) -> Stage:
    """A capture with nothing left unread, since an unread region has no baseline."""
    found = _load(unit, f"power-on/{WHOLE_MAP}")
    if found is None:
        return _missing("power-on state", f"power-on/{WHOLE_MAP}")
    unread = found.get("regions_unread")
    if unread:
        return Stage(
            "power-on state",
            UNMET,
            f"{unread} regions were not read, so they have no power-on value to differ from",
            {"regions unread": unread},
        )
    return Stage(
        "power-on state",
        MET,
        f"{found.get('regions_read')} regions read, none left unread",
    )


def _shapes(unit: Path) -> dict[tuple, list[str]] | None:
    """Blocks grouped by the set of offsets that answered in them.

    An address space repeats, and by a large factor: on the first unit counted
    this way 461 blocks answered and held twelve shapes between them, one of
    them 204 times. Counting a stage's coverage over addresses would then be
    mostly counting how many times it measured the same thing, and the figure
    would move when a unit has more parts rather than when more is known.

    Grouped from the offsets record rather than the address map, because the
    map bounds which blocks exist and not which addresses do: it asks a named
    set of third bytes under each block, so a region beginning elsewhere is
    absent from it. A stage counted against the map is counted against a
    narrower question than it asked.
    """
    found = _load(unit, f"offsets/{WHOLE_MAP}")
    if found is None:
        return None
    grouped: dict[tuple, list[str]] = defaultdict(list)
    for block in found.get("blocks", []):
        answering = tuple(sorted(a.split()[-1] for a in block.get("answered", {})))
        if answering:
            grouped[answering].append(block["address"])
    return dict(grouped)


def _representatives(shapes: dict[tuple, list[str]]) -> dict[str, tuple]:
    """One block per shape, and the offsets a stage has to cover in it."""
    return {sorted(blocks)[0].rsplit(" ", 1)[0]: offsets for offsets, blocks in shapes.items()}


def offsets_and_shapes(unit: Path) -> Stage:
    """Every offset of every block asked, and the shapes those answers fall into."""
    name = "offsets and shapes"
    found = _load(unit, f"offsets/{WHOLE_MAP}")
    if found is None:
        return _missing(name, f"offsets/{WHOLE_MAP}")
    if found.get("stopped"):
        return Stage(
            name,
            UNMET,
            "the run stopped before it finished, so its silences are not vouched for",
            {"blocks asked before it stopped": found.get("blocks_asked", 0)},
        )
    mapped = _load(unit, f"sweep/{WHOLE_MAP}")
    # A block measured to be a window onto another block holds no store of its
    # own, so asking its offsets asks the store it points at once more. Counting
    # those as unasked would leave this unmeetable by measuring, and the only way
    # to meet it would be to measure the same thing again under a name that
    # keeps none of it.
    windows = {
        top
        for finding in (mapped or {}).get("findings", [])
        if finding.get("kind") == "blocks-that-are-a-window"
        for top in finding.get("blocks", [])
    }
    want = len(
        {
            " ".join(r["address"].split()[:2])
            for r in (mapped or {}).get("regions", [])
            if r["address"].split()[0] not in windows
        }
    )
    got = found.get("blocks_asked", 0)
    shapes = _shapes(unit) or {}
    if mapped is not None and got < want:
        return Stage(
            name,
            UNMET,
            f"{got} of the map's {want} blocks were asked at every offset",
            {"blocks not asked": want - got},
        )
    return Stage(
        name,
        UNDECIDED,
        f"{got} blocks asked, falling into {len(shapes)} shapes. Whether each shape's "
        "fold was checked against a second block of that shape is read from the record",
    )


def _covers_the_shapes(unit: Path, stage: str, name: str) -> Stage:
    """A whole-space stage is complete when it covered one block of every shape.

    Measuring the rest of a shape's blocks measures the same thing again, so
    they are not counted for or against. What is counted is whether every shape
    has a block this stage reached, and every offset of it.

    Every record the stage filed counts, not one named whole-map: a space the
    map missed is reached by a later run against a different file, and reading
    only the first would report the addresses it added as never covered.
    """
    records = _records(unit, stage)
    if not records:
        return _missing(name, f"{stage}/")
    found = {
        "regions": [
            region
            for path in records
            for region in (_load(unit, f"{stage}/{path.name}") or {}).get("regions", [])
        ]
    }
    shapes = _shapes(unit)
    if shapes is None:
        return Stage(
            name,
            UNDECIDED,
            f"there is no offsets/{WHOLE_MAP} to group the space into shapes, and the "
            "address map alone bounds which blocks exist rather than which addresses do",
        )
    reached: set[str] = set()
    for region in found.get("regions", []):
        for byte in region.get("bytes", []):
            reached.add(byte["address"])
        for skipped in region.get("skipped", []):
            reached.add(skipped.split(" (")[0])
    want = _representatives(shapes)
    short = {
        block: sum(1 for off in offsets if f"{block} {off}" not in reached)
        for block, offsets in want.items()
    }
    missing = {block: n for block, n in short.items() if n}
    if missing:
        return Stage(
            name,
            UNMET,
            f"{len(want) - len(missing)} of the {len(want)} shapes were covered at "
            "every offset of their representative block",
            {"offsets not covered, by representative": missing},
        )
    return Stage(name, MET, f"every offset of all {len(want)} shapes' representatives was covered")


def accepted_values(unit: Path) -> Stage:
    return _covers_the_shapes(unit, "write-probe", "accepted values")


def independent_storage(unit: Path) -> Stage:
    return _covers_the_shapes(unit, "hold-probe", "independent storage")


def tones_and_effects(unit: Path) -> Stage:
    """A tone map per map-select the unit accepts, and every effect type asked."""
    maps = _records(unit, "tone-map")
    efx = _load(unit, "efx-map/types.json")
    if not maps:
        return _missing("tones and effects", "tone-map/")
    if efx is None:
        return _missing("tones and effects", "efx-map/types.json")
    asked, accepted = efx.get("asked"), efx.get("accepted")
    if efx.get("unanswered"):
        return Stage(
            "tones and effects",
            UNMET,
            f"{len(efx['unanswered'])} effect types went unanswered",
            {"effect types unanswered": len(efx["unanswered"])},
        )
    return Stage(
        "tones and effects",
        MET,
        f"{len(maps)} tone maps, and {asked} effect types asked of which {accepted} were accepted",
    )


def repeatability(unit: Path) -> Stage:
    """A floor for this unit on this chain, which every later figure is read against."""
    floors = _records(unit, "repeat")
    if not floors:
        return _missing("repeatability", "repeat/")
    return Stage(
        "repeatability",
        MET,
        f"{len(floors)} floors measured on this chain: {', '.join(f.stem for f in floors)}",
    )


def block_kinds(unit: Path) -> dict[tuple, list[str]]:
    """Blocks grouped by the sizes of their regions, as a stand-in for their kind.

    Blocks whose regions run to the same lengths in the same order are counted
    once, so that sweeping one of them answers for the rest. This is what folds a
    window back into the store it looks onto, and what keeps sixteen parts from
    being counted as sixteen kinds of work.

    It is a proxy in both directions, and it is reported as one. Two blocks with
    unrelated purposes that happen to be the same length are merged here; the
    sweep records are what would show it, since a block swept under one kind and
    disagreeing with its peers is the disagreement the whole-blocks stage asks
    about. Grouping by contents instead would be worse: parts differ in the
    channel byte each of them holds, so no two would ever merge.
    """
    shape: dict[str, list[int]] = defaultdict(list)
    mapped = _load(unit, f"sweep/{WHOLE_MAP}")
    for region in (mapped or {}).get("regions", []):
        top, second, _ = region["address"].split()
        shape[f"{top} {second}"].append(region["size"])
    kinds: dict[tuple, list[str]] = defaultdict(list)
    for block, sizes in shape.items():
        kinds[tuple(sizes)].append(block)
    return kinds


def _kind_name(kind: tuple, blocks: list[str]) -> str:
    """A kind named by a block in it, since the key is a shape rather than a name."""
    seen = f"{blocks[0]}, sizes {list(kind)}"
    return seen if len(blocks) == 1 else f"{seen}, and {len(blocks) - 1} of the same shape"


def whole_blocks(unit: Path) -> Stage:
    """One block of each kind swept in full, the rest named rather than swept."""
    if _load(unit, f"sweep/{WHOLE_MAP}") is None:
        return _missing("whole blocks", f"sweep/{WHOLE_MAP}")
    swept = set()
    # Over the whole unit rather than one stage: a block is swept by whatever
    # named a block, and reading only the fold's own directory would miss a
    # record that covered one on the way to answering something else.
    for path in unit.rglob("*.json"):
        found = _load(unit, path.relative_to(unit))
        if isinstance(found, dict) and isinstance(found.get("block"), str):
            swept.add(found["block"])
    kinds = block_kinds(unit)
    unswept = {kind: blocks for kind, blocks in kinds.items() if not (set(blocks) & swept)}
    if not unswept:
        return Stage(
            "whole blocks",
            MET,
            f"{len(swept)} blocks swept, covering all {len(kinds)} shapes the map reports",
        )
    addresses = sum(sum(kind) for kind in unswept)
    return Stage(
        "whole blocks",
        UNMET,
        f"{len(swept)} blocks swept, covering {len(kinds) - len(unswept)} of the "
        f"{len(kinds)} kinds the map reports. Blocks whose regions run to the same "
        "lengths are counted once, which is a proxy for their being one kind",
        {
            "kinds not swept": sorted(_kind_name(k, b) for k, b in unswept.items()),
            "addresses in one block of each": addresses,
        },
    )


def effect_response(unit: Path) -> Stage:
    """The route first, then an audible verdict for every effect parameter.

    The verdict is counted at the parameter. A screen that admits every effect
    type has turned nothing away, and the count of parameters left to ask stands
    exactly where it started.
    """
    efx = _load(unit, "efx-map/types.json")
    route = _records(unit, "transfer")
    if efx is None:
        return _missing("effect response", "efx-map/types.json")
    if not route:
        return Stage(
            "effect response",
            UNMET,
            "no transfer record, so whether the analogue input reaches the effects is unasked",
        )
    slots = {len(e.get("parameters", [])) for e in efx.get("effects", [])}
    parameters = sum(len(e.get("parameters", [])) for e in efx.get("effects", []))
    screened = _parameters_screened(unit)
    if screened >= parameters:
        return Stage(
            "effect response",
            MET,
            f"the route is established and all {parameters} effect parameters carry a verdict",
        )
    return Stage(
        "effect response",
        UNMET,
        f"the route is established. {efx.get('accepted')} effect types carry "
        f"{sorted(slots)} parameter slots each, so {parameters} parameters need an "
        f"audible verdict and {screened} have one",
        {"effect parameters to screen": parameters - screened},
    )


def _parameters_screened(unit: Path) -> int:
    """How many effect parameters carry an audible verdict of their own.

    A verdict about an effect type is not a verdict about its parameters, so a
    record that sorts types is not counted here.
    """
    seen = set()
    for path in unit.rglob("*.json"):
        found = _load(unit, path.relative_to(unit))
        if not isinstance(found, dict):
            continue
        for row in found.get("parameters", []):
            if isinstance(row, dict) and "audible" in row and "type" in row:
                seen.add((str(row["type"]), str(row.get("parameter"))))
    return len(seen)


def _by_reading(name: str, unit: Path, *stages: str) -> Stage:
    """A stage whose bar is about controls and bounds, which a file listing cannot see.

    Several of these are answered by more than one command -- an audible verdict
    is reached at an address, at a block and at an effect parameter -- so the
    directories are given together rather than one line of the bar per command.
    """
    present = [p for stage in stages for p in _records(unit, stage)]
    where = ", ".join(f"{stage}/" for stage in stages)
    if not present:
        return Stage(name, UNMET, f"nothing under {where} in this directory")
    return Stage(
        name,
        UNDECIDED,
        f"{len(present)} records under {where}. Whether the controls and the coverage "
        "meet the bar is read from the record, not from the file being there",
    )


def survey(unit: Path) -> dict:
    """Where this unit stands against every line of the bar."""
    unit = Path(unit)
    meta = _load(unit, "meta.json") or {}
    excluded = meta.get(EXCLUSIONS_KEY, {})
    stages = [
        identity(unit),
        address_map(unit),
        offsets_and_shapes(unit),
        power_on(unit),
        _by_reading("windows", unit, "window-probe"),
        accepted_values(unit),
        independent_storage(unit),
        _by_reading("aliases", unit, "alias-scan"),
        _by_reading("resets", unit, "reset-probe"),
        tones_and_effects(unit),
        repeatability(unit),
        _by_reading("audible differences", unit, "contrast", "block", "efx-params"),
        whole_blocks(unit),
        effect_response(unit),
    ]
    stages = [
        Stage(s.name, EXCLUDED, str(excluded[s.name])) if s.name in excluded else s for s in stages
    ]
    remaining: dict = {}
    for stage in stages:
        for what, count in stage.remaining.items():
            remaining[what] = count
    return {
        "unit_id": meta.get("unit_id", unit.name),
        "note": "Structural checks only. A stage reported met was not read, it was counted.",
        "counts_not_hours": (
            "What is left is given as counts. A rate belongs to a chain and a stimulus, "
            "and one measured on this unit is not a fact about the next."
        ),
        "stages": [s.to_json() for s in stages],
        "verdicts": {
            v: sum(1 for s in stages if s.verdict == v) for v in (MET, UNMET, UNDECIDED, EXCLUDED)
        },
        "remaining": remaining,
        "complete": all(s.verdict in (MET, EXCLUDED) for s in stages),
    }


def render(found: dict) -> str:
    """The survey as a person reads it, ordered as the bar is."""
    lines = [f"{found['unit_id']}"]
    width = max(len(s["stage"]) for s in found["stages"])
    for stage in found["stages"]:
        lines.append(f"  {stage['stage']:<{width}}  {stage['verdict']}")
        lines.append(f"  {'':<{width}}  {stage['evidence']}")
    if found["remaining"]:
        lines.append("\nleft to measure")
        for what, count in found["remaining"].items():
            lines.append(f"  {what}: {count}")
    lines.append("\ncomplete" if found["complete"] else "\nnot complete")
    return "\n".join(lines)
