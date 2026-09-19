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
from itertools import product
from pathlib import Path

#: Where what somebody else wrote down is held, beside `data/` and never inside
#: it. A row there is evidence that a page said so, which is a different kind of
#: claim from a measurement, and the directories are kept apart so that reading
#: one against the other stays a comparison rather than a merge.
DOCUMENTS = "documents"

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


def _answering(unit: Path) -> set[str]:
    """Every address an offsets record recorded an answer at.

    What a power-on capture has to cover, and the only list of it the archive
    holds: the address map bounds which blocks exist rather than which addresses
    do, so a capture counted against the map is counted against the narrower
    question the map asked.
    """
    out: set[str] = set()
    for path in _records(unit, "offsets"):
        found = _load(unit, f"offsets/{path.name}") or {}
        for block in found.get("blocks", []):
            out |= set(block.get("answered") or {})
    return out


def power_on(unit: Path) -> Stage:
    """A baseline for every address that answers, since it is the one reading that
    cannot be taken afterwards.

    Counted over addresses rather than over regions, and over every capture in
    the directory rather than one named file. Both because the bar was passing
    while it was neither.

    A region can be read and still come back short: one asked for thirty-two
    bytes and was answered with thirty, and the two addresses it left out answer
    when they are asked on their own. The capture then reports no region unread,
    which is true, while two of its addresses have no value -- so a count of
    regions cannot state what a later stage may read against, and a hole that
    only shows up address by address passes as a full capture.

    The union across captures is the right reading of the directory: a capture
    made over a wider map does not unmake the earlier one, and an address has a
    power-on value if any capture holds it. What it does not carry is agreement
    between them, which is a question about the unit rather than about coverage
    and belongs to whoever reads the two records.
    """
    captures = [_load(unit, f"power-on/{path.name}") or {} for path in _records(unit, "power-on")]
    if not captures:
        return _missing("power-on state", "power-on/")
    held: set[str] = set()
    unread = 0
    for capture in captures:
        held |= set(capture.get("values") or {})
        unread += capture.get("regions_unread") or 0
    answering = _answering(unit)
    without = answering - held
    if not answering:
        return Stage(
            "power-on state",
            UNDECIDED,
            f"{len(held)} addresses hold a power-on value, and no offsets record says which "
            "addresses answer, so there is nothing to hold the capture against",
        )
    if without:
        return Stage(
            "power-on state",
            UNMET,
            f"{len(without)} of the {len(answering)} addresses that answer a read hold no "
            "power-on value, and a power-on value cannot be taken later",
            {"addresses answering with no power-on value": len(without)},
        )
    return Stage(
        "power-on state",
        MET,
        f"{len(held)} addresses hold a power-on value, covering all {len(answering)} that "
        f"answer a read, over {len(captures)} captures with {unread} regions unread",
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
    # Only a block asked at every offset carries a shape. A later run asks a few
    # named offsets of a block the first one skipped, and the set that answered
    # there is an answer to a narrower question: read as a shape it would be a
    # small one, no other block would match it, and every stage folding over
    # shapes would be handed a representative it had never been asked to cover.
    whole = _asked(unit)
    grouped: dict[tuple, list[str]] = defaultdict(list)
    for block in found.get("blocks", []):
        base = " ".join(block["address"].split()[:2])
        if len(whole.get(base, ())) < BLOCK:
            continue
        answering = tuple(sorted(a.split()[-1] for a in block.get("answered", {})))
        if answering:
            grouped[answering].append(block["address"])
    return dict(grouped)


#: How many offsets a block holds. Seven bits, so 00 to 7F.
BLOCK = 128


def _asked(unit: Path) -> dict[str, set[int]]:
    """The offsets each block was put the question at, over every offsets record.

    A stage files more than one record. The first sweeps what the map held, and a
    later one asks the offsets a document names in blocks the first left alone --
    reading only the whole-map record reports those as never asked, which is the
    same defect, in the same directory, as the map that asked two third bytes
    under each block and was read as though it had asked all of them.

    A record names the range it asked by its first address and a count, not by a
    block and a count: `40 40 20` for three is offsets 20 to 22, not 00 to 02.
    """
    out: dict[str, set[int]] = defaultdict(set)
    for path in _records(unit, "offsets"):
        found = _load(unit, f"offsets/{path.name}") or {}
        for block in found.get("blocks", []):
            spelled = block["address"].split()
            start = int(spelled[2], 16)
            out[" ".join(spelled[:2])] |= set(range(start, start + block.get("offsets_asked", 0)))
    return dict(out)


def _representatives(shapes: dict[tuple, list[str]]) -> dict[str, tuple]:
    """One block per shape, and the offsets a stage has to cover in it."""
    return {sorted(blocks)[0].rsplit(" ", 1)[0]: offsets for offsets, blocks in shapes.items()}


def offsets_and_shapes(unit: Path) -> Stage:
    """One block of every kind the map holds, asked at every offset.

    Counted over kinds and not over blocks, which is the reading the stages
    around it already use. An address space repeats: on the unit this was settled
    against, thirty-one of the blocks the map holds are one kind, and a bar over
    blocks reports thirty-one pieces of work where there is one representative
    and a fold. The figure would then move when a machine has more parts rather
    than when more is known about it, and it would keep a unit short of finished
    for declining to sweep a block its documents give no function to -- which the
    scope this repository works to says not to ask.

    The kind is the one the sweep's own region sizes give, not the one the
    answers fall into. The shapes below are what a block's offsets turn out to
    hold, so nothing has them until it has been asked, and a bar folded by them
    could not say what was left to do. What the map holds before anything is
    asked is a length, and that is what a representative is picked by.
    """
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
    want = {
        " ".join(r["address"].split()[:2])
        for r in (mapped or {}).get("regions", [])
        if r["address"].split()[0] not in windows
    }
    # Asked over every offsets record rather than off the whole-map record's own
    # total, and by the offsets each block was actually put the question at rather
    # than by a tally. A later run asks the offsets a document names in blocks the
    # first one skipped, and reading the first record alone reports a block that has
    # since been asked as never asked -- the defect `_asked` exists to close, in the
    # same directory, and this was the one caller that did not use it.
    whole = {block for block, offsets in _asked(unit).items() if len(offsets) >= BLOCK}
    # A block asked at a few named offsets is not a block asked at every offset,
    # and both arrive as a row in this directory. Only the second can stand for
    # its kind: the first was put a narrower question than this stage asks.
    kinds = {
        kind: here
        for kind, blocks in block_kinds(unit).items()
        if (here := [block for block in blocks if block in want])
    }
    uncovered = {
        _kind_name(kind, here): len(here)
        for kind, here in kinds.items()
        if not any(block in whole for block in here)
    }
    shapes = _shapes(unit) or {}
    if mapped is not None and uncovered:
        return Stage(
            name,
            UNMET,
            f"{len(kinds) - len(uncovered)} of the map's {len(kinds)} kinds of block "
            "had one asked at every offset",
            {"kinds with no block asked at every offset": uncovered},
        )
    return Stage(
        name,
        UNDECIDED,
        f"{len(whole & want)} blocks asked at every offset, covering all {len(kinds)} "
        f"kinds the map holds and falling into {len(shapes)} shapes. Whether each "
        "shape's fold was checked against a second block of that shape is read from "
        "the record",
    )


_HEX = "0123456789ABCDEF"


def _covers(template: str, address: str) -> bool:
    """Whether one printed address covers a concrete one.

    A table writes a family of addresses as one row, putting a letter where the
    part, bank or program number goes: `40 1x 0A` is one statement about every
    part rather than sixteen statements. A letter matches any digit and a digit
    matches only itself, in any of the six places -- the low byte carries them
    too, and a reader that only expanded the middle one undercounted what the
    document names by a factor of five.
    """
    printed = template.rstrip("#")
    if len(printed) != len(address):
        return False
    for want, got in zip(printed, address, strict=True):
        if want == " ":
            if got != " ":
                return False
        elif want in _HEX:
            if got != want:
                return False
        elif got not in _HEX:
            return False
    return True


def _named_by_documents(unit: Path, meta: dict) -> dict[str, set[int]] | None:
    """The offsets each block's document rows name, or None if no document is held.

    A unit's record names the documents describing its model, and this walks up
    for the `documents/` tree beside `data/`. Which document belongs to a unit is
    read from the unit rather than guessed from its model name, because a
    document describes a model and this archive measures one unit of it.

    Templates are expanded here and nowhere else. What is stored stays as
    printed, so that the archive keeps holding one row where the document made
    one statement; expansion is a question asked of that row, not a rewriting of
    it.
    """
    named = [str(d) for d in meta.get("documents", [])]
    if not named:
        return None
    root = next((p for p in unit.parents if (p / DOCUMENTS).is_dir()), None)
    if root is None:
        return None
    templates: set[str] = set()
    for document in named:
        rows = _load(root / DOCUMENTS / document, "address-map.json") or {}
        templates |= {r["address"] for r in rows.get("rows", []) if r.get("address")}
    if not templates:
        return None

    out: dict[str, set[int]] = defaultdict(set)
    for template in templates:
        printed = template.rstrip("#")
        places = [i for i, c in enumerate(printed) if c != " " and c not in _HEX]
        for combination in product(_HEX, repeat=len(places)):
            spelled = list(printed)
            for place, digit in zip(places, combination, strict=True):
                spelled[place] = digit
            address = "".join(spelled)
            # An address byte carries seven bits, so the top one is never set and
            # a letter standing for a digit cannot make it so. Expanding over all
            # sixteen values of both places spells 80 to FF as well, which are not
            # addresses the unit has: counting them doubles every block the letter
            # reaches and reports the half that cannot exist as never asked.
            if any(int(byte, 16) > 0x7F for byte in address.split()):
                continue
            out[address[:5]].add(int(address[6:8], 16))
    return dict(out)


def documented_addresses(unit: Path) -> Stage:
    """Every address a published document gives a function to has been asked.

    This is the bar that closes a unit, and the only one whose yardstick is not
    the archive's own. The others are met against a map this repository produced,
    so a hole in that map is a hole in what they can notice -- the write probe
    reported every region of the map covered while three thousand six hundred and
    fifty addresses that answer a read sat outside it. A document was written
    before any of this and cannot be bent by it.

    Counted over one block per shape, like the stages around it. A row naming
    `40 1x 0A` names sixteen parts, and asking sixteen parts asks the same thing
    sixteen times; a bar that grew with the number of parts would say a unit was
    further from finished the more repetition its address space had.

    A block the document names and no shape holds is counted apart rather than
    folded away. It is the case that matters most: nothing was measured there, so
    there is no shape it could belong to, and folding by what has been measured
    would make exactly the unmeasured blocks invisible.

    Asked, not answered. A documented address that stays silent is an answer --
    the document says the model has something there and this unit did not give it
    back, which is a finding rather than a shortfall. What is outstanding is an
    address nobody put the question to.
    """
    name = "documented addresses"
    meta = _load(unit, "meta.json") or {}
    wanted = _named_by_documents(unit, meta)
    if wanted is None:
        return Stage(
            name,
            UNDECIDED,
            "this unit's record names no document held under documents/, so there is "
            "nothing outside the archive to count it against",
        )
    if _load(unit, f"offsets/{WHOLE_MAP}") is None:
        return _missing(name, f"offsets/{WHOLE_MAP}")
    asked = _asked(unit)
    mapped = _load(unit, f"sweep/{WHOLE_MAP}")
    exists = {" ".join(r["address"].split()[:2]) for r in (mapped or {}).get("regions", [])}
    shapes = _shapes(unit) or {}
    representing = {
        block: sorted(blocks)[0].rsplit(" ", 1)[0]
        for blocks in shapes.values()
        for block in [b.rsplit(" ", 1)[0] for b in blocks]
    }

    short: dict[str, int] = defaultdict(int)
    unasked: set[str] = set()
    for block, offsets in wanted.items():
        if exists and block not in exists:
            continue
        # The block itself first, then the block standing for its shape. A run
        # that named these offsets asked them here, and there is nothing to fold
        # when the question was put to this very block; folding is what saves the
        # other fifteen parts from being asked the same thing again.
        reached = asked.get(block, set()) | asked.get(representing.get(block, ""), set())
        missing = len(set(offsets) - reached)
        if not reached:
            unasked.add(block)
        elif missing:
            short[representing.get(block, block)] += missing

    if unasked or short:
        return Stage(
            name,
            UNMET,
            f"{len(wanted)} blocks carry a documented address; "
            f"{len(unasked)} of them were never asked at all",
            {
                "blocks the documents name that were never asked": sorted(unasked),
                "documented offsets not asked, by representative": dict(short),
            },
        )
    return Stage(
        name,
        MET,
        f"every offset named by a document was asked, over {len(wanted)} blocks folded "
        f"into {len(shapes)} shapes",
    )


def _addresses_in(region: dict) -> set[str]:
    """The addresses a region says the stage reached, however that stage spells them.

    Two stages walk the same map and neither writes the same region. The write
    probe lists a `bytes` entry per address with what it did there; the hold
    probe writes a whole region at once and keeps `given` and `read_back` as
    address-to-value maps, because what it measures is whether neighbours can be
    told apart, which is a property of the region rather than of one byte.

    Reading only one of those spellings does not fail. It reports the other
    stage as having covered nothing, which reads as a stage nobody ran -- and
    that was the state of `independent storage` while a hundred and seven of its
    addresses were genuinely unasked and seven hundred were not. A count that is
    wrong in the same direction as the truth is the hardest kind to notice.

    An address the run was told to leave alone counts as reached. It was decided
    rather than overlooked -- `NEVER_WRITE` is the standing example -- and a bar
    that counted it as outstanding could never be met by any amount of measuring.
    """
    found = {byte["address"] for byte in region.get("bytes", [])}
    for key in ("given", "read_back"):
        held = region.get(key)
        if isinstance(held, dict):
            found |= set(held)
    return found | {skipped.split(" (")[0] for skipped in region.get("skipped", [])}


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
        reached |= _addresses_in(region)
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
    refused = _parameters_refused(unit)
    if screened >= parameters:
        return Stage(
            "effect response",
            MET,
            f"the route is established and all {parameters} effect parameters carry a verdict",
        )
    # A slot whose run refused to answer is not a slot waiting to be asked, and
    # putting the two in one figure is what the refusal was written down to stop.
    # It still keeps the stage from being met: something the archive cannot say
    # is not the same as something it has said.
    also = f", and {refused} of them were asked and refused" if refused else ""
    return Stage(
        "effect response",
        UNMET,
        f"the route is established. {efx.get('accepted')} effect types carry "
        f"{sorted(slots)} parameter slots each, so {parameters} parameters need an "
        f"audible verdict and {screened} have one{also}",
        {
            k: v
            for k, v in (
                ("effect parameters to screen", parameters - screened - refused),
                ("effect parameters refused, which asking again the same way will not fill", refused),
            )
            if v
        },
    )


def _parameters_refused(unit: Path) -> int:
    """How many effect parameter slots were asked and would not be answered.

    Counted from the same records as the screened ones and kept apart from them.
    A refused slot and an unasked slot look identical in a total, and only one of
    them is work a later run can do.
    """
    seen = set()
    for path in _records(unit, "efx-params"):
        found = _load(unit, f"efx-params/{path.name}") or {}
        kind = str(found.get("type"))
        for row in (found.get("coverage") or {}).get("refused") or []:
            if isinstance(row, dict):
                seen.add((kind, str(row.get("parameter"))))
    return len(seen)


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
        # Last, because it is the one line of the bar whose yardstick came from
        # outside this repository, and a reader comparing the two wants the
        # archive's account of itself in front of them when they reach it.
        documented_addresses(unit),
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
