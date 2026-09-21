"""What was read out of the measurements, and whether it still stands.

`data/` holds what a unit answered. `documents/` holds what a page stated. This
holds the third kind of claim: what somebody concluded from those two, plus what
the era makes plausible and what anybody outside this project has reported. It is
the only kind here that can be wrong in a way no rerun would catch, so everything
this module enforces exists to make retracting one cheap.

**References run one way.** An inference cites a record; no record cites an
inference. That is the whole of the fence between the archive and the reading of
it, and it has to be physical rather than a convention, because the consumer it
protects is one that never reads a word of prose. Delete this directory and the
archive is unchanged.

**A citation carries the value it was made on.** Two of this archive's stages
have already been re-published with a figure moved by thirty-nine decibels and
another turned into a null, and a conclusion resting on the old number would have
gone on standing with nothing to say it had stopped being supported. So a
citation names the key and records what it held, and `stale` is a query rather
than a memory.

**The queue comes out of the alternatives.** What is worth measuring next is the
observable that separates a reading still standing from the claim, and an
alternative with no way to separate it is either equivalent -- which closes it --
or unanswerable with this rig, which is a result and is written as one.

**An alternative ends three ways, not two.** It falls, and says what felled it in
`ruled_out_by`. It stands, and says what would separate it. Or the measurement
comes back for it and the claim takes it up, which is `taken_up_by` and is not the
first of the three however convenient it would be to file it there: a reading that
turned out to be right, recorded under a key that says it was ruled out, is a file
that says the opposite of what was measured to anybody reading the keys. The claim
that takes one up is revised, so that is a round.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

SCHEMA_VERSION = 1

OPEN_STATES = {
    "standing_untested": {
        "what_is_open": (
            "The claim rests on the era, on a printed page or on somebody's report, "
            "and no measurement has been in a position to contradict it."
        ),
        "reason_at": ("could_have_been_refuted_by",),
    },
    "parked": {
        "what_is_open": (
            "The model was revised as far as this project revises one. What it found "
            "is kept and what it gave up stays open rather than being closed."
        ),
        "reason_at": ("inference", "why_parked"),
    },
}
"""States that are open whatever their alternatives say, and where each says why.

`standing_untested` is a claim resting on the era, on a printed page or on
somebody's report, which no measurement has yet been in a position to contradict.
It is counted as unresolved on purpose: plausibility cannot close anything here,
and a state that read as closed would let a catalogue of reasonable guesses stand
in for a measured one.

**The two reasons are different claims and live in different places.**
`could_have_been_refuted_by` sits beside `refuted_by` at the top of the file
because both are about what the claim says: one names the measurement that would
show it wrong, the other the measurement that could have and was never made.
`why_parked` sits under `inference` beside `rounds`, because parking is a fact
about how far the reading was taken and not about what it states. Reading both out
of the first key printed a parked claim's reason as `None` -- the one sentence
about it a queue exists to carry, dropped by the query meant to keep it.
"""

ROUND_CEILING = 3
"""How many times a claim's model may be revised before it is parked.

Three, which is the number this project has twice found sufficient for a constant
to stop looking like a coincidence. Past it the claim keeps its findings and stops
consuming attention, so that one type cannot absorb the work.
"""

_INDEXED = re.compile(r"^(?P<field>\w+)\[(?P<by>\w+=[^\]]+)\]$")
"""A row of a sweep named by its own fields, as `readings[value=52]`.

More than one field where one will not do. A record whose rows are one byte
written from two places has two rows saying `value=32`, and a citation that
named a row by the byte alone would point at whichever came first -- which is
the silent repointing this naming exists to prevent.
"""


def _parts(key: str) -> list[str]:
    """A dotted key cut into its steps, leaving the dots inside brackets alone.

    A row is named by one of its own fields, and a record names some of its rows
    by the take they came from -- which is a file name with a dot in it. Cutting
    on every dot puts half of that name in one step and half in the next, and the
    citation comes back unresolvable rather than wrong, so a claim resting on a
    figure that is really there reads as stale.
    """
    out, depth, here = [], 0, ""
    for char in key:
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        if char == "." and depth == 0:
            out.append(here)
            here = ""
            continue
        here += char
    out.append(here)
    return out


def resolve(record: dict, key: str):
    """One dotted key out of a record, including `readings[value=52].largest_db`.

    A citation has to be able to name a row of a sweep, because that is the
    granularity a conclusion is drawn at. Naming it by position would break the
    moment a run re-published with an extra setting in the middle -- which is the
    kind of change that must make a claim stale rather than silently repoint it
    at a different row.
    """
    here = record
    for part in _parts(key):
        found = _INDEXED.match(part)
        if found:
            rows = here[found["field"]]
            want = dict(
                pair.split("=", 1) for pair in found["by"].split(",")
            )
            here = next(
                row for row in rows
                if all(str(row.get(k)) == v for k, v in want.items())
            )
            continue
        here = here[part]
    return here


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def every(root: str | Path) -> list[Path]:
    """Every inference file: the JSON under a unit's directory, less its listing.

    The listing is generated from these and is not one of them. Reading it as one
    is how the first pass through here raised a missing-key error on a file it had
    written itself a moment earlier.
    """
    root = Path(root)
    return sorted(
        p for p in (root / "inferences").glob("*/*.json")
        if p.parent.name not in {"schema", "candidates", "models"} and p.name != "index.json"
    )


def stale(root: str | Path) -> list[dict]:
    """Claims resting on figures that have moved since they were made.

    Not a check that the record still exists -- that is a separate failure and a
    louder one. This is the quiet failure: the record was re-published, the number
    changed, and the conclusion drawn from the old one went on standing.
    """
    root = Path(root)
    out = []
    for path in every(root):
        claim = load(path)
        for citation in claim["rests_on"]["measurements"]:
            record_path = root / citation["file"]
            if not record_path.is_file():
                out.append(
                    {
                        "inference": str(path.relative_to(root)),
                        "file": citation["file"],
                        "why": "the record it cites is not in the archive",
                    }
                )
                continue
            record = json.loads(record_path.read_text())
            for key, was in zip(citation["keys"], citation["values"], strict=True):
                try:
                    now = resolve(record, key)
                except (KeyError, StopIteration, IndexError):
                    out.append(
                        {
                            "inference": str(path.relative_to(root)),
                            "file": citation["file"],
                            "key": key,
                            "was": was,
                            "why": "the key it cites is no longer in the record",
                        }
                    )
                    continue
                if now != was:
                    out.append(
                        {
                            "inference": str(path.relative_to(root)),
                            "file": citation["file"],
                            "key": key,
                            "was": was,
                            "now": now,
                            "why": "the record was re-published and this figure moved",
                        }
                    )
    return out


def comparison_set(root: str | Path, spec: dict) -> list[str]:
    """The records a claim was scored against, generated again from the index.

    A claim that chooses its comparison set by hand can be right about the records
    it looked at and wrong about the archive, so the sets here are generated. What
    that buys is only kept if the rule is written down as a rule: a list generated
    once and then committed goes on standing while records are added around it, and
    the claim that rests on it does not fail, it quietly stops being about the
    archive it says it is about. That happened -- three records published under a
    type turned a verdict over and the claim carrying the old one passed every
    check in this file, because the checks were about figures it cited and these
    were records it did not.

    Every field is optional except the stage. A filter naming nothing but the stage
    is a claim scored against every record of it, which is a thing a claim may be.
    """
    root = Path(root)
    index = json.loads((root / spec["index"]).read_text())
    unit = Path(spec["index"]).parent
    out = []
    for entry in index["stages"].get(spec["stage"], []):
        name = entry["file"]
        ends = spec.get("file_ends_with")
        if ends and not name.endswith(ends):
            continue
        record = json.loads((root / unit / name).read_text())
        if spec.get("type") and record.get("type") != spec["type"]:
            continue
        addresses = spec.get("address_in")
        if addresses and record.get("address") not in addresses:
            continue
        out.append(str(unit / name))
    return sorted(out)


def citing_an_inference(root: str | Path) -> list[dict]:
    """Records under data/ that name this directory, which none may.

    The fence is that references run one way. A record that gained a `see also`
    would make the archive's own files carry an interpretation, and every reader
    downstream would have to work out what to subtract -- which is the cost the
    separation exists to avoid.
    """
    root = Path(root)
    out = []
    for path in sorted((root / "data").rglob("*.json")):
        text = path.read_text()
        if "inferences/" in text:
            out.append({"file": str(path.relative_to(root)), "why": "it names inferences/"})
    return out


def already_recorded(root: str | Path, unit: str, run: dict, claim: dict) -> dict | None:
    """Whether the unit already holds records of what a queued run would go and ask.

    Two things land here and they want the same thing done about them. A run made
    and not yet folded in: the queue is derived from the claims, so a run stays on
    it until a claim is revised to take its answer in, and it has twice offered one
    long enough for hardware to be booked against it a second time. And a subject
    an earlier stage already swept for its own reasons, which is not a run that was
    made but is a record that has to be read before the time is booked. Neither is
    a booking, and the unit's listing is generated from its records, so this asks
    the listing rather than asking a reader to remember.

    Two conditions together, because either alone is wrong. Every side the run
    names has to have a record, or there is nothing to read first -- one side of a
    pair usually exists before the run, which is often why the pair was chosen, so
    one side alone says nothing. And at least one side has to have no record the
    claim cites anywhere, or what is here has already been read. Neither the run's
    own prose nor the claim's is consulted: a key saying a run was made is the
    thing that went stale.

    None where the run does not name a type and an address per side, or where the
    unit has no listing. A run that does not say what it is about cannot be looked
    up, and saying so is the answer.
    """
    kinds = run.get("types") or ([run["type"]] if run.get("type") else [])
    addresses = run.get("addresses") or ([run["address"]] if run.get("address") else [])
    if not kinds or len(kinds) != len(addresses):
        return None
    listing = Path(root) / "data" / "units" / unit / "index.json"
    if not listing.exists():
        return None
    entries = json.loads(listing.read_text())["stages"].get(run.get("stage")) or []
    named = json.dumps(claim, ensure_ascii=False)
    sides = []
    for kind, address in zip(kinds, addresses, strict=True):
        files = sorted(
            entry["file"]
            for entry in entries
            if (entry.get("about") or {}).get("type") == kind
            and (entry.get("about") or {}).get("address") == address
        )
        sides.append(
            {
                "side": f"{kind} {address}",
                "records": files,
                "the_claim_names": [f for f in files if f in named],
            }
        )
    if not all(side["records"] for side in sides):
        return None
    if all(side["the_claim_names"] for side in sides):
        return None
    return {"sides": sides}


def why_the_state_holds_it_open(claim: dict) -> str | None:
    """What a claim says about the state that holds it open, wherever it says it.

    None where the state closes nothing, and None where an open state says nothing --
    which is a defect in the claim rather than in the reading of it, and the layout
    suite is what refuses it.
    """
    state = claim["inference"]["state"]
    if state not in OPEN_STATES:
        return None
    found = claim
    for step in OPEN_STATES[state]["reason_at"]:
        if not isinstance(found, dict):
            return None
        found = found.get(step)
    return found


def open_items(root: str | Path) -> list[dict]:
    """What is unresolved, and for each the measurement that would resolve it.

    Ordered by what a run buys: candidates killed per minute of the unit
    sounding. An alternative whose `separated_by` is null is unresolved and
    unqueueable, and it is listed with the reason rather than left out -- a queue
    that silently drops what it cannot schedule reads as a shorter queue.

    A queued run the unit already holds records of is carried with
    `already_recorded` set rather than dropped. What it needs first is those
    records read, not the time booked, and those are different enough that the
    queue must not print them under one heading -- but an item that vanished when
    its subject turned up in the listing would be a question nobody was left to
    answer.

    **Both kinds of item carry the same two sentences**, `reading` and
    `why_there_is_no_run`, because the queue prints them the same way and a caller
    reading one kind's key off the other prints nothing. A claim open by its state
    and a claim held open by an alternative are open for different reasons, and that
    is what `why_open` is for; what a reader wants off either is what is unresolved
    and why no run is booked against it.
    """
    root = Path(root)
    out = []
    for path in every(root):
        claim = load(path)
        state = claim["inference"]["state"]
        name = str(path.relative_to(root))
        if state in {"retracted", "superseded"}:
            continue
        if state in OPEN_STATES:
            out.append(
                {
                    "inference": name,
                    "why_open": state,
                    "reading": OPEN_STATES[state]["what_is_open"],
                    "why_there_is_no_run": why_the_state_holds_it_open(claim),
                    "run": None,
                    "minutes": None,
                }
            )
            # A second item and not a field on the first. What a state holds open is
            # the whole of what the claim gave up, which is prose and carries no run;
            # what would reopen it is one booking with a margin, and the queue orders
            # by minutes. Folded into one item the booking would sort with the null
            # and print under the heading for things no run can answer, which is the
            # opposite of what it is.
            reopen = claim["inference"].get("what_would_reopen") or {}
            if reopen:
                run = reopen.get("run")
                out.append(
                    {
                        "inference": name,
                        "why_open": f"{state}, and a measurement would take it off the shelf",
                        # What is being decided, and never the observable twice over:
                        # the two readings a booking separates are the keys of its own
                        # `predicts`, and a queue that printed the measurement under
                        # both headings said nothing about which question it answers.
                        "reading": (
                            "whether " + " or ".join(reopen["predicts"])
                            if reopen.get("predicts")
                            else reopen.get("observable")
                        ),
                        "observable": reopen.get("observable"),
                        "margin": reopen.get("margin"),
                        "why_there_is_no_run": reopen.get("why_not_separable"),
                        "run": run,
                        "minutes": (run or {}).get("minutes"),
                        "already_recorded": (
                            already_recorded(root, path.parent.name, run, claim)
                            if run
                            else None
                        ),
                    }
                )
        for alternative in claim["alternatives"]:
            if not alternative.get("standing"):
                continue
            if alternative.get("equivalent_under_this_test"):
                continue
            separated = alternative.get("separated_by") or {}
            run = separated.get("run")
            out.append(
                {
                    "inference": name,
                    "why_open": "an alternative reading still stands",
                    "reading": alternative["reading"],
                    "observable": separated.get("observable"),
                    "margin": separated.get("margin"),
                    "why_there_is_no_run": separated.get("why_not_separable"),
                    "run": run,
                    "minutes": (run or {}).get("minutes"),
                    "already_recorded": (
                        already_recorded(root, path.parent.name, run, claim)
                        if run
                        else None
                    ),
                }
            )
    out.sort(key=lambda item: (item["minutes"] is None, item["minutes"] or 0))
    return out


def closed(root: str | Path) -> list[dict]:
    """Claims with nothing standing against them that a measurement could settle.

    An alternative marked equivalent does not hold a claim open. The test does
    not separate the two and saying so is the answer, not a deferral: a reader
    downstream may take either and the record says that is what they are doing.
    """
    root = Path(root)
    out = []
    for path in every(root):
        claim = load(path)
        if claim["inference"]["state"] != "standing":
            continue
        unresolved = [
            a for a in claim["alternatives"]
            if a.get("standing") and not a.get("equivalent_under_this_test")
        ]
        if unresolved:
            continue
        out.append(
            {
                "inference": str(path.relative_to(root)),
                # A claim whose `about` is short of a key is a defect the layout test
            # names; a query over every claim is not the place to raise it, because
            # then one malformed file hides the other forty.
            "types": claim["inference"]["about"].get("types", []),
                "claim": claim["claim"],
                "equivalent_readings": [
                    a["reading"] for a in claim["alternatives"]
                    if a.get("equivalent_under_this_test")
                ],
                "verdict": (claim.get("reproduces") or {}).get("verdict"),
            }
        )
    return out


WHY_THE_DENOMINATOR = (
    "Every parameter row the unit's own document prints against an insertion effect "
    "type, counted as a pair of the type and the address. That is the set the archive's "
    "own aiming rule names first, so it is what a fraction here is a fraction of. It is "
    "evidence about a page and not about the unit, which is the whole reason this is a "
    "query rather than a figure kept anywhere: nothing in `data/` and nothing in a claim "
    "is joined to it."
)

WHY_TWO_FIGURES = (
    "They are two questions and the gap between them is the answer to neither. `named` "
    "is every pair a claim says it is about, which for a claim running across types is "
    "the whole cross-product of its types and its addresses -- so it counts a column "
    "that was swept on one type and asserted over six. `touched` counts only pairs a "
    "record the claim cites names as its own subject, in that record's own `type` and "
    "`address`. One is a ceiling and the other a floor, and a single number quoted for "
    "coverage has always been one of the two with the other not mentioned. Both are "
    "here, and `named_and_not_touched` is what sits between them."
)

WHY_SOME_RECORDS_NAME_NO_ADDRESS = (
    "A record whose subject is a whole block rather than one address -- what every "
    "parameter of a type powers up holding, for instance -- carries no `address`, so it "
    "moves nothing in `touched` however much it read. The count of those is published "
    "beside the figure rather than being folded into it: a floor that quietly counted "
    "them would stop being a floor, and one that hides how many there are cannot be "
    "argued with."
)


def _printed_cells(root: Path, unit: str) -> tuple[dict, str]:
    """Every (type, address) an insertion effect list prints, and which document.

    The address is assembled from the block the effect parameters sit in and the low
    byte the list prints, because the list prints the low byte alone.
    """
    meta = json.loads((root / "data" / "units" / unit / "meta.json").read_text())
    named = (meta.get("documents") or [None])[0]
    if named is None:
        raise ValueError(f"{unit} names no document, so there is nothing to count against")
    listing = json.loads((root / "documents" / named / "effect-list.json").read_text())
    cells: dict[tuple[str, str], str] = {}
    for row in listing["rows"]:
        if "address_lsb" not in row:
            continue
        cells[(f"{row['msb']} {row['lsb']}", f"{EFFECT_BLOCK} {row['address_lsb']}")] = (
            row.get("effect", "")
        )
    return cells, named


EFFECT_BLOCK = "40 03"
"""The block an insertion effect's parameters sit in on this family.

The printed list gives each parameter's low byte and leaves the rest of the address to
the block figure at the front of the document, so the two are joined here rather than
in either file.
"""


def _touched_by(root: Path, claim: dict) -> set[tuple[str, str]]:
    """Every (type, address) a record this claim cites names as its own subject."""
    out = set()
    for cited in claim["rests_on"]["measurements"]:
        path = root / cited["file"]
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        kind, where = record.get("type"), record.get("address")
        if not kind or not where:
            continue
        for one in str(where).split(","):
            out.add((str(kind).strip(), one.strip()))
    return out


WHY_A_SIGNATURE_AND_NOT_AN_ADDRESS = (
    "A class is a printed quantity and never a column of the address space. The same "
    "low byte is a gain on one type, a cutoff on the next and a rate on the one after, "
    "so a reach worked out from a claim's addresses picks up rows that are a different "
    "parameter -- measured here, the largest class claim's addresses reach twenty-two "
    "printed quantities it says nothing about. What a row is matched on is therefore "
    "the parameter the page prints against it and the range it prints beside that, "
    "which is the pair a class is defined by."
)

WHY_THE_SIGNATURES_ARE_NOT_CUT = (
    "How many of the claim's own types a signature reaches is published rather than "
    "used as a bar. A claim's `addresses` is a union across its types, so a quantity "
    "sitting at one of them on two of them arrives here looking like the class's own -- "
    "a compressor's sustain under a claim about gains, for one. Cutting on the count "
    "would be choosing the class's definition after seeing what the cut produced, and "
    "the claim's own text is what defines it. The count is beside each signature so a "
    "reader can take the ones they mean."
)

WHY_A_REACH_IS_NOT_A_VERDICT = (
    "Nothing here is scored. What this returns is which published records lie inside a "
    "class claim's printed reach and outside what the claim names, which is the set any "
    "test of that reach has to start from and the part that must not be picked by eye. "
    "Scoring them is a separate question and not one the models in this project answer "
    "as they stand: a class model carries a type and a chain of that type's addresses, "
    "so rendering it on another type needs that type's chain and not a flag."
)


def reach(root: str | Path, unit: str) -> dict:
    """Where a class claim's printed quantity goes, against where the claim goes.

    A claim whose scope is a class says something about a quantity rather than about a
    type, so the question its scope raises is which of the printed rows carrying that
    quantity it was read on and which it was not. The answer is a set of records, and
    it is derived rather than listed: picking which readings a claim is held against by
    hand is the way a class passes without having been tested anywhere it might fail.
    """
    root = Path(root)
    cells, document = _printed_cells(root, unit)
    rows = json.loads(
        (root / "documents" / document / "effect-list.json").read_text()
    )["rows"]
    printed = {
        (f"{row['msb']} {row['lsb']}", f"{EFFECT_BLOCK} {row['address_lsb']}"): (
            row.get("parameter", ""),
            row.get("data", ""),
        )
        for row in rows
        if "address_lsb" in row
    }
    listing = json.loads((root / "data" / "units" / unit / "index.json").read_text())
    published: dict[tuple[str, str], list[str]] = {}
    for stage, entries in listing["stages"].items():
        for entry in entries:
            about = entry.get("about") or {}
            kind, where = about.get("type"), about.get("address")
            if not kind or not where:
                continue
            for one in str(where).split(","):
                published.setdefault((str(kind).strip(), one.strip()), []).append(
                    f"{stage}:{entry['file']}"
                )

    out = []
    for path in sorted((root / "inferences" / unit).glob("*.json")):
        if path.name == "index.json":
            continue
        claim = load(path)
        about = claim["inference"]["about"]
        if about.get("scope") != "class":
            continue
        cited = {c["file"] for c in claim["rests_on"]["measurements"]}
        named = {
            (kind, where)
            for kind in about.get("types", [])
            for where in about.get("addresses", [])
            if (kind, where) in printed
        }
        types_of: dict[tuple[str, str], set[str]] = {}
        for pair in named:
            types_of.setdefault(printed[pair], set()).add(pair[0])
        signatures = []
        for mark, types in sorted(types_of.items()):
            everywhere = {pair for pair, s in printed.items() if s == mark}
            outside = sorted(everywhere - named)
            signatures.append(
                {
                    "parameter": mark[0],
                    "printed": mark[1],
                    "reaches_named_types": len(types),
                    "printed_rows": len(everywhere),
                    "the_claim_names": len(everywhere) - len(outside),
                    "outside_the_claim": [
                        {
                            "type": kind,
                            "address": where,
                            "records": [
                                name for name in published.get((kind, where), [])
                                if name.split(":", 1)[1] not in cited
                            ],
                        }
                        for kind, where in outside
                    ],
                }
            )
        signatures.sort(key=lambda row: (-row["reaches_named_types"], row["parameter"]))
        out.append(
            {
                "inference": path.name,
                "state": claim["inference"]["state"],
                "types": about.get("types", []),
                "signatures": signatures,
                "records_outside_it": sorted(
                    {
                        name
                        for row in signatures
                        for cell in row["outside_the_claim"]
                        for name in cell["records"]
                    }
                ),
            }
        )
    return {
        "question": (
            "For each claim whose scope is a class, which printed rows carry the "
            "quantity it is about, which of them it names, and which published records "
            "of the rest it has never been held against."
        ),
        "unit_id": unit,
        "document_id": document,
        "printed_rows": len(cells),
        "why_a_signature_and_not_an_address": WHY_A_SIGNATURE_AND_NOT_AN_ADDRESS,
        "why_the_signatures_are_not_cut": WHY_THE_SIGNATURES_ARE_NOT_CUT,
        "why_a_reach_is_not_a_verdict": WHY_A_REACH_IS_NOT_A_VERDICT,
        "claims": out,
    }


def coverage(root: str | Path, unit: str) -> dict:
    """How much of what a document prints the claims are about, two ways round.

    Neither figure is the coverage. One counts what a claim asserts over and the other
    what its records name, and a reader who is given one of them alone cannot tell which
    they have -- which is how a figure half again too large stood in this project for
    months. Both are returned, with the pairs that sit between them.
    """
    root = Path(root)
    cells, document = _printed_cells(root, unit)
    named_by: dict[tuple[str, str], list[str]] = {}
    touched_by: dict[tuple[str, str], list[str]] = {}
    per_claim, no_address = [], 0
    for path in sorted((root / "inferences" / unit).glob("*.json")):
        if path.name == "index.json":
            continue
        claim = load(path)
        about = claim["inference"]["about"]
        named = {
            (kind, where)
            for kind in about.get("types", [])
            for where in about.get("addresses", [])
            if (kind, where) in cells
        }
        touched = {pair for pair in _touched_by(root, claim) if pair in cells}
        no_address += sum(
            1
            for cited in claim["rests_on"]["measurements"]
            if (root / cited["file"]).exists()
            and not json.loads((root / cited["file"]).read_text()).get("address")
        )
        for pair in named:
            named_by.setdefault(pair, []).append(path.name)
        for pair in touched:
            touched_by.setdefault(pair, []).append(path.name)
        per_claim.append(
            {
                "file": path.name,
                "state": claim["inference"]["state"],
                "types": len(about.get("types", [])),
                "addresses": len(about.get("addresses", [])),
                "named": len(named),
                "touched": len(touched),
                "named_and_not_touched": len(named - touched),
            }
        )

    per_type: dict[str, dict] = {}
    for (kind, where), effect in cells.items():
        row = per_type.setdefault(
            kind, {"type": kind, "effect": effect, "printed": 0, "named": 0, "touched": 0}
        )
        row["printed"] += 1
        row["named"] += (kind, where) in named_by
        row["touched"] += (kind, where) in touched_by

    def share(count: int) -> float | None:
        return round(count / len(cells), 4) if cells else None

    return {
        "question": (
            "How many of the parameters this unit's document prints against an insertion "
            "effect type the claims in this directory are about, counted two ways."
        ),
        "unit_id": unit,
        "document_id": document,
        "why_the_denominator": WHY_THE_DENOMINATOR,
        "why_two_figures": WHY_TWO_FIGURES,
        "printed": {"pairs": len(cells), "types": len({k for k, _ in cells})},
        "named": {"pairs": len(named_by), "of_the_printed": share(len(named_by))},
        "touched": {"pairs": len(touched_by), "of_the_printed": share(len(touched_by))},
        "named_and_not_touched": {
            "pairs": len(set(named_by) - set(touched_by)),
            "why": WHY_TWO_FIGURES,
        },
        "touched_and_not_named": {
            "pairs": len(set(touched_by) - set(named_by)),
            "why": (
                "The two are not nested, so neither figure is inside the other. A claim "
                "about one byte cites records of the bytes beside it as controls -- what "
                "the sweep was held against, what the same reading returns on a byte "
                "that should not move -- and those are pairs a record names and the "
                "claim is not about. Counting them into the second figure would make it "
                "a count of records looked at rather than of parameters read."
            ),
        },
        "records_naming_no_address": {
            "count": no_address,
            "why": WHY_SOME_RECORDS_NAME_NO_ADDRESS,
        },
        "types_with_nothing_named": sorted(
            row["type"] for row in per_type.values() if not row["named"]
        ),
        "types_with_nothing_touched": sorted(
            row["type"] for row in per_type.values() if not row["touched"]
        ),
        "why_two_lists_of_types": (
            "The first is types no claim says it is about; the second is types no record "
            "a claim cites names. The second is always the longer, and the difference is "
            "types a claim runs across and has no reading of on that type -- which is the "
            "same gap `named_and_not_touched` counts, read by type instead of by pair. A "
            "queue built from the first books time on a type the archive has already "
            "asserted over."
        ),
        "nothing_is_left_out": (
            "Every claim in the directory is counted, including the two that run across "
            "the whole of it. A figure that quietly dropped them is a different question "
            "answered under the same name, so `per_claim` carries each one's own "
            "contribution and a reader who wants them out can take them out."
        ),
        "per_type": sorted(per_type.values(), key=lambda row: row["type"]),
        "per_claim": per_claim,
    }


def index(root: str | Path, unit: str) -> dict:
    """One listing of a unit's inferences, derived from the files themselves.

    Same arrangement as the archive's own index and for the same reason: an index
    that has gone stale is a failing test rather than a paragraph somebody has to
    notice is wrong. What it says about a claim is what the claim says about
    itself -- the state, the types, the addresses. Never the conclusion, which is
    in the file named beside it.
    """
    root = Path(root)
    entries = []
    for path in sorted((root / "inferences" / unit).glob("*.json")):
        if path.name == "index.json":
            continue
        claim = load(path)
        head = claim["inference"]
        entries.append(
            {
                "file": path.name,
                "state": head["state"],
                "about": head["about"],
                "made_at": head["made_at"],
                "rounds": head.get("rounds", 0),
                "rests_on_records": [c["file"] for c in claim["rests_on"]["measurements"]],
            }
        )
    # Generated with the listing rather than kept beside it, so that a figure for how
    # much of a document the claims are about cannot be quoted from a paragraph that
    # stopped being true. It is dropped rather than raised where the unit names no
    # document or the document holds no effect list: a listing of the claims is still
    # a listing of the claims, and a round script that failed on it would be a claim
    # blocked by a file in another directory.
    try:
        counted = coverage(root, unit)
    except (FileNotFoundError, KeyError, ValueError) as why:
        counted = {"not_counted": str(why)}
    else:
        counted = {
            key: counted[key]
            for key in (
                "document_id",
                "why_the_denominator",
                "why_two_figures",
                "printed",
                "named",
                "touched",
                "named_and_not_touched",
                "touched_and_not_named",
                "records_naming_no_address",
                "types_with_nothing_named",
                "types_with_nothing_touched",
                "why_two_lists_of_types",
                "nothing_is_left_out",
            )
        }

    return {
        "index": {"schema_version": SCHEMA_VERSION, "unit_id": unit},
        "note": (
            "Generated from the inferences in this directory, one entry per file. Nothing here "
            "is a conclusion or a summary of one -- the file named in each entry holds it. What "
            "a claim rests on is listed so that a reader can go the other way, from a record to "
            "the readings made of it, without any record having to name one."
        ),
        "coverage": counted,
        "inferences": entries,
    }
