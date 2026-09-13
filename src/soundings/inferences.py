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
"""

from __future__ import annotations

import json
import re
from pathlib import Path

SCHEMA_VERSION = 1

OPEN_STATES = frozenset({"standing_untested", "parked"})
"""States that are open whatever their alternatives say.

`standing_untested` is a claim resting on the era, on a printed page or on
somebody's report, which no measurement has yet been in a position to contradict.
It is counted as unresolved on purpose: plausibility cannot close anything here,
and a state that read as closed would let a catalogue of reasonable guesses stand
in for a measured one.
"""

ROUND_CEILING = 3
"""How many times a claim's model may be revised before it is parked.

Three, which is the number this project has twice found sufficient for a constant
to stop looking like a coincidence. Past it the claim keeps its findings and stops
consuming attention, so that one type cannot absorb the work.
"""

_INDEXED = re.compile(r"^(?P<field>\w+)\[(?P<key>\w+)=(?P<value>[^\]]+)\]$")


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
            want = found["value"]
            here = next(
                row for row in rows
                if str(row.get(found["key"])) == want
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


def open_items(root: str | Path) -> list[dict]:
    """What is unresolved, and for each the measurement that would resolve it.

    Ordered by what a run buys: candidates killed per minute of the unit
    sounding. An alternative whose `separated_by` is null is unresolved and
    unqueueable, and it is listed with the reason rather than left out -- a queue
    that silently drops what it cannot schedule reads as a shorter queue.
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
                    "could_have_been_refuted_by": claim.get("could_have_been_refuted_by"),
                    "run": None,
                    "minutes": None,
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
                    "why_not_separable": separated.get("why_not_separable"),
                    "run": run,
                    "minutes": (run or {}).get("minutes"),
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
                "types": claim["inference"]["about"]["types"],
                "claim": claim["claim"],
                "equivalent_readings": [
                    a["reading"] for a in claim["alternatives"]
                    if a.get("equivalent_under_this_test")
                ],
                "verdict": (claim.get("reproduces") or {}).get("verdict"),
            }
        )
    return out


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
    return {
        "index": {"schema_version": SCHEMA_VERSION, "unit_id": unit},
        "note": (
            "Generated from the inferences in this directory, one entry per file. Nothing here "
            "is a conclusion or a summary of one -- the file named in each entry holds it. What "
            "a claim rests on is listed so that a reader can go the other way, from a record to "
            "the readings made of it, without any record having to name one."
        ),
        "inferences": entries,
    }
