"""The hand-kept behaviours, held against the records that established them.

`measurements.json` is written by hand. Every other gate here compares one
published record with another, and this file is not one of them -- so it is the
one place in a unit's directory where a claim can go on disagreeing with the
measurements beside it, indefinitely and silently.

It has. Four entries were written from scans that watched the sweep map, which
asks two third bytes under each block and so leaves 3650 answering addresses
outside every region it names. Re-running the scans over both kinds of read
found three stores in that gap, and the summary of one of those entries -- its
identifier included -- said twelve where the records now say thirteen.

Only the shape a machine can check is checked here: a behaviour that lists which
stimulus is stored at which address is a table, and a table has to agree with
the scans. The prose around it is not, and cannot be, checked this way.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

UNITS = Path(__file__).resolve().parents[1] / "data" / "units"


def _units() -> list[Path]:
    return sorted(p for p in UNITS.iterdir() if p.is_dir())


def _attributed(unit: Path) -> dict[str, set[str]]:
    """Every stimulus any alias scan of this unit attributed, and where it landed.

    Over every scan rather than the widest one: which scan reached furthest is
    itself a question about the maps they watched, and a behaviour naming a store
    that any of them found is a behaviour agreeing with the archive.
    """
    out: dict[str, set[str]] = {}
    for path in sorted((unit / "alias-scan").glob("*.json")):
        found = json.loads(path.read_text())
        for row in (found.get("attributed") or []) + (found.get("controls") or []):
            if not isinstance(row, dict):
                continue
            name = str(row.get("stimulus") or row.get("control") or "").split(" ch")[0]
            where = row.get("stores_verbatim") or []
            if name and where:
                out.setdefault(name.upper(), set()).update(where)
    return out


def _tables(unit: Path) -> list[tuple[str, dict]]:
    """Each behaviour that says which stimulus is stored at which address."""
    kept = unit / "measurements.json"
    if not kept.exists():
        return []
    found = json.loads(kept.read_text())
    return [
        (b["id"], b["stored"])
        for b in found.get("behaviours", [])
        if isinstance(b.get("stored"), dict)
    ]


@pytest.mark.parametrize("unit", _units(), ids=lambda p: p.name)
def test_a_behaviour_naming_where_a_stimulus_is_stored_agrees_with_the_scans(
    unit: Path,
) -> None:
    scans = _attributed(unit)
    for behaviour, table in _tables(unit):
        wrong = {
            stimulus: (address, sorted(scans.get(stimulus.upper(), ())))
            for stimulus, address in table.items()
            if address not in scans.get(stimulus.upper(), set())
        }
        assert not wrong, (
            f"{unit.name}/measurements.json, {behaviour}: names a store no alias scan of this "
            f"unit attributes.\n  {wrong}\nA hand-kept table is the one thing here nothing else "
            "checks, so it is the one that goes on disagreeing."
        )


@pytest.mark.parametrize("unit", _units(), ids=lambda p: p.name)
def test_a_behaviour_listing_the_stores_lists_every_one_the_scans_found(unit: Path) -> None:
    """The half that catches a table left behind by a wider scan.

    A store the archive found and the summary does not name is how the count in
    a behaviour's own identifier comes to be wrong: nothing is contradicted, an
    entry is simply missing, and a reader takes the list for the whole of it.
    """
    scans = _attributed(unit)
    for behaviour, table in _tables(unit):
        # Only the stimuli of the kind this table is about: a table of control
        # changes says nothing about an NRPN, and holding it to one would ask it
        # to list every message the unit answers to.
        #
        # And only the ones named in a single word. `CC0 THEN PROGRAM` is a
        # sequence rather than a controller -- what it stores is a fact about the
        # bank latch, which has a behaviour of its own, and this table says in
        # its own words that it leaves those out. A table is owed the stimuli of
        # its kind, not every stimulus whose name begins the same way.
        kinds = {s.split()[0].rstrip("0123456789") for s in table if len(s.split()) == 1}
        missing = {
            stimulus: sorted(where)
            for stimulus, where in scans.items()
            if len(stimulus.split()) == 1
            and stimulus not in {s.upper() for s in table}
            and stimulus.rstrip("0123456789") in kinds
        }
        assert not missing, (
            f"{unit.name}/measurements.json, {behaviour}: an alias scan of this unit attributes "
            f"a store the table does not list.\n  {missing}\nThe count in the identifier and in "
            "the summary is then wrong, and nothing contradicts it -- the entry is only absent."
        )
