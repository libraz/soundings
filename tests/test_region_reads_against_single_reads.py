"""Every published region read, held against a read of one address at a time.

A region read and a single read are two ways of asking the same address, and the
archive already keeps both because they reach different sets of addresses. This
holds them against each other where they overlap, which is the only thing that
catches a reply of the length that was asked for whose bytes after the first are
zero: the length matches, so nothing refuses it, and every byte of it is
published as a value.

The comparison is not the finding. Where the two disagree the record has to say
so, in a value that names its kind, and this asks the record rather than deciding
for it -- in both directions, so a finding that names a block the readings now
agree about is caught as well. The addresses are nowhere in here: which blocks a
unit answers this way is that unit's, and a list of them in a test would arrive
at the next unit as an assumption.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

UNITS = sorted(p for p in (Path(__file__).parents[1] / "data" / "units").iterdir() if p.is_dir())

KIND = "a-region-reply-whose-bytes-after-the-first-are-zero"


def _single_reads(unit: Path) -> dict[str, str]:
    """What the offsets stage got back, asking every offset of a block on its own.

    That stage and no other. A write probe reads each byte one at a time too, and
    that read is the right shape -- but it is taken with the probe, which is after
    the stages that write, so a value differing from a power-on capture there is
    the unit having been written to and not two readings of one state. Held
    against a power-on capture it reports that as a disagreement: on the unit this
    was written against, one address in a documented block.
    """
    out: dict[str, str] = {}
    for path in sorted((unit / "offsets").glob("*.json")):
        for block in json.loads(path.read_text()).get("blocks", []):
            out.update(block.get("answered", {}))
    return out


def _named_blocks(record: dict) -> set[str]:
    return {
        block
        for finding in record.get("findings", [])
        if finding.get("kind") == KIND
        for block in finding.get("blocks", [])
    }


CAPTURES = [
    (unit, path)
    for unit in UNITS
    for path in sorted((unit / "power-on").glob("*.json"))
    if "values" in json.loads(path.read_text())
]


@pytest.mark.parametrize("unit,path", CAPTURES, ids=lambda x: getattr(x, "name", str(x)))
def test_a_capture_that_disagrees_with_a_single_read_says_which_blocks(unit: Path, path: Path):
    record = json.loads(path.read_text())
    singles = _single_reads(unit)
    if not singles:
        pytest.skip(f"{unit.name} has no stage that asks one address at a time")

    disagreed = {
        address[:5]
        for address, value in record["values"].items()
        if address in singles and singles[address] != value
    }
    named = _named_blocks(record)

    assert not disagreed - named, (
        f"{path.name} publishes values that a single-byte read of the same address "
        f"contradicts, in blocks no finding names: {sorted(disagreed - named)}"
    )
    assert not named - disagreed, (
        f"{path.name} names blocks under {KIND} that its own values no longer "
        f"disagree about: {sorted(named - disagreed)}"
    )


def test_the_comparison_is_actually_reaching_addresses():
    """A comparison whose overlap is empty passes every case above and says nothing."""
    compared = 0
    for unit, path in CAPTURES:
        singles = _single_reads(unit)
        compared += sum(1 for a in json.loads(path.read_text())["values"] if a in singles)
    assert compared >= 10000, compared
