"""The set of addresses a stage watches, built from what the unit has answered.

Three stages take a map and bound every negative finding they make by it:
`power-on`, `alias-scan` and `reset-probe`. Which map they are given is
therefore part of what their records mean, and getting it from the sweep alone
was the defect that cost this archive most: the sweep asks a named set of third
bytes under each block, so it produces a list of blocks rather than a list of
addresses, and a message landing outside every region it named was recorded as
landing nowhere.

The two kinds of read do not reach the same addresses, and neither contains the
other:

- A region read returns one reply for a run of addresses. It reaches addresses
  that answer nothing when asked on their own -- one unit has 272 of them -- and
  it is how the sweep's regions were found.
- A single-byte read asks one address and is answered for that address or not at
  all. It reaches addresses that begin where no region does, which is what the
  offsets stage was written to find.

So the set is the union of both, and it is assembled here rather than in a
driver beside the code, because every address in it comes out of the records of
the unit being measured. Nothing below names an address, a block or a size.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

WHY_UNION = (
    "Every address either kind of read has reached on this unit: the regions the sweep found, "
    "and the addresses an offsets record saw answer a single-byte read. Neither set contains "
    "the other. The sweep asks a named set of third bytes under each block, so a region "
    "beginning elsewhere is missing from it; a single-byte read cannot see an address that "
    "comes back only inside a longer read. A stage watching either alone leaves a message able "
    "to land where nothing is looking, and records it as landing nowhere."
)

WHY_WINDOWS_DROPPED = (
    "Blocks measured to be a window onto another block are left out. A value landing in one "
    "lands in the store it points at, and that store is watched here, so watching both asks "
    "the same store twice and reports one store as two."
)

WHY_WINDOWS_KEPT = (
    "Blocks measured to be a window onto another block are kept. What is wanted here is what "
    "each address held, and a window holds what it points at -- so an address left out has no "
    "value recorded for it rather than a value recorded once."
)

WHY_ONE_AT_A_TIME = (
    "Only the addresses an offsets record has already seen answer a single-byte read, asked "
    "one at a time. An address that answers no single read pays a timeout for the asking, and "
    "there are tens of thousands of them inside the regions -- at four tenths of a second each "
    "that is hours to establish what a region read already reaches. What this leaves out is "
    "stated rather than hidden: it is exactly the addresses that come back only inside a "
    "longer read, which is what the other kind of map is for."
)

NOT_A_MEASUREMENT = (
    "This file is not a measurement. It is a list of addresses assembled from the records "
    "named below, so that a stage can be aimed at what this unit has answered rather than at "
    "what one earlier stage happened to look at. Rebuilding it from those records reproduces "
    "it; nothing here was read from the unit."
)


def _sweep_addresses(found: dict) -> set[str]:
    out = set()
    for region in found.get("regions", []):
        start = [int(b, 16) for b in region["address"].split()]
        for step in range(region.get("size") or 0):
            out.add(f"{start[0]:02X} {start[1]:02X} {start[2] + step:02X}")
    return out


def _windows(found: dict) -> set[str]:
    """The top bytes of blocks the sweep recorded as a window onto another block."""
    return {
        top
        for finding in found.get("findings", [])
        if finding.get("kind") == "blocks-that-are-a-window"
        for top in finding.get("blocks", [])
    }


def answering(unit: Path) -> set[str]:
    """Every address an offsets record of this unit saw answer a single-byte read.

    Over every record the stage filed, not one named file: the first run sweeps
    what the map held and a later one asks the offsets a document names in blocks
    the first left alone, and reading only the first reports those as never
    asked.
    """
    out: set[str] = set()
    for path in sorted((unit / "offsets").glob("*.json")):
        found = json.loads(path.read_text())
        for block in found.get("blocks", []):
            out |= set(block.get("answered") or {})
    return out


def _runs(addresses: set[str]) -> list[dict]:
    """The addresses as consecutive runs, so a region read asks in as few as it can."""
    by_block: dict[str, list[int]] = defaultdict(list)
    for address in addresses:
        by_block[address[:5]].append(int(address[6:8], 16))
    out = []
    for block, offsets in sorted(by_block.items()):
        run: list[int] = []
        for offset in sorted(offsets):
            if run and offset == run[-1] + 1:
                run.append(offset)
                continue
            if run:
                out.append({"address": f"{block} {run[0]:02X}", "size": len(run)})
            run = [offset]
        out.append({"address": f"{block} {run[0]:02X}", "size": len(run)})
    return out


def build(unit: Path, *, keep_windows: bool = False, one_at_a_time: bool = False) -> dict:
    """The watch set for this unit, as a map a stage's `--map` can be given.

    `keep_windows` is for a capture of what each address held, where a window
    holding what it points at is a value worth having. It is wrong for a scan
    that attributes a message to a store, which would then report one store
    twice.

    `one_at_a_time` is for a run that must not put a byte on the wrong address:
    a reply to a single read is answered for the address it was asked about or
    not at all, while a reply to a region read carries only the address it starts
    at. It reaches less, so it is one half of a capture rather than a capture.
    """
    sweep = unit / "sweep" / "whole-map.json"
    if not sweep.exists():
        raise FileNotFoundError(
            f"{sweep} is missing. The watch set is built from this unit's own sweep and "
            "offsets records, and there is no default map to fall back on: a set carried "
            "from another unit is a finding about that one arriving as an assumption here."
        )
    found = json.loads(sweep.read_text())
    from_offsets = answering(unit)
    if not from_offsets:
        raise FileNotFoundError(
            f"{unit / 'offsets'} holds no record that says which addresses answer a "
            "single-byte read. The sweep bounds which blocks exist and not which addresses "
            "do, so a map built without the offsets stage carries the hole it was written "
            "to close."
        )
    addresses = from_offsets if one_at_a_time else _sweep_addresses(found) | from_offsets
    windows = _windows(found)
    if not keep_windows:
        addresses = {a for a in addresses if a[:2] not in windows}
    # Two keys rather than one sentence made of both. A published sentence has to
    # exist in the source as it was written, and a run-together pair exists there
    # as neither half -- so joining them puts prose in the archive that a reader
    # following it back cannot find.
    return {
        "why": WHY_ONE_AT_A_TIME if one_at_a_time else WHY_UNION,
        "why_windows": WHY_WINDOWS_KEPT if keep_windows else WHY_WINDOWS_DROPPED,
        "is_not_a_measurement": NOT_A_MEASUREMENT,
        "built_from": sorted(
            [str(sweep.relative_to(unit.parents[2]))]
            + [str(p.relative_to(unit.parents[2])) for p in sorted((unit / "offsets").glob("*.json"))]
        ),
        "windows": {"kept": keep_windows, "blocks": sorted(windows)},
        "addresses": len(addresses),
        "regions": (
            [{"address": a, "size": 1} for a in sorted(addresses)]
            if one_at_a_time
            else _runs(addresses)
        ),
    }


__all__ = ["answering", "build"]
