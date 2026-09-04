"""Reading the archive back, so one measurement can be aimed by an earlier one.

A sweep says which regions exist; a write probe says which of their bytes take
any value. Neither is worth repeating at the head of every later run, and both
are already published. What is read back here is this repository's own output,
so the shapes are the ones the `to_json` methods write.
"""

from __future__ import annotations

import json
from pathlib import Path

Address = tuple[int, int, int]


def stores_reached(paths: list[str | Path]) -> list[str]:
    """Every address these alias scans attributed a message to, verbatim.

    Writing to one of them by SysEx asks whether the location has a third way
    in, and -- because the whole watched space is diffed, not just the byte
    written -- whether the value is also kept anywhere else.

    Read from the records rather than written down beside the code. A list
    written down is a list about the unit it was written from, and it goes stale
    against its own source: the one this replaced named thirteen addresses,
    missed three that the same unit's scans had since attributed, included one
    that no scan in the archive attributes, and described itself as the bytes a
    control change and an NRPN both reach, which was true of eight of them.
    """
    out: set[str] = set()
    for path in paths:
        data = json.loads(Path(path).read_text())
        for entry in data["attributed"]:
            out.update(entry.get("stores_verbatim") or [])
    return sorted(out)


def regions(path: str | Path, prefix: str = "") -> list[tuple[Address, int]]:
    """Every region in an address map, as (start, size), optionally under a prefix."""
    data = json.loads(Path(path).read_text())
    out = []
    for region in data["regions"]:
        if not region["address"].startswith(prefix) or not region["size"]:
            continue
        start = tuple(int(b, 16) for b in region["address"].split())
        out.append((start, region["size"]))
    return out


def accepting_bytes(path: str | Path) -> list[str]:
    """Addresses the write probe found take any value and give it back.

    A reset probe needs somewhere it can put a mark. An address that clamps or
    refuses may keep what it had, and a byte that was never broken tells the
    reset nothing.
    """
    data = json.loads(Path(path).read_text())
    return [
        b["address"]
        for region in data["regions"]
        for b in region["bytes"]
        if b["classification"] == "accepts" and b["restored"]
    ]


def split_off_prefixes(addresses: list[str], prefixes: list[str]) -> tuple[list[str], list[str]]:
    """The addresses to keep, and the ones a prefix asked to leave alone.

    Both halves rather than the kept one, since what was skipped is the caveat
    on everything the run goes on to say, and a caller that only receives the
    remainder has nothing to write down.
    """
    if not prefixes:
        return list(addresses), []
    skipped = sorted({a for a in addresses if any(a.startswith(p) for p in prefixes)})
    dropped = set(skipped)
    return [a for a in addresses if a not in dropped], skipped


def keep_only_prefixes(addresses: list[str], prefixes: list[str]) -> tuple[list[str], list[str]]:
    """The addresses under one of these prefixes, and everything else.

    The inverse of split_off_prefixes, and separate from it rather than the same
    call with the halves swapped: a run bounded to a few blocks and a run that
    excluded a few are two different claims, and reading which one a call made
    from the order of its return values is how they get mixed up.
    """
    if not prefixes:
        return list(addresses), []
    kept = [a for a in addresses if any(a.startswith(p) for p in prefixes)]
    inside = set(kept)
    return kept, sorted({a for a in addresses if a not in inside})
