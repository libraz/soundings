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

NOT_THIS_SHAPE = (
    "{path} is an alias scan this cannot read: it keeps its attributions under a key the "
    "current one does not write, so it was made by an earlier form of the tool. Re-run it "
    "before reading it here. Passing over it instead would be worse than refusing: the "
    "addresses it reached would be missing from the list without the list being any shorter "
    "in a way a caller could see, and a scan aimed by that list would report a null over a "
    "space it never asked about."
)


ANSWERED_SHORT = (
    "This region was asked for more bytes than the unit sent back, and the reply carries the "
    "address it starts at rather than the address of each byte. Counting up from that start "
    "puts every byte after the one the unit skipped on the address below its own, so the "
    "bytes are held here in the order they arrived instead of being placed. Measured rather "
    "than supposed: the same request was answered the same short way every time, with a "
    "checksum that verified, and the addresses the reply leaves out are not the ones that go "
    "silent when the block is asked an offset at a time -- so knowing which addresses answer "
    "does not say which bytes these are. A shorter read of the same block is answered in "
    "full, and that is where the values for its first addresses came from."
)
"""Why a region's bytes are kept as a list rather than against addresses.

The reading that produced them happened and is worth keeping; what cannot be
kept is the claim that byte *n* belongs to the *n*th address asked for. A value
on the wrong address is worse than a missing one, because a missing byte is
visibly missing and a misplaced one is compared, differenced and published.
"""

MAP_ASSUMED = (
    "This record predates the envelope, so nothing in it names the map its run was given, "
    "and which regions came back short cannot be worked out without one. The map named here "
    "was supplied by hand and is an assumption, not something the record states. What makes "
    "it a usable one is that it fits: nearly every region in it holds exactly as many values "
    "as it asked for, which a map belonging to another run would not."
)
"""Why a repair names a map the record it repaired does not.

Kept apart from the repair itself so that a reader meets the assumption at the
same time as what rests on it. A wrong map here would not fail -- it would
report most of the record as short and take back most of its values -- so the
fit is checked rather than trusted, and said either way.
"""

SHORT_REPLY_BLOCK = (
    "The addresses this names in the blocks listed under regions_answered_short came from a "
    "region read whose reply was shorter than the request. The block is what the reply was "
    "for and is right; the offset was counted up from the start of the request and is not. "
    "What the run found still happened -- both snapshots were laid down the same way, so a "
    "byte that moved did move -- but which address moved is not established here."
)
"""Why a finding keeps its verdict and loses its address.

Said on the record that names such an address rather than left to whoever holds
two records against each other, because the defect is invisible from the finding
alone: the address is well formed, sits in a block that exists, and answers.
"""

WATCHED_RECOVERED = (
    "This run predates the record carrying the addresses it watched, and it states them as "
    "a count. A count cannot state the bound its negatives rest on: several maps of this "
    "unit have been watched by these scans and they do not contain one another, so more "
    "regions is not more space. The list here was recovered from the map the invocation "
    "names and the prefix the record names, and was written only because the number of "
    "regions that came back is the number the record already held. It is a recovery of an "
    "argument the run was given, not a second measurement."
)
"""Why a scan says which addresses it watched without the run having said so.

Kept here rather than in whatever recovered it, because a sentence published in
a record has to be findable from the source: a reader who wants to know how a
field got its value has the package to look in, and a driver that is not part of
the distribution is not somewhere they can look.
"""


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

    A record of an older shape is refused by name rather than skipped, per
    NOT_THIS_SHAPE. Written because one such record raised a KeyError naming a
    field rather than the file it came from.
    """
    out: set[str] = set()
    for path in paths:
        data = json.loads(Path(path).read_text())
        if "attributed" not in data:
            raise ValueError(NOT_THIS_SHAPE.format(path=Path(path).name))
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


def markable_bytes(path: str | Path) -> dict[str, list[int]]:
    """Every address a mark can be put in, and the values it was measured to take.

    Wider than `accepting_bytes`, and the difference is most of the documented
    space. A reset probe needs a value the address does not already hold; it does
    not need an address that takes any value at all. Asking only for the latter
    leaves out every byte with a range -- which is to say every byte a document
    gives a function to, since a parameter with four settings clamps and a byte
    nobody defined is the one that accepts anything. On the unit this was written
    against, 25643 bytes accept any value and a further 11686 hold two or more,
    so a third of what the write probe reached was outside every reset probe in
    the archive without anything saying so.

    A byte that took one value has no mark: writing what it already holds breaks
    nothing, and a reset leaving it alone would read as a reset restoring it.
    """
    data = json.loads(Path(path).read_text())
    return {
        b["address"]: [int(v, 16) for v in b["accepted"]]
        for region in data["regions"]
        for b in region["bytes"]
        if b.get("restored") and len(b.get("accepted") or []) > 1
    }


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
