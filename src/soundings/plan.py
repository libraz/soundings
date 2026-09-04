"""Choosing the two settings an address is asked under, from what it was measured to accept.

A `contrast` run compares one address at two values, and the pair is the whole
measurement: both have to be inside what the address accepts or the unit clamps
them to the same byte and answers identically, which reads as a parameter that
does nothing. Picking the pair by hand needs the manual, and a manual is a claim
about a model rather than a measurement of this unit.

It does not need one. The write probe already asked every address what it takes,
by writing and reading back, and its rows carry the accepted range. That is the
right source: it was measured on this unit, it is already in the archive, and a
range it got wrong would show up as a clamp rather than as a plausible number.

**The reference setting is the one nearer what the unit powers up holding.**
Every verdict is measured against the first setting's takes, and the alignment
that fixes it is a correlation -- with a silent reference it is noise against
noise and can land anywhere. The power-on value is the one setting known to make
a sound, so the endpoint closer to it goes first. That matters here more than
elsewhere, because a part parameter has several ways to silence its part: a key
range that excludes the note, a receive channel that moves the part out from
under it, a level of zero.

**An address that accepts one value cannot be asked at all**, and that is a
result rather than an omission -- it says the address is not a parameter with a
setting, whatever else it is. It is reported with the plan rather than dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

RANGE = re.compile(r"^([0-9A-Fa-f]{2})\.\.([0-9A-Fa-f]{2})")

METHOD = (
    "The two settings come from the write probe's own rows rather than from a manual: it "
    "wrote to every address and read it back, so the range it recorded is what this unit "
    "accepted. The pair is the two ends of that range, ordered so the value nearer the one "
    "the unit powers up holding is asked first, since that setting fixes the reference every "
    "other take is aligned against and the power-on value is the one known to make a sound."
)

ONE_VALUE = "accepts one value, so it has no pair to compare and cannot be asked this way"

RANGE_UNREAD = (
    "the write probe never established a range here, because the address would not answer a "
    "one-byte read and there would have been nothing to put back. It is asked over the whole "
    "seven-bit span instead, and a null under it cannot separate an address that ignored the "
    "write from one that took it and reached nothing"
)

NO_RANGE = "the write probe recorded no range for it"


@dataclass(frozen=True)
class Ask:
    """One address and the pair it is to be asked at."""

    address: str
    values: tuple[int, int]
    classification: str
    accepted_range: str
    power_on: int | None
    range_established: bool = True

    @property
    def caveat(self) -> str | None:
        return None if self.range_established else RANGE_UNREAD

    def to_json(self) -> dict:
        out = {
            "address": self.address,
            "values": list(self.values),
            "accepted_range": self.accepted_range,
            "power_on": self.power_on,
            "classification": self.classification,
            "range_established": self.range_established,
        }
        if self.caveat:
            out["caveat"] = self.caveat
        return out


@dataclass(frozen=True)
class Skip:
    address: str
    why: str

    def to_json(self) -> dict:
        return {"address": self.address, "why": self.why}


def _bounds(spec: str) -> tuple[int, int] | None:
    """The two ends of a range the write probe recorded, or None if it recorded none.

    The trailing words some rows carry -- "of the values tried" -- are part of
    what the probe established and are kept in the row, but the numbers are the
    two hex bytes at the front.
    """
    found = RANGE.match(spec.strip())
    if not found:
        return None
    return int(found.group(1), 16), int(found.group(2), 16)


def _ordered(low: int, high: int, power_on: int | None) -> tuple[int, int]:
    """The pair with the setting nearer the power-on value first.

    With no power-on value to go by the lower end leads, which is a choice with
    nothing behind it -- said here rather than dressed up, since the ordering
    only ever matters for which take becomes the reference.
    """
    if power_on is None or abs(low - power_on) <= abs(high - power_on):
        return low, high
    return high, low


def _rows(record: dict, prefix: str) -> list[dict]:
    out = []
    for region in record.get("regions", []):
        for row in region.get("bytes", []):
            if str(row.get("address", "")).startswith(prefix):
                out.append(row)
    return out


def _unread(record: dict, prefix: str) -> list[str]:
    """Addresses the probe listed as skipped, with the reason stripped off.

    It writes them as "40 11 01 (original could not be read)", so the address is
    everything before the bracket.
    """
    out = []
    for region in record.get("regions", []):
        for entry in region.get("skipped", []):
            address = str(entry).split("(")[0].strip()
            if address.startswith(prefix):
                out.append(address)
    return out


def plan_block(record: dict, prefix: str) -> tuple[list[Ask], list[Skip]]:
    """Every address under `prefix`, sorted into the askable and the rest."""
    asks: list[Ask] = []
    skipped: list[Skip] = []
    for row in _rows(record, prefix):
        address = str(row["address"])
        bounds = _bounds(str(row.get("range", "")))
        if bounds is None:
            skipped.append(Skip(address, NO_RANGE))
            continue
        low, high = bounds
        if low == high:
            skipped.append(Skip(address, ONE_VALUE))
            continue
        power_on = int(str(row["original"]), 16) if row.get("original") else None
        asks.append(
            Ask(
                address=address,
                values=_ordered(low, high, power_on),
                classification=str(row.get("classification", "")),
                accepted_range=str(row.get("range", "")),
                power_on=power_on,
            )
        )
    for address in _unread(record, prefix):
        asks.append(
            Ask(
                address=address,
                values=(0, 127),
                classification="",
                accepted_range="",
                power_on=None,
                range_established=False,
            )
        )
    asks.sort(key=lambda a: a.address)
    skipped.sort(key=lambda s: s.address)
    return asks, skipped


def summarise(asks: list[Ask], skipped: list[Skip]) -> str:
    lines = [f"{len(asks)} addresses to ask, {len(skipped)} that cannot be"]
    for ask in asks:
        mark = "" if ask.range_established else "   (range never established)"
        lines.append(
            f"  {ask.address}  {ask.values[0]:3d} against {ask.values[1]:3d}"
            f"   accepts {ask.accepted_range or '?'}{mark}"
        )
    for skip in skipped:
        lines.append(f"  {skip.address}  not asked: {skip.why}")
    return "\n".join(lines)


__all__ = [
    "METHOD",
    "NO_RANGE",
    "ONE_VALUE",
    "RANGE_UNREAD",
    "Ask",
    "Skip",
    "plan_block",
    "summarise",
]
