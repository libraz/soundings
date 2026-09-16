"""Choosing the two settings an address is asked under, from what it was measured to accept.

A `contrast` run compares one address at two values, and the pair is the whole
measurement: both have to be inside what the address accepts or the unit clamps
them to the same byte and answers identically, which reads as a parameter that
does nothing. Picking the pair by hand needs the manual, and a manual is a claim
about a model rather than a measurement of this unit.

For an address carrying a quantity it does not need one. The write probe already
asked every address what it takes, by writing and reading back, and its rows carry
the accepted range. That is the right source: it was measured on this unit, it is
already in the archive, and a range it got wrong would show up as a clamp rather
than as a plausible number.

**For an address carrying a list of states it does need one, and the safeguard
above is why that went unnoticed.** What the write probe measures is what the
store takes, which is not what the engine reaches. One block was found accepting
and reading back every seven-bit value at every one of its addresses, parameters
with two printed states included, so no pair ever clamped and nothing looked
wrong -- and asked at the two ends of that range, not one such parameter on the
whole unit was ever heard. Asked instead at two of its own printed states, the
first two tried answered immediately and loudly. A count of states is a claim
about a model rather than a measurement of this unit, exactly as above; the
difference is that here the alternative is not a weaker measurement but a null
that means nothing. So the count may be handed in, and an address whose pair came
from a page says so in the plan, because that pair rests on something outside the
archive and a reader has to be able to see which ones do.

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

from . import documents

RANGE = re.compile(r"^([0-9A-Fa-f]{2})\.\.([0-9A-Fa-f]{2})")

METHOD = (
    "The two settings come from the write probe's own rows: it wrote to every address and read "
    "it back, so the range it recorded is what this unit accepted. The pair is the two ends of "
    "that range, ordered so the value nearer the one the unit powers up holding is asked first, "
    "since that setting fixes the reference every other take is aligned against and the power-on "
    "value is the one known to make a sound. Where a count of printed states was handed in, that "
    "address is asked at the first and last state instead and never outside what it was measured "
    "to accept, because an address can take a value its parameter has no state for and then the "
    "ends of the range are not two settings. Every row says which of the two its pair came from."
)

FROM_THE_RANGE = "the two ends of the range the write probe measured this address to accept"

FROM_THE_PRINTED_VALUES = (
    "the lowest and highest value the page prints this parameter as having, because the address "
    "accepts values it is not printed as having and the ends of what it accepts are not a pair "
    "of settings. It rests on a page as well as on this unit, which the ends of a measured "
    "range do not"
)

ONE_VALUE = "accepts one value, so it has no pair to compare and cannot be asked this way"

ONE_PRINTED_VALUE = (
    "is printed with one value, so there is nothing to compare it against however much the "
    "address accepts"
)

NONE_OF_ITS_PRINTED_VALUES = (
    "is printed with values, and the address was measured to accept none of them. A page and a "
    "unit disagreeing is a finding and not a licence to write past the range, so nothing is "
    "asked here until one of the two is established"
)

RANGE_UNREAD = (
    "the write probe never established a range here, because the address would not answer a "
    "one-byte read and there would have been nothing to put back. It is asked over the whole "
    "seven-bit span instead, and a null under it cannot separate an address that ignored the "
    "write from one that took it and reached nothing"
)

NO_RANGE = "the write probe recorded no range for it"

FROM_THE_POWER_ON_VALUE = (
    "the value the unit powers up holding, against the end of the range whose printed setting "
    "differs from it. Taken because the page prints one setting at both ends of what this address "
    "accepts, so the pair the ends give asks the parameter twice in the same place -- and a null "
    "read off that pair would say the parameter does nothing when what it says is that the two "
    "values are the same setting of it. Which of the two settings is furthest from which is a "
    "reading of what the column means and is not made here, so the second value is the unit's own "
    "power-on setting rather than a point chosen along the column"
)

ENDS_ARE_ONE_SETTING = (
    "is printed with one setting at both ends of what the address accepts, and the value the unit "
    "powers up holding is printed as that same setting, so no pair this planner can name asks it "
    "in two places"
)


@dataclass(frozen=True)
class Ask:
    """One address and the pair it is to be asked at."""

    address: str
    values: tuple[int, int]
    classification: str
    accepted_range: str
    power_on: int | None
    range_established: bool = True
    values_from: str = FROM_THE_RANGE
    printed_values: str | None = None

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
            "values_from": self.values_from,
        }
        if self.printed_values is not None:
            out["printed_values"] = self.printed_values
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


def _tells_the_pair_apart(
    values: tuple[int, int], settings: dict[int, str], power_on: int | None
) -> tuple[tuple[int, int], str] | None:
    """A pair the page prints two settings for, where the given one is printed as one.

    Only one column of one grid read here needs this, and the page says so in its own
    notation: the quantity wraps, so its two ends are the same place and it prints each
    of them as equal to the other. Asking a wrapped quantity at the ends of its byte
    asks it twice where it is, and the answer is a null that reads exactly like a
    parameter that does nothing.

    Returns None where the pair already names two settings -- which is every other
    column -- so nothing that is asked correctly today is moved.
    """
    first, second = values
    if not documents.one_setting(settings.get(first), settings.get(second)):
        return None
    if power_on is None or documents.one_setting(settings.get(power_on), settings.get(first)):
        return None
    return (power_on, first), FROM_THE_POWER_ON_VALUE


def plan_block(
    record: dict,
    prefix: str,
    printed: dict[str, str] | None = None,
    settings: dict[str, dict[int, str]] | None = None,
) -> tuple[list[Ask], list[Skip]]:
    """Every address under `prefix`, sorted into the askable and the rest.

    `printed` names, by address, the value column an effect list prints against a
    parameter. An address in it is asked at the lowest and highest value that column
    gives, rather than at the ends of what the address accepts, and its row says so.

    A column referring the reader to a table of 128 entries narrows nothing, so the
    address keeps the pair its measured range gives it. `settings` carries what such
    a column does say -- the setting printed at each of the 128 -- which answers a
    different question: whether the pair names two settings or one twice.
    """
    printed = printed or {}
    settings = settings or {}
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
        cell = printed.get(address)
        here = documents.values_printed(cell) if cell else None
        # What a referral says about this address is kept even though it narrows
        # nothing: which values are the same setting is a separate question from
        # which values are settings, and it is asked below once the pair is chosen.
        each = settings.get(address) if here == documents.EVERY_VALUE else None
        if here == documents.EVERY_VALUE:
            cell, here = None, None
        if here is not None:
            # Never outside what the address was measured to take: a page and a unit
            # disagreeing is a finding, not a licence to write past the range.
            within = sorted(value for value in here if low <= value <= high)
            if not within:
                skipped.append(Skip(address, NONE_OF_ITS_PRINTED_VALUES))
                continue
            if len(within) < 2:
                skipped.append(Skip(address, ONE_PRINTED_VALUE))
                continue
            values = _ordered(within[0], within[-1], power_on)
            came_from = FROM_THE_PRINTED_VALUES
        else:
            values, came_from = _ordered(low, high, power_on), FROM_THE_RANGE
        if each:
            moved = _tells_the_pair_apart(values, each, power_on)
            if moved is not None:
                values, came_from = moved
            elif documents.one_setting(each.get(values[0]), each.get(values[1])):
                skipped.append(Skip(address, ENDS_ARE_ONE_SETTING))
                continue
        asks.append(
            Ask(
                address=address,
                values=values,
                classification=str(row.get("classification", "")),
                accepted_range=str(row.get("range", "")),
                power_on=power_on,
                values_from=came_from,
                printed_values=cell,
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
    from_page = sum(a.values_from == FROM_THE_PRINTED_VALUES for a in asks)
    head = f"{len(asks)} addresses to ask, {len(skipped)} that cannot be"
    if from_page:
        head += f", {from_page} of them at values a page printed rather than at what they accept"
    lines = [head]
    for ask in asks:
        mark = "" if ask.range_established else "   (range never established)"
        if ask.printed_values is not None:
            mark += f"   (printed {ask.printed_values})"
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
