"""Find which insertion effects the unit has, by asking for each and reading back.

There is no query for "which effects exist". What there is instead is a type
address that holds two bytes and answers for itself: writing a type the unit has
stores it verbatim, and writing one it does not have leaves something else there.
So asking for all 16384 combinations and reading each back is the whole map.

**The type responds only to a two-byte write.** Writing the high byte alone does
nothing at all -- measured, and it is why an address probe that walks a region a
byte at a time reports this address as unchanging. That is a false negative
rather than a property of the address, and it applies to every multi-byte
parameter in the map.

**What a refusal looks like is measured, not assumed.** A rejected type does not
leave the previous one standing; it puts something else there. Whatever that is,
it is recorded per rejection rather than taken as known, because a fallback that
varies by family would otherwise be invisible.

**Selecting a type loads its parameters.** So each accepted type is read back
with its twenty parameter bytes and its three send levels, and the map carries
what the effect is set to before anyone touches it -- which is the closest thing
to a specification the unit will give up on its own.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import roland
from .midi import MidiLink

TYPE_ADDRESS = "40 03 00"
"""Insertion effect type, high byte then low byte."""

PARAMETER_ADDRESS = "40 03 03"
PARAMETER_COUNT = 20

SEND_ADDRESS = "40 03 17"
SEND_COUNT = 3
"""Send levels from the effect to reverb, chorus and delay, in that order."""

METHOD = (
    "Each of the 16384 type numbers was written to the insertion effect's type address as a "
    "single two-byte message and read straight back. A type the unit has stores verbatim; one "
    "it does not have leaves a different value, which is recorded as it was found rather than "
    "assumed. The type address does not respond to a one-byte write, so it is written whole. "
    "Every accepted type is then read for its twenty parameters and three send levels, which "
    "the unit loads when the type is selected, giving each effect's settings before anything "
    "has changed them."
)


@dataclass
class Effect:
    msb: int
    lsb: int
    parameters: list[int] = field(default_factory=list)
    sends: list[int] = field(default_factory=list)

    @property
    def number(self) -> str:
        return f"{self.msb:02X} {self.lsb:02X}"

    def to_json(self) -> dict:
        return {
            "type": self.number,
            "msb": self.msb,
            "lsb": self.lsb,
            "parameters": self.parameters,
            "sends_to_reverb_chorus_delay": self.sends,
        }


@dataclass
class Survey:
    accepted: list[Effect] = field(default_factory=list)
    refusals: dict[str, int] = field(default_factory=dict)
    """What a refused type left behind, and how many refusals left it."""

    unread: list[str] = field(default_factory=list)
    asked: int = 0
    high_bytes: list[int] = field(default_factory=list)
    """The high bytes swept. Anything outside them is unasked, not absent."""

    @property
    def exhaustive(self) -> bool:
        return sorted(self.high_bytes) == list(range(128))

    @property
    def families(self) -> dict[int, list[Effect]]:
        out: dict[int, list[Effect]] = {}
        for e in self.accepted:
            out.setdefault(e.msb, []).append(e)
        return out

    def to_json(self) -> dict:
        return {
            "method": METHOD,
            "asked": self.asked,
            "accepted": len(self.accepted),
            "effects": [e.to_json() for e in self.accepted],
            "refusal_left_behind": self.refusals,
            "unanswered": self.unread,
            "high_bytes_asked": self.high_bytes,
            "coverage": "Every type number was asked."
            if self.exhaustive
            else "Only the high bytes listed were swept; a family outside them was not asked "
            "for and its absence here means nothing.",
        }


class Asker:
    """Writes a type and reads it back, keeping the unit's answers verbatim."""

    def __init__(self, link: MidiLink, *, device_id: int, settle: float = 0.02):
        self.link = link
        self.device_id = device_id
        self.settle = settle
        self.writes = 0
        self.reads = 0

    def _read(self, address: str, size: int) -> list[int] | None:
        self.reads += 1
        reply = self.link.exchange(roland.rq1(address, size, device_id=self.device_id))
        parsed = roland.parse_dt1(reply)
        return None if parsed is None else list(parsed.data)

    def ask(self, msb: int, lsb: int) -> list[int] | None:
        self.link.send(roland.dt1(TYPE_ADDRESS, [msb, lsb], device_id=self.device_id))
        self.writes += 1
        time.sleep(self.settle)
        return self._read(TYPE_ADDRESS, 2)

    def settings(self) -> tuple[list[int], list[int]]:
        return (
            self._read(PARAMETER_ADDRESS, PARAMETER_COUNT) or [],
            self._read(SEND_ADDRESS, SEND_COUNT) or [],
        )


def survey(asker: Asker, *, high_bytes=None, progress=None) -> Survey:
    """Ask for every type number and keep what came back.

    Nothing is skipped on the way. A family whose members are not contiguous, or
    one sitting where no manual puts it, is exactly what a survey of the whole
    space is for, and sampling the high byte alone would miss a family whose
    lowest member is not zero.

    `high_bytes` narrows the sweep for a resumed or partial run, and the result
    says which were asked, so a narrowed run cannot be read as a complete one.
    """
    high_bytes = list(range(128)) if high_bytes is None else list(high_bytes)
    found = Survey(high_bytes=high_bytes)
    for msb in high_bytes:
        before = len(found.accepted)
        for lsb in range(128):
            found.asked += 1
            got = asker.ask(msb, lsb)
            if got is None:
                found.unread.append(f"{msb:02X} {lsb:02X}")
                continue
            if got == [msb, lsb]:
                parameters, sends = asker.settings()
                found.accepted.append(Effect(msb=msb, lsb=lsb, parameters=parameters, sends=sends))
            else:
                left = " ".join(f"{b:02X}" for b in got)
                found.refusals[left] = found.refusals.get(left, 0) + 1
        if progress and len(found.accepted) > before:
            taken = [e.number for e in found.accepted[before:]]
            progress(f"{msb:02X}: {len(taken)} accepted -- {', '.join(taken)}")
    return found


def summarise(found: Survey) -> str:
    lines = [
        f"{len(found.accepted)} insertion effects, of {found.asked} type numbers asked",
    ]
    families = found.families
    lines.append(f"  in {len(families)} families:")
    for msb in sorted(families):
        members = families[msb]
        lines.append(f"    {msb:02X}: {len(members):2d}  {', '.join(e.number for e in members)}")
    if found.refusals:
        lines.append("  a refused type left behind:")
        for left, count in sorted(found.refusals.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {left}  {count} times")
    if found.unread:
        lines.append(f"  {len(found.unread)} type numbers went unanswered")
    if not found.exhaustive:
        lines.append(
            f"  only {len(found.high_bytes)} of 128 high bytes were swept; a family "
            "outside them was not asked for"
        )
    return "\n".join(lines)


__all__ = [
    "METHOD",
    "PARAMETER_COUNT",
    "TYPE_ADDRESS",
    "Asker",
    "Effect",
    "Survey",
    "summarise",
    "survey",
]
