"""Reading a block's worth of contrast records into one verdict per address.

A part block is asked one address at a time, so what comes back is a directory of
records rather than an answer about the block. Three things have to happen to it
before it is one, and none of them is arithmetic anyone should do by hand.

**The block has to account for itself.** An address is audible, inaudible under
what was tried, unmeasurable, or was never asked -- and the last of those has to
be visible, because a directory holding forty-five records of a forty-seven
address block reads exactly like a complete answer. The plan says how many there
should be and which one cannot be asked at all, and the count is checked against
it rather than against the files that happen to be there.

**A gesture is a rescue for a null, so the two passes are not peers.** The plain
note asks the address in the state the unit powers up in; a gesture asks it with
half a dozen messages moved, which is a narrower question whose verdict holds
only in that state. So an address the plain note heard is answered by the plain
note, and the gesture's job is the addresses it could not.

**A modulator in the gesture makes the repeatability channel meaningless.** That
channel exists for a parameter that switches something moving on: one setting
repeats, the other does not. Under the modulation gesture *both* settings have
the modulator, so there is nothing for it to detect and the gap between the two
settings is the scatter of a free-running phase. Measured on the first address
asked: 29.2 dB against 14.5, which clears the bar comfortably, on an address the
plain note found no difference in at all -- across 51.92 dB against a yardstick
of 51.9 and a level difference of 0.001 dB. The gap is read as evidence only
when the steadier setting is steady in absolute terms, which the plain pass
measured for that same address.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

METHOD = (
    "Each address in the block was asked on its own, at the pair the write probe measured it "
    "to accept, and the records are read back together here. An address the plain note heard "
    "is answered by the plain note; one it could not was asked again with messages moved after "
    "the setting, since a parameter that only decides whether a message is received has nothing "
    "to receive otherwise. The block is counted against the plan rather than against the files "
    "present, so an address that was never asked cannot read as one that answered."
)

WHY_TWO_PASSES = (
    "A gesture is a rescue for a null rather than a better question. It asks the address with "
    "half a dozen messages moved, which is a state the verdict then holds only in, so it is "
    "spent on the addresses the plain note could not hear and on nothing else. Where both "
    "passes spoke, the plain one is the verdict and the gesture is what it took to hear it."
)

WHY_MODULATOR_SCATTER = (
    "Under the modulation gesture both settings carry the modulator, so the channel that reads "
    "one setting repeating worse than the other has nothing to detect and what it reports is "
    "the scatter of a free-running phase. It is read as evidence only where the steadier "
    "setting is steady against what the same address measured under the plain note, which is a "
    "gate blocking the modulation rather than two draws from the same distribution."
)

STEADY_WITHIN_DB = 12.0
"""How near the plain note's own repeatability the steadier setting has to come.

Twice the margin, the same span the repeatability channel already asks a gap to
clear before it is a gap at all. The measured case it has to reject sits 28 dB
outside it, and a real gate would leave the blocked setting repeating as the
plain note does, so the two are not near each other.
"""


@dataclass
class AddressVerdict:
    """One address, over every stimulus it was asked under."""

    address: str
    values: list[int]
    audible: bool
    heard_by: list[str] = field(default_factory=list)
    not_heard_by: list[str] = field(default_factory=list)
    inconclusive_under: list[str] = field(default_factory=list)
    unrepeatable_db: dict[str, list] = field(default_factory=dict)
    discounted: list[str] = field(default_factory=list)
    """Stimuli whose audible verdict was set aside, with the reason in the record."""

    @property
    def still_open(self) -> bool:
        """Whether anything is left to try: nothing heard it, or nothing could."""
        return not self.audible

    @property
    def verdict(self) -> str:
        if self.audible:
            return "audible"
        if self.not_heard_by:
            return "not audible under what was tried"
        return "could not be measured"

    def to_json(self) -> dict:
        out = {
            "address": self.address,
            "values": self.values,
            "verdict": self.verdict,
            "audible": self.audible,
            "heard_by": self.heard_by,
            "not_heard_by": self.not_heard_by,
            "inconclusive_under": self.inconclusive_under,
            "each_setting_unrepeatable_db": self.unrepeatable_db,
        }
        if self.discounted:
            out["set_aside"] = self.discounted
            out["why_set_aside"] = WHY_MODULATOR_SCATTER
        return out


def _entries(record: dict) -> dict[str, dict]:
    return {str(e.get("stimulus_name", "")): e for e in record.get("by_stimulus") or []}


def read_one(record: dict) -> AddressVerdict | None:
    """One record into one verdict. None when it asked nothing."""
    entries = _entries(record)
    if not entries:
        return None
    return AddressVerdict(
        address=str(record.get("address", "")),
        values=list(record.get("values") or []),
        audible=bool(record.get("audible")),
        heard_by=list(record.get("heard_by") or []),
        not_heard_by=list(record.get("not_heard_by") or []),
        inconclusive_under=list(record.get("inconclusive_under") or []),
        unrepeatable_db={
            name: list(e.get("each_setting_unrepeatable_db") or []) for name, e in entries.items()
        },
    )


def survey(root: str | Path) -> dict[str, AddressVerdict]:
    """Every record under a directory, keyed by the address it holds."""
    out: dict[str, AddressVerdict] = {}
    for path in sorted(Path(root).glob("*.json")):
        found = read_one(json.loads(path.read_text()))
        if found is not None and found.address:
            out[found.address] = found
    return out


def _steadier(pair: list) -> float | None:
    """The better of a setting pair's two repeatability figures, if it has two.

    Better is the more negative: these are how far what fails to repeat sits
    below the setting's own signal, so a smaller number is a steadier take.
    """
    usable = [v for v in pair if isinstance(v, int | float)]
    return min(usable) if len(usable) == 2 else None


def modulator_scatter(gesture: AddressVerdict, plain: AddressVerdict | None) -> bool:
    """Whether the modulation gesture's verdict is scatter rather than a gate.

    A gate on the modulation leaves the blocked setting repeating as the plain
    note's takes do. Two draws from a free-running phase do not, however wide the
    gap between them, so the plain pass's own figure for the same address is what
    separates the two.
    """
    if "struck_vibrato" not in gesture.heard_by:
        return False
    if plain is None:
        return True
    here = _steadier(gesture.unrepeatable_db.get("struck_vibrato") or [])
    there = _steadier(plain.unrepeatable_db.get("struck") or [])
    if here is None or there is None:
        return True
    return bool(here > there + STEADY_WITHIN_DB)


def join(plain: dict[str, AddressVerdict], gesture: dict[str, AddressVerdict]) -> list:
    """One verdict per address over both passes, the plain note answering first."""
    out = []
    for address in sorted(set(plain) | set(gesture)):
        first, second = plain.get(address), gesture.get(address)
        if first is not None and first.audible:
            out.append(first)
            continue
        if second is None:
            out.append(first)
            continue
        merged = AddressVerdict(
            address=address,
            values=second.values or (first.values if first else []),
            audible=second.audible,
            heard_by=list(second.heard_by),
            not_heard_by=(first.not_heard_by if first else []) + second.not_heard_by,
            inconclusive_under=(first.inconclusive_under if first else [])
            + second.inconclusive_under,
            unrepeatable_db={**(first.unrepeatable_db if first else {}), **second.unrepeatable_db},
        )
        if modulator_scatter(second, first):
            merged.discounted.append("struck_vibrato")
            merged.heard_by = [n for n in merged.heard_by if n != "struck_vibrato"]
            merged.inconclusive_under = merged.inconclusive_under + ["struck_vibrato"]
            merged.audible = bool(merged.heard_by)
        out.append(merged)
    return out


def against_plan(found: list, planned: dict) -> dict:
    """What the block holds against what the plan said it should.

    A directory of records answers about the records. The plan is what says how
    many addresses the block has, so an address that was never asked shows as
    missing rather than as absent.
    """
    asked = {a["address"] for a in planned.get("ask", [])}
    answered = {f.address for f in found if f is not None}
    return {
        "planned": len(asked),
        "answered": len(answered),
        "never_asked": sorted(asked - answered),
        "cannot_be_asked": planned.get("cannot_be_asked", []),
    }


def summarise(found: list, coverage: dict) -> str:
    piles: dict[str, list[str]] = {}
    lines = []
    for entry in found:
        if entry is None:
            continue
        piles.setdefault(entry.verdict, []).append(entry.address)
        heard = ", ".join(entry.heard_by) if entry.heard_by else "-"
        lines.append(f"  {entry.address}  {entry.verdict:32} {heard}")
    head = [
        f"{coverage['answered']} of {coverage['planned']} addresses answered, "
        f"{len(coverage['cannot_be_asked'])} that cannot be asked"
    ]
    tail = [f"  => {len(v)} {k}" for k, v in sorted(piles.items())]
    if coverage["never_asked"]:
        tail.append(f"  => {len(coverage['never_asked'])} never asked")
    return "\n".join(head + lines + tail)


__all__ = [
    "METHOD",
    "STEADY_WITHIN_DB",
    "WHY_MODULATOR_SCATTER",
    "WHY_TWO_PASSES",
    "AddressVerdict",
    "against_plan",
    "join",
    "modulator_scatter",
    "read_one",
    "summarise",
    "survey",
]
