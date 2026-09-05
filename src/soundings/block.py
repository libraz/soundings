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

WHY_POLYPHONY_PASS = (
    "A third pass asked the addresses the first two could not hear with two notes instead of "
    "one. It is not a narrower question the way a gesture is: the notes are played in the state "
    "the unit powers up in, so a verdict from it is as broad as the plain note's. What it adds "
    "is the only question a single note cannot put at all -- whether the part sounds two voices "
    "at once, and what it does when the voice already sounding is asked for again. A null from "
    "any one-note stimulus on such a parameter is a fact about the stimulus rather than about "
    "the address, so until this pass has run those addresses are unasked rather than silent."
)

WHY_MODULATOR_SCATTER = (
    "Under the modulation gesture both settings carry the modulator, so the channel that reads "
    "one setting repeating worse than the other has nothing to detect and what it reports is "
    "the scatter of a free-running phase. It is read as evidence only where the steadier "
    "setting is steady against what the same address measured under the plain note, which is a "
    "gate blocking the modulation rather than two draws from the same distribution."
)

WHY_LEFT_OUT = (
    "A gesture cannot carry a message that writes the address under test: it would put both "
    "settings at the byte the gesture sends rather than at the two the run asked for. So the "
    "gesture asked here was one message short, and a null under it is narrower than a null "
    "under the whole one. For an address whose only route to being heard is the message that "
    "stores into it, that is narrower to the point of being unanswerable this way."
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

    left_out: dict = field(default_factory=dict)
    """Moves the gesture could not carry here, because they write this address.

    A null under a gesture one message short is narrower than a null under the
    whole one, and for some addresses it is narrower to the point of being
    unanswerable: the message that would reveal the address is the message that
    stores into it. Measured here on the two bytes a bank select and a program
    change land in, which is the only gesture that could have moved them.
    """

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
        if self.left_out:
            out["left_out_of_the_gesture"] = self.left_out
            out["why_left_out"] = WHY_LEFT_OUT
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
        left_out=dict(record.get("left_out_of_the_gesture") or {}),
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


def join(
    plain: dict[str, AddressVerdict],
    gesture: dict[str, AddressVerdict],
    polyphony: dict[str, AddressVerdict] | None = None,
) -> list:
    """One verdict per address over every pass, the plain note answering first.

    The two rescues are peers of each other and neither is a peer of the plain
    note: both are spent only where it heard nothing, and an address either of
    them hears is audible. They are kept apart here rather than merged into one
    pass because only the gesture's verdict is narrowed by the state it was
    asked in, and only the gesture carries a modulator to be discounted.
    """
    polyphony = polyphony or {}
    out = []
    for address in sorted(set(plain) | set(gesture) | set(polyphony)):
        first = plain.get(address)
        second, third = gesture.get(address), polyphony.get(address)
        rescued = [r for r in (second, third) if r is not None]
        if first is not None and first.audible:
            out.append(first)
            continue
        if not rescued:
            out.append(first)
            continue
        merged = AddressVerdict(
            address=address,
            values=next((r.values for r in rescued if r.values), first.values if first else []),
            audible=any(r.audible for r in rescued),
            heard_by=[n for r in rescued for n in r.heard_by],
            not_heard_by=(first.not_heard_by if first else [])
            + [n for r in rescued for n in r.not_heard_by],
            inconclusive_under=(first.inconclusive_under if first else [])
            + [n for r in rescued for n in r.inconclusive_under],
            unrepeatable_db={
                **(first.unrepeatable_db if first else {}),
                **{k: v for r in rescued for k, v in r.unrepeatable_db.items()},
            },
            left_out={
                **(first.left_out if first else {}),
                **{k: v for r in rescued for k, v in r.left_out.items()},
            },
        )
        if second is not None and modulator_scatter(second, first):
            merged.discounted.append("struck_vibrato")
            merged.heard_by = [n for n in merged.heard_by if n != "struck_vibrato"]
            merged.inconclusive_under = merged.inconclusive_under + ["struck_vibrato"]
            merged.audible = bool(merged.heard_by)
        out.append(merged)
    return out


WHY_BALANCE_COUNTS = (
    "A parameter that moves signal between the two channels is invisible to a comparison made "
    "in one of them, and does not read as nothing there: a balance landing somewhere new on "
    "each take reads as the unit failing to repeat itself. So a balance the same run measured "
    "moving counts as the parameter having reached the signal path, which is what audible means "
    "here, and the stimulus it was found under carries the route it was found by."
)


def with_balance(found: list, measured: dict) -> list:
    """Fold a balance record's verdicts into the addresses they were taken on.

    Keyed by the name the takes were saved under, which is the address with its
    spaces turned to dashes -- the same name the driver gave the directory and
    the record.
    """
    by_address = {}
    for entry in measured.get("runs", []):
        if entry.get("moved_between_settings") or entry.get("did_not_repeat_within_a_setting"):
            address = str(entry.get("name", "")).replace("-", " ").upper()
            by_address.setdefault(address, []).append(str(entry.get("stimulus_name", "")))
    for verdict in found:
        if verdict is None or verdict.address not in by_address:
            continue
        for stimulus in by_address[verdict.address]:
            name = f"{stimulus} (balance)"
            if name not in verdict.heard_by:
                verdict.heard_by.append(name)
        verdict.audible = True
        verdict.not_heard_by = [
            n for n in verdict.not_heard_by if n not in by_address[verdict.address]
        ]
        verdict.inconclusive_under = [
            n for n in verdict.inconclusive_under if n not in by_address[verdict.address]
        ]
    return found


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
    "WHY_BALANCE_COUNTS",
    "WHY_LEFT_OUT",
    "STEADY_WITHIN_DB",
    "WHY_MODULATOR_SCATTER",
    "WHY_POLYPHONY_PASS",
    "WHY_TWO_PASSES",
    "AddressVerdict",
    "against_plan",
    "with_balance",
    "join",
    "modulator_scatter",
    "read_one",
    "summarise",
    "survey",
]
