"""Tell an address that holds a value from one that shows somebody else's.

A read that answers is not evidence that the address asked for exists. Some of
this machine's address space is a window: a request there is served by whichever
of two real stores was last addressed, and a write there lands in that same one.
Nothing about a single read says which kind of address it is, and neither does a
write followed by reading it back -- the write points the window and the read
follows it, so the address appears to hold exactly what it was given.

That is what makes this worth a measurement of its own. Twelve blocks of this
unit's space, 25512 addresses, were read as storage by a probe that wrote to each
address and read it straight back, and every one of them came back accepting.
They were byte for byte identical to each other, which is what a window looks
like when nobody has asked it the question that separates the two.

The question that separates them: put two different values in two addresses
believed to be real stores, address one of them, and ask the candidate. A store
answers with its own value either way. A window answers with the value of
whichever store was addressed last.

Both directions are asked, because they can differ and the read alone would not
say so: after each reading, a marker is written through the candidate and the two
stores are read to see which of them moved.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import roland
from .midi import MidiLink

METHOD = (
    "Two addresses believed to hold their own values are given two different values, and each "
    "candidate is asked once after addressing the first and once after addressing the second. "
    "An address that holds a value answers the same both times. An address that is a window "
    "answers with whatever was addressed last. Then a marker is written through the candidate "
    "and both stores are read, because which one a write reaches is a separate question from "
    "which one a read reports and they need not agree."
)

WHY_CONTROLS = (
    "The stores are asked as candidates too, and each must answer with its own mark whichever "
    "of them was addressed last. That is stricter than asking whether they read as holding "
    "their own values, and it has to be: two addresses backed by one storage answer "
    "identically and so read as perfectly ordinary stores, while leaving the run with no two "
    "values to tell any candidate apart, so every candidate reads as holding its own value "
    "too. That is the same output as a run that genuinely found no windows."
)

CAVEAT = (
    "A verdict here is relative to the two stores it was measured against. An address that "
    "does not follow these holds something of its own as far as this can see, which is not the "
    "same as holding what its own address says: it could be a window onto a pair this run "
    "never named."
)

CORRECTS_EARLIER_RECORDS = (
    "Blocks 42 through 4F of this unit are not storage. Every address in them is a window onto "
    "whichever of the two drum setup stores, 41 and 51, was addressed last -- for writes as "
    "well as for reads. That is why they answered as 12 blocks identical to each other and to "
    "41, and why writing to an address and reading it straight back made every one of them "
    "look like it held what it had been given: the write points the window and the read "
    "follows it. Anything an earlier record says about an address in those blocks is about "
    "the window and not about a store of its own. Measured at three offsets, in "
    "drum-map-window-04-24.json, drum-map-window-05-10.json and drum-map-window-11-7b.json."
)

HOLDS_ITS_OWN = "holds its own value"
IS_A_WINDOW = "a window onto whichever store was addressed last"
UNREADABLE = "would not answer"


@dataclass
class Candidate:
    address: str
    answered: dict[str, str] = field(default_factory=dict)
    """Store addressed first -> what the candidate then answered."""

    write_reached: str | None = None
    """Which store a marker written through the candidate landed in."""

    verdict: str = ""

    def to_json(self) -> dict:
        return {
            "address": self.address,
            "answered_after_addressing": self.answered,
            "a_write_through_it_reached": self.write_reached,
            "verdict": self.verdict,
        }


@dataclass
class Result:
    stores: list[str]
    marks: dict[str, str]
    candidates: list[Candidate] = field(default_factory=list)
    controls_held_their_own: bool = False
    restored: bool = False

    def to_json(self) -> dict:
        counts: dict[str, int] = {}
        for c in self.candidates:
            counts[c.verdict] = counts.get(c.verdict, 0) + 1
        return {
            "method": METHOD,
            "stores": self.stores,
            "marks": self.marks,
            "controls": {
                "the_stores_answered_for_themselves": self.controls_held_their_own,
                "why": WHY_CONTROLS,
            },
            "caveat": CAVEAT,
            "restored": self.restored,
            "verdicts": counts,
            "candidates": [c.to_json() for c in self.candidates],
        }


Address = tuple[int, int, int]


def _text(address: Address) -> str:
    return f"{address[0]:02X} {address[1]:02X} {address[2]:02X}"


class Prober:
    def __init__(
        self,
        link: MidiLink,
        *,
        device_id: int = roland.DEFAULT_DEVICE_ID,
        settle: float = 0.05,
        read_timeout: float = 0.5,
    ):
        self.link = link
        self.device_id = device_id
        self.settle = settle
        self.read_timeout = read_timeout

    def read(self, address: Address) -> int | None:
        raw = self.link.exchange(
            roland.rq1(address, 1, device_id=self.device_id), timeout=self.read_timeout
        )
        reply = roland.parse_dt1(raw)
        if reply is None or reply.address != address or reply.size != 1:
            return None
        return reply.data[0]

    def write(self, address: Address, value: int) -> None:
        self.link.send(roland.dt1(address, [value], device_id=self.device_id))
        time.sleep(self.settle)

    def run(
        self,
        stores: tuple[Address, Address],
        candidates: list[Address],
        *,
        marks: tuple[int, int] = (0x2A, 0x55),
        through: int = 0x66,
        progress=None,
    ) -> Result:
        """Ask each candidate which of the two stores it is speaking for, if either."""
        first, second = stores
        originals = {a: self.read(a) for a in stores}
        result = Result(
            stores=[_text(a) for a in stores],
            marks={_text(a): f"{m:02X}" for a, m in zip(stores, marks, strict=True)},
        )
        if any(v is None for v in originals.values()):
            raise ValueError("a store would not answer, so there is nothing to measure against")

        # The stores are candidates too, so a run that calls everything a window
        # says so instead of reporting one.
        asked = list(candidates) + [a for a in stores if a not in candidates]

        for address in asked:
            entry = Candidate(address=_text(address))
            for anchor in stores:
                self.write(first, marks[0])
                self.write(second, marks[1])
                self.read(anchor)
                got = self.read(address)
                entry.answered[_text(anchor)] = "--" if got is None else f"{got:02X}"

            self.write(first, marks[0])
            self.write(second, marks[1])
            self.read(first)
            self.write(address, through)
            moved = [a for a in stores if self.read(a) == through]
            entry.write_reached = _text(moved[0]) if len(moved) == 1 else None

            entry.verdict = self._verdict(entry, stores, marks)
            result.candidates.append(entry)
            if progress:
                progress(f"{entry.address}: {entry.verdict}")

        for address, value in originals.items():
            self.write(address, value)
        result.restored = all(self.read(a) == v for a, v in originals.items())
        # Each store must answer with its own mark whichever one was addressed
        # last. That is stricter than asking whether it reads as holding its own
        # value, and it has to be: two addresses that are secretly one storage
        # both answer with the mark written second, agree with each other, and so
        # read as perfectly ordinary stores -- while making every candidate in
        # the run look like one too, because there are no longer two values to
        # tell apart.
        result.controls_held_their_own = all(
            set(c.answered.values()) == {result.marks[c.address]}
            for c in result.candidates
            if c.address in result.stores
        )
        return result

    @staticmethod
    def _verdict(entry: Candidate, stores, marks) -> str:
        answers = [entry.answered[_text(a)] for a in stores]
        if "--" in answers:
            return UNREADABLE
        # A window answers with the store that was addressed last, so its two
        # answers are the two marks in the order they were addressed in.
        if answers == [f"{m:02X}" for m in marks]:
            return IS_A_WINDOW
        return HOLDS_ITS_OWN


def summarise(result: Result) -> str:
    lines = [f"{len(result.candidates)} addresses asked against {' and '.join(result.stores)}"]
    counts: dict[str, int] = {}
    for c in result.candidates:
        counts[c.verdict] = counts.get(c.verdict, 0) + 1
    for verdict, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {n:4d} {verdict}")
    if not result.controls_held_their_own:
        lines.append(
            "  !! a store did not answer with its own mark whichever was addressed last, so "
            "there were never two values to tell anything apart and no verdict above can be read"
        )
    lines.append("  every store put back" if result.restored else "  !! a store was not put back")
    return "\n".join(lines)


__all__ = ["Candidate", "HOLDS_ITS_OWN", "IS_A_WINDOW", "Prober", "Result", "summarise"]


KEPT_ITS_OWN = "every address answered differently from its neighbours"
ONE_VALUE_BETWEEN_THEM = "every address answered the same, so at most one of them holds anything"
SOME_INDISTINGUISHABLE = "some neighbouring addresses could not be told apart"
NOTHING_READABLE = "no address here would answer"
TOO_FEW_TO_COMPARE = "only one address here would answer, so there was nothing to compare it with"

HOLD_METHOD = (
    "Neighbouring addresses are given different values before any of them is read, and then "
    "the whole run is read back. This is what a probe that writes to one address and reads it "
    "straight back cannot do: on a window the write points it and the read follows, so the "
    "address answers with what it was just given and looks like a store. Written apart from "
    "read, a window can only show the value it was given last, and a run of them all answers "
    "the same."
)

DOES_NOT_SEE_A_MIRRORED_BLOCK = (
    "This does not find a block that mirrors another block. The window at 42 through 4F "
    "redirects the block and keeps the offset, so 42 04 20 and 42 04 21 reach two different "
    "cells of the store they point at -- neighbours there really are distinct, and every one "
    "of those 504 regions passes this as holding independently. What it finds is the other "
    "shape: a run of addresses backed by one cell. The two questions are separate and neither "
    "answer covers the other; window-probe is what asks the first."
)

HOLD_CAVEAT = (
    "Two addresses that answer differently are not one address. That is all this shows. It "
    "does not say either of them holds what its own address is supposed to, nor that anything "
    "here is used for anything. Neighbours that answer the same are not shown to be one "
    "address either: an address bounded to a single value answers with it whichever extreme "
    "it was given, so it cannot be told from its neighbour by this and is reported as "
    "indistinguishable rather than as shared."
)


@dataclass
class Held:
    start: str
    length: int
    given: dict[str, str] = field(default_factory=dict)
    read_back: dict[str, str] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    verdict: str = ""
    indistinguishable_pairs: int = 0
    restored: bool = False

    @property
    def distinct(self) -> int:
        return len(set(self.read_back.values()))

    def to_json(self) -> dict:
        kept = [a for a, v in self.read_back.items() if v == self.given[a]]
        return {
            "start": self.start,
            "length": self.length,
            "addresses_asked": len(self.given),
            "distinct_values_read_back": self.distinct,
            "kept_exactly_what_they_were_given": len(kept),
            "neighbouring_pairs_that_answered_alike": self.indistinguishable_pairs,
            "verdict": self.verdict,
            "restored": self.restored,
            "skipped": self.skipped,
            "given": self.given,
            "read_back": self.read_back,
        }


def hold_verdict(read_back: list[str]) -> tuple[str, int]:
    """The verdict and the count of neighbouring pairs that answered alike.

    A pure reading of the row, so a run already made can be read again when the
    rule changes. Kept beside `hold_probe`, which calls it, because a second
    implementation of a rule is a second rule.
    """
    pairs = list(zip(read_back, read_back[1:], strict=False))
    alike = sum(1 for a, b in pairs if a == b)
    if len(read_back) < 2:
        return (TOO_FEW_TO_COMPARE if read_back else NOTHING_READABLE), alike
    if len(set(read_back)) <= 1:
        return ONE_VALUE_BETWEEN_THEM, alike
    if alike:
        return SOME_INDISTINGUISHABLE, alike
    return KEPT_ITS_OWN, alike


def read_hold_verdicts_again(payload: dict) -> int:
    """Apply the current rule to a saved run, and say how many verdicts moved."""
    moved = 0
    for region in payload["regions"]:
        verdict, alike = hold_verdict(list(region["read_back"].values()))
        if verdict != region["verdict"]:
            region["verdict"] = verdict
            moved += 1
        region["neighbouring_pairs_that_answered_alike"] = alike
    return moved


def hold_probe(prober: Prober, start: Address, length: int, *, progress=None) -> Held:
    """Give a run of addresses different values, then read them all, then put them back.

    The order is the whole method. Every write happens before every read, so an
    address that only ever shows the last thing written anywhere near it has
    nowhere to hide: it answers with that one value while its neighbours answer
    with their own.
    """
    result = Held(start=_text(start), length=length)
    originals: dict[Address, int] = {}
    for i in range(length):
        address = (start[0], start[1], start[2] + i)
        held = prober.read(address)
        if held is None:
            result.skipped.append(f"{_text(address)} (would not answer)")
            continue
        originals[address] = held

    if not originals:
        result.verdict = NOTHING_READABLE
        result.restored = True
        return result

    # Alternating extremes rather than a counter: an address that clamps answers
    # with its own bound either way, so the two neighbours still differ wherever
    # the address has any range at all.
    for i, address in enumerate(originals):
        value = 0x00 if i % 2 == 0 else 0x7F
        result.given[_text(address)] = f"{value:02X}"
        prober.write(address, value)

    for address in originals:
        got = prober.read(address)
        result.read_back[_text(address)] = "--" if got is None else f"{got:02X}"

    # Judged on whether neighbours can be told apart, not on whether each kept
    # exactly what it was given. An address that clamps answers with its own
    # bound rather than with the extreme it was written, which is a fact about
    # its range and says nothing about whose storage it is.
    result.verdict, result.indistinguishable_pairs = hold_verdict(list(result.read_back.values()))

    for address, value in originals.items():
        prober.write(address, value)
    result.restored = all(prober.read(a) == v for a, v in originals.items())
    if progress:
        progress(f"{result.start}: {result.verdict} ({result.distinct} distinct)")
    return result
