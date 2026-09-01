"""Find which tones exist, by asking for each one and seeing whether it was taken.

There is no query for "does this tone exist". What there is instead is the
behaviour the reset work turned up: a bank select is held, a program change
commits the three of them together, and a combination the unit does not have is
discarded whole, leaving the part on the tone it already had. So asking for a
tone and reading the part back afterwards answers the question -- the part
either moved to what was asked for or it did not.

**The part must not already be sitting on the tone being asked for.** If it is,
accepted and discarded read the same, and the whole sweep degenerates into
reporting whichever tone it started on as the only one that exists. The current
tone is tracked and a different known-good one is put in the way whenever the
next request would coincide with it.

**A bank is sampled before it is swept.** Most of the 128 bank numbers hold
nothing, and sweeping all of them costs a half hour to learn that. A handful of
programs spread across the range decides whether a bank is worth the full 128,
and the sample size is recorded with the result, because a bank whose only tone
sits between the sampled programs would be missed and counted as empty.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import parts, roland
from .midi import MidiLink

Address = tuple[int, int, int]

# Spread across the range rather than clustered, since a bank that holds only a
# few tones tends to hold them where the GM set puts that family.
SAMPLE_PROGRAMS = (0, 24, 48, 73, 100, 127)

METHOD = (
    "Each tone was asked for by sending its bank select and a program change, then reading the "
    "part's own tone bytes back. The unit discards a combination it does not have and leaves "
    "the part where it was, so a part that moved to what was asked for is the tone existing. "
    "The part is moved away first whenever it already stands on what is about to be asked."
)


def sampling_caveat(exhaustive: bool) -> str:
    """What a bank's absence from the result is allowed to mean.

    A sampled survey cannot tell an empty bank from one whose only tones sit
    between the sampled programs, and saying so is the difference between a
    result and a claim.
    """
    if exhaustive:
        return (
            "Every bank was asked for all 128 programs, so a bank absent here answered none of "
            "them."
        )
    return (
        "A bank that answered none of the sampled programs was not swept and is absent here. A "
        "bank whose only tones sit between them would read as empty."
    )


@dataclass
class Probe:
    bank: int
    program: int
    accepted: bool


@dataclass
class BankResult:
    bank: int
    programs: list[int] = field(default_factory=list)
    swept: bool = False
    """The whole 0 to 127 range was tried, rather than only the sample."""

    def to_json(self) -> dict:
        return {
            "bank": self.bank,
            "swept_fully": self.swept,
            "tones": len(self.programs),
            "programs": self.programs,
        }


class Asker:
    """Asks the unit for a tone and reports whether it took it."""

    def __init__(
        self,
        link: MidiLink,
        *,
        channel: int = 0,
        map_select: int = 0,
        device_id: int = roland.DEFAULT_DEVICE_ID,
        settle: float = 0.04,
    ):
        self.link = link
        self.channel = channel
        self.map_select = map_select
        self.device_id = device_id
        self.settle = settle
        self.tone_block: Address = parts.part_block(parts.TONE, channel)
        self.current: tuple[int, int] | None = None
        self.asks = 0
        self.unread = 0

    def _read_tone(self) -> tuple[int, int] | None:
        request = roland.rq1(self.tone_block, 2, device_id=self.device_id)
        reply = roland.parse_dt1(self.link.exchange(request, timeout=0.6))
        if reply is None or reply.address != self.tone_block or reply.size != 2:
            return None
        return (reply.data[0], reply.data[1])

    def _send(self, bank: int, program: int) -> None:
        status = 0xB0 | (self.channel & 0x0F)
        self.link.send([status, 0, bank & 0x7F])
        self.link.send([status, 32, self.map_select & 0x7F])
        self.link.send([0xC0 | (self.channel & 0x0F), program & 0x7F])
        time.sleep(self.settle)

    def settle_on(self, bank: int, program: int) -> bool:
        """Move the part somewhere known, so the next request is a real question."""
        self._send(bank, program)
        self.current = self._read_tone()
        return self.current == (bank, program)

    def ask(self, bank: int, program: int) -> bool:
        """Ask for one tone. True when the part moved to it."""
        if self.current == (bank, program):
            # Standing on the answer makes the question unanswerable. Anything
            # already known to be taken will do, so long as it is not this.
            elsewhere = (0, 0) if (bank, program) != (0, 0) else (0, 48)
            self.settle_on(*elsewhere)
        self.asks += 1
        self._send(bank, program)
        seen = self._read_tone()
        if seen is None:
            self.unread += 1
            return False
        self.current = seen
        return seen == (bank, program)


def survey(
    asker: Asker,
    *,
    banks=range(128),
    sample=SAMPLE_PROGRAMS,
    exhaustive: bool = False,
    progress=None,
) -> list[BankResult]:
    """Sample every bank, then sweep the ones that answered to anything.

    `exhaustive` drops the sampling and asks for all 128 programs in every bank.
    It costs about three times as much and is the only form with nothing to
    caveat: a bank whose one tone sits between the sampled programs is invisible
    to the sampled form and comes back as empty, which is indistinguishable from
    a bank that holds nothing.
    """
    found: list[BankResult] = []
    if exhaustive:
        for bank in banks:
            programs = [p for p in range(128) if asker.ask(bank, p)]
            if not programs:
                continue
            found.append(BankResult(bank=bank, programs=programs, swept=True))
            if progress:
                progress(f"bank {bank}: {len(programs)} tones")
        return found

    for bank in banks:
        hits = [p for p in sample if asker.ask(bank, p)]
        if not hits:
            continue
        result = BankResult(bank=bank, programs=list(hits))
        found.append(result)
        if progress:
            progress(f"bank {bank}: {len(hits)} of {len(sample)} sampled programs answered")
    for result in found:
        result.programs = [p for p in range(128) if asker.ask(result.bank, p)]
        result.swept = True
        if progress:
            progress(f"bank {result.bank}: {len(result.programs)} tones")
    return found


def summarise(
    found: list[BankResult], sample: int, asks: int, unread: int, *, exhaustive: bool = False
) -> str:
    total = sum(len(b.programs) for b in found)
    lines = [f"{total} tones across {len(found)} banks, from {asks} requests"]
    for b in found:
        runs = _runs(b.programs)
        lines.append(f"  bank {b.bank:3d}: {len(b.programs):3d} tones  {runs}")
    lines.append(
        "  every bank was asked for all 128 programs, so a bank absent above answered none"
        if exhaustive
        else f"  banks holding nothing at {sample} sampled programs were not swept, so a bank "
        "whose only tones sit between them reads as empty"
    )
    if unread:
        lines.append(f"  !! {unread} reads produced nothing usable")
    return "\n".join(lines)


def _runs(programs: list[int]) -> str:
    """Compress a program list to ranges, which is how a tone map reads."""
    if not programs:
        return ""
    out, start, previous = [], programs[0], programs[0]
    for p in programs[1:]:
        if p != previous + 1:
            out.append(f"{start}-{previous}" if start != previous else f"{start}")
            start = p
        previous = p
    out.append(f"{start}-{previous}" if start != previous else f"{start}")
    return ", ".join(out)


__all__ = ["SAMPLE_PROGRAMS", "Asker", "BankResult", "Probe", "summarise", "survey"]
