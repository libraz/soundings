"""Find what each address accepts, by writing to it and reading it back.

A write is not acknowledged. DT1 produces no reply, so the only evidence that
anything happened is what the address reads as afterwards, and the classification
this module produces is exactly that and nothing more:

- `accepts`   the value written comes back unchanged, over the whole range tried
- `clamps`    a value outside some range comes back as the edge of that range
- `refuses out of range`  a value outside the range leaves the address as it was
- `quantises` the value comes back changed in a way that is not a clamp
- `ignores`   every write leaves the address reading as it did before
- `unchanging, so a clamp and a refusal cannot be told apart` -- the address
  rests at exactly the value a clamp would produce, so no write can separate
  the two, and neither is asserted

Telling a clamp from a refusal takes putting a known good value in between. Both
leave the address reading as something other than what was written, and after an
ascending probe both leave it reading the top of the range -- the accepted value
just below is also the bound a clamp would return.

**None of these says the unit sounds different.** An address that stores
what it is given may still be wired to nothing, and separating a parameter that
is kept from one that is used takes audio, not a read back. Storing is the
weaker claim and it is the only one made here.

Everything is restored. The original bytes are read before anything is written,
written back afterwards, and read again to confirm they took -- and a region
whose original could not be read in the first place is never written to at all,
because there would be nothing to put back. A failed restore stops the run
rather than being noted and passed over: every measurement after it would be
starting from a state nobody chose.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import roland
from .midi import MidiLink

# Writing here is not a parameter change. SYSTEM MODE SET reinitialises the unit,
# which would discard the very state this module is in the middle of restoring.
NOTE = "Classifications describe what an address stores, not what it does. Nothing here was heard."

SINGLE_BYTE_LIMIT = (
    "Each byte was written on its own, and a byte the unit would not answer a one byte "
    "request for was never written to, since there would have been nothing to put back. "
    "Those are listed as skipped. So this describes the addresses that answer singly, and "
    "a parameter reachable only as a whole falls outside it. Which of the skipped bytes are "
    "that, rather than simply undefined, is not settled here."
)

WENT_DEAF = (
    "An address that had been answering stopped, so the unit was no longer talking and "
    "nothing read after that point would have been about an address. The run stopped there "
    "and kept what it had already measured; the regions it never reached are absent rather "
    "than empty."
)

NEVER_WRITE = {
    (0x40, 0x00, 0x7F),
}

UNDECIDABLE = "unchanging, so a clamp and a refusal cannot be told apart"
"""The one verdict that asserts nothing, and so the one the ladder can overturn."""


class RestoreFailed(RuntimeError):
    """An address would not take its original value back, so the unit is not as it was.

    Carries the region it happened in, so the bytes measured before it survive.
    They were measured and restored like any others, and throwing them away with
    the failure would discard the only record of what led up to it.
    """

    def __init__(self, message: str, region: RegionProbe | None = None):
        super().__init__(message)
        self.region = region


@dataclass
class ByteProbe:
    address: str
    original: int
    written: list[tuple[int, int]] = field(default_factory=list)
    """(value written, value read back) in the order tried."""

    low: int | None = None
    """What the address read as after writing 00. Its minimum, if it clamps."""

    high: int | None = None
    """What the address read as after writing 7F. Its maximum, if it clamps."""

    classification: str = ""
    restored: bool = False

    restore_tries: int = 1
    """Writes it took to put the original back. Recorded when it took more than one.

    Measured over a whole address space: one write in 213328 did not arrive, and
    the address held the last value that did -- long enough after to be read back
    five times. That is a fact about the link under sustained traffic, not about
    the address, so it is kept where a reader can weigh it rather than smoothed
    away by the retry that recovers from it.
    """

    @property
    def accepted(self) -> list[int]:
        return sorted({w for w, r in self.written if r == w})

    @property
    def range(self) -> str | None:
        """The bounds, when they were measured. A refusing address has no range until asked."""
        if self.classification == "refuses out of range":
            taken = self.accepted
            return f"{min(taken):02X}..{max(taken):02X} of the values tried" if taken else None
        if self.low is None or self.high is None or self.high < self.low:
            return None
        return f"{self.low:02X}..{self.high:02X}"

    def to_json(self) -> dict:
        return {
            "address": self.address,
            "original": f"{self.original:02X}",
            "wrote_read": [[f"{w:02X}", f"{r:02X}"] for w, r in self.written],
            "accepted": [f"{v:02X}" for v in self.accepted],
            "range": self.range,
            "classification": self.classification,
            "restored": self.restored,
            **({} if self.restore_tries == 1 else {"restore_tries": self.restore_tries}),
        }


@dataclass
class RegionProbe:
    start: str
    length: int
    bytes: list[ByteProbe] = field(default_factory=list)
    region_restored: bool = False
    """The whole region was re-read at the end and matched the snapshot taken first."""

    skipped: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        counts: dict[str, int] = {}
        for b in self.bytes:
            counts[b.classification] = counts.get(b.classification, 0) + 1
        return {
            "start": self.start,
            "length": self.length,
            "region_restored": self.region_restored,
            "classifications": counts,
            "skipped": self.skipped,
            "bytes": [b.to_json() for b in self.bytes],
        }


class Writer:
    def __init__(
        self,
        link: MidiLink,
        *,
        device_id: int = roland.DEFAULT_DEVICE_ID,
        settle: float = 0.02,
        read_timeout: float = 0.5,
    ):
        self.link = link
        self.device_id = device_id
        self.settle = settle
        self.read_timeout = read_timeout
        self.writes = 0
        self.reads = 0

    def read_byte(self, address: tuple[int, int, int]) -> int | None:
        """Read one byte, refusing anything that is not this address's own reply."""
        self.reads += 1
        request = roland.rq1(address, 1, device_id=self.device_id)
        raw = self.link.exchange(request, timeout=self.read_timeout)
        if roland.malformation(raw) is not None:
            while self.link.receive(timeout=0.4):
                pass
            raw = self.link.exchange(request, timeout=self.read_timeout)
        reply = roland.parse_dt1(raw)
        if reply is None or reply.address != address or reply.size != 1:
            return None
        return reply.data[0]

    def answering(self, address: tuple[int, int, int]) -> bool:
        """Whether an address that was answering still is.

        A long run needs this because a unit that stops talking does not look
        like anything going wrong: every byte then reads as unreadable, which is
        skipped rather than written to, so the probe walks the rest of the map
        producing empty regions and a clean exit. Asked between regions against
        an address already known to answer, it separates a map of silent
        addresses from a silent machine.
        """
        return self.read_byte(address) is not None

    def write_byte(self, address: tuple[int, int, int], value: int) -> None:
        if tuple(address) in NEVER_WRITE:
            raise ValueError(f"{address} is on the never-write list")
        self.writes += 1
        self.link.send(roland.dt1(address, [value], device_id=self.device_id))
        time.sleep(self.settle)

    def write_then_read(self, address: tuple[int, int, int], value: int) -> int | None:
        self.write_byte(address, value)
        return self.read_byte(address)

    def probe_byte(self, address: tuple[int, int, int], original: int) -> ByteProbe:
        """Write to the address and read it back, deriving its range rather than assuming one.

        The extremes come first because a clamping address answers them with its
        own bounds: write 00 and the read back is the minimum, write 7F and it is
        the maximum. Only then are interior values chosen, from inside the range
        that was just measured.

        Choosing them in advance is what an earlier version did, and it read a
        clamp as a distortion: it probed 40, which for a macro selector bounded
        at 07 is outside the range, so the value came back changed and the
        address was filed as quantising. An interior point has to be interior to
        the measured range, not to the byte.
        """
        addr = f"{address[0]:02X} {address[1]:02X} {address[2]:02X}"
        probe = ByteProbe(address=addr, original=original)

        def attempt(value: int) -> int | None:
            got = self.write_then_read(address, value)
            if got is None:
                probe.classification = "unreadable after a write"
                return None
            probe.written.append((value, got))
            return got

        low = attempt(0x00)
        high = None if low is None else attempt(0x7F)
        if low is not None and high is not None:
            probe.low = low
            probe.high = high
            if high > low:
                for value in sorted({(low + high) // 2, low + 1, high - 1}):
                    if low < value < high and attempt(value) is None:
                        break
                probe.classification = self._classify(probe, original, low, high)
            else:
                probe.classification = self._out_of_range_behaviour(address, probe, low)
            # An address with a full 00..7F range has nothing outside it to test.
            # Any other one has its out-of-range behaviour asserted so far from a
            # single value, which is one point too few to describe a rule.
            if (low, high) != (0x00, 0x7F):
                self._ladder(address, probe, low)
                probe.classification = self._after_the_ladder(probe)

        probe.restore_tries = self.restore(address, original)
        probe.restored = probe.restore_tries > 0
        return probe

    RESTORE_TRIES = 3

    def restore(self, address: tuple[int, int, int], original: int) -> int:
        """Put a byte back, and say how many writes it took. 0 means it would not go.

        Retried because a lost message is not a fact about the address, and with
        one attempt it stops the run as though it were. Measured: over 213328
        writes exactly one failed to arrive, and what it looked like afterwards
        was an address holding a value it had been given several writes earlier,
        with everything sent since gone. A second write, sent after longer, put
        it back at once.

        The later tries wait longer rather than repeating the same thing faster.
        The one failure came in the middle of sustained traffic, which is when a
        link coalesces or drops frames, so what the retry has to give it is time.
        """
        for attempt in range(1, self.RESTORE_TRIES + 1):
            if attempt > 1:
                time.sleep(self.settle * 10)
            self.write_byte(address, original)
            if self.read_byte(address) == original:
                return attempt
        return 0

    @staticmethod
    def _after_the_ladder(probe: ByteProbe) -> str:
        """Read the verdict again, now that the evidence for it exists.

        A verdict reached before the ladder is reached without what the ladder
        finds. `_out_of_range_behaviour` needs a value the address has already
        taken to put in between, and an address resting at the same value that
        both 00 and 7F leave it at has none to offer -- so it reports that a
        clamp and a refusal cannot be told apart, and then the ladder goes and
        finds three. Measured: the 64 one-of-four selectors at 00 01 xx all take
        00 to 03 and refuse everything above, and the 16 of them resting at 00
        were filed as indistinguishable while their own rows showed 01, 02 and
        03 going in and coming back.

        Only that verdict is revisited, because it is the only one the ladder
        contradicts. Of 11544 addresses filed as clamping, every one the ladder
        pushed past its bound read that bound back.
        """
        if probe.classification != UNDECIDABLE or len(probe.accepted) < 2:
            return probe.classification
        # The ladder writes a value the address takes before each trial, so a
        # value that does not survive left the address holding that one: it was
        # refused rather than clamped to a bound it does not have.
        return "refuses out of range"

    LADDER = (1, 2, 3, 4, 5, 6, 7, 8, 11, 15, 31, 63, 126)

    def _ladder(self, address: tuple[int, int, int], probe: ByteProbe, marker: int) -> None:
        """Find which values an address will take, resetting to a known value between each.

        The reset is the whole point. Walking a ladder upwards without one makes
        a refusing address look exactly like a clamping one: 07 is accepted, then
        08 is refused and reads back as 07, which is also what a clamp to 07
        would return. Every value above it then reads 07 too, and the address
        files as clamping to a bound it does not have. Writing a known low value
        first separates them -- a refusal reads back as that value, a clamp reads
        back as the bound.

        Written out one value at a time rather than bisected. A bisection assumes
        the accepted values form a single run, and that is part of what is being
        measured; a selector with a hole would come back as a range covering it.
        """
        for value in self.LADDER:
            if self.write_then_read(address, marker) != marker:
                return
            got = self.write_then_read(address, value)
            if got is None:
                return
            probe.written.append((value, got))

    def _out_of_range_behaviour(
        self, address: tuple[int, int, int], probe: ByteProbe, low: int
    ) -> str:
        """Separate a clamp to the bottom from a write that was simply refused.

        Writing 00 and then 7F and reading the same value back both times admits
        two readings: the address clamped 7F down to that value, or it refused 7F
        and kept what the previous write left. They are different behaviours and
        the two probes cannot tell them apart, because the previous write happens
        to have left exactly the value a bottom clamp would produce.

        Putting a known good value in between settles it. Write something the
        address has already accepted, then 7F: if the accepted value survives,
        the out of range write was refused; if it turns into the bound, it was
        clamped.
        """
        marker = next((r for w, r in probe.written if w == r and r != low), None)
        if marker is None:
            marker = probe.original if probe.original != low else None
        if marker is None:
            # There is no second value to put in between, because the address has
            # never held one: it rests where a clamp would leave it. The two
            # readings stay open rather than one of them being picked.
            return UNDECIDABLE
        placed = self.write_then_read(address, marker)
        probe.written.append((marker, placed if placed is not None else -1))
        if placed != marker:
            return "clamps"
        after = self.write_then_read(address, 0x7F)
        probe.written.append((0x7F, after if after is not None else -1))
        if after == marker:
            return "refuses out of range"
        return "clamps"

    @staticmethod
    def _classify(probe: ByteProbe, original: int, low: int, high: int) -> str:
        if all(r == original for _, r in probe.written):
            return "ignores"
        if high < low:
            return "quantises"
        interior = [(w, r) for w, r in probe.written if low < w < high]
        if not all(w == r for w, r in interior):
            return "quantises"
        if low == 0x00 and high == 0x7F:
            return "accepts"
        return "clamps"

    def probe_region(
        self, start: tuple[int, int, int], length: int, *, progress=None
    ) -> RegionProbe:
        """Probe each byte of a region, having first proved the region is restorable."""
        s = f"{start[0]:02X} {start[1]:02X} {start[2]:02X}"
        result = RegionProbe(start=s, length=length)

        snapshot: dict[tuple[int, int, int], int] = {}
        for i in range(length):
            address = (start[0], start[1], start[2] + i)
            addr = f"{address[0]:02X} {address[1]:02X} {address[2]:02X}"
            if tuple(address) in NEVER_WRITE:
                result.skipped.append(f"{addr} (never-write list)")
                continue
            value = self.read_byte(address)
            if value is None:
                result.skipped.append(f"{addr} (original could not be read)")
                continue
            snapshot[address] = value

        if progress:
            progress(f"{s}: {len(snapshot)} of {length} bytes readable, so writable and restorable")

        for address, original in snapshot.items():
            probe = self.probe_byte(address, original)
            result.bytes.append(probe)
            if not probe.restored:
                raise RestoreFailed(
                    f"{probe.address} would not take back its original {original:02X} "
                    f"in {self.RESTORE_TRIES} tries; stopping so nothing is measured "
                    f"from a state nobody chose",
                    result,
                )

        result.region_restored = all(self.read_byte(a) == v for a, v in snapshot.items())
        if progress:
            counts: dict[str, int] = {}
            for b in result.bytes:
                counts[b.classification] = counts.get(b.classification, 0) + 1
            progress(f"{s}: {counts}, region restored: {result.region_restored}")
        return result


def summarise(regions: list[RegionProbe]) -> str:
    lines = []
    counts: dict[str, int] = {}
    for region in regions:
        for b in region.bytes:
            counts[b.classification] = counts.get(b.classification, 0) + 1
    total = sum(counts.values())
    lines.append(f"{total} bytes probed across {len(regions)} regions")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {v:4d} {k}")
    unrestored = [r.start for r in regions if not r.region_restored]
    if unrestored:
        lines.append(f"  !! regions not restored: {', '.join(unrestored)}")
    else:
        lines.append("  every region read back as it did before the writes")
    lines.append("  storing a value is not the same as using it; nothing here was heard, only read")
    return "\n".join(lines)


__all__ = ["ByteProbe", "RegionProbe", "RestoreFailed", "Writer", "summarise"]
