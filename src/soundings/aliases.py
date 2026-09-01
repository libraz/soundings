"""Find where a control change is stored, by watching the address space change.

The method is a difference: read a set of regions, send one message, read them
again, and see which bytes moved. What makes it a measurement rather than a
coincidence is everything around that.

**A control run first, with no message sent.** Two snapshots taken back to back
must be identical. Anything that moves on its own -- a running LFO, a counter, a
level meter -- would otherwise be attributed to whichever message happened to be
in flight, and the attribution would look exactly like a real one. Addresses
that fail the control are excluded by name rather than by hope.

**Two values, not one.** A byte is only attributed to a message if it moved for
both values sent and moved to something different each time. A byte that changes
once has changed; a byte that follows what it was told is storing it.

**The value sent is compared with the value stored.** An address that ends up
holding exactly what was transmitted is a different finding from one that holds
something derived from it, and the pair is recorded so the difference survives.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import roland
from .midi import MidiLink


@dataclass
class Change:
    address: str
    """The address of the byte itself, not of the region it was read in."""

    before: int
    after: int


@dataclass
class Attribution:
    label: str
    sent: list[int]
    """The two values transmitted, in order."""

    per_byte: dict[str, list[int]] = field(default_factory=dict)
    """Address -> what it read as after each value."""

    verbatim: list[str] = field(default_factory=list)
    """Addresses that ended up holding exactly what was sent, both times."""

    def to_json(self) -> dict:
        return {
            "control": self.label,
            "sent": [f"{v:02X}" for v in self.sent],
            "stores_verbatim": self.verbatim,
            "bytes": {a: [f"{v:02X}" for v in vs] for a, vs in sorted(self.per_byte.items())},
        }


class Snapshotter:
    """Reads a fixed set of regions, the same way every time."""

    def __init__(
        self,
        link: MidiLink,
        regions: list[tuple[tuple[int, int, int], int]],
        *,
        device_id: int = roland.DEFAULT_DEVICE_ID,
        timeout: float = 0.4,
    ):
        self.link = link
        self.regions = regions
        self.device_id = device_id
        self.timeout = timeout
        self.reads = 0
        self.unread = 0

    def take(self) -> dict[tuple[int, int, int], int]:
        """Read every region and flatten it to one byte per address.

        A region that fails to read is left out rather than recorded as zeros: an
        absent byte cannot differ from itself, which keeps a failed read from
        appearing as a change on the next comparison.
        """
        out: dict[tuple[int, int, int], int] = {}
        for start, length in self.regions:
            self.reads += 1
            request = roland.rq1(start, length, device_id=self.device_id)
            raw = self.link.exchange(request, timeout=self.timeout)
            if roland.malformation(raw) is not None:
                while self.link.receive(timeout=0.3):
                    pass
                raw = self.link.exchange(request, timeout=self.timeout)
            reply = roland.parse_dt1(raw)
            if reply is None or reply.address != start:
                self.unread += 1
                continue
            for i, value in enumerate(reply.data):
                out[(start[0], start[1], start[2] + i)] = value
        return out


def differences(
    before: dict[tuple[int, int, int], int], after: dict[tuple[int, int, int], int]
) -> list[Change]:
    """Bytes present in both snapshots that hold a different value in the second."""
    changed = []
    for address, old in before.items():
        new = after.get(address)
        if new is not None and new != old:
            changed.append(Change(f"{address[0]:02X} {address[1]:02X} {address[2]:02X}", old, new))
    return sorted(changed, key=lambda c: c.address)


def control_run(shot: Snapshotter, *, rounds: int = 3, settle: float = 0.15) -> set[str]:
    """Names the addresses that change with nothing sent to them.

    Run before anything is attributed. Its output is a exclusion list, and an
    empty one is the result worth having -- it is what makes a later difference
    mean the message caused it.
    """
    restless: set[str] = set()
    previous = shot.take()
    for _ in range(rounds):
        time.sleep(settle)
        current = shot.take()
        restless.update(c.address for c in differences(previous, current))
        previous = current
    return restless


class Scanner:
    def __init__(
        self,
        shot: Snapshotter,
        *,
        restless: set[str],
        settle: float = 0.1,
    ):
        self.shot = shot
        self.restless = restless
        self.settle = settle

    def attribute(self, label: str, send: list[list[int]], values: list[int]) -> Attribution | None:
        """Send each message in turn and keep the bytes that followed all of them.

        Returns None when nothing followed, which is the common answer: most
        control changes are not stored anywhere this can see.
        """
        result = Attribution(label=label, sent=values)
        moved: dict[str, list[int]] = {}
        before = self.shot.take()
        for message in send:
            self.shot.link.send(message)
            time.sleep(self.settle)
            after = self.shot.take()
            for change in differences(before, after):
                if change.address in self.restless:
                    continue
                moved.setdefault(change.address, []).append(change.after)
            before = after

        for address, seen in moved.items():
            if len(seen) == len(send) and len(set(seen)) == len(send):
                result.per_byte[address] = seen
                if seen == values:
                    result.verbatim.append(address)
        return result if result.per_byte else None


def control_change(channel: int, controller: int, value: int) -> list[int]:
    return [0xB0 | (channel & 0x0F), controller & 0x7F, value & 0x7F]


def summarise(found: list[Attribution], restless: set[str], unread: int) -> str:
    lines = [f"{len(found)} controls were stored somewhere this could see"]
    for a in found:
        where = ", ".join(f"{k}={'/'.join(f'{v:02X}' for v in vs)}" for k, vs in a.per_byte.items())
        mark = "  (verbatim)" if a.verbatim else ""
        lines.append(f"  {a.label:28} -> {where}{mark}")
    if restless:
        lines.append(
            f"  {len(restless)} addresses moved with nothing sent and were excluded: "
            + ", ".join(sorted(restless)[:8])
        )
    else:
        lines.append("  nothing moved with nothing sent, so a difference means the message did it")
    if unread:
        lines.append(f"  !! {unread} region reads failed and were left out of the comparison")
    return "\n".join(lines)


__all__ = [
    "Attribution",
    "Change",
    "Scanner",
    "Snapshotter",
    "control_change",
    "control_run",
    "differences",
    "summarise",
]
