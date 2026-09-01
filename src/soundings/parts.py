"""Where a channel's settings live in the address space, and putting them back.

Two things a scan needs that are not measurements. The first is arithmetic: GS
does not lay the parts out in channel order, so an address computed the obvious
way answers, and answers about the wrong part. The second is bookkeeping: a scan
that writes has to leave the unit where it found it, including the state that is
held but not readable.
"""

from __future__ import annotations

import time

from . import roland
from .midi import MidiLink

Address = tuple[int, int, int]

TONE = 0x10
"""The per-part block holding bank and program."""

MAPPED = 0x40
"""The per-part block CC32 writes into."""


def part_block(family: int, channel: int) -> Address:
    """The per-part block for a channel, in one of the families of them.

    GS numbers the parts so that channel 10 comes first: 40 f0 is part 10, and
    40 f1 through 40 fF are channels 1 to 9 and 11 to 16 in order. `family` is
    the whole high nibble of the middle byte -- `TONE` for the part block that
    holds the tone, `MAPPED` for the one CC32 writes into -- so passing 0x00 by
    mistake addresses the patch common block instead, which answers, and which
    holds something entirely unrelated.
    """
    index = 0 if channel == 9 else (channel + 1 if channel < 9 else channel)
    return (0x40, family | index, 0x00)


def master_tune_cents(link: MidiLink, *, device_id: int) -> float | None:
    """MASTER TUNE as cents off A440, from the four nibbles the unit stores it in."""
    reply = roland.parse_dt1(link.exchange(roland.rq1((0x40, 0x00, 0x00), 4, device_id=device_id)))
    if reply is None or reply.size != 4:
        return None
    packed = 0
    for nibble in reply.data:
        packed = (packed << 4) | (nibble & 0x0F)
    return (packed - 0x400) / 10.0


class Restorer:
    """Remembers what a scan is about to overwrite, and puts it back afterwards.

    Every read drains first. This runs among hundreds of request-and-reply pairs,
    and a reply still in flight is answered to whichever request asks next.
    """

    def __init__(
        self,
        link: MidiLink,
        *,
        device_id: int = roland.DEFAULT_DEVICE_ID,
        settle: float = 0.2,
    ):
        self.link = link
        self.device_id = device_id
        self.settle = settle

    def _read(self, address: Address, size: int) -> list[int] | None:
        while self.link.receive(timeout=0.05):
            pass
        request = roland.rq1(address, size, device_id=self.device_id)
        reply = roland.parse_dt1(self.link.exchange(request, timeout=0.6))
        if reply is None or reply.address != address:
            return None
        return list(reply.data)

    def remember(self, addresses: list[Address]) -> dict[Address, int]:
        """Read one byte at each address, leaving out any that will not answer.

        What comes back is both the state to restore and the list of addresses
        worth writing at all: a byte with no original here is one there would be
        nothing to put back for.
        """
        out: dict[Address, int] = {}
        for address in addresses:
            data = self._read(address, 1)
            if data is not None and len(data) == 1:
                out[address] = data[0]
        return out

    def put_back(self, originals: dict[Address, int]) -> str:
        """Put back every byte the scan wrote, and say so only after re-reading it.

        A scan that writes has to end where it started or the next measurement is
        taken from a state nobody chose.
        """
        for address, value in originals.items():
            self.link.send(roland.dt1(address, [value], device_id=self.device_id))
        time.sleep(self.settle)
        after = self.remember(list(originals))
        wrong = [
            f"{a[0]:02X} {a[1]:02X} {a[2]:02X}" for a, v in originals.items() if after.get(a) != v
        ]
        if wrong:
            return f"NOT restored: {', '.join(wrong)}"
        return f"{len(originals)} bytes put back and re-read"

    def clear_bank_latch(self, channel: int) -> str:
        """Put the bank select latch back where the scan found it.

        Bank select is held without changing anything readable until a program
        change commits the three of them together, and a pair the unit does not
        have is discarded whole. So a scan that sends CC0 or CC32 on their own --
        which a controller sweep does, having no reason to send a program change
        -- ends with a latch set to whatever it tried last, and every program
        change in the next run is thrown away. Nothing in the address space says
        so.

        The committing program change is the tone the part already holds, and the
        bank halves are read back rather than assumed, so the commit puts the part
        exactly where it was rather than somewhere tidy.
        """
        tone = part_block(TONE, channel)
        part = self._read(tone, 2)
        mapped = self._read(part_block(MAPPED, channel), 1)
        if part is None or mapped is None:
            return "could not be read back, so the bank latch was left as the scan left it"
        self.link.send([0xB0 | (channel & 0x0F), 0, part[0]])
        self.link.send([0xB0 | (channel & 0x0F), 32, mapped[0]])
        self.link.send([0xC0 | (channel & 0x0F), part[1]])
        time.sleep(0.25)
        after = self._read(tone, 2)
        if after != part:
            return f"restoring it moved the part from {part} to {after}"
        return f"committed bank {part[0]}, map {mapped[0]}, program {part[1]}"
