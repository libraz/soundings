"""Opening the machine, and refusing to measure down a path that has not been proved.

Both ways this measurement breaks are silent, so no command that touches the unit
starts without the selftest passing first. Putting that here rather than at the
head of each command means a new command cannot be written without it.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager

from ..midi import MidiLink
from ..selftest import midi_selftest


class Refused(Exception):
    """A run stopped before it could produce anything worth reading.

    Carries the sentence shown to the reader. The command line prints it and
    exits non-zero; nothing else catches it, because there is nothing else to be
    done about a path that cannot be trusted.
    """


@contextmanager
def verified_link(
    args: argparse.Namespace,
    *,
    refusing: str,
    show_port: bool = False,
    announce: str | None = None,
) -> Iterator[MidiLink]:
    """Open the port, prove the round trip, and hand over the link.

    `refusing` is the verb the refusal is phrased with -- "sweeping", "writing"
    -- so that a reader who sees the message knows what did not happen rather
    than only that something did not.
    """
    with MidiLink(args.port) as link:
        if show_port:
            print(f"MIDI: {link.ports.output_name}")
        if announce:
            print(announce)
        report = midi_selftest(link, repeats=args.verify_reads, device_id=args.device_id)
        print(report)
        if not report.passed:
            raise Refused(f"\nSelftest failed. Not {refusing}.")
        yield link


def prepared(link: MidiLink, specs, *, device_id: int, settle: float) -> bool:
    """Put the unit in the state a run needs, and prove each write took.

    Here rather than beside any one command, for the same reason the selftest is:
    an unverified preparation is the other way a run measures down a path it never
    established. The two failures look alike from outside -- the unit answers, the
    take comes back, the numbers are plausible -- and both are silent.

    Reading each address back is what separates a preparation the unit obeyed from
    one it ignored. A parameter wider than a byte is the common case: the unit
    takes the write, keeps what it had, and the run goes on measuring the state it
    meant to leave behind.
    """
    import time

    from .. import roland

    for address, values in specs:
        link.send(roland.dt1(address, list(values), device_id=device_id))
        time.sleep(settle)
        reply = link.exchange(roland.rq1(address, len(values), device_id=device_id))
        parsed = roland.parse_dt1(reply)
        got = None if parsed is None else list(parsed.data)
        if got != list(values):
            wanted = " ".join(f"{v:02X}" for v in values)
            found = "no reply" if got is None else " ".join(f"{v:02X}" for v in got)
            print(
                f"\n    {address} was set to {wanted} and reads back {found}. The state this "
                "run needs is not there, so anything measured from here would be about the "
                "state rather than about what the run was asked."
            )
            return False
    return True


def took(link: MidiLink, address: str, value: int, *, device_id: int) -> tuple[bool, str]:
    """Whether the address holds what was just written to it, and what it holds.

    The preparation is read back and the setting under test was not, which is the
    same failure at the other end of the run: an address that took the write and
    kept what it had leaves every take a take of one setting, and the pair of
    them reads as a parameter that does nothing. That null is indistinguishable
    from a real one and there is nothing in the record to tell them apart.

    An address that answers no one-byte read cannot be checked this way, and that
    is reported as unverified rather than as a failure -- two addresses in a part
    block are readable only as part of a wider region, and refusing them would
    drop them from the sweep for being unreadable rather than measuring them.
    """
    from .. import roland

    while link.receive(timeout=0.02):
        pass
    parsed = roland.parse_dt1(link.exchange(roland.rq1(address, 1, device_id=device_id)))
    if parsed is None or len(parsed.data) != 1:
        return True, "no reply to a one byte read, so the write could not be checked"
    got = parsed.data[0]
    return got == (value & 0x7F), f"{got:02X}"
