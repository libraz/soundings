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

    The proof is made in the space the probe address is known to answer in, not
    in the one the run is about to ask in. They are different questions: this one
    asks whether the cable carries a message and its reply intact, and an address
    that answers nothing -- which is the ordinary state of a space being explored
    -- would report a sound path as a broken one and stop the run. A space that
    does not answer at all is a finding for the stage to make, not a reason to
    refuse to start it.
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

    A DT1 says which address it is for, and a reply for another one is not an
    answer to this read: the sweeps all check it and this did not, which is how a
    read of an address that answers none at all came back holding a neighbour's
    byte.
    """
    import time

    from .. import roland

    for address, values in specs:
        link.send(roland.dt1(address, list(values), device_id=device_id))
        time.sleep(settle)
        reply = link.exchange(roland.rq1(address, len(values), device_id=device_id))
        parsed = roland.parse_dt1(reply)
        for_this_address = parsed is not None and tuple(parsed.address) == tuple(
            roland.address_bytes(address)
        )
        got = list(parsed.data) if for_this_address else None
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


def holds(link: MidiLink, address: str, *, device_id: int) -> int | None:
    """The byte an address holds, or None when it answers no one-byte read.

    The preparation is read back and the setting under test was not, which is the
    same failure at the other end of the run: an address that took the write and
    kept what it had leaves every take a take of one setting, and the pair of
    them reads as a parameter that does nothing. That null is indistinguishable
    from a real one and there is nothing in the record to tell them apart.

    What comes back is the byte rather than a verdict on it. An address that
    clamps a write, or rounds it to something it has, is still answering the
    question as long as the two settings land apart, so the comparison is between
    the two readings and not between a reading and what was asked.

    An address that answers no one-byte read is None rather than a failure -- two
    in a part block are readable only as part of a wider region, and refusing
    them would drop them from the sweep for being unreadable.

    **The reply's own address is checked against the one asked for.** A DT1 is
    self-describing and nothing here forced the two to agree, so a reply left
    over from an earlier exchange satisfied a read of an address that answers
    none at all -- measured on this unit, a one-byte read of an address readable
    only inside a wider region came back holding a neighbour's byte, and the run
    refused a measurable address on the strength of it.
    """
    from .. import roland

    while link.receive(timeout=0.02):
        pass
    wanted = roland.address_bytes(address)
    parsed = roland.parse_dt1(link.exchange(roland.rq1(address, 1, device_id=device_id)))
    if parsed is None or len(parsed.data) != 1 or tuple(parsed.address) != tuple(wanted):
        return None
    return int(parsed.data[0])
