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
