"""Command line entry point."""

from __future__ import annotations

import argparse
import sys

from .. import hardware, record, roland
from . import catalogue, contents, extent, inject, offline, ports, scan, sound, state, wire
from .session import Refused

COMMANDS = (wire, extent, contents, state, catalogue, scan, ports, sound, inject, offline)
"""The modules that hold the subcommands, in the order --help lists them.

Each registers its own parsers and carries the handlers for them, so a flag and
the code that reads it stay in one file. Grouped by the question a command asks
rather than by what it asks with: the cable, which addresses are there, what
they hold, the state a reset puts back, what the unit's catalogues list, a scan
that writes, a pair of runs with the cable moved between them, an audio
interface, an audio interface alone, and nothing at all.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="soundings", description=__doc__)
    parser.add_argument("--port", help="substring of the MIDI port name")
    parser.add_argument("--device-id", type=lambda s: int(s, 0), default=roland.DEFAULT_DEVICE_ID)
    # Every command drives the unit unless it says otherwise, which is the safe
    # way round: a command added without a thought about this waits its turn
    # rather than joining a run already in progress. `offline` clears the flag on
    # its own parsers, and it is the group that reads takes with nothing attached.
    parser.set_defaults(needs_unit=True)
    sub = parser.add_subparsers(dest="command", required=True)
    for module in COMMANDS:
        module.register(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Declared once, here, because this is the only place that knows both what
    # was asked for and what it was asked of. Every record written by the run
    # takes its identity from it without its writer being told to.
    record.invoked(
        record.Invocation(
            stage=args.command,
            argv=list(argv if argv is not None else sys.argv[1:]),
            midi_device_id=f"{args.device_id:02X}",
        )
    )
    try:
        if not getattr(args, "needs_unit", True):
            return args.func(args)
        with hardware.held(args.command):
            return args.func(args)
    except hardware.Busy as busy:
        print(busy)
        return 1
    except Refused as stopped:
        print(stopped)
        return 1
