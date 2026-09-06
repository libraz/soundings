"""Command line entry point."""

from __future__ import annotations

import argparse

from .. import hardware, roland
from . import inject, offline, ports, scan, sound, space, wire
from .session import Refused

COMMANDS = (wire, space, scan, ports, sound, inject, offline)
"""The modules that hold the subcommands, in the order --help lists them.

Each registers its own parsers and carries the handlers for them, so a flag and
the code that reads it stay in one file. Grouped by what a command needs to run:
the cable, the address space, a scan that writes, a pair of runs with the cable
moved between them, an audio interface, an audio interface alone, and nothing at
all.
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
