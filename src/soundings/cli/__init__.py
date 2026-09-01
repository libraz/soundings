"""Command line entry point."""

from __future__ import annotations

import argparse

from .. import roland
from . import offline, scan, sound, space, wire
from .session import Refused

COMMANDS = (wire, space, scan, sound, offline)
"""The modules that hold the subcommands, in the order --help lists them.

Each registers its own parsers and carries the handlers for them, so a flag and
the code that reads it stay in one file. Grouped by what a command needs to run:
the cable, the address space, a scan that writes, an audio interface, and
nothing at all.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="soundings", description=__doc__)
    parser.add_argument("--port", help="substring of the MIDI port name")
    parser.add_argument("--device-id", type=lambda s: int(s, 0), default=roland.DEFAULT_DEVICE_ID)
    sub = parser.add_subparsers(dest="command", required=True)
    for module in COMMANDS:
        module.register(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Refused as stopped:
        print(stopped)
        return 1
