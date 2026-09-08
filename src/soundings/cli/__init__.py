"""Command line entry point."""

from __future__ import annotations

import argparse
import sys

from .. import hardware, record, roland
from . import (
    blocks,
    catalogue,
    contents,
    documents,
    effects,
    extent,
    inject,
    pairs,
    ports,
    runs,
    scan,
    sound,
    standing,
    state,
    wire,
)
from .session import Refused

COMMANDS = (
    wire,
    extent,
    contents,
    state,
    catalogue,
    scan,
    ports,
    sound,
    inject,
    pairs,
    runs,
    blocks,
    effects,
    standing,
    documents,
)
"""The modules that hold the subcommands, in the order --help lists them.

Each registers its own parsers and carries the handlers for them, so a flag and
the code that reads it stay in one file. Grouped by the question a command asks
rather than by what it asks with: the cable, which addresses are there, what they
hold, the state a reset puts back, what the unit's catalogues list, a scan that
writes, a pair of runs with the cable moved between them, an audio interface, an
audio interface alone, what an effect did between two takes, what one saved run
shows, what a block's records come to, how the effect types sort, where a unit
stands, and -- last, because it is the only one that is not a measurement at all
-- what a published document states about a unit somebody else made.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="soundings", description=__doc__)
    parser.add_argument("--port", help="substring of the MIDI port name")
    parser.add_argument("--device-id", type=lambda s: int(s, 0), default=roland.DEFAULT_DEVICE_ID)
    # Every command drives the unit unless it says otherwise, which is the safe
    # way round: a command added without a thought about this waits its turn
    # rather than joining a run already in progress. The modules that read what
    # an earlier run left behind clear the flag on their own parsers, so asking a
    # saved block's records a question does not queue behind the sweep capturing
    # the next one.
    parser.set_defaults(needs_unit=True)
    sub = parser.add_subparsers(dest="command", required=True)
    for module in COMMANDS:
        module.register(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # A device id only where the run addressed one. The flag that says so is the
    # same one that decides whether the unit is held: a command reading saved
    # takes or a published document never opens a port, and stamping the option's
    # default on its record would say a device answered when none was asked.
    asks_the_unit = getattr(args, "needs_unit", True)
    # Declared once, here, because this is the only place that knows both what
    # was asked for and what it was asked of. Every record written by the run
    # takes its identity from it without its writer being told to.
    record.invoked(
        record.Invocation(
            stage=args.command,
            argv=list(argv if argv is not None else sys.argv[1:]),
            midi_device_id=f"{args.device_id:02X}" if asks_the_unit else None,
        )
    )
    try:
        if not asks_the_unit:
            return args.func(args)
        with hardware.held(args.command):
            return args.func(args)
    except hardware.Busy as busy:
        print(busy)
        return 1
    except Refused as stopped:
        print(stopped)
        return 1
