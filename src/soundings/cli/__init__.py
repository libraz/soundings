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
    readings,
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
    readings,
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
    parser.add_argument(
        "--model-id",
        type=lambda s: int(s, 0),
        default=roland.GS_MODEL_ID,
        help="the Roland model ID to address, which with the device ID decides what three "
        "address bytes name. A unit answering under more than one opens a separate space in "
        "each. Only the stages that need nothing of a family accept a value other than GS",
    )
    # Every command drives the unit unless it says otherwise, which is the safe
    # way round: a command added without a thought about this waits its turn
    # rather than joining a run already in progress. The modules that read what
    # an earlier run left behind clear the flag on their own parsers, so asking a
    # saved block's records a question does not queue behind the sweep capturing
    # the next one.
    #
    # `takes_model_id` is the same way round and for the same reason. A stage
    # whose messages, addresses or catalogue are GS's own produces nothing
    # meaningful under another model id -- a GS Reset is a write to one address
    # in one space, and elsewhere it is a write to whatever those bytes happen to
    # mean there. The stages that need only RQ1 and DT1 say so on their own
    # parsers, so a command added without a thought about it refuses rather than
    # sends, and its record cannot claim a space it did not ask in.
    parser.set_defaults(needs_unit=True, takes_model_id=False)
    sub = parser.add_subparsers(dest="command", required=True)
    for module in COMMANDS:
        module.register(sub)
    return parser


NOT_ADDRESSED_BY_MODEL_ID = (
    "{stage} cannot be aimed at model id {model:02X}. What it sends is either GS's own -- a "
    "message, a fixed address or a catalogue that means nothing in another family's space, "
    "where those bytes name whatever they happen to name there -- or it carries no model id at "
    "all. Either way the run would ask in one space and record another. The stages that need "
    "only RQ1 and DT1 take --model-id; this one would need the other family defined first, and "
    "nothing here holds a definition of one."
)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # A device id only where the run addressed one. The flag that says so is the
    # same one that decides whether the unit is held: a command reading saved
    # takes or a published document never opens a port, and stamping the option's
    # default on its record would say a device answered when none was asked.
    asks_the_unit = getattr(args, "needs_unit", True)
    if args.model_id != roland.GS_MODEL_ID and not getattr(args, "takes_model_id", False):
        print(NOT_ADDRESSED_BY_MODEL_ID.format(stage=args.command, model=args.model_id))
        return 1
    # Declared once, here, because this is the only place that knows both what
    # was asked for and what it was asked of. Every record written by the run
    # takes its identity from it without its writer being told to.
    record.invoked(
        record.Invocation(
            stage=args.command,
            argv=list(argv if argv is not None else sys.argv[1:]),
            midi_device_id=f"{args.device_id:02X}" if asks_the_unit else None,
            midi_model_id=f"{args.model_id:02X}" if asks_the_unit else None,
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
