"""The command line surface, held against a recorded copy of itself.

Every flag here is part of how a measurement is described: a default that moves
silently changes what a later run means, and a help string is often the only
place a caveat is written down. None of that is exercised by the measurement
tests, because none of it needs a machine to be wrong.

So the surface is recorded rather than asserted piecemeal. Regenerate the file
with `python tests/test_cli_surface.py` and read the diff; a change that is
meant is one line of review, and a change that is not shows up as one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from soundings.cli import build_parser

SURFACE = Path(__file__).parent / "data" / "cli-surface.json"


def _round_trips(value):
    """A tuple as the list JSON reads it back as.

    argparse gives a two-argument flag a tuple metavar. Recorded as a tuple it
    comes back from the file as a list and never compares equal again, so the
    gate fails on every run until the flag is taken out -- which is the gate
    refusing the surface rather than the surface being wrong.
    """
    return list(value) if isinstance(value, tuple) else value


def _actions(parser: argparse.ArgumentParser) -> list[dict]:
    return [
        {
            "option_strings": action.option_strings,
            "dest": action.dest,
            "nargs": action.nargs,
            "default": repr(action.default),
            "required": action.required,
            "help": action.help,
            "metavar": _round_trips(action.metavar),
            "choices": None if action.choices is None else list(action.choices),
            "class": type(action).__name__,
        }
        for action in parser._actions
        if not isinstance(action, argparse._SubParsersAction)
    ]


def surface() -> dict:
    """Every subcommand, in order, with each of its options."""
    parser = build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return {
        "global": _actions(parser),
        "order": list(sub.choices),
        "commands": {
            name: {
                "help": next(c.help for c in sub._choices_actions if c.dest == name),
                "handler": sub.choices[name]._defaults["func"].__name__,
                # Recorded here because it is not a flag and would otherwise
                # change invisibly. Which stages accept a model id other than GS
                # is the list of stages that need nothing of a family, and a
                # stage joining it is the claim that its messages and addresses
                # are not one family's own -- one line of review, not a default
                # that moves in silence.
                "takes_model_id": sub.choices[name]._defaults.get("takes_model_id", False),
                # The same reasoning, for the other default that is not a flag.
                # Whether a command holds the unit decides whether it can run
                # while something else is measuring, and whether what it writes
                # carries a device id -- and a command that reads records only
                # has no business holding either.
                "needs_unit": sub.choices[name]._defaults.get("needs_unit", True),
                "options": _actions(sub.choices[name]),
            }
            for name in sub.choices
        },
    }


def test_surface_is_unchanged():
    assert surface() == json.loads(SURFACE.read_text())


def test_every_command_has_a_handler():
    for name, command in surface()["commands"].items():
        assert command["handler"], name


if __name__ == "__main__":
    SURFACE.parent.mkdir(parents=True, exist_ok=True)
    SURFACE.write_text(json.dumps(surface(), indent=2) + "\n")
    print(f"wrote {SURFACE}")
