"""Where a unit stands against the bar for a finished one.

The only command whose subject is a unit rather than a run. Everything else here
reads what one measurement left behind; this counts a whole directory of them
against the stages a unit is finished when it has, and names what is left.

It exits non-zero while the unit is short of the bar, which is what lets the
question be asked mechanically rather than read: a stage quietly missing from a
directory is then caught by whatever asks, rather than by whoever remembers.
"""

from __future__ import annotations

import argparse

from . import options, report


def register(sub) -> None:
    p = sub.add_parser(
        "complete",
        help="count a unit's directory against the bar for a finished one, and say "
        "what is left, with no machine attached",
    )
    p.add_argument("unit", help="a directory under data/units")
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_complete)


def cmd_complete(args: argparse.Namespace) -> int:
    """Say where a unit stands, and fail while it is short of the bar.

    A non-zero exit is what lets this be asked mechanically rather than read, so
    a stage quietly going missing from a directory is caught by whatever asks.
    """
    from .. import completion

    found = completion.survey(args.unit)
    print(completion.render(found))
    report.write_json(args.out, found)
    return 0 if found["complete"] else 1
