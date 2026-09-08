"""Rolling per-address records up into one statement about a block.

Three commands over records that other stages left. One decides which pair of
values each address in a block should be asked at, and the other two read the
resulting per-address verdicts back into a single answer -- about a block of the
address space, or about the parameters of one insertion effect type.

The plan is the authority in both directions: it says how many addresses there
were to ask, so a short set of verdicts reads as incomplete rather than as
complete, and a record it does not name is left out rather than counted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import options, report


def register(sub) -> None:
    p = sub.add_parser(
        "plan",
        help="say what pair of values each address in a block should be asked at, "
        "from what the write probe measured it to accept",
    )
    p.add_argument("write_probe", help="a write-probe record covering the block")
    p.add_argument(
        "block",
        help="the leading bytes of the addresses to plan, e.g. '40 11' for part 1",
    )
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_plan)

    p = sub.add_parser(
        "block",
        help="read a directory of per-address contrast records into one verdict per "
        "address, counted against the plan the block was asked from",
    )
    p.add_argument("plan", help="the plan the block was asked from, which says how many")
    p.add_argument(
        "plain",
        nargs="?",
        help="records from the pass that asked the plain note. Omitted only for a block "
        "whose plan asks nothing, where there is no pass to point at",
    )
    p.add_argument(
        "--gesture",
        help="records from the pass that moved messages after the setting. A gesture "
        "is a rescue for a null, so it answers only where the plain note could not",
    )
    p.add_argument(
        "--polyphony",
        help="records from the pass that played two notes. A parameter about polyphony "
        "sounds identical under a single note however it is set, so an address left null "
        "by the one-note passes has not been asked rather than answered",
    )
    p.add_argument(
        "--superseded-by",
        action="append",
        default=[],
        metavar="RECORD",
        help="a published record that asked some of these addresses and disagrees. Each such "
        "address is named with that record and the verdict it carries there, so a reader "
        "landing on this file does not take an answer the archive has since bettered. Only "
        "disagreements are marked",
    )
    p.add_argument(
        "--balance",
        action="append",
        default=[],
        metavar="RECORD",
        help="a balance record, once per pass. A parameter that moves signal between the "
        "channels is invisible to a comparison made in one of them, so what it found is "
        "folded in as another way of having reached the signal path. Each pass has its own "
        "takes and its own balance, so give the plain pass's and the gesture pass's both",
    )
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_block)

    p = sub.add_parser(
        "efx-params",
        help="read a directory of per-address records into one verdict per parameter of "
        "one insertion effect type, with no machine attached",
    )
    p.add_argument("records", help="a directory of contrast records, one per parameter address")
    p.add_argument(
        "--type", required=True, metavar="MSB LSB", help="the type they were taken under"
    )
    p.add_argument(
        "--types-from", required=True, help="an efx-type-map record, for the settings it loads"
    )
    p.add_argument(
        "--control", required=True, help="the routed-against-bypassed record from the same run"
    )
    p.add_argument(
        "--prepare",
        type=options.write_spec,
        action="append",
        default=[],
        metavar="ADDR=BYTES",
        help="the state the run was taken in, recorded with it since a verdict holds in it",
    )
    p.add_argument(
        "--supersede",
        action="append",
        default=[],
        metavar="ADDR=RECORD",
        help="an address whose first pair was withdrawn, and the record that replaced it. "
        "A pair that left one setting silent compares sound with silence, and the reason "
        "travels with the row rather than with whoever remembers the directory",
    )
    p.add_argument(
        "--slots",
        nargs="+",
        default=[],
        metavar="ADDR",
        help="the type's parameter addresses in slot order, which is what makes a missing "
        "record readable. Without them the slot is the position in the sorted file names, so "
        "an address the run could not measure renumbers every parameter after it and pairs "
        "each with the default of the slot before, and the count reports the type as having "
        "one parameter fewer than it has. Given rather than derived because which address is "
        "which slot is a fact about the unit",
    )
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_efx_params)


def cmd_plan(args: argparse.Namespace) -> int:
    """Turn a write probe's measured ranges into the pair each address is asked at."""
    from .. import plan

    record = json.loads(Path(args.write_probe).read_text())
    asks, skipped = plan.plan_block(record, args.block)
    if not asks and not skipped:
        print(f"no address under {args.block!r} in {args.write_probe}")
        return 1
    print(plan.summarise(asks, skipped))

    report.write_json(
        args.out,
        {
            "block": args.block,
            "from": str(args.write_probe),
            "method": plan.METHOD,
            "ask": [a.to_json() for a in asks],
            "cannot_be_asked": [s.to_json() for s in skipped],
        },
    )
    return 0


def cmd_block(args: argparse.Namespace) -> int:
    """Read a block's per-address records into one answer about the block."""
    from .. import block

    planned = json.loads(Path(args.plan).read_text())
    plain, outside = (
        block.split_by_plan(block.survey(args.plain), planned) if args.plain else ({}, [])
    )
    gesture, more = (
        block.split_by_plan(block.survey(args.gesture), planned) if args.gesture else ({}, [])
    )
    outside += more
    polyphony, more = (
        block.split_by_plan(block.survey(args.polyphony), planned) if args.polyphony else ({}, [])
    )
    outside += more
    for address in sorted(set(outside)):
        print(f"  {address}: a record the plan does not name, left out of the block")
    # An empty pass is a mistake only where the plan expected one. A block whose
    # every address accepts a single value has no pair to compare and so no
    # contrast record to point at, and refusing to fold it would leave the one
    # thing measured about it -- that nothing there can be moved -- unpublished,
    # while the block went on being counted as a sweep somebody still owes.
    if not plain and not gesture and not polyphony and planned.get("ask"):
        print(f"no contrast records under {args.plain}")
        return 1

    found = [f for f in block.join(plain, gesture, polyphony) if f is not None]
    # One per pass rather than one for the block: each pass has its own takes and
    # its own balance, and a parameter that moves the balance under the gesture
    # is as much a finding as one that moves it under the plain note.
    for where in args.balance:
        found = block.with_balance(found, json.loads(Path(where).read_text()))
    found = block.with_the_plans_caveats(found, planned)
    # Only where the two actually disagree. A record that asked the same address
    # and answered the same way supersedes nothing, and marking it would tell a
    # reader to go elsewhere for the answer already in front of them.
    answered_elsewhere: dict[str, dict] = {}
    for where in args.superseded_by:
        other = json.loads(Path(where).read_text())
        mine = {f.address: f.verdict for f in found}
        for row in other.get("addresses", []):
            if row["address"] in mine and row["verdict"] != mine[row["address"]]:
                answered_elsewhere[row["address"]] = {
                    "record": Path(where).name,
                    "verdict": row["verdict"],
                }
    if answered_elsewhere:
        found = block.superseded(found, answered_elsewhere)
        print(f"  {len(answered_elsewhere)} addresses answered by another record")
    coverage = block.against_plan(found, planned)
    print(block.summarise(found, coverage))
    # Named rather than counted: an address left to try is the next run's list,
    # and a number is not a list.
    open_still = [f.address for f in found if f.still_open]
    if open_still:
        print(f"\nstill open: {' | '.join(open_still)}")

    report.write_json(
        args.out,
        {
            "block": planned.get("block"),
            "method": block.method_for(bool(args.gesture), asked=bool(planned.get("ask"))),
            # Where the addresses came from, which for a block nothing could be
            # asked of is the whole of its evidence: the record holds no verdict
            # of its own, and a reader has to be able to reach the run that
            # established there was nothing to ask.
            "planned_from": planned.get("from"),
            "asked_in": {
                name: found_states
                for name, where in (
                    ("plain", args.plain),
                    ("gesture", args.gesture),
                    ("polyphony", args.polyphony),
                )
                if where and (found_states := block.states(where))
            },
            "why_asked_in": block.WHY_ASKED_IN,
            **(
                {
                    "records_the_plan_does_not_name": sorted(set(outside)),
                    "why_outside_the_plan": block.WHY_OUTSIDE_THE_PLAN,
                }
                if outside
                else {}
            ),
            **({"two_passes": block.WHY_TWO_PASSES} if args.gesture else {}),
            **({"polyphony_pass": block.WHY_POLYPHONY_PASS} if args.polyphony else {}),
            **({"balance_counts": block.WHY_BALANCE_COUNTS} if args.balance else {}),
            "chose_the_values": planned.get("method"),
            "coverage": coverage,
            "still_open": open_still,
            "addresses": [f.to_json() for f in found],
        },
    )
    return 0


def cmd_efx_params(args) -> int:
    """One verdict per parameter of one type, from records already captured."""
    from .. import efxparams

    types = json.loads(Path(args.types_from).read_text())
    loads = [e["parameters"] for e in types["effects"] if e["type"] == args.type]
    if not loads:
        print(f"{args.types_from} has no type {args.type}")
        return 1
    found = efxparams.read_directory(
        args.records,
        args.type,
        loads[0],
        args.control,
        [{"address": a, "bytes": " ".join(f"{v:02X}" for v in vs)} for a, vs in args.prepare],
        supersede=dict(s.split("=", 1) for s in args.supersede),
        slots=args.slots or None,
    )
    for name, count in found["results"].items():
        print(f"  {name}: {count}")
    if (coverage := found.get("coverage")) and (missing := coverage["never_asked"]):
        print(f"  => {len(missing)} never asked: {' | '.join(missing)}")
    report.write_json(args.out, found)
    return 0
