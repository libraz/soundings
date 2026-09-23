"""Asking what has been read out of the measurements, and what is still open.

Four questions and no measuring. `open` is the queue and it is derived, not
kept: what to run next is whatever separates a reading still standing from the
claim, and if nothing does, the queue says so instead of dropping the item.
`closed` is the other side of the same query. `stale` is the one that fires on
its own, when a record a claim rests on is re-published with a figure moved.
`index` regenerates the listing.

None of them touches the unit. They are here rather than in a separate tool for
the same reason the document commands are: a reader who has to find a second
program to check a claim will not check it.

`render` and `stage` carry a type's candidate models through stages 7-9: the first
draws one model on one directory of takes, the second compares a class of them with
the unit's own takes and, asked to, writes where the type stands.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import inferences


def register(sub) -> None:
    p = sub.add_parser(
        "inferences",
        help="what was read out of the measurements, what it rests on, and what is "
        "still open",
    )
    p.set_defaults(needs_unit=False, func=_dispatch)
    inner = p.add_subparsers(dest="action", required=True)

    for name, help_text in (
        ("open", "readings still standing against a claim, and what would separate them"),
        ("closed", "claims with nothing standing against them a measurement could settle"),
        ("stale", "claims resting on figures that have moved since they were made"),
    ):
        one = inner.add_parser(name, help=help_text)
        one.add_argument("--root", default=".", type=Path)

    listing = inner.add_parser("index", help="regenerate a unit's listing from its files")
    listing.add_argument("unit_id")
    listing.add_argument("--root", default=".", type=Path)
    listing.add_argument("--write", action="store_true")

    counted = inner.add_parser(
        "coverage",
        help="how much of what a unit's document prints the claims are about, counted "
        "as what they say they cover and as what their records name",
    )
    counted.add_argument("unit_id")
    counted.add_argument("--root", default=".", type=Path)
    counted.add_argument(
        "--per-type",
        action="store_true",
        help="print a row per printed type rather than the totals alone",
    )

    reached = inner.add_parser(
        "reach",
        help="for each claim about a class, which printed rows carry its quantity and "
        "which published records of them it has never been held against",
    )
    reached.add_argument("unit_id")
    reached.add_argument("--root", default=".", type=Path)

    drawn = inner.add_parser(
        "render",
        help="draw one graph model on a directory of the unit's takes, into "
        ".cache/rendered/<model-id>/, from the bypass takes the ledger finds for it",
    )
    drawn.add_argument("model", type=Path, help="a graph model file")
    drawn.add_argument("takes", type=Path, help="a directory under .cache/takes")
    drawn.add_argument("--root", default=".", type=Path)

    staged = inner.add_parser(
        "stage",
        help="compare a type's candidate models with the unit's own takes and say which "
        "stage it reached; without --class, run the identity control alone",
    )
    staged.add_argument("unit_id")
    staged.add_argument("type", help="the effect type, e.g. 01-24")
    staged.add_argument(
        "--class",
        dest="class_name",
        help="the candidates' class, e.g. whole-0124. Without it only the identity model "
        "is compared, which says whether this comparison could fail anything at all",
    )
    staged.add_argument(
        "--exclude",
        metavar="REASON",
        help="record the type as out of scope for stages 7-9 instead of comparing "
        "anything, e.g. no_effect for a type that passes its input through; needs --write",
    )
    staged.add_argument(
        "--write",
        action="store_true",
        help="write inferences/<unit>/stages/<MM-LL>.json; needs --class or --exclude",
    )
    staged.add_argument("--root", default=".", type=Path)


def _dispatch(args) -> int:
    return {
        "open": _open,
        "closed": _closed,
        "stale": _stale,
        "index": _index,
        "coverage": _coverage,
        "reach": _reach,
        "render": _render,
        "stage": _stage,
    }[args.action](args)


def _open(args) -> int:
    items = inferences.open_items(args.root)
    if not items:
        print("nothing open")
        return 0
    made = [i for i in items if i.get("already_recorded")]
    runnable = [i for i in items if i["run"] and not i.get("already_recorded")]
    blocked = [i for i in items if not i["run"]]
    if made:
        # Not under the heading below, and above it. What these need first is the
        # records read, and a queue that printed them beside the bookings is how
        # the same run gets made twice.
        print("== the unit already holds records of this, and the claim cites none")
        for item in made:
            run = item["run"]
            print(
                f"  {run.get('minutes', '?'):>4} min  {run.get('stage', '?')}  "
                f"{item['inference']}"
            )
            for side in item["already_recorded"]["sides"]:
                unread = [
                    f for f in side["records"] if f not in side["the_claim_names"]
                ]
                print(f"    {side['side']}: {', '.join(unread) or 'all cited'}")
        print("\n  read these before booking the time above")
        print()
    if runnable:
        print("== what a run would settle, soonest first")
        for item in runnable:
            run = item["run"]
            print(
                f"  {run.get('minutes', '?'):>4} min  {run.get('stage', '?')} "
                f"{run.get('type', '')} {run.get('address', '')}"
            )
            print(f"            {item['reading']}")
            print(f"            {item.get('observable', '')}")
        print(f"\n  {sum(i['minutes'] or 0 for i in runnable):.0f} minutes with the unit sounding")
    if blocked:
        # Listed rather than left out. A queue that silently drops what it cannot
        # schedule reads as a shorter queue, and the reason one of these cannot be
        # scheduled is usually the most useful sentence about the type.
        print("\n== open and not answerable by a run this rig can make")
        for item in blocked:
            print(f"  {item['inference']}  ({item['why_open']})")
            print(f"    {item['reading']}")
            print(f"    {item['why_there_is_no_run']}")
    return 0


def _closed(args) -> int:
    for item in inferences.closed(args.root):
        print(f"{', '.join(item['types'])}  {item['verdict']}")
        print(f"  {item['claim']}")
        for reading in item["equivalent_readings"]:
            print(f"  equivalent under this test: {reading}")
    return 0


def _stale(args) -> int:
    found = inferences.stale(args.root)
    for item in found:
        print(f"{item['inference']}")
        print(f"  {item['file']} {item.get('key', '')}")
        print(f"  {item['why']}: was {item.get('was')}, now {item.get('now')}")
    if not found:
        print("every claim rests on figures the records still hold")
    return 1 if found else 0


def _coverage(args) -> int:
    """Both figures and the gap between them, because one of them alone is misread."""
    found = inferences.coverage(args.root, args.unit_id)
    printed = found["printed"]
    print(
        f"{printed['pairs']} parameters printed against {printed['types']} types in "
        f"{found['document_id']}"
    )
    for name in ("named", "touched"):
        block = found[name]
        print(
            f"  {name:>8}  {block['pairs']:>4} of {printed['pairs']}  "
            f"{block['of_the_printed'] * 100:5.1f}%"
        )
    print(
        f"  between   {found['named_and_not_touched']['pairs']:>4}  pairs a claim is "
        "about and no record it cites names"
    )
    print(
        f"  the other way {found['touched_and_not_named']['pairs']:>4}  pairs a record "
        "names and no claim citing it is about, which is what a control is"
    )
    print(
        f"  {found['records_naming_no_address']['count']} cited records name a type and "
        "no address, so they move the second figure not at all"
    )
    for name in ("types_with_nothing_named", "types_with_nothing_touched"):
        types = found[name]
        print(f"  {len(types)} {name.replace('_', ' ')}: {', '.join(types)}")
    if args.per_type:
        print()
        for row in found["per_type"]:
            print(
                f"  {row['type']}  {row['effect'][:28]:<28} "
                f"{row['named']:>3} named {row['touched']:>3} touched "
                f"of {row['printed']:>2} printed"
            )
    return 0


def _reach(args) -> int:
    """What a class claim's quantity reaches, and which of it the claim never saw."""
    found = inferences.reach(args.root, args.unit_id)
    if not found["claims"]:
        print("no claim here is about a class")
        return 0
    for item in found["claims"]:
        print(f"{item['inference']}  ({item['state']}, {len(item['types'])} types)")
        for row in item["signatures"]:
            outside = row["outside_the_claim"]
            with_records = [cell for cell in outside if cell["records"]]
            print(
                f"  {row['parameter']:<14} {row['printed'][:22]:<22} "
                f"printed on {row['printed_rows']:>3}, named {row['the_claim_names']:>3}, "
                f"outside {len(outside):>3}, published and uncited {len(with_records):>2}"
                f"   (reaches {row['reaches_named_types']} of its own types)"
            )
            for cell in with_records:
                print(
                    f"      {cell['type']} {cell['address']}  "
                    + ", ".join(name.split("/")[-1] for name in cell["records"])
                )
        print()
    return 0


def _index(args) -> int:
    listing = inferences.index(args.root, args.unit_id)
    where = Path(args.root) / "inferences" / args.unit_id / "index.json"
    if args.write:
        where.write_text(json.dumps(listing, indent=2) + "\n")
        print(f"wrote {where}")
    else:
        print(json.dumps(listing, indent=2))
    return 0


def _render(args) -> int:
    from .. import ledger, stages

    root = Path(args.root)
    takes_root = (root / ledger.TAKES_ROOT).resolve()
    try:
        rel = Path(args.takes).resolve().relative_to(takes_root).as_posix()
    except ValueError:
        print(f"{args.takes} is not under {ledger.TAKES_ROOT}")
        return 2
    try:
        drawn = stages.render_directory(stages.candidate(args.model, root), root, rel)
    except stages.LedgerBehind as behind:
        print(behind)
        return 1
    kept = json.loads((drawn / "takes-manifest.json").read_text())["rendered"]
    print(f"drew {kept['model']} ({kept['model_sha256'][:12]}) on {rel}")
    print(f"  channels {kept['channels']}, input from {kept['input_from']}")
    if kept["unrenderable"]:
        print(f"  unrenderable at {kept['unrenderable']}")
    print(f"  into {drawn}")
    return 0


def _stage(args) -> int:
    from .. import stages

    if args.exclude and (args.class_name or not args.write):
        print("--exclude is written instead of a comparison: give it --write and no --class")
        return 2
    if args.write and not (args.class_name or args.exclude):
        print("--write needs --class: the identity control alone decides no stage")
        return 2
    root = Path(args.root)
    type_ = stages.type_of(args.type)
    where = root / "inferences" / args.unit_id / "stages" / f"{type_.replace(' ', '-')}.json"
    if args.exclude:
        found = {
            "type": type_,
            "excluded": {"reason": args.exclude},
            "invocation": {"unit": args.unit_id, "type": type_, "exclude": args.exclude},
        }
        where.parent.mkdir(parents=True, exist_ok=True)
        where.write_text(json.dumps(found, indent=2) + "\n")
        print(f"wrote {where}")
        return 0
    try:
        found = stages.stage(root, args.unit_id, type_, class_name=args.class_name)
    except stages.LedgerBehind as behind:
        print(behind)
        return 1
    if "stopped" in found and "p1" not in found and "control" not in found:
        print(f"{type_} stopped at {found['stopped']['at']}: {found['stopped']['gate']}")
        print(f"  needs {found['stopped']['needs']}")
        return 0
    block = found.get("p1") or found
    comparison = block["comparison"]
    print(f"{type_} compared by {comparison['used']} ({comparison['chosen_by']}), "
          f"{len(block['compared'])} settings in "
          f"{len({row['dir'] for row in block['compared']})} directories")
    control = block["control"]
    if "bypass_check" in control:
        print(f"  bypass check: {control['bypass_check']}")
    gates = control["gates"]
    print(
        f"  identity control: {'separated' if control['separated'] else 'did not separate'}"
        f"  gross {gates['gross']['residual_over_span']} of a span of "
        f"{gates['gross']['span']} {gates['measured_in']}, breakdown "
        f"{'passed' if gates['breakdown']['passed'] else 'failed'}"
    )
    if not args.class_name:
        return 0
    for phase in ("p1", "p2"):
        if phase in found:
            print(f"  {phase}: {found[phase]['verdict']}, held out "
                  f"{len(found[phase]['held_out_settings'])}, kept out of the ranking "
                  f"{found[phase]['excluded_from_ranking']}")
    if "stopped" in found:
        print(f"  stopped at {found['stopped']['at']}: {found['stopped']['gate']}")
    if args.write:
        where.parent.mkdir(parents=True, exist_ok=True)
        where.write_text(json.dumps(found, indent=2) + "\n")
        print(f"wrote {where}")
    return 0
