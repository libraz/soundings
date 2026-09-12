"""Sorting every insertion effect type into the ones that move and the ones that do not.

Two commands, over the same question by two routes that rest on different
properties. One tracks a delay between a bypassed take and a routed one and reads
its modulation directly. The other reads what routing did to the unit's own
repeatability, since a modulator the takes cannot track still stops them agreeing
with each other.

They are kept apart rather than reconciled. A type the two routes disagree about
is reported as a disagreement, because the disagreement is the finding: agreeing
routes say the verdict does not depend on which property was measured, and there
is nothing to say that with if one has been folded into the other.
"""

from __future__ import annotations

import argparse

from . import options, report


def register(sub) -> None:
    p = sub.add_parser(
        "efx-motion",
        help="sort a unit's insertion effects into the ones that move and the ones "
        "that stand still, from saved takes, with no machine attached",
    )
    p.add_argument(
        "takes",
        help="a directory holding one subdirectory of takes per effect type, each "
        "with the takes-manifest.json a --save run wrote",
    )
    p.add_argument(
        "--bypassed",
        default="0",
        help="the setting in each manifest that had the part routed past the effect",
    )
    p.add_argument(
        "--routed",
        default="1",
        help="the setting that had the part routed through it",
    )
    p.add_argument(
        "--types-from",
        help="an efx-type-map record naming every type the unit accepts. Any of them "
        "with no takes under the directory is reported as unsurveyed, so a capture "
        "that dropped a type cannot leave a short list of verdicts reading as a "
        "complete one",
    )
    p.add_argument("--max-delay", type=float, default=60.0, help="milliseconds of delay searched")
    p.add_argument("--min-rate", type=float, default=0.05, help="slowest modulation searched, Hz")
    p.add_argument("--max-rate", type=float, default=20.0, help="fastest modulation searched, Hz")
    p.add_argument(
        "--lead",
        type=float,
        default=0.6,
        help="seconds of silence at the head of a take, for any manifest that does not "
        "record its stimulus's own. The noise floor is measured in it and frames below "
        "that floor are dropped, without which a take that decays into silence cannot "
        "recover its own control",
    )
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_efx_motion)

    p = sub.add_parser(
        "efx-sort",
        help="sort a unit's insertion effects into the ones that move and the ones "
        "that stand still, from what each did to the unit's own repeatability",
    )
    p.add_argument(
        "records",
        help="a directory of contrast records, one per effect type, named for the type",
    )
    p.add_argument(
        "--types-from",
        help="an efx-type-map record naming every type the unit accepts, so a type with "
        "no record is reported as unsurveyed rather than silently absent",
    )
    p.add_argument(
        "--tracked",
        help="an efx-motion record to compare against. The two routes rest on different "
        "properties, so a type they disagree about is reported as a disagreement rather "
        "than reconciled",
    )
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_efx_sort)


def cmd_efx_motion(args: argparse.Namespace) -> int:
    """Sort every insertion effect type into moving, static, or unable to say."""
    from .. import efxmotion

    def said(entry) -> None:
        print(f"  {entry.type_id:8} {entry.verdict}")

    found = efxmotion.survey(
        args.takes,
        dry=args.bypassed,
        wet=args.routed,
        search_ms=(0.0, args.max_delay),
        rate_range=(args.min_rate, args.max_rate),
        lead_s=args.lead,
        progress=said,
    )
    if not found:
        print(f"no take directories with a manifest under {args.takes}")
        return 1

    print()
    print(efxmotion.summarise(found))
    piles = efxmotion.partition(found)

    unsurveyed: list[str] = []
    if args.types_from:
        unsurveyed = efxmotion.missing(found, efxmotion.accepted_types(args.types_from))
        if unsurveyed:
            print(f"  !! {len(unsurveyed)} types the unit accepts have no takes: {unsurveyed}")

    report.write_json(
        args.out,
        {
            "takes": str(args.takes),
            "searched_rate_hz": [args.min_rate, args.max_rate],
            "searched_delay_ms": [0.0, args.max_delay],
            "method": efxmotion.METHOD,
            "one_pair_per_type": efxmotion.WHY_ONE_PAIR,
            "nothing_is_named": efxmotion.NOT_NAMED,
            "why_channel": efxmotion.WHY_CHANNEL,
            "moving": piles["moving"],
            "static": piles["static"],
            "could_not_say": {
                "types": piles["could_not_say"],
                "why": efxmotion.WHY_COULD_NOT_SAY,
            },
            **(
                {
                    "types_accepted": str(args.types_from),
                    "unsurveyed": {"types": unsurveyed, "why": efxmotion.WHY_MISSING},
                }
                if args.types_from
                else {}
            ),
            "types": [f.to_json() for f in found],
        },
    )
    return 0 if not unsurveyed else 1


def cmd_efx_sort(args: argparse.Namespace) -> int:
    """Sort every insertion effect type by what it did to the unit's repeatability."""
    import json

    from .. import efxmotion, efxsort

    found = efxsort.survey(args.records, progress=lambda e: print(f"  {e.type_id:8} {e.verdict}"))
    if not found:
        print(f"no contrast records under {args.records}")
        return 1

    print()
    print(efxsort.summarise(found))
    piles = efxsort.partition(found)
    steadier = efxsort.steadier_when_routed(found)
    if steadier:
        print(
            f"  {len(steadier)} made the takes agree better routed than bypassed, which is the "
            f"opposite of a modulator: {steadier}"
        )
    gap = efxsort.inside_the_gap(found)
    if gap:
        print(
            f"  of those, {len(gap)} plainly did something and did it by a number the bar "
            f"cannot read: {gap}"
        )

    unsurveyed: list[str] = []
    if args.types_from:
        accepted = efxmotion.accepted_types(args.types_from)
        seen = {f.type_id for f in found}
        unsurveyed = [t for t in accepted if t not in seen]
        if unsurveyed:
            print(f"  !! {len(unsurveyed)} types the unit accepts have no record: {unsurveyed}")

    clashes: list[dict] = []
    if args.tracked:
        tracked = json.loads(open(args.tracked).read())
        other = [_Tracked(t["type"], t["verdict"]) for t in tracked.get("types", [])]
        clashes = efxsort.disagreements(found, other)
        declined = efxsort.declined_elsewhere(found, other)
        if declined:
            print(
                f"  the delay track stood aside on {len(declined)} of the types answered here, "
                "so an empty disagreement list is not the two routes agreeing"
            )
        for clash in clashes:
            print(
                f"  !! {clash['type']}: repeatability says {clash['by_repeatability']}, "
                f"the delay track says {clash['by_delay_track']}"
            )

    report.write_json(
        args.out,
        {
            "records": str(args.records),
            "method": efxsort.METHOD,
            "why_the_asymmetry_is_the_evidence": efxsort.WHY_ASYMMETRY,
            "steadier_when_routed": {
                "types": efxsort.steadier_when_routed(found),
                "why": efxsort.WHY_STEADIER,
            },
            "moving": piles["moving"],
            "static": piles["static"],
            "could_not_say": {
                "types": piles["could_not_say"],
                "why": efxsort.WHY_NOT_AUDIBLE,
                "inside_the_calibration_gap": {
                    "types": efxsort.inside_the_gap(found),
                    "why": efxsort.WHY_INSIDE_THE_GAP,
                },
            },
            **(
                {
                    "types_accepted": str(args.types_from),
                    "unsurveyed": {"types": unsurveyed, "why": efxmotion.WHY_MISSING},
                }
                if args.types_from
                else {}
            ),
            **(
                {
                    "compared_with": str(args.tracked),
                    "disagreements": clashes,
                    "answered_here_and_declined_there": declined,
                    "why_two_routes": efxsort.WHY_TWO_ROUTES,
                }
                if args.tracked
                else {}
            ),
            "types": [f.to_json() for f in found],
        },
    )
    return 0 if not unsurveyed else 1


class _Tracked:
    """Just enough of an efx-motion entry for the comparison to read."""

    def __init__(self, type_id: str, verdict: str) -> None:
        self.type_id = type_id
        self.verdict = verdict
