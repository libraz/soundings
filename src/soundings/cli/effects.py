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
        "--frame-ms",
        type=float,
        default=10.0,
        help="length of the frame each delay is read over, ms. This sets the "
        "deepest swing the track can follow at a given rate -- shorter follows a "
        "deeper one and loses a little coverage",
    )
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
        "efx-partials",
        help="read every insertion effect's modulator off the partials of one held "
        "tone per type, from saved takes, with no machine attached",
    )
    p.add_argument(
        "takes",
        help="one directory holding a held-tone take per effect type and at least one "
        "with the part bypassed. The files are the subject: a take store rewrites its "
        "manifest when it closes, and which files the manifest forgot is reported",
    )
    p.add_argument(
        "--carrier-hz",
        type=float,
        required=True,
        help="the fundamental of the note the takes hold. Its orders are what the "
        "reading is taken on, and the demodulator's window is one period of it, which "
        "is what puts the neighbouring orders on a null",
    )
    p.add_argument(
        "--setting",
        default=r"(?P<type>[0-9A-Fa-f]{2}-[0-9A-Fa-f]{2})(?:-\d+)?(?:\.wav)?$",
        help="a regular expression capturing a group named 'type' out of each take's "
        "setting or file name. Anchored at the end by default, because a file name "
        "carries the stimulus before the setting and an unanchored pattern reads the "
        "stimulus: a take of 'held-16' collapses every type into one",
    )
    p.add_argument(
        "--bypassed",
        default="bypassed",
        help="what names the takes with the part routed past the effect. The first of "
        "them decides which orders are read; the rest are read as repeats of it",
    )
    p.add_argument("--lead", type=float, default=0.6, help="seconds of silence at the head")
    p.add_argument(
        "--hold",
        type=float,
        default=8.0,
        help="seconds the note is held for. Half a second is cut from each end of it, "
        "because an attack and a release are not the steady tone this reads",
    )
    p.add_argument("--min-rate", type=float, default=0.20, help="slowest rate the grid reaches, Hz")
    p.add_argument("--max-rate", type=float, default=8.0, help="fastest rate the grid reaches, Hz")
    p.add_argument(
        "--step-hz",
        type=float,
        default=0.01,
        help="how finely the grid is walked. This is not the resolution -- two rates "
        "closer than one over the length read are one peak however fine the grid is -- "
        "it is only how precisely that peak can be placed",
    )
    p.add_argument(
        "--control-at",
        type=float,
        default=0.45,
        help="the rate the injected controls are put in at, Hz. Best set where the "
        "types being read actually sit, since what a reading can recover depends on "
        "how many cycles the take holds",
    )
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_efx_partials)

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
        window=args.frame_ms / 1000.0,
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
            "slowest_shown_hz": efxmotion.slowest_shown(found),
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


def cmd_efx_partials(args: argparse.Namespace) -> int:
    """Read every type's modulator off the partials of one held tone per type."""
    from .. import efxpartials, takes

    def said(row) -> None:
        peak = row.get("phase", {}).get("peak", {})
        level = row.get("level", {}).get("peak", {})
        kind = row.get("the_level_swing_is") or ""
        fitted = (row.get("comb") or {}).get("excursion_ms")
        print(
            f"  {row['type']:6} phase {peak.get('hz', 0):5.2f} Hz "
            f"({peak.get('stands_over_bypassed') or 0:6.1f}x)  "
            f"level {level.get('hz', 0):5.2f} Hz "
            f"({level.get('stands_over_bypassed') or 0:6.1f}x)  "
            f"{kind:20s}" + (f" {fitted:8.3f} ms" if fitted is not None else "")
        )

    found = efxpartials.survey(
        args.takes,
        carrier_hz=args.carrier_hz,
        setting=takes.capturing(args.setting, "type"),
        bypassed=args.bypassed,
        lead_s=args.lead,
        hold_s=args.hold,
        grid_hz=(args.min_rate, args.max_rate),
        step_hz=args.step_hz,
        control_at_hz=args.control_at,
        progress=said,
    )
    rows = found["types"]
    if not rows:
        print(f"no takes under {args.takes} whose setting the pattern named")
        return 1

    moving = efxpartials.moving(rows)
    named = efxpartials.named_an_excursion(rows)
    forgot = found["takes"]["not_listed"]
    print()
    print(f"{len(moving)} of {len(rows)} types show something moving over their own grid")
    print(f"{len(named)} of those have an excursion that survived every gate")
    if forgot:
        print(f"  !! the manifest did not list {len(forgot)} of the takes that were read")

    report.write_json(args.out, {"takes": str(args.takes), **found, "excursions": named})
    return 0


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
                "why": efxsort.WHY_UNDECIDED,
                "grounds": efxsort.grounds(found),
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
