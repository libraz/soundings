"""Measure kept takes, with no machine attached.

Device time is the scarce resource, so a take is recorded once and asked
questions afterwards. Everything here reads a dry and a wet take of the same note
and says what the effect did between them -- one pair at a time, or a directory
of them at once. None of it opens a MIDI port.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import options, report


def register(sub) -> None:
    """Everything here reads takes that were already recorded.

    Each parser clears `needs_unit`, so these do not queue behind a sweep that is
    still running -- which matters, because reading a block's records back while
    the next block is being captured is the ordinary way to work.
    """
    p = sub.add_parser(
        "motion",
        help="say what an effect does over time -- its modulation rate, depth and "
        "shape -- from a dry and a wet take, with no machine attached",
    )
    p.add_argument("dry", help="a take with the effect off")
    p.add_argument("wet", help="the same note with the effect on")
    p.add_argument(
        "--max-delay",
        type=float,
        default=60.0,
        help="milliseconds of delay searched. A null is a fact about this range",
    )
    p.add_argument("--min-rate", type=float, default=0.05, help="slowest modulation searched, Hz")
    p.add_argument("--max-rate", type=float, default=20.0, help="fastest modulation searched, Hz")
    p.add_argument(
        "--lead",
        type=float,
        default=0.6,
        help="seconds of silence at the head of the take. The noise floor is measured "
        "in it, and a frame that does not stand over that floor is dropped: a normalised "
        "correlation cannot tell silence from sound, so an untracked frame of noise "
        "otherwise reports a confident delay at whatever lag its noise peaked at. Every "
        "stimulus in the catalogue records 0.6",
    )
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_motion)

    p = sub.add_parser(
        "decay",
        help="say how long an effect's tail takes to die in each octave band, from a "
        "dry and a wet take, with no machine attached",
    )
    p.add_argument("dry", help="a take with the effect off")
    p.add_argument("wet", help="the same note with the effect on")
    p.add_argument(
        "--type",
        metavar="MSB LSB",
        help="the insertion effect type the pair was taken under. Without it the record "
        "says how long a tail took to die without saying whose tail it was, and a "
        "directory of them is identified by its filenames -- which is an index kept by "
        "hand beside records that could carry it themselves",
    )
    p.add_argument(
        "--lead",
        type=float,
        default=0.5,
        help="seconds of silence at the head of the take; the per-band noise floor is "
        "measured in it, and it is what says where a tail stops being a tail",
    )
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_decay)

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

    p = sub.add_parser(
        "verdict",
        help="ask an audible verdict of takes recorded in an earlier session, with no "
        "machine attached",
    )
    p.add_argument("takes", help="a directory a --save run left, with its takes-manifest.json")
    p.add_argument(
        "--first",
        help="the setting compared from; defaults to the first the manifest records",
    )
    p.add_argument(
        "--second",
        help="the setting compared to; defaults to the last. A run that swept more than "
        "two settings has to be told which pair, since any choice defaulted to here would "
        "be a measurement decision wearing a default's clothes",
    )
    p.add_argument(
        "--margin",
        type=float,
        default=6.0,
        help="dB the change must clear the unit's own repeatability by",
    )
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_verdict)

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
    p.add_argument("plain", help="records from the pass that asked the plain note")
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
        "balance",
        help="say what a parameter did to the level difference between the two channels, "
        "which a comparison made in one of them cannot see",
    )
    p.add_argument(
        "takes",
        help="a directory a --save run left, or one holding several of them",
    )
    p.add_argument(
        "--margin",
        type=float,
        default=6.0,
        help="dB a movement must clear the steadier setting's own scatter by",
    )
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_balance)

    p = sub.add_parser(
        "vibrato",
        help="say whether a take's pitch was modulated, and at what rate and depth, "
        "which no comparison of two takes can measure",
    )
    p.add_argument("takes", help="a directory a --save run left, with its takes-manifest.json")
    p.add_argument(
        "--lead",
        type=float,
        default=0.6,
        help="seconds of silence at the head of a take. The noise floor is measured in "
        "it, and a frame that does not stand over that floor is dropped: a frequency "
        "estimated from silence is the noise's own strongest period reported as a note",
    )
    p.add_argument("--min-rate", type=float, default=0.5, help="slowest modulation searched, Hz")
    p.add_argument("--max-rate", type=float, default=15.0, help="fastest modulation searched, Hz")
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_vibrato)

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
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_efx_params)

    p = sub.add_parser(
        "complete",
        help="count a unit's directory against the bar for a finished one, and say "
        "what is left, with no machine attached",
    )
    p.add_argument("unit", help="a directory under data/units")
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_complete)


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
    )
    for name, count in found["results"].items():
        print(f"  {name}: {count}")
    report.write_json(args.out, found)
    return 0


def cmd_complete(args) -> int:
    """Say where a unit stands, and fail while it is short of the bar.

    A non-zero exit is what lets this be asked mechanically rather than read, so
    a stage quietly going missing from a directory is caught by whatever asks.
    """
    from .. import completion

    found = completion.survey(args.unit)
    print(completion.render(found))
    report.write_json(args.out, found)
    return 0 if found["complete"] else 1


def _pair(dry_path: str, wet_path: str):
    """Load two takes and hand back the loudest channel of each, plus the rate."""
    from ..takes import loudest, read

    dry, dry_rate = read(dry_path)
    wet, wet_rate = read(wet_path)
    if dry_rate != wet_rate:
        raise SystemExit(f"the two takes were captured at {dry_rate} and {wet_rate} Hz")
    return loudest(dry), loudest(wet), dry_rate


def cmd_motion(args: argparse.Namespace) -> int:
    """Say what an effect does over time, from a dry and a wet take of the same note."""
    from .. import motion

    dry, wet, rate = _pair(args.dry, args.wet)
    span = (0.0, args.max_delay)
    rates = (args.min_rate, args.max_rate)
    found = motion.measure(dry, wet, rate, search_ms=span, rate_range=rates, lead_s=args.lead)
    # Always, not only when the answer is a null. A run that reports motion is
    # not excused the control either: knowing the tracker works on this material
    # is what says a recovered rate is the effect's and not the search's.
    vouched = motion.control(dry, wet, rate, search_ms=span, rate_range=rates, lead_s=args.lead)

    print(f"{args.dry} against {args.wet}, {rate} Hz")
    print(found.describe())
    ladder = ", ".join(
        f"{a['injected_depth_ms']}{'' if a['recovered'] else ' (missed)'}"
        for a in vouched["attempts"]
    )
    print(
        f"  control: the take's own return, {vouched['return_level_db']} dB under the direct "
        f"path, swept at {vouched['injected_rate_hz']} Hz by these many ms peak to peak: {ladder}"
    )
    if vouched["detectable_ms"] is not None:
        deep, shallow = vouched["detectable_ms"]
        print(
            f"  => the null above holds for a swing between {shallow} and {deep} ms peak to "
            "peak, and says nothing about one outside that band"
        )
    else:
        print(
            "  !! no injected swing was recovered at any depth. Nothing above is a finding "
            "about the effect."
        )

    report.write_json(
        args.out,
        {
            "dry": str(args.dry),
            "wet": str(args.wet),
            "sample_rate": rate,
            "searched_rate_hz": [args.min_rate, args.max_rate],
            "lead_s": args.lead,
            "positive_control": vouched,
            **(
                {"control_failed": motion.CONTROL_FAILED}
                if vouched["detectable_ms"] is None
                else {}
            ),
            **found.to_json(),
        },
    )
    return 0 if vouched["detectable_ms"] is not None else 1


def cmd_decay(args: argparse.Namespace) -> int:
    """Say how long an effect's tail takes to die, per octave band."""
    from .. import decay as dec

    dry, wet, rate = _pair(args.dry, args.wet)
    tail, noise = dec.isolate_tail(dry, wet, rate, lead=args.lead)
    found = dec.measure(tail, rate, noise=noise)
    print(f"{args.dry} against {args.wet}, {rate} Hz")
    print(found.describe())

    report.write_json(
        args.out,
        {
            "type_id": args.type,
            "dry": str(args.dry),
            "wet": str(args.wet),
            "sample_rate": rate,
            "lead_s": args.lead,
            **found.to_json(),
        },
    )
    return 0


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


def cmd_verdict(args: argparse.Namespace) -> int:
    """Judge a saved pair of settings, without the machine that recorded them."""
    from .. import audible, rejudge

    try:
        overall, manifest = rejudge.judge_directory(
            args.takes, first=args.first, second=args.second, margin_db=args.margin
        )
    except (rejudge.NothingToJudge, FileNotFoundError) as exc:
        print(exc)
        return 1

    print(f"{args.takes}: settings {rejudge.settings_in(manifest)}")
    for verdict in overall.verdicts:
        print(f"  {verdict.describe()}")
    print()
    print(overall.describe())

    report.write_json(
        args.out,
        {
            "takes": str(args.takes),
            "controller": manifest.get("controller"),
            "address": manifest.get("address"),
            "settings_compared": [
                args.first if args.first is not None else rejudge.settings_in(manifest)[0],
                args.second if args.second is not None else rejudge.settings_in(manifest)[-1],
            ],
            "stimuli": manifest.get("stimuli", []),
            "method": audible.METHOD,
            "judged_offline": rejudge.METHOD_SUFFIX,
            **overall.to_json(),
        },
    )
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """Turn a write probe's measured ranges into the pair each address is asked at."""
    import json
    from pathlib import Path

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


def cmd_balance(args: argparse.Namespace) -> int:
    """Say what each saved run did to the balance between the two channels."""
    from pathlib import Path

    from .. import balance

    root = Path(args.takes)
    # One run, or a directory of them. The manifest is what says which, since a
    # run's directory holds one and a directory of runs holds none.
    roots = [root] if (root / "takes-manifest.json").exists() else sorted(root.glob("*"))
    found = []
    for one in roots:
        if not (one / "takes-manifest.json").exists():
            continue
        for verdict in balance.measure(one, margin_db=args.margin):
            print(f"{one.name}: {verdict.describe()}")
            found.append((one.name, verdict))
    if not found:
        print(f"no saved takes under {args.takes}")
        return 1

    moved = [n for n, v in found if v.moved_between_settings or v.did_not_repeat]
    unmeasured = [n for n, v in found if v.not_measured]
    # The two are counted apart because they mean opposite things. A run that was
    # asked and said no belongs under the denominator; a run that could not be
    # asked does not, and folding it in reports a null the measurement never made.
    print(
        f"\n{len(moved)} of {len(found) - len(unmeasured)} moved the balance: "
        f"{' | '.join(moved) or 'none'}"
    )
    if unmeasured:
        print(f"{len(unmeasured)} could not be measured: {' | '.join(unmeasured)}")

    report.write_json(
        args.out,
        {
            "takes": str(args.takes),
            "method": balance.METHOD,
            "why_the_total_is_reported": balance.WHY_NOT_A_LEVEL,
            "moved_the_balance": moved,
            "runs": [{"name": n, **v.to_json()} for n, v in found],
        },
    )
    return 0


def cmd_vibrato(args: argparse.Namespace) -> int:
    """Say what each setting did to the pitch, from inside single takes."""
    import json
    from pathlib import Path

    from .. import vibrato
    from ..takes import loudest, read

    root = Path(args.takes)
    manifest = json.loads((root / "takes-manifest.json").read_text())
    search = (args.min_rate, args.max_rate)

    by_setting: dict[str, list] = {}
    for entry in manifest.get("takes", []):
        frames, sample_rate = read(root / entry["file"])
        found = vibrato.measure(loudest(frames), sample_rate, lead_s=args.lead, search_hz=search)
        key = f"{entry['stimulus']} at {entry['setting']}"
        by_setting.setdefault(key, []).append(found)
        print(f"  {key} take {entry['take']}: {found.describe()}")

    # The control goes into the setting that showed nothing, for the reason
    # WHY_ONE_SETTING_CARRIES_THE_CONTROL gives to the reader of the record.
    quiet = min(
        by_setting,
        key=lambda k: sum(1 for f in by_setting[k] if f.found),
        default=None,
    )
    vouched = None
    if quiet is not None:
        entry = next(e for e in manifest["takes"] if f"{e['stimulus']} at {e['setting']}" == quiet)
        frames, sample_rate = read(root / entry["file"])
        vouched = vibrato.control(loudest(frames), sample_rate, lead_s=args.lead, search_hz=search)
        print(
            f"\ncontrol, injected into {quiet}: recovered "
            f"{vouched['depths_recovered_cents'] or 'nothing'} of "
            f"{list(vibrato.CONTROL_DEPTHS_CENTS)} cents"
        )

    report.write_json(
        args.out,
        vibrato.record(
            by_setting,
            takes=str(args.takes),
            searched_hz=search,
            control=vouched,
            control_taken_from=quiet,
        ),
    )
    return 0


def cmd_block(args: argparse.Namespace) -> int:
    """Read a block's per-address records into one answer about the block."""
    import json
    from pathlib import Path

    from .. import block

    planned = json.loads(Path(args.plan).read_text())
    plain, outside = block.split_by_plan(block.survey(args.plain), planned)
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
    if not plain and not gesture and not polyphony:
        print(f"no contrast records under {args.plain}")
        return 1

    found = [f for f in block.join(plain, gesture, polyphony) if f is not None]
    # One per pass rather than one for the block: each pass has its own takes and
    # its own balance, and a parameter that moves the balance under the gesture
    # is as much a finding as one that moves it under the plain note.
    for where in args.balance:
        found = block.with_balance(found, json.loads(Path(where).read_text()))
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
            "method": block.METHOD,
            **(
                {
                    "records_the_plan_does_not_name": sorted(set(outside)),
                    "why_outside_the_plan": block.WHY_OUTSIDE_THE_PLAN,
                }
                if outside
                else {}
            ),
            "two_passes": block.WHY_TWO_PASSES,
            **({"polyphony_pass": block.WHY_POLYPHONY_PASS} if args.polyphony else {}),
            **({"balance_counts": block.WHY_BALANCE_COUNTS} if args.balance else {}),
            "chose_the_values": planned.get("method"),
            "coverage": coverage,
            "still_open": open_still,
            "addresses": [f.to_json() for f in found],
        },
    )
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
