"""What one saved run of takes shows, asked after the session ended.

Five commands over the directory a `--save` run leaves. One re-reaches the
audible verdict the session reached, and the rest ask what that verdict cannot
see: a parameter that moves signal between the channels is invisible to a
comparison made in one of them, and a parameter that modulates pitch is invisible
to a comparison of two takes at all. The separation between the channels is then
asked again band by band, which is what tells one channel scaled down from one
carrying something with a shape of its own. A parameter that mixes two signals
onto the same channels is invisible to every one of those, and is asked by
splitting the take in time instead: where an effect returns late enough, the two
it mixes are in different parts of one take.

None of them opens a MIDI port. The takes carry the settings, the stimuli and the
label, so the only thing missing from the directory is the machine, and the
machine is not consulted after the last take is recorded.
"""

from __future__ import annotations

import argparse

from . import options, report


def register(sub) -> None:
    from .. import efxbands

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
    p.add_argument(
        "--separated-as-far-as",
        metavar="RECORD",
        help="a record of this unit whose widest separation bounds what this one may "
        "report. A separation that stops has to be shown to be the unit's and not the "
        "chain's, and how far these two inputs can be driven apart is a fact about the "
        "rig that some other address already measured -- so it is read out of that "
        "record rather than held beside the code, where it would reach the next unit "
        "as an assumption",
    )
    options.add_subject(p)
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_balance)

    p = sub.add_parser(
        "balance-bands",
        help="say what a parameter did to the separation between the two channels band by "
        "band, which tells one channel scaled from one with a shape of its own",
    )
    p.add_argument(
        "takes",
        help="a directory a --save run left, or one holding several of them",
    )
    p.add_argument(
        "--bands",
        default="third-octave",
        choices=sorted(efxbands.BAND_SETS),
        help="the set of band centres, each read at the width that belongs to it",
    )
    p.add_argument(
        "--margin",
        type=float,
        default=6.0,
        help="dB a setting's spread across the bands must clear the flattest setting's by",
    )
    options.add_subject(p)
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_balance_bands)

    p = sub.add_parser(
        "arrival",
        help="say what a parameter put in the window a take's direct sound arrives in and "
        "what it put in the window the effect's return arrives in",
    )
    p.add_argument(
        "takes",
        help="a directory a --save run left, or one holding several of them. The manifest "
        "has to carry the windows the run measured, since where a return lands is a "
        "measurement and not a thing this can work out from a take",
    )
    p.add_argument(
        "--margin",
        type=float,
        default=6.0,
        help="dB a movement must clear the steadiest setting's own scatter by",
    )
    options.add_subject(p)
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_arrival)

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
    options.add_subject(p)
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_vibrato)



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


def cmd_balance(args: argparse.Namespace) -> int:
    """Say what each saved run did to the balance between the two channels."""
    from pathlib import Path

    from .. import balance, takes
    from .. import record as envelope

    root = Path(args.takes)
    # One run, or a directory of them. The manifest is what says which, since a
    # run's directory holds one and a directory of runs holds none.
    roots = [root] if (root / "takes-manifest.json").exists() else sorted(root.glob("*"))
    found = []
    asked = {}
    for one in roots:
        if not (one / "takes-manifest.json").exists():
            continue
        # What this run was of, from the run rather than from the command line. A
        # record binding several runs across several types has one invocation and no
        # single subject, so the only place each run's own can come from is the
        # manifest its driver closed over.
        method = takes.method_of(one)
        about = options.about_of(method)
        asked[one.name] = method.get("question")
        for verdict in balance.measure(one, margin_db=args.margin):
            print(f"{one.name}: {verdict.describe()}")
            found.append((one.name, about, verdict))
    if not found:
        print(f"no saved takes under {args.takes}")
        return 1

    moved = [n for n, _a, v in found if v.moved_between_settings or v.did_not_repeat]
    unmeasured = [n for n, _a, v in found if v.not_measured]
    # The two are counted apart because they mean opposite things. A run that was
    # asked and said no belongs under the denominator; a run that could not be
    # asked does not, and folding it in reports a null the measurement never made.
    print(
        f"\n{len(moved)} of {len(found) - len(unmeasured)} moved the balance: "
        f"{' | '.join(moved) or 'none'}"
    )
    if unmeasured:
        print(f"{len(unmeasured)} could not be measured: {' | '.join(unmeasured)}")

    # A run this could not read as a balance is a run that measured levels and no
    # separation, and what it asked is in its own manifest -- so the finding states
    # the run's question and points at the rows that answer it. Derived from the
    # verdict rather than named on the command line: a record that has to be told
    # which of its runs found something is a record with a table kept by hand.
    # One per run and not one per stimulus: a run asked under three voices is one
    # run that could not be read, and three copies of its question would read as
    # three findings.
    refused = {n: a for n, a, v in found if v.not_measured and v.over_the_lead_on}
    findings = [
        envelope.finding(
            asked.get(n) or balance.MONO_SOURCE,
            runs=[n],
            **({"held": a["held"]} if a.get("held") else {}),
            read_in=f"runs[name={n}].by_setting[].over_the_lead_db",
            why=balance.WHY_A_REFUSED_RUN_STILL_REPORTS_ITS_LEVELS,
        )
        for n, a in refused.items()
    ]

    report.write_json(
        args.out,
        {
            "takes": str(args.takes),
            **options.subject_of(args),
            "method": balance.METHOD,
            "why_the_total_is_reported": balance.WHY_NOT_A_LEVEL,
            "why_the_pair_is_in_input_order": balance.WHY_THE_PAIR_IS_IN_INPUT_ORDER,
            "moved_the_balance": moved,
            **(
                {
                    "what_bounds_the_separation": {
                        "how_far_this_rig_has_separated": balance.how_far_a_record_separated(
                            args.separated_as_far_as
                        ),
                        "why": balance.WHY_A_SEPARATION_NEEDS_A_CEILING,
                    }
                }
                if args.separated_as_far_as
                else {}
            ),
            "runs": [
                {"name": n, **({"about": a} if a else {}), **v.to_json()}
                for n, a, v in found
            ],
            **({"findings": findings} if findings else {}),
        },
    )
    return 0


def cmd_balance_bands(args: argparse.Namespace) -> int:
    """Say what each saved run did to the separation, one band at a time."""
    from pathlib import Path

    from .. import balance, takes

    root = Path(args.takes)
    roots = [root] if (root / "takes-manifest.json").exists() else sorted(root.glob("*"))
    found = []
    for one in roots:
        if not (one / "takes-manifest.json").exists():
            continue
        # What this run was of, from the run rather than from the command line. A
        # record binding several runs across several types has one invocation and no
        # single subject, so the only place each run's own can come from is the
        # manifest its driver closed over.
        about = options.about_of(takes.method_of(one))
        for verdict in balance.by_band(one, margin_db=args.margin, band_set=args.bands):
            print(f"{one.name}: {verdict.describe()}")
            found.append((one.name, about, verdict))
    if not found:
        print(f"no saved takes under {args.takes}")
        return 1

    shaped = [n for n, _a, v in found if v.depends_on_frequency]
    unmeasured = [n for n, _a, v in found if v.not_measured]
    print(
        f"\n{len(shaped)} of {len(found) - len(unmeasured)} separate differently by band: "
        f"{' | '.join(shaped) or 'none'}"
    )
    if unmeasured:
        print(f"{len(unmeasured)} could not be measured: {' | '.join(unmeasured)}")

    report.write_json(
        args.out,
        {
            "takes": str(args.takes),
            **options.subject_of(args),
            "method": balance.SEPARATION_BY_BAND,
            "why_the_windows_are_one_length": balance.WHY_THE_WINDOWS_ARE_ONE_LENGTH,
            "why_a_flat_separation_is_the_yardstick": (
                balance.WHY_A_FLAT_SEPARATION_IS_THE_YARDSTICK
            ),
            "why_a_band_has_to_repeat": balance.WHY_A_BAND_HAS_TO_REPEAT,
            "why_the_pair_is_in_input_order": balance.WHY_THE_PAIR_IS_IN_INPUT_ORDER,
            "band_above_the_floor_db": balance.BAND_ABOVE_THE_FLOOR_DB,
            "band_repeats_within_db": balance.BAND_REPEATS_WITHIN_DB,
            "separated_differently_by_band": shaped,
            "runs": [
                {"name": n, **({"about": a} if a else {}), **v.to_json()}
                for n, a, v in found
            ],
        },
    )
    return 0


def cmd_arrival(args: argparse.Namespace) -> int:
    """Say what each saved run put in the early window and what it put in the late one."""
    from pathlib import Path

    from .. import arrival, takes

    root = Path(args.takes)
    roots = [root] if (root / "takes-manifest.json").exists() else sorted(root.glob("*"))
    found = []
    for one in roots:
        if not (one / "takes-manifest.json").exists():
            continue
        # What this run was of, from the run rather than from the command line. A
        # record binding several runs across several types has one invocation and no
        # single subject, so the only place each run's own can come from is the
        # manifest its driver closed over.
        about = options.about_of(takes.method_of(one))
        for verdict in arrival.measure(one, margin_db=args.margin):
            print(f"{one.name}: {verdict.describe()}")
            found.append((one.name, about, verdict))
    if not found:
        print(f"no saved takes under {args.takes}")
        return 1

    moved = [n for n, _a, v in found if v.moved_between_settings]
    unmeasured = [n for n, _a, v in found if v.not_measured]
    print(
        f"\n{len(moved)} of {len(found) - len(unmeasured)} moved signal between the two "
        f"windows: {' | '.join(moved) or 'none'}"
    )
    if unmeasured:
        print(f"{len(unmeasured)} could not be measured: {' | '.join(unmeasured)}")

    report.write_json(
        args.out,
        {
            "takes": str(args.takes),
            **options.subject_of(args),
            "method": arrival.METHOD,
            "why_the_windows_are_one_length": arrival.WHY_THE_WINDOWS_ARE_ONE_LENGTH,
            "why_the_windows_are_measured_and_not_printed": (
                arrival.WHY_THE_WINDOWS_ARE_MEASURED_AND_NOT_PRINTED
            ),
            "why_the_lead_is_the_floor": arrival.WHY_THE_LEAD_IS_THE_FLOOR,
            "why_a_return_that_overlaps_is_refused": (
                arrival.WHY_A_RETURN_THAT_OVERLAPS_IS_REFUSED
            ),
            "moved_between_the_windows": moved,
            "runs": [
                {"name": n, **({"about": a} if a else {}), **v.to_json()}
                for n, a, v in found
            ],
        },
    )
    return 0


def cmd_vibrato(args: argparse.Namespace) -> int:
    """Say what each setting did to the pitch, from inside single takes."""
    import json
    from pathlib import Path

    from .. import vibrato
    from ..takes import channel, channel_reaching, read

    root = Path(args.takes)
    manifest = json.loads((root / "takes-manifest.json").read_text())
    search = (args.min_rate, args.max_rate)

    # One channel for the whole run, chosen from the takes that reached highest.
    # Per take, a setting that turns the part down far enough is answered from an
    # input the unit is not on -- and a pitch track of an idle preamp finds no
    # vibrato, which is the same answer as a setting that has none.
    picked, reached = channel_reaching(root, [e["file"] for e in manifest.get("takes", [])])

    by_setting: dict[str, list] = {}
    for entry in manifest.get("takes", []):
        frames, sample_rate = read(root / entry["file"])
        found = vibrato.measure(
            channel(frames, picked), sample_rate, lead_s=args.lead, search_hz=search
        )
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
        vouched = vibrato.control(
            channel(frames, picked), sample_rate, lead_s=args.lead, search_hz=search
        )
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
            subject=options.subject_of(args, manifest),
            searched_hz=search,
            control=vouched,
            control_taken_from=quiet,
            channel={"read": picked, "reached_db": reached, "why": vibrato.WHY_CHANNEL},
        ),
    )
    return 0
