"""What one saved run of takes shows, asked after the session ended.

Three commands over the directory a `--save` run leaves. One re-reaches the
audible verdict the session reached, and the other two ask what that verdict
cannot see: a parameter that moves signal between the channels is invisible to a
comparison made in one of them, and a parameter that modulates pitch is invisible
to a comparison of two takes at all.

None of them opens a MIDI port. The takes carry the settings, the stimuli and the
label, so the only thing missing from the directory is the machine, and the
machine is not consulted after the last take is recorded.
"""

from __future__ import annotations

import argparse

from . import options, report


def register(sub) -> None:
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
            "why_the_pair_is_in_input_order": balance.WHY_THE_PAIR_IS_IN_INPUT_ORDER,
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
            searched_hz=search,
            control=vouched,
            control_taken_from=quiet,
            channel={"read": picked, "reached_db": reached, "why": vibrato.WHY_CHANNEL},
        ),
    )
    return 0
