"""What an effect did between a dry take and a wet one.

Two commands, each given one pair of takes of the same note -- the effect off,
then the effect on -- and each recovering a quantity from the difference: how the
wet take moves over time, and how long its tail takes to die in each octave band.

Neither opens a MIDI port. Device time is the scarce resource, so a take is
recorded once and asked questions afterwards.
"""

from __future__ import annotations

import argparse
from itertools import combinations

from . import options, report
from .blocks import BAND_SET_NAMES


def register(sub) -> None:
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
    p.add_argument(
        "--frame-ms",
        type=float,
        default=10.0,
        help="length of the frame each delay is read over, ms. This sets the "
        "deepest swing the track can follow at a given rate -- shorter follows a "
        "deeper one and loses a little coverage",
    )
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
    options.add_subject(p)
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_motion)

    p = sub.add_parser(
        "decay",
        help="say how long an effect's tail takes to die in each octave band, from a "
        "dry and a wet take, with no machine attached",
    )
    p.add_argument("dry", help="a take with the effect off")
    p.add_argument("wet", help="the same note with the effect on")
    options.add_subject(p)
    p.add_argument(
        "--floor",
        nargs=2,
        metavar=("TAKE", "TAKE"),
        help="two takes of ONE setting, subtracted from each other to measure what the "
        "subtraction itself leaves. The lead-in bounds the interface noise only; this "
        "bounds the error under the note, which is the larger of the two and is what a "
        "type that did nothing at all produces a full set of decay times out of",
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
        "phase",
        help="say what an effect did to the phase of each band, from a dry and a wet "
        "take, with no machine attached",
    )
    p.add_argument("dry", help="a take with the effect flat")
    p.add_argument("wet", help="the same note with the effect doing something")
    p.add_argument(
        "--control",
        nargs="+",
        metavar="TAKE",
        help="two or more takes of ONE setting, read exactly as the pair above, and "
        "every pair of them measured. A phase between two takes of a note is the "
        "alignment's and the stimulus's as well as the effect's, and this is what the "
        "method returns when the answer is nothing. Every pair rather than one because "
        "one pair is one draw. Without it the record says what a phase was and not "
        "whether it was one",
    )
    p.add_argument(
        "--band-set",
        choices=BAND_SET_NAMES,
        default="third-octave",
        help="which set of bands the phase is reported over",
    )
    options.add_subject(p)
    p.add_argument(
        "--block",
        type=float,
        default=None,
        help="seconds aligned and transformed as one. The unit's clock and the "
        "converter's do not run at the same rate, so a block long enough for them to "
        "move a sample reports a phase that is turning inside it and a coherence that "
        "has collapsed",
    )
    p.add_argument(
        "--lead",
        type=float,
        default=0.6,
        help="seconds of silence at the head of the take, dropped before anything is "
        "measured. Every stimulus in the catalogue records 0.6",
    )
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_phase)


def _pair(dry_path: str, wet_path: str, *, on: int | None = None):
    """Load two takes, read both from one channel. `takes.read_pair` is where it lives."""
    from ..takes import read_pair

    return read_pair(dry_path, wet_path, on=on)


def cmd_motion(args: argparse.Namespace) -> int:
    """Say what an effect does over time, from a dry and a wet take of the same note."""
    from .. import motion

    dry, wet, rate, picked = _pair(args.dry, args.wet)
    span = (0.0, args.max_delay)
    rates = (args.min_rate, args.max_rate)
    found = motion.measure(
        dry,
        wet,
        rate,
        search_ms=span,
        rate_range=rates,
        lead_s=args.lead,
        window=args.frame_ms / 1000.0,
    )
    # Always, not only when the answer is a null. A run that reports motion is
    # not excused the control either: knowing the tracker works on this material
    # is what says a recovered rate is the effect's and not the search's.
    vouched = motion.control(
        dry,
        wet,
        rate,
        search_ms=span,
        rate_range=rates,
        lead_s=args.lead,
        window=args.frame_ms / 1000.0,
    )

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
            **options.subject_of(args),
            "dry": str(args.dry),
            "wet": str(args.wet),
            "sample_rate": rate,
            "channel": picked,
            "searched_rate_hz": [args.min_rate, args.max_rate],
            "lead_s": args.lead,
            "positive_control": vouched,
            **(
                {"control_failed": motion.CONTROL_FAILED}
                if vouched["detectable_ms"] is None
                else {}
            ),
            **found.to_json(),
            # After the track's own fields, because it is the one verdict here
            # that reads both of them: a track that could be followed says
            # nothing on its own until the control says this material could have
            # given a modulation up.
            "conclusive": motion.is_conclusive(found, vouched),
        },
    )
    return 0 if vouched["detectable_ms"] is not None else 1


def cmd_decay(args: argparse.Namespace) -> int:
    """Say how long an effect's tail takes to die, per octave band."""
    from .. import decay as dec

    dry, wet, rate, picked = _pair(args.dry, args.wet)
    tail, noise = dec.isolate_tail(dry, wet, rate, lead=args.lead)
    floor = None
    if args.floor:
        first, second, floor_rate, _ = _pair(*args.floor, on=picked["read"])
        if floor_rate != rate:
            print(f"the floor pair is {floor_rate} Hz and the dry and wet are {rate} Hz")
            return 1
        floor = dec.subtraction_floor(first, second)
    found = dec.measure(tail, rate, noise=noise, floor=floor)
    print(f"{args.dry} against {args.wet}, {rate} Hz")
    print(found.describe())

    report.write_json(
        args.out,
        {
            **options.subject_of(args),
            "dry": str(args.dry),
            "wet": str(args.wet),
            "floor_from": [str(p) for p in args.floor] if args.floor else None,
            "sample_rate": rate,
            "channel": picked,
            "lead_s": args.lead,
            **found.to_json(),
        },
    )
    return 0


def cmd_phase(args: argparse.Namespace) -> int:
    """Say what an effect did to the phase of each band, from a dry and a wet take."""
    from .. import efxbands
    from .. import phase as ph

    centres, width = efxbands.BAND_SETS[args.band_set]
    dry, wet, rate, picked = _pair(args.dry, args.wet)
    head = int(args.lead * rate)
    how = {"bands": centres, "width_octaves": width}
    if args.block is not None:
        how["block_s"] = args.block
    found = ph.measure(dry[head:], wet[head:], rate, **how)

    vouched = None
    if args.control:
        if len(args.control) < 2:
            print("--control needs two or more takes of one setting")
            return 2
        # The control read on the channel the pair was read on, not on one chosen
        # again: a bound measured on the other leg of the unit bounds a comparison
        # that was never made.
        repeats = []
        for first, second in combinations(args.control, 2):
            a, b, control_rate, _ = _pair(first, second, on=picked["read"])
            if control_rate != rate:
                print(f"a control pair is {control_rate} Hz and the pair is {rate} Hz")
                return 1
            repeats.append((a[head:], b[head:]))
        vouched = ph.control(repeats, rate, **how)

    print(f"{args.dry} against {args.wet}, {rate} Hz")
    print(ph.describe(found, vouched))

    report.write_json(
        args.out,
        {
            **options.subject_of(args),
            "dry": str(args.dry),
            "wet": str(args.wet),
            "control_from": [str(p) for p in args.control] if args.control else None,
            "sample_rate": rate,
            "channel": picked,
            "lead_s": args.lead,
            "limits": ph.LIMITS,
            **found,
            **({"control": vouched} if vouched else {}),
            **(
                {"control_failed": ph.CONTROL_FAILED}
                if vouched is not None and vouched["largest_deg"] is None
                else {}
            ),
            "conclusive": ph.is_conclusive(found, vouched) if vouched else None,
        },
    )
    return 0 if vouched is None or vouched["largest_deg"] is not None else 1
