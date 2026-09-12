"""What an effect did between a dry take and a wet one.

Two commands, each given one pair of takes of the same note -- the effect off,
then the effect on -- and each recovering a quantity from the difference: how the
wet take moves over time, and how long its tail takes to die in each octave band.

Neither opens a MIDI port. Device time is the scarce resource, so a take is
recorded once and asked questions afterwards.
"""

from __future__ import annotations

import argparse

from . import options, report


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


WHY_CHANNEL = (
    "Which channel of the interface both takes were read from, and the highest each channel "
    "reached across the two. One channel for the pair rather than the loudest of each take: this "
    "measurement subtracts one take from the other, and an interface carries inputs the unit is "
    "not on which are not silent, so a take whose output fell below one of them would be answered "
    "from that input and subtracted from a different one -- leaving a residual of nothing that "
    "reads as an effect doing nothing."
)


def _pair(dry_path: str, wet_path: str, *, on: int | None = None):
    """Load two takes, read both from one channel, and say which and what each reached.

    `on` names the channel instead of choosing one, which is what a second pair
    read as a control over the first needs: a floor measured on the other leg of
    the unit bounds a subtraction that was never made.
    """
    from ..takes import channel, channel_across, read

    dry, dry_rate = read(dry_path)
    wet, wet_rate = read(wet_path)
    if dry_rate != wet_rate:
        raise SystemExit(f"the two takes were captured at {dry_rate} and {wet_rate} Hz")
    chosen, reached = channel_across(dry, wet)
    picked = chosen if on is None else on
    said = {"read": picked, "reached_db": reached, "why": WHY_CHANNEL}
    return channel(dry, picked), channel(wet, picked), dry_rate, said


def cmd_motion(args: argparse.Namespace) -> int:
    """Say what an effect does over time, from a dry and a wet take of the same note."""
    from .. import motion

    dry, wet, rate, picked = _pair(args.dry, args.wet)
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
            "type_id": args.type,
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
