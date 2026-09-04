"""Measure kept takes, with no machine attached.

Device time is the scarce resource, so a take is recorded once and asked
questions afterwards. Both commands here read a dry and a wet take of the same
note and say what the effect did between them; neither opens a MIDI port.
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
    options.add_out(p)
    p.set_defaults(func=cmd_motion)

    p = sub.add_parser(
        "decay",
        help="say how long an effect's tail takes to die in each octave band, from a "
        "dry and a wet take, with no machine attached",
    )
    p.add_argument("dry", help="a take with the effect off")
    p.add_argument("wet", help="the same note with the effect on")
    p.add_argument(
        "--lead",
        type=float,
        default=0.5,
        help="seconds of silence at the head of the take; the per-band noise floor is "
        "measured in it, and it is what says where a tail stops being a tail",
    )
    options.add_out(p)
    p.set_defaults(func=cmd_decay)


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
    found = motion.measure(dry, wet, rate, search_ms=span, rate_range=rates)
    # Always, not only when the answer is a null. A run that reports motion is
    # not excused the control either: knowing the tracker works on this material
    # is what says a recovered rate is the effect's and not the search's.
    vouched = motion.control(dry, wet, rate, search_ms=span, rate_range=rates)

    print(f"{args.dry} against {args.wet}, {rate} Hz")
    print(found.describe())
    ladder = ", ".join(
        f"{a['injected_depth_ms']}{'' if a['recovered'] else ' (missed)'}"
        for a in vouched["attempts"]
    )
    print(
        f"  control: swings at {vouched['injected_rate_hz']} Hz injected at the real return's "
        f"level ({vouched['return_level_db']} dB), peak to peak in ms: {ladder}"
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
            "dry": str(args.dry),
            "wet": str(args.wet),
            "sample_rate": rate,
            "lead_s": args.lead,
            **found.to_json(),
        },
    )
    return 0
