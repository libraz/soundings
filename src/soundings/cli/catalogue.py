"""Which tones and which insertion effects the unit actually has.

Two commands that ask for something by number and then read back what the unit
took. A number that is not there leaves the previous selection standing, so the
answer is never the request echoed: it is what the part or the effect block holds
afterwards, which is the only thing that distinguishes an accepted number from an
ignored one.

Both put the unit back to a known reset when they finish, because a survey that
ends leaves whatever it asked for last selected, and the next measurement would
inherit it.
"""

from __future__ import annotations

import argparse

from . import options, report
from .session import verified_link


def register(sub) -> None:
    p = sub.add_parser(
        "tone-map",
        help="find which tones exist, by asking for each and seeing whether it was taken",
    )
    options.add_channel(p)
    p.add_argument(
        "--map-select",
        type=options.number,
        default=0,
        help="the CC32 value held for the whole survey",
    )
    p.add_argument(
        "--exhaustive",
        action="store_true",
        help="ask every bank for all 128 programs instead of sampling first; three times "
        "the cost and the only form with nothing to caveat",
    )
    p.add_argument("--banks-from", type=options.number, default=0)
    p.add_argument("--banks-to", type=options.number, default=127)
    p.add_argument(
        "--settle",
        type=float,
        default=0.04,
        help="pause after a program change before reading the part back; the tone was "
        "measured to reach the part block in a median 12.6 ms",
    )
    options.add_verify_reads(p)
    options.add_out(p)
    p.set_defaults(func=cmd_tone_map)

    p = sub.add_parser(
        "efx-map",
        help="find which insertion effects exist, by asking for each type and reading "
        "it back with the settings it loads",
    )
    p.add_argument(
        "--settle",
        type=float,
        default=0.02,
        help="pause after selecting a type before reading it back",
    )
    options.add_verify_reads(p)
    options.add_out(p)
    p.set_defaults(func=cmd_efx_map)


def cmd_tone_map(args: argparse.Namespace) -> int:
    from ..resets import Prober, named
    from ..tonemap import METHOD, SAMPLE_PROGRAMS, Asker, sampling_caveat, summarise, survey

    with verified_link(args, refusing="asking") as link:
        gs_reset = named("GS Reset", args.device_id)
        prober = Prober(link, baseline={}, device_id=args.device_id)
        prober.apply(gs_reset)

        asker = Asker(
            link,
            channel=args.channel,
            map_select=args.map_select,
            device_id=args.device_id,
            settle=args.settle,
        )
        # Nothing can be asked until the part is standing somewhere the sweep
        # knows about, or the first answer is a comparison against a guess.
        if not asker.settle_on(0, 0):
            print(
                "\nThe part would not take bank 0 program 0, so nothing here would mean anything."
            )
            return 1

        banks = range(args.banks_from, args.banks_to + 1)
        print(
            f"\nmap {args.map_select}, channel {args.channel + 1}, "
            f"banks {banks.start}-{banks.stop - 1}"
        )
        found = survey(
            asker,
            banks=banks,
            exhaustive=args.exhaustive,
            progress=lambda m: print(f"  {m}"),
        )
        prober.apply(gs_reset)

    print()
    print(
        summarise(found, len(SAMPLE_PROGRAMS), asker.asks, asker.unread, exhaustive=args.exhaustive)
    )

    report.write_json(
        args.out,
        {
            "device_id": f"{args.device_id:02X}",
            "channel": args.channel + 1,
            "map_select": args.map_select,
            "method": METHOD,
            "sampled_before_sweeping": None if args.exhaustive else list(SAMPLE_PROGRAMS),
            "sampling_caveat": sampling_caveat(args.exhaustive),
            "requests": asker.asks,
            "reads_unusable": asker.unread,
            "banks": [b.to_json() for b in found],
        },
    )
    return 0 if not asker.unread else 1


def cmd_efx_map(args: argparse.Namespace) -> int:
    from ..efxmap import METHOD, Asker, summarise, survey
    from ..resets import Prober, named

    with verified_link(args, refusing="asking") as link:
        gs_reset = named("GS Reset", args.device_id)
        prober = Prober(link, baseline={}, device_id=args.device_id)
        prober.apply(gs_reset)

        asker = Asker(link, device_id=args.device_id, settle=args.settle)
        print("\nasking for all 16384 insertion effect type numbers")
        found = survey(asker, progress=lambda m: print(f"  {m}"))
        # The type is left where the sweep ended otherwise, so the next thing to
        # touch the unit would inherit an effect nobody selected.
        prober.apply(gs_reset)

    print()
    print(summarise(found))
    print(f"  {asker.writes} writes, {asker.reads} reads")

    report.write_json(
        args.out,
        {
            "device_id": f"{args.device_id:02X}",
            "settle_s": args.settle,
            "method": METHOD,
            **found.to_json(),
        },
    )
    return 0 if not found.unread else 1
