"""Mapping the address space, and asking it what it holds.

Four commands that talk to the unit over MIDI alone. None of them listens to
anything: what they establish is which addresses exist, what they accept, what a
reset puts back, and which tones the unit has.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .. import archive
from . import options, report
from .session import verified_link


def register(sub) -> None:
    p = sub.add_parser("sweep", help="map the address space by asking the machine")
    p.add_argument(
        "--margin",
        type=float,
        default=3.0,
        help="multiple of the measured reply time allowed before a probe is called a miss; "
        "a reply that outruns it is not lost, it arrives while the next probe is listening",
    )
    p.add_argument(
        "--canary",
        default="40 01 30",
        help="an address known to answer; used both to calibrate reply timing and to tell "
        "a silent machine from a silent address",
    )
    p.add_argument(
        "--sizes",
        type=lambda s: tuple(int(x, 0) for x in s.split(",")),
        default=(1, 16, 64),
        help="request sizes the reply timing is fitted over",
    )
    options.add_verify_reads(
        p, default=30, help="repeat reads in the selftest that gates the sweep"
    )
    p.add_argument(
        "--ceiling",
        type=options.number,
        default=64,
        help="largest size requested when bounding a region; a request far larger "
        "than any real region asks the machine for something no file would",
    )
    p.add_argument(
        "--low",
        default="00,30",
        help="comma-separated third bytes to try under each pair; a region need not begin at 00",
    )
    p.add_argument("--top-from", type=options.number, default=0x00)
    p.add_argument(
        "--top-to",
        type=options.number,
        default=None,
        help="restrict the top-byte range; default is the whole space",
    )
    options.add_out(p)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser(
        "write-probe",
        help="find what an address accepts by writing to it and reading it back",
    )
    p.add_argument(
        "regions",
        nargs="+",
        metavar="ADDR:COUNT",
        help="'40 01 30:24' probes 24 consecutive bytes from 40 01 30",
    )
    p.add_argument(
        "--settle",
        type=float,
        default=0.02,
        help="pause between a write and the read back that checks it",
    )
    options.add_verify_reads(p)
    options.add_out(p)
    p.set_defaults(func=cmd_write_probe)

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
        "reset-probe",
        help="find what each reset restores, by breaking the state first",
    )
    p.add_argument(
        "--baseline",
        default="data/units/roland-sc8850-01/power-on-state.json",
        help="the power-on capture every reset is compared against",
    )
    p.add_argument("--map", default="data/units/roland-sc8850-01/address-map.json")
    p.add_argument(
        "--write-probe",
        default="data/units/roland-sc8850-01/write-probe.json",
        help="where the addresses that take any value are read from",
    )
    p.add_argument(
        "--include-mode-set",
        action="store_true",
        help="also send System Mode Set, which reinitialises the unit rather than "
        "resetting its parameters",
    )
    p.add_argument(
        "--from-reset",
        default="GS Reset",
        help="sent before each reset under test, so all of them start from one state",
    )
    p.add_argument("--settle", type=float, default=0.6, help="pause after a reset before reading")
    options.add_verify_reads(p)
    options.add_out(p)
    p.set_defaults(func=cmd_reset_probe)


def cmd_sweep(args: argparse.Namespace) -> int:
    from ..sweep import MidiCalibrationError, Sweeper, calibrate, summarise

    with verified_link(
        args,
        refusing="sweeping",
        show_port=True,
        announce="Verifying the path before sweeping (a dropped byte becomes a missing row)",
    ) as link:
        canary = tuple(int(b, 16) for b in args.canary.split())
        try:
            timing = calibrate(link, addresses=[canary], device_id=args.device_id, sizes=args.sizes)
        except MidiCalibrationError as exc:
            print(f"\nCannot calibrate a deadline: {exc}")
            return 1
        timing.margin = args.margin
        # Level 1 only ever reads one byte; the bulk deadline is re-measured
        # against addresses level 1 finds, which is the first point it can be.
        print(f"\nLevel 1 timing: {timing.describe()}")

        sweeper = Sweeper(
            link,
            timing=timing,
            device_id=args.device_id,
            canary=canary,
            ceiling=args.ceiling,
        )
        top = range(args.top_from, args.top_to + 1) if args.top_to is not None else None
        low = tuple(int(b, 16) for b in args.low.split(",")) if args.low else (0x00,)
        result = sweeper.run(top_bytes=top, low_bytes=low, progress=lambda m: print(f"  {m}"))

    print()
    print(summarise(result))
    report.write_json(args.out, result.to_json())
    return 0 if result.complete and not result.lagged else 1


def cmd_write_probe(args: argparse.Namespace) -> int:
    from ..writeback import NOTE, RestoreFailed, Writer, summarise

    regions = [
        (tuple(int(b, 16) for b in spec.split(":")[0].split()), int(spec.split(":")[1], 0))
        for spec in args.regions
    ]
    with verified_link(
        args,
        refusing="writing",
        show_port=True,
        announce="Verifying the path before writing (a write is never acknowledged)",
    ) as link:
        writer = Writer(link, device_id=args.device_id, settle=args.settle)
        done = []
        try:
            for start, length in regions:
                done.append(writer.probe_region(start, length, progress=lambda m: print(f"  {m}")))
        except RestoreFailed as exc:
            print(f"\nSTOPPED: {exc}")
            return 1

    print()
    print(summarise(done))
    print(f"  {writer.writes} writes, {writer.reads} reads")

    report.write_json(
        args.out,
        {
            "device_id": f"{args.device_id:02X}",
            "settle_s": args.settle,
            "note": NOTE,
            "regions": [r.to_json() for r in done],
        },
    )
    return 0 if all(r.region_restored for r in done) else 1


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


def cmd_reset_probe(args: argparse.Namespace) -> int:
    from ..aliases import Snapshotter
    from ..resets import (
        METHOD,
        WHY_PRECEDED,
        Prober,
        ResetResult,
        catalogue,
        compare,
        mode_set,
        named,
        summarise,
    )

    captured = json.loads(Path(args.baseline).read_text())
    baseline = {
        tuple(int(b, 16) for b in a.split()): int(v, 16) for a, v in captured["values"].items()
    }
    regions = archive.regions(args.map)
    targets = [
        tuple(int(b, 16) for b in a.split())
        for a in list(archive.ALIASED_BYTES) + archive.accepting_bytes(args.write_probe)
    ]
    resets = catalogue(args.device_id)
    if args.include_mode_set:
        resets.append(mode_set(args.device_id))

    print(f"baseline: {args.baseline}, {len(baseline)} bytes")
    print(f"{len(targets)} addresses to mark, {len(regions)} regions read after each reset")

    results = []
    with verified_link(args, refusing="resetting") as link:
        prober = Prober(link, baseline=baseline, device_id=args.device_id, settle=args.settle)
        shot = Snapshotter(link, regions, device_id=args.device_id)
        # Put the unit back to a known state before each one, so the three
        # numbers can be compared with each other rather than each being read
        # against wherever the previous reset happened to leave things. The
        # state chosen is the one this same probe measured as reproducing the
        # power-on capture byte for byte.
        opener = named(args.from_reset, args.device_id)
        for reset in resets:
            print(f"\n{reset.label}")
            prober.apply(opener)
            marked, refused = prober.mark(targets)
            print(f"  marked {len(marked)} of {len(targets)} bytes")
            before = shot.unread
            prober.apply(reset)
            after = shot.take()
            result = ResetResult(
                label=reset.label,
                message=" ".join(f"{b:02X}" for b in reset.message),
                note=reset.note,
            )
            result.marked = {f"{a[0]:02X} {a[1]:02X} {a[2]:02X}": v for a, v in marked.items()}
            result.refused_the_mark = [f"{a[0]:02X} {a[1]:02X} {a[2]:02X}" for a in refused]
            result.regions_unread = shot.unread - before
            compare(result, marked, after, baseline)
            results.append(result)
            print(
                f"  {len(result.restored)} restored, {len(result.left_marked)} still marked, "
                f"{len(result.differs_from_power_on)} bytes differ from power-on"
            )

        # Leave the unit on the reset that comes closest to the power-on state,
        # so the next measurement does not start from whatever the last one
        # under test left. Ranked on the whole map rather than on the marked
        # bytes: every reset here restores those, and ranking on them alone
        # picks whichever happened to be tried first.
        best = min(results, key=lambda r: (len(r.differs_from_power_on), -len(r.restored)))
        link.send(next(x for x in resets if x.label == best.label).message)
        time.sleep(args.settle)
        print(f"\nleft the unit on {best.label}")

    print()
    print(summarise(results))

    report.write_json(
        args.out,
        {
            "device_id": f"{args.device_id:02X}",
            "baseline": args.baseline,
            "method": METHOD,
            "each_preceded_by": args.from_reset,
            "why_preceded": WHY_PRECEDED,
            "order": [r.label for r in results],
            "left_on": best.label,
            "resets": [r.to_json() for r in results],
        },
    )
    return 0
