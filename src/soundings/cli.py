"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import capture as cap
from . import roland
from .midi import MidiLink, list_ports
from .selftest import midi_selftest, timeline_selftest


def cmd_devices(args: argparse.Namespace) -> int:
    ins, outs = list_ports()
    print("MIDI inputs (what the machine sends to us):")
    for n in ins or ["  (none)"]:
        print(f"  {n}")
    print("MIDI outputs (what we send to the machine):")
    for n in outs or ["  (none)"]:
        print(f"  {n}")
    print("Audio inputs:")
    for d in cap.list_devices():
        print(f"  {d['name']}  ({d['max_input_channels']} ch, {int(d['default_samplerate'])} Hz)")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    ok = True
    with MidiLink(args.port) as link:
        print(f"MIDI in : {link.ports.input_name}")
        print(f"MIDI out: {link.ports.output_name}")
        print("\nMIDI round trip")
        report = midi_selftest(link, repeats=args.repeats, device_id=args.device_id)
        print(report)
        ok = ok and report.passed

        if not args.no_audio:
            print("\nCapture timeline")
            report = timeline_selftest(link, device=args.audio, ticks=args.ticks)
            print(report)
            ok = ok and report.passed

    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_read(args: argparse.Namespace) -> int:
    with MidiLink(args.port) as link:
        request = roland.rq1(args.address, args.size, device_id=args.device_id)
        reply = roland.parse_dt1(link.exchange(request, timeout=args.timeout))
        if reply is None:
            print(f"{args.address}  size={args.size}  -> no reply")
            return 1
        data = " ".join(f"{b:02X}" for b in reply.data)
        flag = "" if reply.checksum_ok else "  [CHECKSUM MISMATCH]"
        print(f"{args.address}  requested {args.size}, returned {reply.size}: {data}{flag}")
    return 0


def cmd_identity(args: argparse.Namespace) -> int:
    with MidiLink(args.port) as link:
        raw = link.exchange(roland.IDENTITY_REQUEST, timeout=1.0)
        if not raw:
            print("no reply")
            return 1
        print(" ".join(f"{b:02X}" for b in raw))
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    from .sweep import MidiCalibrationError, Sweeper, calibrate, summarise

    with MidiLink(args.port) as link:
        print(f"MIDI: {link.ports.output_name}")
        print("Verifying the path before sweeping (a dropped byte becomes a missing row)")
        report = midi_selftest(link, repeats=args.verify_reads, device_id=args.device_id)
        print(report)
        if not report.passed:
            print("\nSelftest failed. Not sweeping.")
            return 1

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

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.to_json(), indent=2) + "\n")
        print(f"\nwrote {path}")
    return 0 if result.complete and not result.lagged else 1


def cmd_write_probe(args: argparse.Namespace) -> int:
    from .writeback import RestoreFailed, Writer, summarise

    regions = [
        (tuple(int(b, 16) for b in spec.split(":")[0].split()), int(spec.split(":")[1], 0))
        for spec in args.regions
    ]
    with MidiLink(args.port) as link:
        print(f"MIDI: {link.ports.output_name}")
        print("Verifying the path before writing (a write is never acknowledged)")
        report = midi_selftest(link, repeats=args.verify_reads, device_id=args.device_id)
        print(report)
        if not report.passed:
            print("\nSelftest failed. Not writing.")
            return 1

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

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "device_id": f"{args.device_id:02X}",
                    "settle_s": args.settle,
                    "note": "Classifications describe what an address stores, not what it does. "
                    "Nothing here was heard.",
                    "regions": [r.to_json() for r in done],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {path}")
    return 0 if all(r.region_restored for r in done) else 1


def _regions_from_map(path: str, prefix: str) -> list[tuple[tuple[int, int, int], int]]:
    data = json.loads(Path(path).read_text())
    out = []
    for r in data["regions"]:
        if not r["address"].startswith(prefix) or not r["size"]:
            continue
        start = tuple(int(b, 16) for b in r["address"].split())
        out.append((start, r["size"]))
    return out


def cmd_alias_scan(args: argparse.Namespace) -> int:
    from .aliases import Scanner, Snapshotter, control_change, control_run, summarise

    regions = _regions_from_map(args.map, args.prefix)
    asked = (
        list(range(0, 120)) if args.controllers is None else [int(c, 0) for c in args.controllers]
    )
    # A run that finds nothing cannot say whether the unit stores nothing or the
    # scan was broken, so one controller known to be stored is always included.
    controllers = asked if args.control_cc in asked else [args.control_cc, *asked]
    with MidiLink(args.port) as link:
        print(f"MIDI: {link.ports.output_name}")
        report = midi_selftest(link, repeats=args.verify_reads, device_id=args.device_id)
        print(report)
        if not report.passed:
            print("\nSelftest failed. Not scanning.")
            return 1

        # Park the RPN and NRPN selectors before anything else, so a data entry
        # controller later in the scan writes to a known nothing rather than to
        # whatever parameter was last selected on this channel.
        for cc, value in ((101, 127), (100, 127), (99, 127), (98, 127)):
            link.send(control_change(args.channel, cc, value))

        shot = Snapshotter(link, regions, device_id=args.device_id)
        print(f"\n{len(regions)} regions under {args.prefix!r}, one snapshot each round")
        print("Control round: reading twice over with nothing sent")
        started = time.monotonic()
        restless = control_run(shot)
        print(
            f"  {len(restless)} addresses moved on their own"
            f" ({(time.monotonic() - started) / 4:.1f}s per snapshot)"
        )

        scanner = Scanner(shot, restless=restless)
        found = []
        for cc in controllers:
            messages = [control_change(args.channel, cc, v) for v in args.values]
            hit = scanner.attribute(f"CC{cc} ch{args.channel + 1}", messages, list(args.values))
            if hit is not None:
                found.append(hit)
                print(f"  CC{cc}: {len(hit.per_byte)} bytes follow it")

    control_hit = next((a for a in found if a.label.startswith(f"CC{args.control_cc} ")), None)
    print()
    print(summarise(found, restless, shot.unread))
    if control_hit is None:
        print(
            f"\n!! CC{args.control_cc} is known to be stored and this run did not see it. "
            "Every negative above is a statement about the scan, not about the unit."
        )

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "device_id": f"{args.device_id:02X}",
                    "channel": args.channel + 1,
                    "positive_control": {
                        "controller": args.control_cc,
                        "detected": control_hit is not None,
                        "at": sorted(control_hit.per_byte) if control_hit else [],
                        "why": "Included in every run. Without it a scan that finds nothing "
                        "cannot be told from a scan that cannot find anything.",
                    },
                    "region_prefix": args.prefix,
                    "regions_watched": len(regions),
                    "restless_addresses": sorted(restless),
                    "region_reads_failed": shot.unread,
                    "controllers_scanned": f"{min(controllers)}-{max(controllers)}",
                    "not_scanned": "Controllers 120 to 127 are channel mode messages. "
                    "Sending one resets the channel state every later attribution is measured "
                    "against, so they need a scan of their own.",
                    "rpn_parked": "RPN and NRPN were set to 7F 7F before the scan.",
                    "note": "A byte listed here followed the control both times it was sent. "
                    "That says where the value is kept, not that anything uses it.",
                    "controls": [a.to_json() for a in found],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {path}")
    return 0 if control_hit is not None else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="soundings", description=__doc__)
    parser.add_argument("--port", help="substring of the MIDI port name")
    parser.add_argument("--device-id", type=lambda s: int(s, 0), default=roland.DEFAULT_DEVICE_ID)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("devices", help="list MIDI and audio devices").set_defaults(func=cmd_devices)

    p = sub.add_parser("selftest", help="prove the measurement path before trusting it")
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--ticks", type=int, default=10)
    p.add_argument("--audio", help="substring of the audio input device name")
    p.add_argument("--no-audio", action="store_true", help="MIDI checks only")
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser("read", help="read one address with RQ1")
    p.add_argument("address", help="three hex bytes, e.g. '40 01 30'")
    p.add_argument("size", type=lambda s: int(s, 0), nargs="?", default=1)
    p.add_argument("--timeout", type=float, default=1.0)
    p.set_defaults(func=cmd_read)

    sub.add_parser("identity", help="send an Identity Request").set_defaults(func=cmd_identity)

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
    p.add_argument(
        "--verify-reads",
        type=int,
        default=30,
        help="repeat reads in the selftest that gates the sweep",
    )
    p.add_argument(
        "--ceiling",
        type=lambda s: int(s, 0),
        default=64,
        help="largest size requested when bounding a region; a request far larger "
        "than any real region asks the machine for something no file would",
    )
    p.add_argument(
        "--low",
        default="00,30",
        help="comma-separated third bytes to try under each pair; a region need not begin at 00",
    )
    p.add_argument("--top-from", type=lambda s: int(s, 0), default=0x00)
    p.add_argument(
        "--top-to",
        type=lambda s: int(s, 0),
        default=None,
        help="restrict the top-byte range; default is the whole space",
    )
    p.add_argument("--out", help="write the result as JSON")
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
    p.add_argument("--verify-reads", type=int, default=20)
    p.add_argument("--out", help="write the result as JSON")
    p.set_defaults(func=cmd_write_probe)

    p = sub.add_parser(
        "alias-scan",
        help="find where a control change is stored, by diffing the address space around it",
    )
    p.add_argument(
        "--map",
        default="data/units/roland-sc8850-01/address-map.json",
        help="address map the watched regions are taken from",
    )
    p.add_argument("--prefix", default="40 ", help="only watch regions whose address starts here")
    p.add_argument("--channel", type=int, default=0, help="zero based MIDI channel")
    p.add_argument(
        "--values",
        type=lambda s: tuple(int(v, 0) for v in s.split(",")),
        default=(0x20, 0x60),
        help="the two values sent; a byte must follow both to be attributed",
    )
    p.add_argument("--controllers", nargs="*", help="controller numbers; default is 0 to 119")
    p.add_argument(
        "--control-cc",
        type=int,
        default=7,
        help="a controller known to be stored, always scanned, so a run that finds nothing "
        "can be told from a run that could not have found anything",
    )
    p.add_argument("--verify-reads", type=int, default=20)
    p.add_argument("--out", help="write the result as JSON")
    p.set_defaults(func=cmd_alias_scan)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
