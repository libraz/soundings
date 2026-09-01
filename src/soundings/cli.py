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


def _part_block(family: int, channel: int) -> tuple[int, int, int]:
    """The per-part block for a channel, in one of the families of them.

    GS numbers the parts so that channel 10 comes first: 40 f0 is part 10, and
    40 f1 through 40 fF are channels 1 to 9 and 11 to 16 in order. `family` is
    the whole high nibble of the middle byte -- 0x10 for the part block that
    holds the tone, 0x40 for the one CC32 writes into -- so passing 0x00 by
    mistake addresses the patch common block instead, which answers, and which
    holds something entirely unrelated.
    """
    index = 0 if channel == 9 else (channel + 1 if channel < 9 else channel)
    return (0x40, family | index, 0x00)


def _clear_bank_latch(link, channel: int, device_id: int) -> str:
    """Put the bank select latch back where the scan found it.

    Bank select is held without changing anything readable until a program
    change commits the three of them together, and a pair the unit does not have
    is discarded whole. So a scan that sends CC0 or CC32 on their own -- which a
    controller sweep does, having no reason to send a program change -- ends with
    a latch set to whatever it tried last, and every program change in the next
    run is thrown away. Nothing in the address space says so.

    The committing program change is the tone the part already holds, and the
    bank halves are read back rather than assumed, so the commit puts the part
    exactly where it was rather than somewhere tidy.
    """
    from . import roland as r

    tone = _part_block(0x10, channel)

    def read(address, size):
        # Drain first: this runs straight after hundreds of request-and-reply
        # pairs, and a reply still in flight is answered to whichever request
        # asks next.
        while link.receive(timeout=0.05):
            pass
        reply = r.parse_dt1(link.exchange(r.rq1(address, size, device_id=device_id), timeout=0.6))
        return None if reply is None or reply.address != address else list(reply.data)

    part = read(tone, 2)
    mapped = read(_part_block(0x40, channel), 1)
    if part is None or mapped is None:
        return "could not be read back, so the bank latch was left as the scan left it"
    link.send([0xB0 | (channel & 0x0F), 0, part[0]])
    link.send([0xB0 | (channel & 0x0F), 32, mapped[0]])
    link.send([0xC0 | (channel & 0x0F), part[1]])
    time.sleep(0.25)
    after = read(tone, 2)
    if after != part:
        return f"restoring it moved the part from {part} to {after}"
    return f"committed bank {part[0]}, map {mapped[0]}, program {part[1]}"


def _regions_from_map(path: str, prefix: str) -> list[tuple[tuple[int, int, int], int]]:
    data = json.loads(Path(path).read_text())
    out = []
    for r in data["regions"]:
        if not r["address"].startswith(prefix) or not r["size"]:
            continue
        start = tuple(int(b, 16) for b in r["address"].split())
        out.append((start, r["size"]))
    return out


def _stimuli(args: argparse.Namespace) -> list:
    """Build the list of things to send, for the kind asked for."""
    from . import aliases as al

    ch = args.channel
    if args.kind == "cc":
        numbers = (
            list(range(0, 120))
            if args.controllers is None
            else [int(c, 0) for c in args.controllers]
        )
        return [al.cc(ch, n) for n in numbers]
    if args.kind == "nrpn":
        return [al.nrpn(ch, msb, lsb, name) for msb, lsb, name in al.GS_NRPN]
    if args.kind == "drum-nrpn":
        note = args.note
        return [al.nrpn(ch, msb, note, f"{name} note {note}") for msb, name in al.GS_DRUM_NRPN]
    if args.kind == "rpn":
        return [al.rpn(ch, msb, lsb, name, values=v) for msb, lsb, name, v in al.GS_RPN]
    if args.kind == "channel":
        return [
            al.program_change(ch),
            # Bank MSB 8 rather than a low number: the pair is accepted or
            # rejected whole, and program 0 does not exist in banks 1, 2 or 3, so
            # those values measure the rejection instead of the storage.
            al.bank_then_program(ch, 0, values=(0x00, 0x08)),
            al.bank_then_program(ch, 32),
            al.pitch_bend(ch),
            al.channel_pressure(ch),
        ]
    raise ValueError(args.kind)


def cmd_alias_scan(args: argparse.Namespace) -> int:
    from .aliases import Scanner, Snapshotter, cc, control_change, control_run, summarise

    regions = _regions_from_map(args.map, args.prefix)
    # A run that finds nothing cannot say whether the unit stores nothing or the
    # scan was broken, so one control change known to be stored is always sent.
    # Sent at both ends rather than once: a control that passes at the start says
    # the reading worked at the start, and a run has been seen to miss something
    # after that point.
    control = cc(args.channel, args.control_cc)
    stimuli = [control, *_stimuli(args), control]
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
        for number, value in ((101, 127), (100, 127), (99, 127), (98, 127)):
            link.send(control_change(args.channel, number, value))

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
        control_passes = 0
        control_hit = None
        for stimulus in stimuli:
            hit = scanner.attribute(stimulus)
            if stimulus is control:
                control_passes += hit is not None
                control_hit = control_hit or hit
                print(f"  control CC{args.control_cc}: {'seen' if hit else 'NOT SEEN'}")
                continue
            if hit is not None:
                found.append(hit)
                print(f"  {stimulus.label}: {', '.join(hit.addresses)}")

        latch = _clear_bank_latch(link, args.channel, args.device_id)
        print(f"  bank latch: {latch}")

    # The control proves the snapshots and the diff work. It does not prove that
    # a message of the kind under test was received, because it is not one of
    # them, so a kind that lands nothing anywhere is reported as unreached
    # rather than as absent.
    reached = args.kind == "cc" or any(a.kind == args.kind for a in found)
    bracketed = control_passes == 2
    print()
    print(summarise(found, restless, shot.unread))
    if scanner.recovered:
        print(
            f"  !! {len(scanner.recovered)} stimuli were missed by the first pass and found by "
            f"the retry: {', '.join(scanner.recovered)}"
        )
    for label, addresses in scanner.residue.items():
        print(f"  !! {label} did not put back: {', '.join(addresses)}")
    if not bracketed:
        print(
            f"\n!! CC{args.control_cc} is known to be stored and this run saw it "
            f"{control_passes} of the 2 times it was sent. Every negative above is a statement "
            "about the scan, not about the unit."
        )
    elif not reached:
        print(
            f"\n!! nothing of kind {args.kind!r} was attributed anywhere. The control was seen, so "
            "the reading works; but no message of this kind landed, and 'the unit stores none of "
            "these' cannot be told from 'these never arrived'."
        )

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "device_id": f"{args.device_id:02X}",
                    "channel": args.channel + 1,
                    "kind": args.kind,
                    "positive_control": {
                        "controller": args.control_cc,
                        "sent": 2,
                        "detected": control_passes,
                        "at": control_hit.addresses if control_hit else [],
                        "why": "Sent before the first stimulus and after the last. Without it a "
                        "scan that finds nothing cannot be told from a scan that cannot find "
                        "anything; sent only once, it says nothing about the rest of the run.",
                    },
                    "left_changed_afterwards": {
                        "by_stimulus": scanner.residue,
                        "why": "Every stimulus ends on the value it started with, so anything "
                        "listed here is state one message carried into the next.",
                    },
                    "missed_by_the_first_pass": {
                        "stimuli": scanner.recovered,
                        "why": "Each stimulus that lands nothing is retried with its two values "
                        "swapped. Anything listed here is something a single pass would have "
                        "reported as absent.",
                    },
                    "kind_reached": {
                        "value": reached,
                        "why": "The control is a control change, so it cannot show that a "
                        "message of another kind arrived. Where this is false, every negative "
                        "in the run is about the path, not about the unit.",
                    },
                    "method": "Each stimulus was sent at its low value, snapshotted, sent at its "
                    "high value, snapshotted, and sent at its low value again. A byte is listed "
                    "only if it moved both times, to a different value each time.",
                    "region_prefix": args.prefix,
                    "regions_watched": len(regions),
                    "stimuli_sent": [s.label for s in stimuli],
                    "restless_addresses": sorted(restless),
                    "region_reads_failed": shot.unread,
                    "not_scanned": "Controllers 120 to 127 are channel mode messages. "
                    "Sending one resets the channel state every later attribution is measured "
                    "against, so they need a scan of their own.",
                    "rpn_parked": "RPN and NRPN were set to 7F 7F before the scan.",
                    "bank_latch_on_exit": latch,
                    "note": "A byte listed here followed the stimulus out and back. That says "
                    "where the value is kept, not that anything uses it.",
                    "attributed": [a.to_json() for a in found],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {path}")
    return 0 if bracketed and reached else 1


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
        help="find where a message is stored, by diffing the address space around it",
    )
    p.add_argument(
        "--map",
        default="data/units/roland-sc8850-01/address-map.json",
        help="address map the watched regions are taken from",
    )
    p.add_argument("--prefix", default="40 ", help="only watch regions whose address starts here")
    p.add_argument("--channel", type=int, default=0, help="zero based MIDI channel")
    p.add_argument(
        "--kind",
        default="cc",
        choices=("cc", "nrpn", "drum-nrpn", "rpn", "channel"),
        help="what to send; two kinds landing on one address is what makes them aliases",
    )
    p.add_argument(
        "--note",
        type=lambda s: int(s, 0),
        default=36,
        help="drum note the per-note NRPNs address, for --kind drum-nrpn",
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
