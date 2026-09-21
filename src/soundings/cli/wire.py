"""What is on the other end of the cable, and whether the cable works.

Nothing here measures the unit. These are the commands that answer the questions
asked before a measurement: which ports exist, whether the round trip is sound,
and what one address says right now.
"""

from __future__ import annotations

import argparse

from .. import capture as cap
from .. import roland
from ..midi import MidiLink, list_ports
from ..selftest import midi_selftest, timeline_selftest
from . import options, report


def register(sub) -> None:
    sub.add_parser("devices", help="list MIDI and audio devices").set_defaults(func=cmd_devices)

    p = sub.add_parser("selftest", help="prove the measurement path before trusting it")
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--ticks", type=int, default=10)
    p.add_argument(
        "--probe",
        default=None,
        help="an address that answers, read repeatedly to prove the round trip. Needed with "
        "--model-id, since the default belongs to the GS space and would answer nothing in "
        "another one -- which would report a healthy path as a broken one",
    )
    p.add_argument(
        "--probe-size",
        type=options.number,
        default=None,
        help="bytes to ask the probe address for. Long enough that a truncated reply is "
        "visible; an address that answers only one byte proves the path over one byte",
    )
    options.add_audio(p)
    p.add_argument("--no-audio", action="store_true", help="MIDI checks only")
    p.set_defaults(func=cmd_selftest, takes_model_id=True)

    p = sub.add_parser("read", help="read one address with RQ1")
    p.add_argument("address", help="three hex bytes, e.g. '40 01 30'")
    p.add_argument("size", type=options.number, nargs="?", default=1)
    p.add_argument("--timeout", type=float, default=1.0)
    p.set_defaults(func=cmd_read, takes_model_id=True)

    sub.add_parser("identity", help="send an Identity Request").set_defaults(func=cmd_identity)

    p = sub.add_parser(
        "output-map",
        help="say which capture channels each socket pair on the back of the unit "
        "arrived on, read from takes already saved, with no machine attached",
    )
    p.add_argument(
        "takes",
        help="a directory of takes with the takes-manifest.json a --save run wrote, "
        "holding the same note struck with the part on each output in turn",
    )
    p.add_argument(
        "--output",
        required=True,
        metavar="REGEX",
        help="a pattern over each take's setting with a group named `output`, which is "
        "the socket pair the part was put on for that take",
    )
    p.add_argument(
        "--window",
        type=float,
        default=1.0,
        help="seconds either side of the note's arrival that the rise is read over",
    )
    p.add_argument(
        "--margin",
        type=float,
        default=None,
        help="decibels a carrying channel's rise must stand over a non-carrying one's. "
        "A floor on the evidence rather than a threshold on the rise: what decides the "
        "mapping is that the two groups separate",
    )
    p.add_argument(
        "--device",
        default=None,
        help="the capture device the takes were made on, named because re-cabling "
        "changes the answer and nothing in a take would show that",
    )
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_output_map)


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
        probe = {}
        if args.probe:
            probe["probe_address"] = args.probe
        if args.probe_size is not None:
            probe["probe_size"] = args.probe_size
        report = midi_selftest(
            link,
            repeats=args.repeats,
            device_id=args.device_id,
            model_id=args.model_id,
            **probe,
        )
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
        request = roland.rq1(
            args.address, args.size, device_id=args.device_id, model_id=args.model_id
        )
        reply = roland.parse_dt1(link.exchange(request, timeout=args.timeout))
        if reply is None:
            print(f"{args.address}  size={args.size}  -> no reply")
            return 1
        data = " ".join(f"{b:02X}" for b in reply.data)
        flag = "" if reply.checksum_ok else "  [CHECKSUM MISMATCH]"
        # The model id is printed rather than checked. This is the command an
        # operator reaches for to find out which spaces a unit answers in, and
        # refusing a reply from another one would hide the answer.
        space = "" if reply.model_id == args.model_id else f"  [model {reply.model_id:02X}]"
        print(f"{args.address}  requested {args.size}, returned {reply.size}: {data}{flag}{space}")
    return 0


def cmd_identity(args: argparse.Namespace) -> int:
    with MidiLink(args.port) as link:
        raw = link.exchange(roland.IDENTITY_REQUEST, timeout=1.0)
        if not raw:
            print("no reply")
            return 1
        print(" ".join(f"{b:02X}" for b in raw))
    return 0


def cmd_output_map(args: argparse.Namespace) -> int:
    """Which capture channel carries which output, read from takes already saved."""
    from .. import outputmap

    found = outputmap.read_directory(
        args.takes,
        output=args.output,
        window_s=args.window,
        margin_db=args.margin if args.margin is not None else outputmap.MARGIN_DB,
        device=args.device,
    )
    width = found["method"]["channels"]
    print(f"{found['method']['takes']} takes, {width} channels")
    for row in found["readings"]:
        print(
            f"  {row['setting']:<24} output {row['output']:<6}"
            + "".join(f"{v:+8.1f}" for v in row["rise_db"])
            + "  dB over the second before the note"
        )
    print()
    for name, channels in found["output_pairs"].items():
        print(
            f"  output pair {name} -> "
            + ("channels " + ", ".join(str(c) for c in channels) if channels else "NOT AGREED")
        )
    apart = found["how_it_separated"]
    if apart["worst_apart_db"] is not None:
        print(
            f"  the groups stood {apart['worst_apart_db']:.1f} to "
            f"{apart['best_apart_db']:.1f} dB apart, against a margin of "
            f"{found['method']['margin_db']:.1f}"
        )
    if refused := outputmap.verdicts(found):
        for line in refused:
            print(f"  REFUSED: {line}")
        report.write_json(args.out, found)
        return 1
    report.write_json(args.out, found)
    return 0
