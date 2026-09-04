"""What a MIDI input reaches, in two runs with the cable moved between them.

The only pair of commands here that has to be run in order, on either side of
something a person does. `port-send` transmits into the input under test and
reads nothing, because that input answers nothing; `port-read` reads through the
input that does answer and finds what the first one left. Neither is worth
anything alone, which is why the second refuses without the first one's record.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import archive, ports, roland
from ..midi import MidiLink
from . import options, report
from .session import Refused, verified_link


def register(sub) -> None:
    p = sub.add_parser(
        "port-send",
        help="send into a MIDI input that answers nothing, leaving the result in the unit",
    )
    p.add_argument(
        "--input",
        required=True,
        help="which of the unit's MIDI inputs the cable is in, e.g. 'MIDI IN B'. Stated by "
        "whoever moved the cable and not measured: both inputs are reached over one cable from "
        "one host output, so nothing in the data can tell them apart",
    )
    p.add_argument(
        "--channels",
        type=int,
        default=16,
        help="how many channels the per-channel controller is sent on, from --channel upward",
    )
    options.add_channel(p)
    options.add_out(p)
    p.set_defaults(func=cmd_port_send)

    p = sub.add_parser(
        "port-read",
        help="find what a port-send left in the unit, read through the input that answers",
    )
    p.add_argument("--sent", required=True, help="the record port-send wrote")
    p.add_argument("--map", required=True, help="address map the regions to read are taken from")
    p.add_argument(
        "--baseline",
        required=True,
        help="the state to compare against, normally this unit's power-on capture. There is no "
        "snapshot at the head of this run because the input under test could not be read from, "
        "so everything that moved since the baseline is credited to the sending phase",
    )
    p.add_argument(
        "--input",
        required=True,
        help="which MIDI input the cable is in now, which must be the one that answers",
    )
    p.add_argument(
        "--no-reset",
        action="store_true",
        help="leave the unit holding what the sending phase wrote. The default sends a GS Reset "
        "once everything has been read, since the state was the message and is spent",
    )
    options.add_verify_reads(p)
    options.add_out(p)
    p.set_defaults(func=cmd_port_read)


def cmd_port_send(args: argparse.Namespace) -> int:
    sent = ports.catalogue(
        channels=args.channels, device_id=args.device_id, first_channel=args.channel
    )
    # Checked before the port is opened. A catalogue with a repeated value cannot
    # be read back afterwards at all, and finding that out after the cable has
    # been moved costs the run and the move both.
    repeated = ports.repeated_values(sent)
    if repeated:
        raise Refused(
            "two stimuli would be sent at one value and neither could be credited afterwards: "
            + "; ".join(f"{v:02X} by {' and '.join(labels)}" for v, labels in repeated.items())
        )

    with MidiLink(args.port) as link:
        print(f"MIDI: {link.ports.output_name}  ->  {args.input} (asserted)")
        answered = ports.answers_sysex(link, device_id=args.device_id)
        print(f"\n  identity request: {answered['identity_request']}")
        print(f"  data request:     {answered['data_request']}")
        if answered["answered"]:
            print(
                "\n  This input answers requests, so it can be measured by alias-scan directly "
                "and does not need the cable move this pair of commands is for."
            )
        print(f"\nsending {len(sent)} stimuli, reading nothing back")
        ports.send(link, sent)

    print("  done. Nothing was restored -- what was written is the measurement.")
    print(f"\nMove the cable to the input that answers, then run port-read --sent <{args.out}>")

    report.write_json(
        args.out,
        {
            "device_id": f"{args.device_id:02X}",
            "phase": "send",
            "input": args.input,
            "input_is_asserted_not_measured": ports.INPUT_IS_ASSERTED,
            "answers_sysex": answered,
            "first_channel": args.channel + 1,
            "channels": args.channels,
            "method": ports.METHOD,
            "values_are_unique": ports.WHY_VALUES_UNIQUE,
            "nothing_was_restored": ports.WHY_NOT_RESTORED,
            "sent": [s.to_json() for s in sent],
        },
    )
    return 0


def _sent_from(path: str) -> list[ports.Sent]:
    """Rebuild the sending phase's catalogue from its record.

    Read back rather than regenerated from the same arguments: a catalogue built
    twice is two claims that happen to agree, and an edit to the builder between
    the halves would be invisible. What was sent is what the record says was sent.
    """
    data = json.loads(Path(path).read_text())
    if data.get("phase") != "send":
        raise Refused(f"{path} is not a port-send record")
    return [
        ports.Sent(
            label=s["stimulus"],
            kind=s["kind"],
            value=int(s["value"], 16),
            messages=[[int(b, 16) for b in m.split()] for m in s["messages"]],
        )
        for s in data["sent"]
    ]


def cmd_port_read(args: argparse.Namespace) -> int:
    from ..aliases import Snapshotter

    sent = _sent_from(args.sent)
    sending = json.loads(Path(args.sent).read_text())
    baseline_record = json.loads(Path(args.baseline).read_text())
    baseline = {a: int(v, 16) for a, v in baseline_record["values"].items()}
    regions = archive.regions(args.map)

    with verified_link(args, refusing="reading back", show_port=True) as link:
        print(f"reading through {args.input} (asserted)")
        shot = Snapshotter(link, regions, device_id=args.device_id)
        print(f"\n{len(regions)} regions, one snapshot")
        raw = shot.take()
        after = {f"{a[0]:02X} {a[1]:02X} {a[2]:02X}": v for a, v in raw.items()}
        print(f"  {len(after)} bytes read, {shot.unread} regions unanswered")

        landed, unexplained, unknown = ports.locate(sent, baseline, after)
        put_back = "left as the sending phase wrote it"
        if not args.no_reset:
            link.send(roland.gs_reset(device_id=args.device_id))
            put_back = "GS Reset"
        print(f"  unit: {put_back}")

    print()
    print(ports.summarise(landed, unexplained, unknown))

    report.write_json(
        args.out,
        {
            "device_id": f"{args.device_id:02X}",
            "phase": "read",
            "sent_through": sending["input"],
            "read_through": args.input,
            "input_is_asserted_not_measured": ports.INPUT_IS_ASSERTED,
            "sending_phase_answered_sysex": sending["answers_sysex"],
            "method": ports.METHOD,
            "baseline": {
                "from": Path(args.baseline).name,
                "captured": baseline_record.get("captured", ""),
                "bytes": len(baseline),
                "why": ports.WHY_BASELINE,
            },
            "regions_read": len(regions) - shot.unread,
            "regions_unread": shot.unread,
            "bytes_read": len(after),
            "blocks_reached": ports.blocks_reached(landed),
            "landed": {label: addresses for label, addresses in landed.items()},
            "landed_nowhere": {
                "stimuli": [label for label, addresses in landed.items() if not addresses],
                "regions_unread": shot.unread,
                "why": ports.WHY_LANDED_NOWHERE,
            },
            "moved_holding_no_value_this_run_sent": {
                "bytes": {a: [f"{was:02X}", f"{now:02X}"] for a, [was, now] in unexplained.items()},
                "why": ports.WHY_UNEXPLAINED,
            },
            "read_without_a_baseline": unknown,
            "nothing_was_restored": ports.WHY_NOT_RESTORED,
            "unit_afterwards": put_back,
        },
    )
    return 0
