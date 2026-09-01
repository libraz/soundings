"""Find where a message is stored, by diffing the address space around it.

The one command that both sends and writes, and so the one that has to put the
unit back. What it establishes is where a value lands, which is what makes two
different messages aliases of each other.
"""

from __future__ import annotations

import argparse
import time

from .. import archive, parts
from . import options, report
from .session import verified_link


def register(sub) -> None:
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
    options.add_channel(p)
    p.add_argument(
        "--kind",
        default="cc",
        choices=("cc", "nrpn", "drum-nrpn", "rpn", "channel", "address"),
        help="what to send; two kinds landing on one address is what makes them aliases",
    )
    p.add_argument(
        "--addresses",
        nargs="*",
        metavar="ADDR",
        help="for --kind address: bytes to write, default being the ones a control change "
        "and an NRPN were both measured to reach",
    )
    p.add_argument(
        "--values",
        type=options.pair,
        default=(0x20, 0x60),
        help="for --kind address: the two values written; both must be inside what the "
        "address accepts, or a clamp answers them identically and reads as storing nothing",
    )
    p.add_argument(
        "--note",
        type=options.number,
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
    options.add_verify_reads(p)
    options.add_out(p)
    p.set_defaults(func=cmd_alias_scan)


def _addresses(args: argparse.Namespace) -> list[tuple[int, int, int]]:
    from ..writeback import NEVER_WRITE

    given = args.addresses or list(archive.ALIASED_BYTES)
    out = []
    for spec in given:
        address = tuple(int(b, 16) for b in spec.split())
        if len(address) != 3:
            raise SystemExit(f"an address is three hex bytes, not {spec!r}")
        if address in NEVER_WRITE:
            raise SystemExit(f"{spec} is on the never-write list")
        out.append(address)
    return out


def _stimuli(args: argparse.Namespace) -> list:
    """Build the list of things to send, for the kind asked for."""
    from .. import aliases as al

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
    if args.kind == "address":
        return [
            al.address_write(a, device_id=args.device_id, values=args.values)
            for a in _addresses(args)
        ]
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
    from ..aliases import (
        METHOD,
        NOT_SCANNED,
        NOTE,
        RPN_PARKED,
        WHY_CONTROL,
        WHY_KIND_REACHED,
        WHY_RECOVERED,
        WHY_RESIDUE,
        Scanner,
        Snapshotter,
        cc,
        control_change,
        control_run,
        summarise,
    )

    regions = archive.regions(args.map, args.prefix)
    # A run that finds nothing cannot say whether the unit stores nothing or the
    # scan was broken, so one control change known to be stored is always sent.
    # Sent at both ends rather than once: a control that passes at the start says
    # the reading worked at the start, and a run has been seen to miss something
    # after that point.
    control = cc(args.channel, args.control_cc)
    with verified_link(args, refusing="scanning", show_port=True) as link:
        # Park the RPN and NRPN selectors before anything else, so a data entry
        # controller later in the scan writes to a known nothing rather than to
        # whatever parameter was last selected on this channel.
        for number, value in ((101, 127), (100, 127), (99, 127), (98, 127)):
            link.send(control_change(args.channel, number, value))

        # Read every byte the scan will write before anything is written, and
        # drop from the run any that would not answer: a byte with no original
        # is one there would be nothing to put back for.
        restorer = parts.Restorer(link, device_id=args.device_id)
        originals = {}
        if args.kind == "address":
            wanted = _addresses(args)
            originals = restorer.remember(wanted)
            args.addresses = [f"{a[0]:02X} {a[1]:02X} {a[2]:02X}" for a in originals]
            print(
                f"\n{len(originals)} of {len(wanted)} target bytes read, so writable and restorable"
            )
        stimuli = [control, *_stimuli(args), control]

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

        restored = restorer.put_back(originals) if originals else "nothing written"
        if originals:
            print(f"  restore: {restored}")
        latch = restorer.clear_bank_latch(args.channel)
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

    report.write_json(
        args.out,
        {
            "device_id": f"{args.device_id:02X}",
            "channel": args.channel + 1,
            "kind": args.kind,
            "positive_control": {
                "controller": args.control_cc,
                "sent": 2,
                "detected": control_passes,
                "at": control_hit.addresses if control_hit else [],
                "why": WHY_CONTROL,
            },
            "left_changed_afterwards": {
                "by_stimulus": scanner.residue,
                "why": WHY_RESIDUE,
            },
            "missed_by_the_first_pass": {
                "stimuli": scanner.recovered,
                "why": WHY_RECOVERED,
            },
            "kind_reached": {
                "value": reached,
                "why": WHY_KIND_REACHED,
            },
            "method": METHOD,
            "region_prefix": args.prefix,
            "regions_watched": len(regions),
            "stimuli_sent": [s.label for s in stimuli],
            "restless_addresses": sorted(restless),
            "region_reads_failed": shot.unread,
            "not_scanned": NOT_SCANNED,
            "rpn_parked": RPN_PARKED,
            "bank_latch_on_exit": latch,
            "written_bytes_restored": restored,
            "note": NOTE,
            "attributed": [a.to_json() for a in found],
        },
    )
    return 0 if bracketed and reached else 1
