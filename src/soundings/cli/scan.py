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
        required=True,
        help="address map the watched regions are taken from",
    )
    p.add_argument(
        "--prefix",
        default="",
        help="only watch regions whose address starts here. Every negative finding a run "
        "produces is bounded by what it watched, so a prefix turns 'this message is stored "
        "nowhere' into 'nowhere in the part of the space that was looked at'. The default "
        "watches all of it",
    )
    options.add_channel(p)
    p.add_argument(
        "--kind",
        default="cc",
        choices=("cc", "nrpn", "drum-nrpn", "rpn", "channel", "address", "universal", "mode"),
        help="what to send; two kinds landing on one address is what makes them aliases",
    )
    p.add_argument(
        "--addresses",
        nargs="*",
        metavar="ADDR",
        help="for --kind address: bytes to write",
    )
    p.add_argument(
        "--addresses-from",
        nargs="*",
        default=[],
        metavar="SCAN",
        help="for --kind address: take the bytes to write from what these alias scans "
        "attributed. Which addresses are worth asking a third way into is a finding about "
        "the unit, so it is read from that unit's own records rather than assumed",
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
        help="drum note addressed one note at a time, for --kind drum-nrpn and for the "
        "key-based controls of --kind universal",
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

    # Named addresses win outright rather than adding to the ones read from a
    # record. The run narrows this list to what it could read and put back, and
    # writes the remainder here; adding the record's addresses every time would
    # put the dropped ones back in after they had been dropped.
    given = list(args.addresses) if args.addresses else archive.stores_reached(args.addresses_from)
    if not given:
        raise SystemExit(
            "--kind address needs addresses: --addresses, or --addresses-from pointed at "
            "this unit's alias scans. There is no default, because which addresses are "
            "worth writing to is a finding about the unit rather than about the method"
        )
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
    if args.kind == "universal":
        return al.universal_stimuli(ch, args.note)
    if args.kind == "mode":
        return al.mode_stimuli(ch)
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


def _kind_reached(kind: str, stimuli: list, found: list) -> bool:
    """Whether a message of the family under test was shown to arrive at all.

    The positive control proves the snapshots and the diff work, but it is a
    control change and so cannot show that anything else got through. Without
    this, a family that lands nothing reads as absent when it may never have
    been delivered.

    Asked of the stimuli the run actually sent, not of the name they were
    selected by. `--kind drum-nrpn` sends NRPNs, which is what they are and what
    they are recorded as, so comparing the selector with a stimulus's own kind
    compared two different things and could never agree: a run that landed 4
    stimuli across 13 blocks announced that nothing of its kind had been
    attributed anywhere, and wrote that into its record. An attribution is
    itself proof of arrival, so a run that finds something can never report
    this as unreached.
    """
    if kind == "cc":
        return True
    sent = {s.label for s in stimuli}
    return any(a.label in sent for a in found)


def cmd_alias_scan(args: argparse.Namespace) -> int:
    from ..aliases import (
        METHOD,
        MODE_NORMAL,
        MODE_PAIRED,
        MODE_RESTORED,
        NOTE,
        RPN_PARKED,
        WHY_CONTROL,
        WHY_KIND_REACHED,
        WHY_LANDED_OUTSIDE,
        WHY_MODE_PAIRED,
        WHY_RECOVERED,
        WHY_RESIDUE,
        WHY_UNREAD,
        Scanner,
        Snapshotter,
        cc,
        control_change,
        control_run,
        landed_outside_its_own_block,
        not_scanned,
        summarise,
    )

    regions = archive.regions(args.map, args.prefix)
    # Resolved before the port is opened, so a run with nowhere to write refuses
    # without having occupied the unit to find that out.
    targets = _addresses(args) if args.kind == "address" else []
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
            wanted = targets
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
        unread_by_stimulus: dict[str, int] = {}
        for stimulus in stimuli:
            before = shot.unread
            hit = scanner.attribute(stimulus)
            # Whose null the unanswered reads weaken. A cumulative total says
            # only that the run had some, which leaves every negative in it
            # equally suspect and none of them accountable.
            missed = shot.unread - before
            if missed:
                unread_by_stimulus[stimulus.label] = (
                    unread_by_stimulus.get(stimulus.label, 0) + missed
                )
            if stimulus is control:
                control_passes += hit is not None
                control_hit = control_hit or hit
                print(f"  control CC{args.control_cc}: {'seen' if hit else 'NOT SEEN'}")
                continue
            if hit is not None:
                found.append(hit)
                print(f"  {stimulus.label}: {', '.join(hit.addresses)}")

        # After the trailing control rather than before it, so the control is
        # asked in whatever mode the last stimulus left. A mode this unit honours
        # would break the channel messages that follow it, and the control is the
        # one message in the run positioned to notice.
        if args.kind == "mode":
            for number, value in MODE_NORMAL:
                link.send(control_change(args.channel, number, value))
            print("  put back: omni on, poly, local control on")

        restored = restorer.put_back(originals) if originals else "nothing written"
        if originals:
            print(f"  restore: {restored}")
        latch = restorer.clear_bank_latch(args.channel)
        print(f"  bank latch: {latch}")

    reached = _kind_reached(args.kind, stimuli, found)
    outside = landed_outside_its_own_block(found)
    bracketed = control_passes == 2
    print()
    print(summarise(found, restless, shot.unread))
    if args.kind == "address":
        if outside:
            for label, addresses in outside.items():
                print(f"  {label} was also seen outside its block, at {', '.join(addresses)}")
            print("  so a write landing in another block is something this run can report")
        else:
            print(
                "  no write landed outside the block it was addressed to, so this run was never "
                "shown able to report one, and says nothing about mirroring"
            )
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
            **(
                {
                    "landed_outside_its_own_block": {
                        "by_stimulus": outside,
                        "why": WHY_LANDED_OUTSIDE,
                    }
                }
                if args.kind == "address"
                else {}
            ),
            **(
                {
                    "channel_mode_restored": {
                        "sent": [f"CC{n} {v}" for n, v in MODE_NORMAL],
                        "why": MODE_RESTORED,
                    },
                    "stimuli_naming_two_controllers": {
                        "stimuli": list(MODE_PAIRED),
                        "why": WHY_MODE_PAIRED,
                    },
                }
                if args.kind == "mode"
                else {}
            ),
            "method": METHOD,
            "region_prefix": args.prefix,
            "regions_watched": len(regions),
            # Every negative here is bounded by exactly this, and a count cannot
            # state that bound: three maps of one unit have been watched by these
            # scans and none of them contains the others, so which run looked
            # wider is not something a number of regions answers. Without the
            # addresses themselves, "this message is stored nowhere" cannot be
            # checked by a reader, and two runs of the same kind cannot be told
            # apart from two runs of the same question.
            "regions_watched_are": [
                {"address": " ".join(f"{b:02X}" for b in start), "size": size}
                for start, size in regions
            ],
            "stimuli_sent": [s.label for s in stimuli],
            "restless_addresses": sorted(restless),
            "region_reads_failed": shot.unread,
            "region_reads": {
                "attempted": shot.reads,
                "unanswered": shot.unread,
                "by_stimulus": unread_by_stimulus,
                "why": WHY_UNREAD,
            },
            "not_scanned": not_scanned(args.kind),
            "rpn_parked": RPN_PARKED,
            "bank_latch_on_exit": latch,
            "written_bytes_restored": restored,
            "note": NOTE,
            "attributed": [a.to_json() for a in found],
        },
    )
    return 0 if bracketed and reached else 1
