"""The state the unit comes up in, and what each reset puts back of it.

The two commands here are one measurement in two halves that cannot run
together. The first is available only once per power cycle -- reading the whole
space before anything has written to it -- and the second is meaningless without
it, because a reset can only be credited against a state somebody recorded while
it was still untouched.

So the capture is a run of its own, kept as a file, and the reset probe is scored
against that file rather than against whatever the unit happens to hold.
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
    p = sub.add_parser(
        "power-on",
        help="capture the state the unit powers up in, which is available only until "
        "something writes to it",
    )
    p.add_argument(
        "--map",
        required=True,
        help="address map the regions are read from. Required rather than defaulted: a "
        "capture read through another unit's map is a baseline for a space this unit was "
        "never asked about",
    )
    p.add_argument(
        "--unit-id",
        required=True,
        help="the unit this is a capture of, as its directory names it",
    )
    p.add_argument(
        "--captured",
        default="immediately after a power cycle, before anything was sent to the unit",
        help="what was done before this run, in the words of whoever did it. The run cannot "
        "check it and records it as a claim",
    )
    options.add_verify_reads(p)
    options.add_out(p)
    p.set_defaults(func=cmd_power_on)

    p = sub.add_parser(
        "reset-probe",
        help="find what each reset restores, by breaking the state first",
    )
    p.add_argument(
        "--baseline",
        required=True,
        help="the power-on capture every reset is compared against",
    )
    p.add_argument("--map", required=True, help="address map read after each reset")
    p.add_argument(
        "--write-probe",
        required=True,
        help="where the addresses that take any value are read from. A partial probe leaves "
        "the marks partial, and a reset is then credited only across what was broken",
    )
    p.add_argument(
        "--skip-prefix",
        nargs="*",
        default=[],
        metavar="PREFIX",
        help="do not mark addresses starting here; for blocks measured to hold nothing, whose "
        "marks land in the store they are a window onto and say only what it did",
    )
    p.add_argument(
        "--mark-prefix",
        nargs="*",
        default=[],
        metavar="PREFIX",
        help="mark only addresses starting here. Everything outside is left at whatever it "
        "held, so the run says nothing about it -- which is the bound to state when a subject "
        "is aimed at part of the space rather than at all of it",
    )
    p.add_argument(
        "--canary",
        default="40 01 30",
        help="an address known to answer, asked while the marks go in. A unit that stops "
        "answering reads as one refusing every remaining mark, and the subject then scores "
        "for restoring a space the run never broke",
    )
    p.add_argument(
        "--subjects",
        default="resets",
        choices=("resets", "channel-mode"),
        help="what to break the state with. 'channel-mode' sends controllers 120, 121 and 123, "
        "which act rather than set and so cannot be measured by an alias scan's out-and-back",
    )
    options.add_channel(p)
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


def cmd_power_on(args: argparse.Namespace) -> int:
    """Read the whole map twice, writing nothing, and keep what both reads agreed on."""
    from .. import roland
    from ..aliases import Snapshotter
    from ..resets import power_on_record

    regions = archive.regions(args.map)
    print(f"{len(regions)} regions from {args.map}, read twice with nothing written")

    with verified_link(
        args,
        refusing="capturing",
        show_port=True,
        announce="Reading only. A capture taken after anything writes is not a power-on state",
    ) as link:
        identity = link.exchange(roland.IDENTITY_REQUEST, timeout=1.0)
        shot = Snapshotter(link, regions, device_id=args.device_id)
        first = shot.take()
        print(f"  first read: {len(first)} bytes, {shot.unread} regions unread")
        unread_after_first = shot.unread
        second = shot.take()
        print(f"  second read: {len(second)} bytes, {shot.unread - unread_after_first} unread")

    record = power_on_record(
        first,
        second,
        unit_id=args.unit_id,
        identity_reply=" ".join(f"{b:02X}" for b in identity) if identity else "no reply",
        captured=args.captured,
        regions_read=shot.reads,
        regions_unread=shot.unread,
        # Deduplicated: the map is read twice, so a region answered short is
        # answered short twice, and a list saying so twice reads as two regions.
        answered_short=sorted(set(shot.short)),
    )
    print()
    if record["regions_answered_short"]:
        print(f"  {len(record['regions_answered_short'])} regions answered short and were refused")
    print(f"  {len(record['values'])} bytes both reads agreed on")
    print(f"  {len(record['read_disagreed_at'])} disagreed and were left out")
    print(f"  {len(record['read_only_once'])} were answered by one read only")

    report.write_json(args.out, record)
    return 0 if not record["read_disagreed_at"] and not shot.unread else 1


def cmd_reset_probe(args: argparse.Namespace) -> int:
    from ..aliases import Snapshotter
    from ..resets import (
        METHOD,
        WHY_CHANNEL_MODE_MARKED,
        WHY_PRECEDED,
        WHY_STOPPED,
        Prober,
        ResetResult,
        UnitWentQuiet,
        catalogue,
        channel_mode_catalogue,
        compare,
        left_unmarked,
        mode_set,
        named,
        outcomes_agree,
        summarise,
    )

    captured = json.loads(Path(args.baseline).read_text())
    baseline = {
        tuple(int(b, 16) for b in a.split()): int(v, 16) for a, v in captured["values"].items()
    }
    regions = archive.regions(args.map)
    # Every address a mark can be put in comes from the write probe. A list of
    # aliased bytes used to be added here as well, from when the probe covered
    # one block: none of those thirteen were in that probe's accepting set and
    # all of them are in a whole-map one, so the addition was covering for the
    # probe rather than adding to it.
    wanted = archive.accepting_bytes(args.write_probe)
    if args.mark_prefix:
        kept, skipped = archive.keep_only_prefixes(wanted, args.mark_prefix)
    else:
        kept, skipped = archive.split_off_prefixes(wanted, args.skip_prefix)
    # Regions overlap, so an address is offered more than once. Marking it twice
    # measures nothing further, and counting it twice makes the tally of what was
    # marked fall short of the target list by the number of repeats -- which reads
    # exactly like that many addresses having refused.
    targets = list(dict.fromkeys(tuple(int(b, 16) for b in a.split()) for a in kept))
    if args.subjects == "channel-mode":
        resets = channel_mode_catalogue(args.channel)
    else:
        resets = catalogue(args.device_id)
    if args.include_mode_set:
        resets.append(mode_set(args.device_id))

    print(f"baseline: {args.baseline}, {len(baseline)} bytes")
    print(f"{len(targets)} addresses to mark, {len(regions)} regions read after each reset")
    if skipped:
        where = (
            f"outside {', '.join(args.mark_prefix)}"
            if args.mark_prefix
            else f"under {', '.join(args.skip_prefix)}"
        )
        print(f"{len(skipped)} left unmarked {where}")

    results = []
    stopped: str | None = None
    canary = tuple(int(b, 16) for b in args.canary.split())
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
            try:
                marked, refused = prober.mark(
                    targets, progress=lambda m: print(f"  {m}"), canary=canary
                )
            except UnitWentQuiet as exc:
                # Nothing after this point would be a measurement: the subject
                # would be credited for every byte the run stopped being able to
                # break, and would score better the worse the failure was.
                stopped = f"{reset.label}: {exc}"
                print(f"\nSTOPPED: {exc}")
                break
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
        # A channel mode message is not a state to leave the unit in: it is aimed
        # at one part and the run has just broken the rest of the space with
        # marks. So the subject that ranked best is reported and the opener is
        # what actually goes out. A run that stopped is in the same position for
        # a different reason -- it broke a space and measured nothing.
        best = (
            min(results, key=lambda r: (len(r.differs_from_power_on), -len(r.restored)))
            if results and args.subjects == "resets" and not stopped
            else None
        )
        left_on = best.label if best else opener.label
        link.send(
            next(x for x in resets if x.label == best.label).message if best else opener.message
        )
        time.sleep(args.settle)
        print(f"\nleft the unit on {left_on}")

    print()
    print(summarise(results))

    record = {
        "device_id": f"{args.device_id:02X}",
        "baseline": args.baseline,
        "method": METHOD,
        "each_preceded_by": args.from_reset,
        "why_preceded": WHY_PRECEDED,
        "write_probe": args.write_probe,
        "subjects": args.subjects,
        "left_unmarked": left_unmarked(
            confined_to=args.mark_prefix,
            skipped=args.skip_prefix,
            addresses=len(skipped),
        ),
        **(
            {
                "channel": args.channel + 1,
                "marked_by_sysex_not_by_controller": {"why": WHY_CHANNEL_MODE_MARKED},
            }
            if args.subjects == "channel-mode"
            else {}
        ),
        "order": [r.label for r in results],
        "stopped": {"at": stopped, "why": WHY_STOPPED},
        "left_on": left_on,
        "resets": [r.to_json() for r in results],
    }
    # The subjects' outcomes are held against each other before the record is
    # written, so a run whose subjects all answered identically says so in the
    # file rather than leaving a reader to notice three equal counts.
    agreement = outcomes_agree(record)
    if agreement["every_subject_sorted_every_byte_the_same_way"]:
        print(
            f"\n  all {len(results)} subjects sorted every marked byte the same way, so nothing "
            "here is attributable to any one of them"
        )
    report.write_json(args.out, record)
    return 1 if stopped else 0
