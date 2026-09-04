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
        nargs="*",
        metavar="ADDR:COUNT",
        help="'40 01 30:24' probes 24 consecutive bytes from 40 01 30",
    )
    p.add_argument(
        "--map",
        help="probe every region an address map found, instead of naming them. The map is "
        "what the unit answered when it was asked, so this covers what exists rather than "
        "what a manual lists",
    )
    p.add_argument(
        "--prefix",
        nargs="*",
        default=[],
        help="for --map: keep only regions whose address starts with one of these, in the "
        "order given. A run long enough to be interrupted should meet the blocks that "
        "differ from each other before the ones that repeat",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="skip regions already in --out and add to it. A region is written as soon as "
        "it is finished, so an interrupted run resumes from the last whole region rather "
        "than from the start",
    )
    p.add_argument(
        "--canary",
        default="40 01 30",
        help="an address known to answer, asked between regions. A unit that stops talking "
        "reads as a map of unreadable bytes, which is skipped rather than written to, so "
        "without this the probe finishes cleanly having measured nothing",
    )
    p.add_argument(
        "--verify",
        help="read every address a saved run touched, compare it with what that run "
        "recorded it holding beforehand, and write the outcome back into the file. Writes "
        "nothing to the unit. A run restores as it goes and checks each region as it "
        "finishes, but only this sees the whole space at once and after the fact, which is "
        "what catches a byte left behind by a run that did not end on its own terms",
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
        "window-probe",
        help="tell an address that holds a value from one that shows another address's, "
        "which writing to it and reading it back cannot",
    )
    p.add_argument(
        "--stores",
        nargs=2,
        required=True,
        metavar="ADDR",
        help="two addresses believed to hold their own values. Everything is measured "
        "relative to these, and they are asked as candidates too so a run that calls "
        "everything a window says so rather than reporting one",
    )
    p.add_argument(
        "candidates",
        nargs="+",
        metavar="ADDR",
        help="addresses to ask. Each is read after addressing each store in turn, and "
        "then written through to see which store the write reaches",
    )
    p.add_argument(
        "--settle",
        type=float,
        default=0.05,
        help="pause after a write before the read that follows it",
    )
    options.add_verify_reads(p)
    options.add_out(p)
    p.set_defaults(func=cmd_window_probe)

    p = sub.add_parser(
        "hold-probe",
        help="find whether neighbouring addresses hold their own values, by giving them "
        "different ones before reading any of them back",
    )
    p.add_argument(
        "regions",
        nargs="*",
        metavar="ADDR:COUNT",
        help="'20 00 00:64' asks 64 consecutive addresses from 20 00 00",
    )
    p.add_argument("--map", help="ask every region an address map found, instead of naming them")
    p.add_argument(
        "--prefix",
        nargs="*",
        default=[],
        help="for --map: keep only regions whose address starts with one of these, in order",
    )
    p.add_argument(
        "--settle",
        type=float,
        default=0.02,
        help="pause after each write. Nothing is read until every write is done, so this is "
        "not a settle before a read back",
    )
    options.add_verify_reads(p)
    options.add_out(p)
    p.set_defaults(func=cmd_hold_probe)

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


def _probe_regions(args: argparse.Namespace) -> list[tuple[tuple[int, ...], int]]:
    """The regions to probe, named on the command line or taken from a map.

    A prefix list orders as well as filters, because which regions get measured
    first is the whole of what an interrupted run leaves behind.
    """
    named = [
        (tuple(int(b, 16) for b in spec.split(":")[0].split()), int(spec.split(":")[1], 0))
        for spec in args.regions
    ]
    if not args.map:
        if not named:
            raise SystemExit("name at least one ADDR:COUNT, or pass --map")
        return named
    if args.prefix:
        return named + [r for p in args.prefix for r in archive.regions(args.map, p)]
    return named + archive.regions(args.map)


def _still_to_do(
    regions: list[tuple[tuple[int, ...], int]], kept: list[dict]
) -> list[tuple[tuple[int, ...], int]]:
    """The regions a resumed run has left, matched the way the file writes them.

    Start and length together, not start alone: the same address probed for a
    different number of bytes is a different measurement, and taking it as done
    would leave the tail of the longer one silently unprobed.

    Only a region that read back whole counts as done. The region a run stopped
    in is written out for its evidence, but it stopped part way through, and
    treating the entry as coverage would leave the rest of it unprobed for good.
    """
    seen = {(r["start"], r["length"]) for r in kept if r["region_restored"]}
    return [(s, n) for s, n in regions if (" ".join(f"{b:02X}" for b in s), n) not in seen]


def _superseded(kept: list[dict], regions: list[tuple[tuple[int, ...], int]]) -> list[dict]:
    """What to carry forward, with any region about to be measured again dropped.

    A region is re-probed whole, so keeping the partial entry beside the new one
    would put the same address in the file twice with two different answers.
    """
    redo = {(" ".join(f"{b:02X}" for b in s), n) for s, n in regions}
    return [r for r in kept if (r["start"], r["length"]) not in redo]


def cmd_verify_saved_probe(args: argparse.Namespace) -> int:
    """Read back every address a saved run touched, and say whether it is as it was."""
    from ..writeback import READ_BACK_AFTERWARDS, Writer

    payload = json.loads(Path(args.verify).read_text())
    was = {b["address"]: int(b["original"], 16) for r in payload["regions"] for b in r["bytes"]}
    print(f"{len(was)} addresses to read back, writing nothing")

    def ask(writer, wanted: dict[str, int]) -> tuple[list, list]:
        differ, silent = [], []
        for i, (addr, before) in enumerate(sorted(wanted.items())):
            got = writer.read_byte(tuple(int(b, 16) for b in addr.split()))
            if got is None:
                silent.append(addr)
            elif got != before:
                differ.append({"address": addr, "was": f"{before:02X}", "holds": f"{got:02X}"})
            if i and i % 5000 == 0:
                print(f"  {i}/{len(wanted)}")
        return differ, silent

    with verified_link(args, refusing="reading back", show_port=True) as link:
        writer = Writer(link, device_id=args.device_id, settle=args.settle)
        differ, unanswered = ask(writer, was)
        # Asked again at the end rather than on the spot: a failure here comes in
        # unbroken runs, so an immediate retry asks inside the same stall.
        suspect = {d["address"]: was[d["address"]] for d in differ}
        suspect.update({a: was[a] for a in unanswered})
        asked_again = len(suspect)
        if suspect:
            print(f"\n  asking {asked_again} again")
            differ, unanswered = ask(writer, suspect)

    payload["read_back_afterwards"] = {
        "method": READ_BACK_AFTERWARDS,
        "addresses": len(was),
        "asked_again": asked_again,
        "differ_from_what_they_held_before": differ,
        "would_not_answer": unanswered,
    }
    Path(args.verify).write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\n{len(was)} addresses read back, {asked_again} of them asked a second time")
    print(f"  differ from what they held before the probe: {len(differ)}")
    for d in differ[:20]:
        print(f"    {d['address']}: was {d['was']}, holds {d['holds']}")
    print(f"  would not answer: {len(unanswered)}")
    print(f"\nwrote {args.verify}")
    return 0 if not differ and not unanswered else 1


def cmd_write_probe(args: argparse.Namespace) -> int:
    from ..writeback import NOTE, SINGLE_BYTE_LIMIT, WENT_DEAF, RestoreFailed, Writer, summarise

    if args.verify:
        return cmd_verify_saved_probe(args)

    regions = _probe_regions(args)
    done: list = []
    kept: list[dict] = []
    if args.resume and args.out and Path(args.out).exists():
        kept = json.loads(Path(args.out).read_text())["regions"]
        before = len(regions)
        regions = _still_to_do(regions, kept)
        kept = _superseded(kept, regions)
        print(f"resuming: {len(kept)} regions already measured, {before - len(regions)} skipped")

    canary = tuple(int(b, 16) for b in args.canary.split())
    stopped: str | None = None
    print(f"{len(regions)} regions, {sum(n for _, n in regions)} bytes to probe")

    def write_out() -> None:
        report.write_json(
            args.out,
            {
                "device_id": f"{args.device_id:02X}",
                "settle_s": args.settle,
                "note": NOTE,
                "single_byte_limit": SINGLE_BYTE_LIMIT,
                "complete": stopped is None and not regions,
                "stopped": stopped,
                "regions": kept + [r.to_json() for r in done],
            },
        )

    with verified_link(
        args,
        refusing="writing",
        show_port=True,
        announce="Verifying the path before writing (a write is never acknowledged)",
    ) as link:
        writer = Writer(link, device_id=args.device_id, settle=args.settle)
        remaining = list(regions)
        try:
            for start, length in regions:
                done.append(writer.probe_region(start, length, progress=lambda m: print(f"  {m}")))
                remaining.pop(0)
                # Written per region rather than at the end: a run this long is
                # interrupted by things that do not come back to close a file.
                write_out()
                if not writer.answering(canary):
                    stopped = WENT_DEAF
                    print(f"\nSTOPPED: {args.canary} stopped answering")
                    break
        except RestoreFailed as exc:
            # The region it happened in is kept: the bytes before the failure
            # were measured and restored like any others, and they are the only
            # record of what led up to it.
            if exc.region is not None:
                done.append(exc.region)
            stopped = str(exc)
            print(f"\nSTOPPED: {exc}")
        finally:
            regions = remaining

    print()
    print(summarise(done))
    print(f"  {writer.writes} writes, {writer.reads} reads")
    write_out()
    return 0 if stopped is None and all(r.region_restored for r in done) else 1


def _address(spec: str) -> tuple[int, int, int]:
    parts = spec.split()
    if len(parts) != 3:
        raise SystemExit(f"an address is three hex bytes, not {spec!r}")
    return tuple(int(b, 16) for b in parts)


def cmd_window_probe(args: argparse.Namespace) -> int:
    """Ask each address whether it answers for itself or for whatever was addressed last."""
    from ..window import Prober, summarise
    from ..writeback import NEVER_WRITE

    stores = tuple(_address(s) for s in args.stores)
    candidates = [_address(s) for s in args.candidates]
    for address in list(stores) + candidates:
        if address in NEVER_WRITE:
            raise SystemExit(f"{_address} is on the never-write list")

    with verified_link(
        args,
        refusing="probing",
        show_port=True,
        announce="Verifying the path before writing (a write is never acknowledged)",
    ) as link:
        prober = Prober(link, device_id=args.device_id, settle=args.settle)
        result = prober.run(stores, candidates, progress=lambda m: print(f"  {m}"))

    print()
    print(summarise(result))
    report.write_json(args.out, {"device_id": f"{args.device_id:02X}", **result.to_json()})
    return 0 if result.restored and result.controls_held_their_own else 1


def cmd_hold_probe(args: argparse.Namespace) -> int:
    """Ask each run of addresses whether its members hold values of their own."""
    from ..window import Prober, hold_probe, hold_record

    regions = _probe_regions(args)
    print(f"{len(regions)} regions, {sum(n for _, n in regions)} addresses")

    done = []
    with verified_link(
        args,
        refusing="probing",
        show_port=True,
        announce="Verifying the path before writing (a write is never acknowledged)",
    ) as link:
        prober = Prober(link, device_id=args.device_id, settle=args.settle)
        for start, length in regions:
            done.append(hold_probe(prober, start, length, progress=lambda m: print(f"  {m}")))

    record = hold_record(done)
    print()
    for verdict, n in sorted(record["verdicts"].items(), key=lambda item: -item[1]):
        print(f"  {n:4d} regions: {verdict}")
    unrestored = [h.start for h in done if not h.restored]
    print(
        f"  !! not put back: {', '.join(unrestored)}" if unrestored else "  every region put back"
    )

    report.write_json(args.out, {"device_id": f"{args.device_id:02X}", **record})
    return 0 if not unrestored else 1


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
    )
    print()
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
