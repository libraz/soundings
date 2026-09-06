"""What an address holds, and whether the value it hands back is its own.

Three commands that write to the unit and put back what they found. Between them
they separate questions a read alone cannot: which values an address accepts,
whether neighbouring addresses hold values independently of each other, and
whether an address answers for itself or shows whatever was addressed last.

Everything here restores as it goes, and says in its record whether it managed
to. A run that broke the state and could not put it back is a run whose later
answers are about a machine nobody else can reproduce.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import archive
from . import options, report
from .session import prepared as prepare_state
from .session import verified_link


def register(sub) -> None:
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
        "--prepare",
        type=options.write_spec,
        action="append",
        default=[],
        metavar="ADDR=BYTES",
        help="state to put the unit in first, written once and read back to prove it took: "
        "'40 03 00=04 02' loads an insertion effect. What an address accepts can depend on "
        "it -- a block of effect parameters probed with no effect loaded reports the ranges "
        "of whatever was there instead, and a plan built from that asks the wrong pair. "
        "Recorded with the result, since the ranges only hold in the state they were "
        "measured in",
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
                **(
                    {
                        "prepared": [
                            {"address": a, "bytes": " ".join(f"{v:02X}" for v in vs)}
                            for a, vs in args.prepare
                        ],
                        "prepared_caveat": (
                            "Every range here was measured with the unit in this state and "
                            "holds in it. An address whose accepted values depend on what "
                            "is loaded reports a different range under a different load, so "
                            "a plan built from this record asks about this state and no other."
                        ),
                    }
                    if args.prepare
                    else {}
                ),
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
        if args.prepare and not prepare_state(
            link, args.prepare, device_id=args.device_id, settle=args.settle
        ):
            # Refused rather than noted: every range this run would report is a
            # range of the state it was actually in, and a probe of the wrong
            # state is indistinguishable afterwards from a probe of the right one.
            return 1
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
