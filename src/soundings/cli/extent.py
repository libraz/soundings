"""Which addresses are there, and how far a region actually reaches.

Three commands that write nothing. Each asks the unit for a byte and takes an
answer as the address existing, so every result here is a negative with one
positive control behind it: an address known to answer, asked between questions,
because a unit that stopped talking answers nothing to all of them and that is
indistinguishable from a space that holds nothing.
"""

from __future__ import annotations

import argparse
import json
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
        "boundary",
        help="ask whether each mapped region ends where the map says it ends, by "
        "reading past it one byte at a time",
    )
    p.add_argument("map", help="an address map, whose regions are the ones asked about")
    p.add_argument(
        "--reach",
        type=int,
        default=16,
        help="addresses read past a region that answered past its end, before moving on. "
        "A region that answers all of them is reported as reaching at least this far "
        "rather than as ending here",
    )
    p.add_argument(
        "--canary",
        default=None,
        help="an address known to answer, asked between regions. Every result here is a "
        "negative, and a unit that stopped talking answers nothing to all of them. "
        "Defaults to the first region the map holds",
    )
    p.add_argument("--every", type=int, default=25, help="regions between canary questions")
    p.add_argument("--timeout", type=float, default=0.3)
    p.add_argument(
        "--resume",
        action="store_true",
        help="skip regions already in --out and add to it. A region is written as soon as "
        "it is asked, so an interrupted run carries on from the last whole one",
    )
    options.add_verify_reads(p)
    options.add_out(p)
    p.set_defaults(func=cmd_boundary)

    p = sub.add_parser(
        "offsets",
        help="find which addresses in a block answer a single-byte read, which neither a "
        "map bounded by block reads nor a read forward from a region's end shows",
    )
    p.add_argument(
        "blocks",
        nargs="+",
        metavar="ADDR:COUNT",
        help="'40 40 00:128' asks all 128 offsets of that block",
    )
    p.add_argument(
        "--canary",
        required=True,
        help="an address known to answer, asked between blocks. Most offsets answer nothing "
        "and a unit that stopped talking answers nothing to all of them",
    )
    p.add_argument("--every", type=int, default=1, help="blocks between canary questions")
    p.add_argument(
        "--resume",
        action="store_true",
        help="keep the blocks already in --out and ask only the rest. The block a stopped "
        "run ended on is asked again, since it is the one no canary answered for",
    )
    p.add_argument("--timeout", type=float, default=0.3)
    options.add_verify_reads(p)
    options.add_out(p)
    p.set_defaults(func=cmd_offsets)


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


def _single_byte_reader(link, args):
    """Ask one address for one byte, and hand back None for anything that is not its answer."""
    from .. import roland

    def answers(address: str) -> list[int] | None:
        reply = roland.parse_dt1(
            link.exchange(roland.rq1(address, 1, device_id=args.device_id), timeout=args.timeout)
        )
        if reply is None or list(reply.address) != roland.address_bytes(address):
            # A reply for another address is not an answer to this read. The
            # sweeps all check it; a read that did not once came back holding a
            # neighbour's byte.
            return None
        return list(reply.data)

    return answers


def cmd_offsets(args: argparse.Namespace) -> int:
    """Ask every offset of a block, since reading forward from a region's end misses a gap."""
    from .. import boundary

    blocks: list[boundary.Block] = []
    deaf = False
    wanted = args.blocks
    if args.resume and args.out and Path(args.out).exists():
        blocks = boundary.restore_blocks(json.loads(Path(args.out).read_text()))
        already = {(b.address, b.asked) for b in blocks}
        before = len(wanted)
        wanted = [
            spec
            for spec in wanted
            if (spec.partition(":")[0].strip(), int(spec.partition(":")[2])) not in already
        ]
        print(f"resuming: {len(blocks)} blocks already asked, {before - len(wanted)} skipped")

    with verified_link(args, refusing="reading a block's offsets", show_port=True) as link:
        answers = _single_byte_reader(link, args)
        for asked, spec in enumerate(wanted):
            start, _, count = spec.partition(":")
            block = boundary.Block(address=start.strip())
            packed = boundary.past_the_end(block.address, 0)
            for step in range(int(count)):
                here = boundary.unpack(packed + step)
                block.asked += 1
                data = answers(here)
                if data is not None:
                    block.answered[here] = " ".join(f"{b:02X}" for b in data)
            print(f"  {block.address}: {len(block.answered)} of {block.asked} offsets answered")
            blocks.append(block)
            report.write_json(args.out, boundary.scanned(blocks, args.canary, deaf))
            if asked % args.every == args.every - 1 and answers(args.canary) is None:
                deaf = True
                print(f"\nSTOPPED: {args.canary} stopped answering")
                break

    result = boundary.scanned(blocks, args.canary, deaf)
    print(f"\n{result['offsets_that_answered']} of {result['offsets_asked']} offsets answered")
    report.write_json(args.out, result)
    return 1 if deaf else 0


def cmd_boundary(args: argparse.Namespace) -> int:
    """Read one byte past every mapped region, and keep going where one answers."""
    from .. import boundary

    regions = archive.regions(args.map)
    canary = args.canary or " ".join(f"{b:02X}" for b in regions[0][0])
    found: list[boundary.Region] = []
    deaf = False
    if args.resume and args.out and Path(args.out).exists():
        found = [boundary.restore(r) for r in json.loads(Path(args.out).read_text())["regions"]]
        already = {r.address for r in found}
        before = len(regions)
        regions = [r for r in regions if " ".join(f"{b:02X}" for b in r[0]) not in already]
        print(f"resuming: {len(found)} regions already asked, {before - len(regions)} skipped")

    def write_out() -> None:
        # Written per region rather than at the end. A run this long is ended by
        # things that do not come back to close a file: the MIDI layer refusing a
        # client kills the process outright, and a record written only at the end
        # is a record of nothing.
        report.write_json(args.out, boundary.summarise(found, canary, deaf))

    with verified_link(args, refusing="reading past the mapped regions", show_port=True) as link:
        answers = _single_byte_reader(link, args)
        print(f"{len(regions)} regions, canary {canary}")
        for asked, (address, size) in enumerate(regions):
            start = " ".join(f"{b:02X}" for b in address)
            region = boundary.Region(address=start, mapped_size=size)
            packed = boundary.past_the_end(start, size)
            for step in range(args.reach):
                here = boundary.unpack(packed + step)
                data = answers(here)
                if data is None:
                    break
                if not region.answered_beyond:
                    region.stopped_at = here
                region.answered_beyond += 1
                region.values.append(" ".join(f"{b:02X}" for b in data))
            if region.short:
                print(f"  {start}: mapped {size}, answered {region.answered_beyond} past the end")
            found.append(region)
            write_out()
            if asked % args.every == args.every - 1 and answers(canary) is None:
                deaf = True
                print(f"\nSTOPPED: {canary} stopped answering")
                break

    result = boundary.summarise(found, canary, deaf)
    print(
        f"\n{result['regions_reaching_past_their_mapped_end']} of {result['asked']} regions "
        f"reached past their mapped end, {result['addresses_found_that_way']} addresses in all"
    )
    report.write_json(args.out, result)
    return 1 if deaf else 0
