"""Rolling per-address records up into one statement about a block.

Three commands over records that other stages left. One decides which pair of
values each address in a block should be asked at, and the other two read the
resulting per-address verdicts back into a single answer -- about a block of the
address space, or about the parameters of one insertion effect type.

The plan is the authority in both directions: it says how many addresses there
were to ask, so a short set of verdicts reads as incomplete rather than as
complete, and a record it does not name is left out rather than counted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import options, report

BAND_SET_NAMES = ("third-octave", "twelfth-octave")
"""The band sets `efx-bands` offers by name, spelled here so the parser can be
built without loading the reader. `efxbands.BAND_SETS` is what they resolve to,
and a test holds the two lists against each other rather than a reader doing it."""


def register(sub) -> None:
    p = sub.add_parser(
        "plan",
        help="say what pair of values each address in a block should be asked at, "
        "from what the write probe measured it to accept",
    )
    p.add_argument("write_probe", help="a write-probe record covering the block")
    p.add_argument(
        "block",
        help="the leading bytes of the addresses to plan, e.g. '40 11' for part 1",
    )
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_plan)

    p = sub.add_parser(
        "block",
        help="read a directory of per-address contrast records into one verdict per "
        "address, counted against the plan the block was asked from",
    )
    p.add_argument("plan", help="the plan the block was asked from, which says how many")
    p.add_argument(
        "plain",
        nargs="?",
        help="records from the pass that asked the plain note. Omitted only for a block "
        "whose plan asks nothing, where there is no pass to point at",
    )
    p.add_argument(
        "--gesture",
        help="records from the pass that moved messages after the setting. A gesture "
        "is a rescue for a null, so it answers only where the plain note could not",
    )
    p.add_argument(
        "--polyphony",
        help="records from the pass that played two notes. A parameter about polyphony "
        "sounds identical under a single note however it is set, so an address left null "
        "by the one-note passes has not been asked rather than answered",
    )
    p.add_argument(
        "--superseded-by",
        action="append",
        default=[],
        metavar="RECORD",
        help="a published record that asked some of these addresses and disagrees. Each such "
        "address is named with that record and the verdict it carries there, so a reader "
        "landing on this file does not take an answer the archive has since bettered. Only "
        "disagreements are marked",
    )
    p.add_argument(
        "--balance",
        action="append",
        default=[],
        metavar="RECORD",
        help="a balance record, once per pass. A parameter that moves signal between the "
        "channels is invisible to a comparison made in one of them, so what it found is "
        "folded in as another way of having reached the signal path. Each pass has its own "
        "takes and its own balance, so give the plain pass's and the gesture pass's both",
    )
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_block)

    p = sub.add_parser(
        "efx-rate",
        help="read what one insertion effect's rate slot modulated at, setting by "
        "setting, from takes already saved, with no machine attached",
    )
    p.add_argument(
        "takes",
        help="a directory of takes with the takes-manifest.json a --save run wrote",
    )
    p.add_argument("--type", metavar="MSB LSB", help="the type the takes were made under")
    p.add_argument("--slot", metavar="ADDR", help="the address that was swept")
    p.add_argument(
        "--setting",
        required=True,
        metavar="REGEX",
        help="a pattern over each take's setting with a group named `value`, which is "
        "the byte it was taken at, or named `type` under --untouched. A run names its "
        "takes however its own question needed, so how the setting is read back out "
        "belongs in the invocation -- where it lands in the record, and a reader can "
        "check it rather than trust it",
    )
    p.add_argument(
        "--untouched",
        action="store_true",
        help="the takes were made with a type loaded and no parameter written, so there "
        "is no byte and the pattern names the type instead. This is the baseline a "
        "sweep of the same type is read against: a sweep says what changed with the "
        "byte and cannot say what was already there",
    )
    p.add_argument(
        "--held",
        type=options.write_spec,
        action="append",
        default=[],
        metavar="ADDR=BYTES",
        help="an address the run had written while it read, and what it held. A type "
        "with two modulators returns whichever dominates, so a reading taken with the "
        "other stage turned down is a different reading and nothing in the number says so",
    )
    p.add_argument(
        "--settled",
        type=float,
        help="seconds the run waited after writing the setting before recording. One "
        "family accelerates for about four seconds and a take begun before that returns "
        "the ramp's average, so a run that did not wait records that it did not",
    )
    p.add_argument(
        "--lead", type=float, default=0.6, help="seconds of silence at the head of a take"
    )
    p.add_argument(
        "--hold",
        type=float,
        help="seconds the note was held, where the manifest's own take length is not "
        "one second longer than it",
    )
    p.add_argument(
        "--lines",
        action="store_true",
        help="also report the lines the partials' level spectra share. A vote returns "
        "one answer and lands between two modulators; this returns both, which is what "
        "a type with a modulator per stage needs",
    )
    p.add_argument(
        "--channel",
        type=int,
        metavar="N",
        help="the interface channel to read every take from. Defaults to whichever "
        "reached highest across the takes read, chosen once for the run: an input the "
        "unit is not on is not silent, so reading each take's own loudest channel gives "
        "a setting that silenced the output a level and a rate belonging to that input",
    )
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_efx_rate)

    p = sub.add_parser(
        "efx-bands",
        help="read what one insertion effect's parameter did to the level of each third "
        "octave, setting by setting, from takes already saved, with no machine attached",
    )
    p.add_argument(
        "takes",
        help="a directory of takes with the takes-manifest.json a --save run wrote",
    )
    p.add_argument(
        "--type", required=True, metavar="MSB LSB", help="the type the takes were made under"
    )
    p.add_argument("--slot", required=True, metavar="ADDR", help="the address that was swept")
    p.add_argument(
        "--setting",
        required=True,
        metavar="REGEX",
        help="a pattern over each take's setting with a group named `value`, which is "
        "the byte it was taken at. A run names its takes however its own question "
        "needed, so how the setting is read back out belongs in the invocation -- where "
        "it lands in the record, and a reader can check it rather than trust it",
    )
    p.add_argument(
        "--reference",
        required=True,
        metavar="REGEX",
        help="a pattern naming the repeats of the setting the run held flat. Every "
        "profile is reported against their mean and has to clear their spread, and they "
        "come from this directory because what a stimulus fails to repeat belongs to the "
        "session it was recorded in",
    )
    p.add_argument(
        "--control",
        metavar="REGEX",
        help="a pattern naming the takes made with the part routed past the effect. "
        "Without it the record cannot say whether the flat setting was the effect doing "
        "nothing, and every deviation below would imply a unity it never measured",
    )
    p.add_argument(
        "--silence",
        metavar="REGEX",
        help="a pattern naming the takes made with the same chain and nothing played. "
        "A setting that turns the output down far enough returns the room and the "
        "converter, and without this the floor's own shape is published as a profile",
    )
    p.add_argument(
        "--stimulus",
        help="what was sounded through the effect. A band a stimulus does not reach "
        "cannot report what the effect did there, so which stimulus carried the take "
        "bounds the record rather than decorating it",
    )
    p.add_argument(
        "--held",
        type=options.write_spec,
        action="append",
        default=[],
        metavar="ADDR=BYTES",
        help="an address the run had written while it read, and what it held. A band "
        "profile is the whole chain's, so a parameter read with another of the type's "
        "stages moved is a reading of something else and nothing in the numbers says so",
    )
    p.add_argument(
        "--reference-held",
        type=options.write_spec,
        action="append",
        default=[],
        metavar="ADDR=BYTES",
        help="an address the reference takes were made under, where that differs from "
        "the sweep. The reference is a state and not one of the readings, and on a "
        "printed range whose last position is the stage switched out it is a value of "
        "the byte being swept -- so without this the record reports every profile "
        "against something it cannot name",
    )
    p.add_argument(
        "--band",
        type=float,
        action="append",
        metavar="HZ",
        help="a band centre to measure, repeatable. Defaults to third octaves from 100 "
        "to 12500, because an octave band cannot separate two corners printed one octave "
        "apart",
    )
    p.add_argument(
        "--band-set",
        choices=BAND_SET_NAMES,
        default="third-octave",
        help="how fine to read, where --band is not given. A band wider than the "
        "deviation in it reports that deviation shallower than it was, and a deviation "
        "still growing where the set ends is reported as though it had stopped there; "
        "twelfth-octave is four times finer and reaches two octaves lower, over the "
        "same takes",
    )
    p.add_argument(
        "--channel",
        type=int,
        metavar="N",
        help="the interface channel to read every take from. Defaults to whichever is "
        "loudest in the reference takes, chosen once for the run: an input the unit is "
        "not on is not silent, so reading each take's own loudest channel makes a "
        "setting that turns the output down a full profile of something else",
    )
    p.add_argument(
        "--lead", type=float, default=0.6, help="seconds of silence at the head of a take"
    )
    p.add_argument(
        "--hold",
        type=float,
        help="seconds the stimulus was held, where the manifest's own take length is not "
        "one second longer than it",
    )
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_efx_bands)

    p = sub.add_parser(
        "efx-time",
        help="read where one insertion effect's delay slot put the copy it returns, "
        "setting by setting, from takes already saved, with no machine attached",
    )
    p.add_argument(
        "takes",
        help="a directory of takes with the takes-manifest.json a --save run wrote",
    )
    p.add_argument(
        "--type", required=True, metavar="MSB LSB", help="the type the takes were made under"
    )
    p.add_argument("--slot", required=True, metavar="ADDR", help="the address that was swept")
    p.add_argument(
        "--setting",
        required=True,
        metavar="REGEX",
        help="a pattern over each take's setting with a group named `value`, which is "
        "the byte it was taken at. Two takes matching one value is how this stage gets "
        "its floor, so a run that repeated a setting needs no separate pattern for it",
    )
    p.add_argument(
        "--control",
        required=True,
        metavar="REGEX",
        help="a pattern naming the takes made with the part routed past the effect. "
        "Required rather than optional here: their cepstrum is subtracted from every "
        "reading, so without them each row would carry whatever the stimulus and the "
        "converters put into a log spectrum and report it as a copy",
    )
    p.add_argument(
        "--stimulus",
        help="what was sounded through the effect. The reading wants a source whose own "
        "log spectrum is smooth, and a held note's is a comb of its own, so which "
        "stimulus carried the take bounds the record rather than decorating it",
    )
    p.add_argument(
        "--held",
        type=options.write_spec,
        action="append",
        default=[],
        metavar="ADDR=BYTES",
        help="an address the run had written while it read, and what it held. A time "
        "read out of a cepstrum is a time through the whole chain, so a feedback path "
        "left open returns further copies and a balance carrying only the return leaves "
        "nothing for the copy to beat against and no peak at all",
    )
    p.add_argument(
        "--frame",
        type=int,
        default=65536,
        help="samples per cepstral frame. It bounds the reading at both ends -- half of "
        "it is the longest delay that can be placed, and a delay that moves inside one "
        "smears until nothing stands out -- so a longer printed range needs a longer "
        "frame and buys it with fewer frames to average",
    )
    p.add_argument(
        "--hop", type=int, default=16384, help="samples between frames"
    )
    p.add_argument(
        "--shortest",
        type=float,
        default=0.4,
        help="milliseconds below which no peak is read. Under it the cepstrum carries "
        "the shape of the source rather than anything in the path",
    )
    p.add_argument(
        "--longest",
        type=float,
        default=620.0,
        help="milliseconds above which no peak is read. Set past the printed end of the "
        "range, so a byte that runs further than the page says shows as a reading rather "
        "than as the search's own edge",
    )
    p.add_argument(
        "--peaks-apart",
        type=float,
        default=1.0,
        help="milliseconds set aside either side of a peak before the next is taken. The "
        "top of a broad peak is several quefrencies wide, and without this the same peak "
        "is returned three times",
    )
    p.add_argument(
        "--channel",
        type=int,
        metavar="N",
        help="the interface channel to read every take from. Defaults to whichever is "
        "loudest in the takes with the effect out, chosen once for the run: an input the "
        "unit is not on is not silent, and a take read from one returns a full plausible "
        "cepstrum of something else",
    )
    p.add_argument(
        "--lead", type=float, default=0.6, help="seconds of silence at the head of a take"
    )
    p.add_argument(
        "--hold",
        type=float,
        help="seconds the stimulus was held, where the manifest's own take length is not "
        "one second longer than it",
    )
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_efx_time)

    p = sub.add_parser(
        "efx-params",
        help="read a directory of per-address records into one verdict per parameter of "
        "one insertion effect type, with no machine attached",
    )
    p.add_argument("records", help="a directory of contrast records, one per parameter address")
    p.add_argument(
        "--type", required=True, metavar="MSB LSB", help="the type they were taken under"
    )
    p.add_argument(
        "--types-from", required=True, help="an efx-type-map record, for the settings it loads"
    )
    p.add_argument(
        "--control", required=True, help="the routed-against-bypassed record from the same run"
    )
    p.add_argument(
        "--prepare",
        type=options.write_spec,
        action="append",
        default=[],
        metavar="ADDR=BYTES",
        help="the state the run was taken in, recorded with it since a verdict holds in it",
    )
    p.add_argument(
        "--supersede",
        action="append",
        default=[],
        metavar="ADDR=RECORD",
        help="an address whose first pair was withdrawn, and the record that replaced it. "
        "A pair that left one setting silent compares sound with silence, and the reason "
        "travels with the row rather than with whoever remembers the directory",
    )
    p.add_argument(
        "--slots",
        nargs="+",
        default=[],
        metavar="ADDR",
        help="the type's parameter addresses in slot order, which is what makes a missing "
        "record readable. Without them the slot is the position in the sorted file names, so "
        "an address the run could not measure renumbers every parameter after it and pairs "
        "each with the default of the slot before, and the count reports the type as having "
        "one parameter fewer than it has. Given rather than derived because which address is "
        "which slot is a fact about the unit",
    )
    options.add_out(p)
    p.set_defaults(needs_unit=False, func=cmd_efx_params)


def cmd_plan(args: argparse.Namespace) -> int:
    """Turn a write probe's measured ranges into the pair each address is asked at."""
    from .. import plan

    record = json.loads(Path(args.write_probe).read_text())
    asks, skipped = plan.plan_block(record, args.block)
    if not asks and not skipped:
        print(f"no address under {args.block!r} in {args.write_probe}")
        return 1
    print(plan.summarise(asks, skipped))

    report.write_json(
        args.out,
        {
            "block": args.block,
            "from": str(args.write_probe),
            "method": plan.METHOD,
            "ask": [a.to_json() for a in asks],
            "cannot_be_asked": [s.to_json() for s in skipped],
        },
    )
    return 0


def cmd_block(args: argparse.Namespace) -> int:
    """Read a block's per-address records into one answer about the block."""
    from .. import block

    planned = json.loads(Path(args.plan).read_text())
    plain, outside = (
        block.split_by_plan(block.survey(args.plain), planned) if args.plain else ({}, [])
    )
    gesture, more = (
        block.split_by_plan(block.survey(args.gesture), planned) if args.gesture else ({}, [])
    )
    outside += more
    polyphony, more = (
        block.split_by_plan(block.survey(args.polyphony), planned) if args.polyphony else ({}, [])
    )
    outside += more
    for address in sorted(set(outside)):
        print(f"  {address}: a record the plan does not name, left out of the block")
    # An empty pass is a mistake only where the plan expected one. A block whose
    # every address accepts a single value has no pair to compare and so no
    # contrast record to point at, and refusing to fold it would leave the one
    # thing measured about it -- that nothing there can be moved -- unpublished,
    # while the block went on being counted as a sweep somebody still owes.
    if not plain and not gesture and not polyphony and planned.get("ask"):
        print(f"no contrast records under {args.plain}")
        return 1

    found = [f for f in block.join(plain, gesture, polyphony) if f is not None]
    # One per pass rather than one for the block: each pass has its own takes and
    # its own balance, and a parameter that moves the balance under the gesture
    # is as much a finding as one that moves it under the plain note.
    for where in args.balance:
        found = block.with_balance(found, json.loads(Path(where).read_text()))
    found = block.with_the_plans_caveats(found, planned)
    # Only where the two actually disagree. A record that asked the same address
    # and answered the same way supersedes nothing, and marking it would tell a
    # reader to go elsewhere for the answer already in front of them.
    answered_elsewhere: dict[str, dict] = {}
    for where in args.superseded_by:
        other = json.loads(Path(where).read_text())
        mine = {f.address: f.verdict for f in found}
        for row in other.get("addresses", []):
            if row["address"] in mine and row["verdict"] != mine[row["address"]]:
                answered_elsewhere[row["address"]] = {
                    "record": Path(where).name,
                    "verdict": row["verdict"],
                }
    if answered_elsewhere:
        found = block.superseded(found, answered_elsewhere)
        print(f"  {len(answered_elsewhere)} addresses answered by another record")
    coverage = block.against_plan(found, planned)
    print(block.summarise(found, coverage))
    # Named rather than counted: an address left to try is the next run's list,
    # and a number is not a list.
    open_still = [f.address for f in found if f.still_open]
    if open_still:
        print(f"\nstill open: {' | '.join(open_still)}")

    report.write_json(
        args.out,
        {
            "block": planned.get("block"),
            "method": block.method_for(bool(args.gesture), asked=bool(planned.get("ask"))),
            # Where the addresses came from, which for a block nothing could be
            # asked of is the whole of its evidence: the record holds no verdict
            # of its own, and a reader has to be able to reach the run that
            # established there was nothing to ask.
            "planned_from": planned.get("from"),
            "asked_in": {
                name: found_states
                for name, where in (
                    ("plain", args.plain),
                    ("gesture", args.gesture),
                    ("polyphony", args.polyphony),
                )
                if where and (found_states := block.states(where))
            },
            "why_asked_in": block.WHY_ASKED_IN,
            **(
                {
                    "records_the_plan_does_not_name": sorted(set(outside)),
                    "why_outside_the_plan": block.WHY_OUTSIDE_THE_PLAN,
                }
                if outside
                else {}
            ),
            **({"two_passes": block.WHY_TWO_PASSES} if args.gesture else {}),
            **({"polyphony_pass": block.WHY_POLYPHONY_PASS} if args.polyphony else {}),
            **({"balance_counts": block.WHY_BALANCE_COUNTS} if args.balance else {}),
            "chose_the_values": planned.get("method"),
            "coverage": coverage,
            "still_open": open_still,
            "addresses": [f.to_json() for f in found],
        },
    )
    return 0


def cmd_efx_rate(args) -> int:
    """What one rate slot modulates at, read from takes already saved."""
    from .. import efxrate

    def said(reading) -> None:
        at = reading.get("value")
        head = f"{at:5d}" if at is not None else f"{reading['type']:>5s}"
        found = reading["rate_hz"]
        print(
            f"  {head} -> "
            + (f"{found:8.4f} Hz" if found is not None else f"{'--':>11s}")
            + f"  {reading['agreeing']}/{reading['of']} partials"
            f"  floor {reading['slowest_measurable_hz']}  {reading['heard_db']:.0f} dBFS"
        )

    if args.untouched:
        if args.type or args.slot:
            print("--untouched takes no --type or --slot: nothing was written, and the")
            print("pattern names the type because that is what separates the takes")
            return 2
        found = efxrate.read_untouched(
            args.takes,
            setting=args.setting,
            settled_s=args.settled,
            lead_s=args.lead,
            hold_s=args.hold,
            shared_lines=args.lines,
            channel=args.channel,
            progress=said,
        )
    else:
        if not args.type or not args.slot:
            print("--type and --slot are required unless --untouched")
            return 2
        found = efxrate.read_directory(
            args.takes,
            type_id=args.type,
            address=args.slot,
            setting=args.setting,
            held=[{"address": a, "bytes": " ".join(f"{b:02X}" for b in v)} for a, v in args.held],
            settled_s=args.settled,
            lead_s=args.lead,
            hold_s=args.hold,
            shared_lines=args.lines,
            channel=args.channel,
            progress=said,
        )
    picked = found["channel"]
    print(
        f"  read from channel {picked['read']} of "
        f"{len(picked['reached_db'])} ({picked['chosen_by']}): "
        + " ".join(f"{v:.0f}" for v in picked["reached_db"])
        + " dBFS"
    )
    if not found["readings"]:
        print(
            f"no take under {args.takes} has a setting matching {args.setting!r}; "
            f"{found['takes_not_matching']['count']} were looked at"
        )
        return 1
    if (missed := found["takes_not_matching"]["count"]):
        print(f"  ({missed} takes under the same directory did not match the pattern)")
    report.write_json(args.out, found)
    return 0


def _span_said(reading) -> str:
    """Where the deviation had halved, and whether it levelled off before the end.

    An open end is printed as such rather than left blank: a profile that never
    comes back down inside the set and one that comes back down at the last band
    are different answers, and a blank reads as the second.
    """
    below, above = reading["half_below_hz"], reading["half_above_hz"]
    if below is None and above is None:
        return ""
    ends = [
        reading["settled_below_db"] if below is None else None,
        reading["settled_above_db"] if above is None else None,
    ]
    still = max((abs(v) for v in ends if v is not None), default=None)
    return (
        f"  half {'<' if below is None else f'{below:.0f}'}"
        f"-{'>' if above is None else f'{above:.0f}'} Hz"
        + ("" if still is None else f" ({still:.2f} dB left at the end)")
    )


def _steepest_said(reading) -> str:
    """How fast the deviation ran at its fastest, where it has a figure at all."""
    slope = reading.get("steepest_db_per_octave")
    if slope is None:
        return ""
    return f"  {slope:+.1f} dB/oct at {reading['steepest_at_hz']:.0f} Hz"


def cmd_efx_bands(args) -> int:
    """What one parameter did to each band, read from takes already saved."""
    from .. import efxbands

    # Centres given one at a time are read a third of an octave wide, because a
    # list of centres does not say how far each reaches and nothing else in the
    # invocation would say it either.
    named = efxbands.BAND_SETS[args.band_set]

    def said(reading) -> None:
        largest = reading["largest_db"]
        moved = len(reading["outside_the_floor_hz"])
        print(
            f"  {reading['value']:5d} -> "
            + (
                f"{largest:+7.2f} dB at {reading['largest_at_hz']:>6.0f} Hz"
                if largest is not None
                else f"{'inside the floor':>25s}"
            )
            + f"  {moved:2d} bands  {reading['heard_db']:.0f} dBFS"
            + _span_said(reading)
            + _steepest_said(reading)
        )

    found = efxbands.read_directory(
        args.takes,
        type_id=args.type,
        address=args.slot,
        setting=args.setting,
        reference=args.reference,
        control=args.control,
        silence=args.silence,
        stimulus=args.stimulus,
        held=[
            {"address": a, "bytes": " ".join(f"{b:02X}" for b in v)} for a, v in args.held
        ],
        reference_held=[
            {"address": a, "bytes": " ".join(f"{b:02X}" for b in v)}
            for a, v in args.reference_held
        ],
        bands_hz=args.band or named[0],
        band_width_octaves=1 / 3 if args.band else named[1],
        channel=args.channel,
        lead_s=args.lead,
        hold_s=args.hold,
        progress=said,
    )
    picked = found["channel"]
    print(
        f"  read from channel {picked['read']} of "
        f"{len(picked['reference_db'])} ({picked['chosen_by']}): "
        + " ".join(f"{v:.0f}" for v in picked["reference_db"])
        + " dBFS"
    )
    if (beside := found["other_channel"]["read"]) is not None:
        apart = [abs(r["apart_db"]) for r in found["readings"] if r.get("apart_db") is not None]
        moved = [abs(r["other_db"]) for r in found["readings"] if r.get("other_db") is not None]
        if apart:
            print(
                f"  channel {beside} moved up to {max(moved):.2f} dB of its own and "
                f"stayed within {max(apart):.2f} dB of channel {picked['read']}"
            )
    if not found["readings"]:
        print(
            f"no take under {args.takes} has a setting matching {args.setting!r}; "
            f"{found['takes_not_matching']['count']} were looked at"
        )
        return 1
    if not found["control"]["takes"]:
        print("  (no --control: the record cannot say whether the flat setting was unity)")
    if not found["silence"]["takes"]:
        print("  (no --silence: a setting that turns the output off reads as a profile)")
    if (missed := found["takes_not_matching"]["count"]):
        print(f"  ({missed} takes under the same directory did not match the pattern)")
    if (astray := picked["loudest_elsewhere"]):
        print(
            f"  ({len(astray)} takes are loudest on another channel; read from "
            f"{picked['read']} anyway, and named in the record)"
        )
    report.write_json(args.out, found)
    return 0


def cmd_efx_time(args) -> int:
    """Where a delay slot put its copy, read from takes already saved."""
    from .. import efxtime

    def said(reading) -> None:
        also = "  ".join(
            f"{q:.1f}x{s:.0f}"
            for q, s in zip(reading["also_ms"], reading["also_stands"], strict=True)
        )
        print(
            f"  {reading['value']:5d} -> "
            + (
                f"{reading['ms']:9.3f} ms  x{reading['stands']:6.1f}"
                if reading["admitted"]
                else f"{'under the roughness':>19s}  x{reading['stands']:6.1f}"
            )
            + f"   also {also}"
        )

    found = efxtime.read_directory(
        args.takes,
        type_id=args.type,
        address=args.slot,
        setting=args.setting,
        control=args.control,
        stimulus=args.stimulus,
        held=[
            {"address": a, "bytes": " ".join(f"{b:02X}" for b in v)} for a, v in args.held
        ],
        channel=args.channel,
        frame=args.frame,
        hop=args.hop,
        searched_ms=(args.shortest, args.longest),
        apart_ms=args.peaks_apart,
        lead_s=args.lead,
        hold_s=args.hold,
        progress=said,
    )
    picked = found["channel"]
    print(
        f"  read from channel {picked['read']} of "
        f"{len(picked['reference_db'])} ({picked['chosen_by']}): "
        + " ".join(f"{v:.0f}" for v in picked["reference_db"])
        + " dBFS"
    )
    if not found["readings"]:
        print(
            f"no take under {args.takes} has a setting matching {args.setting!r}; "
            f"{found['takes_not_matching']['count']} were looked at"
        )
        return 1
    out = found["with_the_effect_out"]
    print(
        f"  with the effect out the chain's own best peak is {out['ms']:.3f} ms at "
        f"x{out['stands']:.1f}"
        + ("  <- which would be read as a delay" if out["would_be_read_as_a_delay"] else "")
    )
    left = len(found["settings_asked"]) - len(found["settings_admitted"])
    print(
        f"  {len(found['settings_admitted'])} settings admitted, {left} under the "
        f"roughness, over {found['frames_averaged']} frames a take"
    )
    if found["floor_ms"] is None:
        print("  (no setting was taken twice: this run measured no floor)")
    else:
        print(
            f"  the same setting twice lands {found['floor_ms']:.3f} ms apart, on a "
            f"grid of {found['quefrency_step_ms']:.4f} ms"
        )
    if (missed := found["takes_not_matching"]["count"]):
        print(f"  ({missed} takes under the same directory did not match the pattern)")
    if (astray := picked["loudest_elsewhere"]):
        print(
            f"  ({len(astray)} takes are loudest on another channel; read from "
            f"{picked['read']} anyway, and named in the record)"
        )
    report.write_json(args.out, found)
    return 0


def cmd_efx_params(args) -> int:
    """One verdict per parameter of one type, from records already captured."""
    from .. import efxparams

    types = json.loads(Path(args.types_from).read_text())
    loads = [e["parameters"] for e in types["effects"] if e["type"] == args.type]
    if not loads:
        print(f"{args.types_from} has no type {args.type}")
        return 1
    found = efxparams.read_directory(
        args.records,
        args.type,
        loads[0],
        args.control,
        [{"address": a, "bytes": " ".join(f"{v:02X}" for v in vs)} for a, vs in args.prepare],
        supersede=dict(s.split("=", 1) for s in args.supersede),
        slots=args.slots or None,
    )
    for name, count in found["results"].items():
        print(f"  {name}: {count}")
    if (coverage := found.get("coverage")) and (missing := coverage["never_asked"]):
        print(f"  => {len(missing)} never asked: {' | '.join(missing)}")
    report.write_json(args.out, found)
    return 0
