"""Listen to the unit, and say what is a difference and what is only noise.

The two commands that need an audio interface as well as a MIDI one. Both rest
on the same fact: nothing measured from a pair of takes means anything until the
unit has been shown to repeat itself, so `repeat` is what makes `contrast`
readable rather than merely confident.
"""

from __future__ import annotations

import argparse
import time

from .. import clock, gestures, parts, perform, roland
from ..stimuli import BROAD as _STIMULUS_BROAD
from ..stimuli import CATALOGUE as _STIMULUS_CATALOGUE
from ..stimuli import DEFAULT as _STIMULUS_DEFAULT
from ..stimuli import SWITCH as _STIMULUS_SWITCH
from . import options, report
from .session import holds as setting_holds
from .session import prepared as prepare_state
from .session import verified_link

_STIMULUS_NAMES = tuple(_STIMULUS_CATALOGUE)

_HOW_MANY = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight")
"""Spelled out, because the help around them is prose and `5` reads as a flag.

Counted from the sets rather than written beside them: the switch set was
described as four while holding five, which is the whole failure mode of a
number kept next to the thing it counts.
"""

WHY_READ_BACK = (
    "The setting under test was read back after each write, as the preparation already was. An "
    "address that takes a write and keeps what it had leaves both sets of takes at one setting, "
    "which reads as a parameter that does nothing -- a null indistinguishable from a real one "
    "with nothing in the record to tell them apart. An address that answers no one-byte read is "
    "recorded as unchecked rather than refused, since some are readable only as part of a wider "
    "region and refusing them would drop them from the sweep for being unreadable."
)


def settings_reset_before(values, between: bool) -> list:
    """Which of the two settings is preceded by a reset of its own.

    The first is not: the run resets before the stimulus anyway, and resetting
    again would cost a reset to reach the state it is already in. The second is,
    and only where the run asked -- which is the whole difference between asking
    a switch and asking it in the state the previous setting left.
    """
    return list(values[1:]) if between else []


WHY_RESET_BETWEEN = (
    "The unit was reset and set up again before the second setting as well as the first. Without "
    "it the second setting inherits whatever the first one's takes left behind, which is what "
    "makes a parameter that decides whether a message is received invisible: the message moves "
    "something under the setting that receives it, and under the setting that ignores it that "
    "something is still exactly where it was moved to. A reset rather than one parameter put "
    "back, because a gated message cannot be undone by another message of the same kind -- the "
    "switch ignores that one too -- and pitch bend, modulation and channel pressure have no "
    "address on this unit for a write to reach. What it costs is that each setting is asked "
    "from the power-on state rather than from the one before it."
)

WHY_AS_HELD = (
    "What the two settings are is what the unit holds after each write, not what the run asked "
    "for. An address that clamps, or that rounds to what it has, still answers the question as "
    "long as the two land somewhere different -- refusing it because a byte came back changed "
    "would drop a measurable parameter for being narrower than the plan guessed, and the plan's "
    "range is itself derived rather than given. Only a pair that collapses to one byte is "
    "refused, since there the two sets of takes are takes of the same setting."
)


def register(sub) -> None:
    p = sub.add_parser(
        "repeat",
        help="find whether two takes of the same note can be compared, which every "
        "difference measurement rests on",
    )
    options.add_channel(p)
    p.add_argument("--program", type=int, default=0)
    p.add_argument("--note", type=int, default=60)
    p.add_argument("--velocity", type=int, default=100)
    p.add_argument("--takes", type=int, default=6)
    p.add_argument("--hold", type=float, default=1.0, help="seconds the note is held")
    p.add_argument("--seconds", type=float, default=3.0, help="length of each take")
    p.add_argument(
        "--lead",
        type=float,
        default=0.6,
        help="silence before the note; the noise floor is measured in it, so a residual "
        "has something it could not have gone below",
    )
    p.add_argument("--between", type=float, default=0.8, help="seconds between takes")
    p.add_argument("--reverb", type=int, default=0, help="CC91 send during the takes")
    p.add_argument("--chorus", type=int, default=0, help="CC93 send during the takes")
    p.add_argument(
        "--clock",
        action="store_true",
        help="also hold one long note and read its frequency, which carries the unit's "
        "tuning and the two sample clocks' ratio together",
    )
    p.add_argument(
        "--clock-program",
        type=options.bank_program,
        nargs="+",
        default=[(0, 16), (0, 19), (0, 73), (0, 79), (0, 80), (8, 80), (8, 81)],
        help="programs to hold; most voices are not a frequency reference, so several "
        "are tried and the ones that wobble are reported as wobbling rather than read",
    )
    p.add_argument("--clock-note", type=int, default=69)
    p.add_argument("--clock-seconds", type=float, default=20.0)
    options.add_audio(p)
    p.add_argument(
        "--min-rise",
        type=float,
        default=12.0,
        help="dB the note must rise above the silence before it; below this the take holds "
        "no sound from the unit, which would otherwise read as the unit not repeating",
    )
    options.add_max_lead_in(p)
    options.add_verify_reads(p)
    options.add_save(p)
    options.add_out(p)
    p.set_defaults(func=cmd_repeat)

    p = sub.add_parser(
        "contrast",
        help="find whether changing one parameter changes the sound, measured against "
        "how well the unit repeats itself",
    )
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--cc", type=int, help="controller number to change")
    target.add_argument("--address", help="three hex bytes to write instead, e.g. '40 01 30'")
    p.add_argument(
        "--values",
        type=options.pair,
        default=(0, 127),
        help="the two settings compared; both must be inside what the parameter accepts, "
        "or a clamp answers them identically and reads as inaudible",
    )
    options.add_channel(p)
    p.add_argument(
        "--stimulus",
        nargs="+",
        default=list(_STIMULUS_DEFAULT),
        metavar="NAME",
        help="notes to ask the parameter under: "
        + ", ".join(_STIMULUS_NAMES)
        + f", plus 'broad' for the {_HOW_MANY[len(_STIMULUS_BROAD)]} that answer most "
        f"parameters, 'switch' for the {_HOW_MANY[len(_STIMULUS_SWITCH)]} a byte with two "
        "values needs, and 'all'. Audible under any is audible; a null carries the list of "
        "what was tried, because an inaudible result is as much a fact about the note as "
        "about the parameter",
    )
    p.add_argument(
        "--prepare",
        type=options.write_spec,
        action="append",
        default=[],
        metavar="ADDR=BYTES",
        help="state the parameter needs before it can be heard at all, written after "
        "each reset and read back to prove it took: '40 41 22=01' routes part 1 through "
        "the insertion effect, '40 03 00=02 01' selects one. An effect parameter asked "
        "with nothing routed to the effect answers inaudible every time, and that is a "
        "verdict on the routing wearing the parameter's name. Recorded with the result, "
        "since the verdict only holds in the state it was taken in",
    )
    p.add_argument(
        "--reset-between-settings",
        action="store_true",
        help="reset the unit and set it up again before the second setting as well as "
        "before the first, for a parameter that decides whether a message is received. "
        "Such a parameter is invisible otherwise: with it on the message lands and moves "
        "something, and with it off that something is still where the first setting's "
        "message left it, so the two settings sound alike however different they are. A "
        "reset rather than putting one parameter back, because a message that is gated "
        "cannot be undone by another message of the same kind -- the switch ignores that "
        "one too -- and pitch bend, modulation and pressure have no address on this unit "
        "to put back by SysEx",
    )
    p.add_argument("--takes", type=int, default=4, help="takes per setting per stimulus")
    p.add_argument("--between", type=float, default=0.8)
    p.add_argument("--settle", type=float, default=0.4, help="seconds after changing the setting")
    p.add_argument(
        "--margin",
        type=float,
        default=6.0,
        help="dB the change must clear the unit's own repeatability by before it is called audible",
    )
    options.add_audio(p)
    p.add_argument("--min-rise", type=float, default=12.0)
    options.add_max_lead_in(p)
    options.add_verify_reads(p)
    options.add_save(p)
    options.add_out(p)
    p.set_defaults(func=cmd_contrast)


def _store(where: str | None):
    """Open a directory for the takes, when the run was asked to keep them."""
    if not where:
        return None
    from ..takes import Store

    return Store.open(where)


def _lead_in_ok(groups, rate: float, before: float, limit: float) -> bool:
    """Refuse a run whose lead-in was not silent, naming why it matters."""
    from .. import stability

    worst = stability.loudest_lead_in([t for g in groups for t in g], rate, before=before)
    if worst <= limit:
        return True
    print(
        f"\nThe loudest lead-in sits at {worst:.1f} dBFS, against {limit:.0f} dBFS asked for. "
        "Something was sounding before the note: the tail of the take before it, or another "
        "process driving the same unit. The noise floor sets the yardstick every number here "
        "is judged against, so nothing measured from these takes would mean anything."
    )
    return False


def _part_offset(args: argparse.Namespace, channel: int) -> int | None:
    """The offset in this channel's part block the run is writing, if it is writing one.

    A controller run writes no address, and an address outside the part block --
    an effect parameter, a drum setup byte -- is nothing a gesture here touches.
    Both come back None, which takes nothing out of the gesture.
    """
    if args.cc is not None or not args.address:
        return None
    try:
        top, middle, low = (int(b, 16) for b in args.address.split())
    except ValueError:
        return None
    block = parts.part_block(parts.TONE, channel)
    return low if (top, middle) == block[:2] else None


def _without_the_address(stim, args: argparse.Namespace):
    """The stimulus with any move that writes the address under test taken out."""
    import dataclasses

    kept, dropped = gestures.without(stim.moves, _part_offset(args, stim.on(args.channel)))
    if not dropped:
        return stim, []
    return dataclasses.replace(stim, moves=kept), dropped


def cmd_contrast(args: argparse.Namespace) -> int:
    """Play the same note under two settings and say whether the unit sounded different."""
    from .. import audible, stability, stimuli
    from ..resets import Prober, named

    def setting(value: int, channel: int) -> list[list[int]]:
        """The messages that put the parameter at `value`, on the stimulus's channel.

        The channel is the stimulus's, not the command's: a drum stimulus sounds
        on channel 10, and a controller sent to the command's channel would
        change a part that is not the one being listened to. That fails as a
        null -- the parameter reads inaudible -- rather than as an error.

        An address carries its own part number and cannot follow the stimulus
        that way, so an address run has to be given a stimulus on the part the
        address addresses.
        """
        if args.cc is not None:
            return [[0xB0 | channel, args.cc & 0x7F, value & 0x7F]]
        return [roland.dt1(args.address, [value], device_id=args.device_id)]

    try:
        asked = stimuli.resolve(args.stimulus)
    except KeyError as exc:
        print(exc)
        return 1

    where = f"CC{args.cc}" if args.cc is not None else f"address {args.address}"
    label = f"{where} {args.values[0]} against {args.values[1]}"
    overall = audible.Overall(label=label)
    store = _store(args.save)
    # What was actually sent, which is what the record has to carry: a gesture
    # one message short is a different question from the whole one.
    sent: list = []
    left_out: dict[str, list] = {}
    read_back: list[dict] = []
    rests: list[dict] = []
    resets_between: list[dict] = []
    as_held: list[int | None] = []

    with verified_link(args, refusing="recording") as link:
        gs_reset = named("GS Reset", args.device_id)
        prober = Prober(link, baseline={}, device_id=args.device_id)
        prober.apply(gs_reset)

        # What the pair actually lands on, before any of it is played. An address
        # that clamps still answers as long as the two land apart, and where they
        # land is what the label has to say.
        if args.address:
            for value in args.values:
                link.send(roland.dt1(args.address, [value], device_id=args.device_id))
                time.sleep(args.settle)
                as_held.append(setting_holds(link, args.address, device_id=args.device_id))
            if as_held[0] is not None and as_held[0] == as_held[1]:
                print(
                    f"\n{args.address} was asked at {args.values[0]} and {args.values[1]} and "
                    f"holds {as_held[0]:02X} after both. The two sets of takes would be takes of "
                    "one setting, so nothing measured from them would be about the parameter."
                )
                return 1
            if any(h is not None and h != v for h, v in zip(as_held, args.values, strict=False)):
                landed = " and ".join("unreadable" if h is None else f"{h:02X}" for h in as_held)
                label = f"{where} {args.values[0]} against {args.values[1]}, holding {landed}"
                overall = audible.Overall(label=label)
                print(f"\n{args.address} lands on {landed}, not on what was asked. {WHY_AS_HELD}")

        def from_the_top(stim, channel) -> bool:
            """Put the unit back where a stimulus starts from.

            The reset, the preparation, and the stimulus's own setup: one block,
            because they are one state and re-establishing part of it is how a
            run ends up measuring the part that was left. Called once per
            stimulus, and once per setting as well where the run asks for it.
            """
            prober.apply(gs_reset)
            if not prepare_state(link, args.prepare, device_id=args.device_id, settle=args.settle):
                return False
            for prepared, value in stim.writes:
                link.send(roland.dt1(prepared, [value], device_id=args.device_id))
            for message in (
                [0xC0 | channel, stim.program & 0x7F],
                [0xB0 | channel, 7, 127],
                [0xB0 | channel, 11, 127],
            ):
                link.send(message)
            time.sleep(0.3)
            return True

        print(f"\n{label}\n  {len(asked)} stimulus/stimuli: {', '.join(s.name for s in asked)}")
        for stim in asked:
            channel = stim.on(args.channel)
            # A gesture that writes the address under test would put both
            # settings at its own value, so the run would measure the gesture.
            stim, dropped = _without_the_address(stim, args)
            if dropped:
                print(
                    f"  {stim.name}: left out of the gesture, it writes the address under "
                    f"test -- {gestures.describe(tuple(dropped))}"
                )
                left_out.setdefault(stim.name, []).extend(m.to_json() for m in dropped)
            sent.append(stim)
            # Reset between stimuli, so a setting left by the previous one cannot
            # follow the parameter into the next and be read as part of it.
            if not from_the_top(stim, channel):
                return 1
            print(f"\n  {stim.name}: {stim.describe()}")

            captured: list[list] = []
            for value in args.values:
                # And again before the second setting, where the run asks for it.
                # A parameter that decides whether a message is received is
                # invisible without this: the message moves something under the
                # first setting and, ignored under the second, leaves that
                # something exactly where it put it.
                if value in settings_reset_before(args.values, args.reset_between_settings):
                    if not from_the_top(stim, channel):
                        return 1
                    resets_between.append({"stimulus": stim.name, "before_setting": value})
                for message in setting(value, channel):
                    link.send(message)
                time.sleep(args.settle)
                # Read the setting back, as the preparation already is. Against
                # what the pre-flight measured the write to land on rather than
                # against what was asked: an address that clamps lands somewhere
                # else every time and is still a setting, while one that has
                # drifted from where it landed is not the setting being asked.
                if args.address:
                    expected = as_held[args.values.index(value)]
                    now = setting_holds(link, args.address, device_id=args.device_id)
                    shown = "unreadable" if now is None else f"{now:02X}"
                    ok = now is None or expected is None or now == expected
                    read_back.append({"setting": value, "reads": shown, "took": ok})
                    if not ok:
                        print(
                            f"\n    {args.address} was set to {value}, landed on {expected:02X} "
                            f"before the run and reads back {shown} now. The setting moved under "
                            "the run, so the takes are not all of one setting."
                        )
                        return 1
                takes = []
                for index in range(args.takes):
                    # After the setting, and before every take rather than once
                    # per setting: a note-off is the one thing between two takes,
                    # and whether it clears a pressure or a pedal is exactly the
                    # sort of thing this is here to find out rather than assume.
                    for move in stim.moves:
                        for message in move.messages(channel):
                            link.send(message)
                    if stim.moves:
                        time.sleep(0.15)
                    # Take it again when its own lead-in was not quiet, rather
                    # than measuring through a tail. The lead-in is where the
                    # floor comes from and the floor is the yardstick, so a tail
                    # left by the take before does not add noise to the answer --
                    # it raises the bar the parameter is then asked to clear.
                    for attempt in range(perform.RETAKES):
                        recording, reopened = perform.record_notes_retrying(
                            link,
                            device=args.audio,
                            channel=channel,
                            notes=stim.played(),
                            seconds=stim.seconds,
                            lead=stim.lead,
                        )
                        was = perform.lead_in_dbfs(recording, stim.lead * 0.8)
                        quiet = was <= args.max_lead_in
                        if attempt or reopened or not quiet:
                            rests.append(
                                {
                                    "stimulus": stim.name,
                                    "setting": value,
                                    "take": index,
                                    "attempt": attempt + 1,
                                    "lead_in_dbfs": None if was == float("-inf") else round(was, 1),
                                    "was_quiet": quiet,
                                    "device_reopened": reopened,
                                }
                            )
                        if quiet:
                            break
                        time.sleep(perform.REST_S)
                    if not recording.healthy:
                        print(f"    lossy capture: {recording.health_report()}")
                        return 1
                    takes.append(recording)
                    if store is not None:
                        store.keep(recording, stimulus=stim.name, setting=str(value), take=index)
                    time.sleep(args.between)
                print(f"    {where} = {value}: {len(takes)} takes")
                captured.append(takes)

            # Which input the unit arrived on, and whether it arrived at all, are
            # both asked of the setting that sounded loudest rather than of the
            # first one. A parameter that silences the note at one of its two
            # values is the strongest audible result there is, and asking the
            # first setting alone read that as a dead input and threw the takes
            # away -- while choosing the channel from a silent take picks
            # whichever input carried the most noise.
            rate = captured[0][0].sample_rate
            before = stim.lead * 0.8
            index = perform.loudest_channel(max((t[0] for t in captured), key=perform.peak))
            groups = [[t.channel(index) for t in takes] for takes in captured]
            rises = [stability.signal_over_silence(g[0], rate, before=before) for g in groups]
            shown = ", ".join(f"{v} at {r:.1f} dB" for v, r in zip(args.values, rises, strict=True))
            print(f"    channel {index}: note over the lead-in, {shown}")
            if not max(rises) > args.min_rise:
                print(
                    f"\n    The note never rose {args.min_rise} dB above the silence before "
                    "it at either setting, so these takes hold no sound from the unit. Name "
                    "the right input with --audio."
                )
                return 1
            if not _lead_in_ok(groups, rate, before, args.max_lead_in):
                return 1

            verdict = audible.judge(
                groups[0],
                groups[1],
                rate,
                label=label,
                stimulus=stim.describe(),
                stimulus_name=stim.name,
                silence_before=before,
                margin_db=args.margin,
            )
            overall.verdicts.append(verdict)
            print(f"    {verdict.describe()}")
        prober.apply(gs_reset)

    print()
    print(overall.describe())

    if store is not None:
        manifest = store.close(
            label=label,
            controller=args.cc,
            address=None if args.cc is not None else args.address,
            values=list(args.values),
            channel=args.channel,
            stimuli=[stim.to_json() for stim in sent],
        )
        print(f"\nkept {len(store.entries)} takes under {manifest.parent}")

    report.write_json(
        args.out,
        {
            "device_id": f"{args.device_id:02X}",
            "controller": args.cc,
            "address": None if args.cc is not None else args.address,
            "values": list(args.values),
            **(
                {
                    "values_as_held": [None if h is None else f"{h:02X}" for h in as_held],
                    "why_values_as_held": WHY_AS_HELD,
                }
                if any(h is not None and h != v for h, v in zip(as_held, args.values, strict=False))
                else {}
            ),
            "prepared": [
                {"address": a, "bytes": " ".join(f"{v:02X}" for v in vs)} for a, vs in args.prepare
            ],
            **({"prepared_caveat": audible.PREPARED_CAVEAT} if args.prepare else {}),
            "reset_before_each_setting": bool(args.reset_between_settings),
            **(
                {"why_reset_between": WHY_RESET_BETWEEN, "reset_at": resets_between}
                if resets_between
                else {}
            ),
            "stimuli": [stim.to_json() for stim in sent],
            **(
                {"setting_read_back": read_back, "why_read_back": WHY_READ_BACK}
                if read_back
                else {}
            ),
            **({"retaken": rests, "why_retaken": perform.WHY_RETAKEN} if rests else {}),
            **(
                {"left_out_of_the_gesture": left_out, "why_left_out": gestures.WHY_DROPPED}
                if left_out
                else {}
            ),
            "method": audible.METHOD,
            **overall.to_json(),
        },
    )
    return 0


def cmd_repeat(args: argparse.Namespace) -> int:
    from .. import stability
    from ..resets import Prober, named

    setup = [
        [0xC0 | args.channel, args.program & 0x7F],
        [0xB0 | args.channel, 7, 127],
        [0xB0 | args.channel, 11, 127],
        [0xB0 | args.channel, 91, args.reverb & 0x7F],
        [0xB0 | args.channel, 93, args.chorus & 0x7F],
    ]
    store = _store(args.save)

    with verified_link(args, refusing="recording") as link:
        gs_reset = named("GS Reset", args.device_id)
        prober = Prober(link, baseline={}, device_id=args.device_id)
        prober.apply(gs_reset)
        cents = parts.master_tune_cents(link, device_id=args.device_id)

        for message in setup:
            link.send(message)
        time.sleep(0.3)

        print(
            f"\nprogram {args.program}, note {args.note}, velocity {args.velocity}, "
            f"reverb {args.reverb}, chorus {args.chorus}"
        )
        takes = []
        for index in range(args.takes):
            recording = perform.record_note(
                link,
                device=args.audio,
                channel=args.channel,
                note=args.note,
                velocity=args.velocity,
                hold=args.hold,
                seconds=args.seconds,
                lead=args.lead,
            )
            print(f"  take {index + 1}: {recording.health_report()}")
            if not recording.healthy:
                print("  the capture was lossy, so nothing measured from it would mean anything")
                return 1
            takes.append(recording)
            if store is not None:
                store.keep(
                    recording,
                    stimulus=f"p{args.program}n{args.note}",
                    setting=f"rev{args.reverb}cho{args.chorus}",
                    take=index,
                )
            time.sleep(args.between)

        index = perform.loudest_channel(takes[0])
        signals = [t.channel(index) for t in takes]
        rate = takes[0].sample_rate
        rise = stability.signal_over_silence(signals[0], rate, before=args.lead * 0.8)
        print(f"  channel {index} of {takes[0].device}: note {rise:.1f} dB over the lead-in")
        if not rise > args.min_rise:
            print(
                f"\nThe note never rose {args.min_rise} dB above the silence before it, so this "
                "take holds no sound from the unit. Name the right input with --audio before "
                "reading anything into a residual."
            )
            return 1
        if not _lead_in_ok([signals], rate, args.lead * 0.8, args.max_lead_in):
            return 1
        comparisons = [
            stability.compare(signals[0], s, rate, silence_before=args.lead * 0.8)
            for s in signals[1:]
        ]
        print()
        print(stability.summarise(comparisons, label=f"program {args.program} note {args.note}"))

        pitch = None
        if args.clock:
            expected = clock.equal_temperament(args.clock_note, cents)
            print(
                f"\nclock: note {args.clock_note}, {expected:.4f} Hz expected, "
                f"master tune {cents:+.1f} cents"
                if cents is not None
                else f"\nclock: note {args.clock_note}, {expected:.4f} Hz expected"
            )
            pitch = clock.measure(
                link,
                programs=args.clock_program,
                expected_hz=expected,
                note=args.clock_note,
                seconds=args.clock_seconds,
                channel=args.channel,
                device_id=args.device_id,
                audio=args.audio,
                progress=lambda m: print(f"  {m}"),
            )

        prober.apply(gs_reset)

    if store is not None:
        manifest = store.close(
            program=args.program,
            note=args.note,
            velocity=args.velocity,
            reverb_send=args.reverb,
            chorus_send=args.chorus,
            channel=args.channel,
        )
        print(f"\nkept {len(store.entries)} takes under {manifest.parent}")

    report.write_json(
        args.out,
        {
            "device_id": f"{args.device_id:02X}",
            "method": stability.METHOD,
            "program": args.program,
            "note": args.note,
            "velocity": args.velocity,
            "reverb_send": args.reverb,
            "chorus_send": args.chorus,
            "takes": args.takes,
            "sample_rate": takes[0].sample_rate,
            "comparisons": [c.to_json() for c in comparisons],
            "master_tune_cents": cents,
            "pitch": pitch.to_json() if pitch is not None else None,
        },
    )
    return 0
