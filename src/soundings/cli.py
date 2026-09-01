"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import archive, clock, parts, perform, roland
from . import capture as cap
from .midi import MidiLink, list_ports
from .selftest import midi_selftest, timeline_selftest
from .stimuli import CATALOGUE as _STIMULUS_CATALOGUE
from .stimuli import DEFAULT as _STIMULUS_DEFAULT

_STIMULUS_NAMES = tuple(_STIMULUS_CATALOGUE)

_SAVE_HELP = (
    "directory to keep every take in, as float WAV with a manifest. Device time is the "
    "scarce thing here and a verdict thrown away with its audio has to be re-recorded to "
    "be asked anything else; kept takes can be measured again with the machine unplugged"
)

_LEAD_IN_HELP = (
    "dBFS the lead-in must stay under; above this something was sounding before the note and "
    "the noise floor, which is the yardstick for everything else, is wrong. Absolute rather "
    "than relative to the take, or a quiet stimulus reads the same as a contaminated one"
)


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
        report = midi_selftest(link, repeats=args.repeats, device_id=args.device_id)
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
        request = roland.rq1(args.address, args.size, device_id=args.device_id)
        reply = roland.parse_dt1(link.exchange(request, timeout=args.timeout))
        if reply is None:
            print(f"{args.address}  size={args.size}  -> no reply")
            return 1
        data = " ".join(f"{b:02X}" for b in reply.data)
        flag = "" if reply.checksum_ok else "  [CHECKSUM MISMATCH]"
        print(f"{args.address}  requested {args.size}, returned {reply.size}: {data}{flag}")
    return 0


def cmd_identity(args: argparse.Namespace) -> int:
    with MidiLink(args.port) as link:
        raw = link.exchange(roland.IDENTITY_REQUEST, timeout=1.0)
        if not raw:
            print("no reply")
            return 1
        print(" ".join(f"{b:02X}" for b in raw))
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    from .sweep import MidiCalibrationError, Sweeper, calibrate, summarise

    with MidiLink(args.port) as link:
        print(f"MIDI: {link.ports.output_name}")
        print("Verifying the path before sweeping (a dropped byte becomes a missing row)")
        report = midi_selftest(link, repeats=args.verify_reads, device_id=args.device_id)
        print(report)
        if not report.passed:
            print("\nSelftest failed. Not sweeping.")
            return 1

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

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.to_json(), indent=2) + "\n")
        print(f"\nwrote {path}")
    return 0 if result.complete and not result.lagged else 1


def cmd_write_probe(args: argparse.Namespace) -> int:
    from .writeback import RestoreFailed, Writer, summarise

    regions = [
        (tuple(int(b, 16) for b in spec.split(":")[0].split()), int(spec.split(":")[1], 0))
        for spec in args.regions
    ]
    with MidiLink(args.port) as link:
        print(f"MIDI: {link.ports.output_name}")
        print("Verifying the path before writing (a write is never acknowledged)")
        report = midi_selftest(link, repeats=args.verify_reads, device_id=args.device_id)
        print(report)
        if not report.passed:
            print("\nSelftest failed. Not writing.")
            return 1

        writer = Writer(link, device_id=args.device_id, settle=args.settle)
        done = []
        try:
            for start, length in regions:
                done.append(writer.probe_region(start, length, progress=lambda m: print(f"  {m}")))
        except RestoreFailed as exc:
            print(f"\nSTOPPED: {exc}")
            return 1

    print()
    print(summarise(done))
    print(f"  {writer.writes} writes, {writer.reads} reads")

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "device_id": f"{args.device_id:02X}",
                    "settle_s": args.settle,
                    "note": "Classifications describe what an address stores, not what it does. "
                    "Nothing here was heard.",
                    "regions": [r.to_json() for r in done],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {path}")
    return 0 if all(r.region_restored for r in done) else 1


def cmd_reset_probe(args: argparse.Namespace) -> int:
    from .aliases import Snapshotter
    from .resets import Prober, ResetResult, catalogue, compare, mode_set, named, summarise

    captured = json.loads(Path(args.baseline).read_text())
    baseline = {
        tuple(int(b, 16) for b in a.split()): int(v, 16) for a, v in captured["values"].items()
    }
    regions = archive.regions(args.map)
    targets = [
        tuple(int(b, 16) for b in a.split())
        for a in list(archive.ALIASED_BYTES) + archive.accepting_bytes(args.write_probe)
    ]
    resets = catalogue(args.device_id)
    if args.include_mode_set:
        resets.append(mode_set(args.device_id))

    print(f"baseline: {args.baseline}, {len(baseline)} bytes")
    print(f"{len(targets)} addresses to mark, {len(regions)} regions read after each reset")

    results = []
    with MidiLink(args.port) as link:
        report = midi_selftest(link, repeats=args.verify_reads, device_id=args.device_id)
        print(report)
        if not report.passed:
            print("\nSelftest failed. Not resetting.")
            return 1

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
            marked, refused = prober.mark(targets)
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
        best = min(results, key=lambda r: (len(r.differs_from_power_on), -len(r.restored)))
        link.send(next(x for x in resets if x.label == best.label).message)
        time.sleep(args.settle)
        print(f"\nleft the unit on {best.label}")

    print()
    print(summarise(results))

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "device_id": f"{args.device_id:02X}",
                    "baseline": args.baseline,
                    "method": "Each reset was preceded by writing a mark into every address the "
                    "write probe found accepts any value, so that a byte the reset leaves alone "
                    "reads as the mark rather than as its default. Only bytes read back as "
                    "holding the mark are counted.",
                    "each_preceded_by": args.from_reset,
                    "why_preceded": "So the three are comparable with each other rather than each "
                    "being read against wherever the previous one left the unit. This same probe "
                    "measured that reset as reproducing the power-on capture byte for byte, which "
                    "is what makes it usable as a starting line.",
                    "order": [r.label for r in results],
                    "left_on": best.label,
                    "resets": [r.to_json() for r in results],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {path}")
    return 0


def cmd_tone_map(args: argparse.Namespace) -> int:
    from .resets import Prober, named
    from .tonemap import SAMPLE_PROGRAMS, Asker, summarise, survey

    with MidiLink(args.port) as link:
        report = midi_selftest(link, repeats=args.verify_reads, device_id=args.device_id)
        print(report)
        if not report.passed:
            print("\nSelftest failed. Not asking.")
            return 1

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

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "device_id": f"{args.device_id:02X}",
                    "channel": args.channel + 1,
                    "map_select": args.map_select,
                    "method": "Each tone was asked for by sending its bank select and a program "
                    "change, then reading the part's own tone bytes back. The unit discards a "
                    "combination it does not have and leaves the part where it was, so a part "
                    "that moved to what was asked for is the tone existing. The part is moved "
                    "away first whenever it already stands on what is about to be asked.",
                    "sampled_before_sweeping": None if args.exhaustive else list(SAMPLE_PROGRAMS),
                    "sampling_caveat": (
                        "Every bank was asked for all 128 programs, so a bank absent here "
                        "answered none of them."
                        if args.exhaustive
                        else "A bank that answered none of the sampled programs was not swept "
                        "and is absent here. A bank whose only tones sit between them would "
                        "read as empty."
                    ),
                    "requests": asker.asks,
                    "reads_unusable": asker.unread,
                    "banks": [b.to_json() for b in found],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {path}")
    return 0 if not asker.unread else 1


def _addresses(args: argparse.Namespace) -> list[tuple[int, int, int]]:
    from .writeback import NEVER_WRITE

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
    from . import aliases as al

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


def _store(where: str | None):
    """Open a directory for the takes, when the run was asked to keep them."""
    if not where:
        return None
    from .takes import Store

    return Store.open(where)


def _lead_in_ok(groups, rate: float, before: float, limit: float) -> bool:
    """Refuse a run whose lead-in was not silent, naming why it matters."""
    from . import stability

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


def cmd_contrast(args: argparse.Namespace) -> int:
    """Play the same note under two settings and say whether the unit sounded different."""
    from . import audible, stability, stimuli
    from .resets import Prober, named

    def setting(value: int) -> list[list[int]]:
        if args.cc is not None:
            return [[0xB0 | args.channel, args.cc & 0x7F, value & 0x7F]]
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

    with MidiLink(args.port) as link:
        report = midi_selftest(link, repeats=args.verify_reads, device_id=args.device_id)
        print(report)
        if not report.passed:
            print("\nSelftest failed. Not recording.")
            return 1

        gs_reset = named("GS Reset", args.device_id)
        prober = Prober(link, baseline={}, device_id=args.device_id)
        prober.apply(gs_reset)

        print(f"\n{label}\n  {len(asked)} stimulus/stimuli: {', '.join(s.name for s in asked)}")
        for stim in asked:
            # Reset between stimuli, so a setting left by the previous one cannot
            # follow the parameter into the next and be read as part of it.
            prober.apply(gs_reset)
            for message in (
                [0xC0 | args.channel, stim.program & 0x7F],
                [0xB0 | args.channel, 7, 127],
                [0xB0 | args.channel, 11, 127],
            ):
                link.send(message)
            time.sleep(0.3)
            print(f"\n  {stim.name}: {stim.describe()}")

            captured: list[list] = []
            for value in args.values:
                for message in setting(value):
                    link.send(message)
                time.sleep(args.settle)
                takes = []
                for index in range(args.takes):
                    recording = perform.record_note(
                        link,
                        device=args.audio,
                        channel=args.channel,
                        note=stim.note,
                        velocity=stim.velocity,
                        hold=stim.hold,
                        seconds=stim.seconds,
                        lead=stim.lead,
                    )
                    if not recording.healthy:
                        print(f"    lossy capture: {recording.health_report()}")
                        return 1
                    takes.append(recording)
                    if store is not None:
                        store.keep(recording, stimulus=stim.name, setting=str(value), take=index)
                    time.sleep(args.between)
                print(f"    {where} = {value}: {len(takes)} takes")
                captured.append(takes)

            index = perform.loudest_channel(captured[0][0])
            rate = captured[0][0].sample_rate
            groups = [[t.channel(index) for t in takes] for takes in captured]
            before = stim.lead * 0.8
            rise = stability.signal_over_silence(groups[0][0], rate, before=before)
            print(f"    channel {index}: note {rise:.1f} dB over the lead-in")
            if not rise > args.min_rise:
                print(
                    f"\n    The note never rose {args.min_rise} dB above the silence before "
                    "it, so these takes hold no sound from the unit. Name the right input "
                    "with --audio."
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
            stimuli=[stim.to_json() for stim in asked],
        )
        print(f"\nkept {len(store.entries)} takes under {manifest.parent}")

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "device_id": f"{args.device_id:02X}",
                    "controller": args.cc,
                    "address": None if args.cc is not None else args.address,
                    "values": list(args.values),
                    "stimuli": [stim.to_json() for stim in asked],
                    "method": "The same note was played several times under each setting. Takes "
                    "of one setting are compared with each other to measure what the unit fails "
                    "to repeat, and takes of the two settings are compared the same way to "
                    "measure the change. The second must clear the first by the margin, so a "
                    "unit that repeats badly cannot be read as a parameter that does something. "
                    "Level is judged separately, because the alignment divides out the best "
                    "fitting gain and a parameter that only changes level would otherwise "
                    "leave no trace. A parameter is asked under one or more named stimuli, "
                    "because a null is a fact about the note as much as about the parameter, "
                    "and the answer over a set of them is a union.",
                    **overall.to_json(),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {path}")
    return 0


def _bank_program(spec: str) -> tuple[int, int]:
    """'80' or '8:80' -- a bare number is the GM bank, which is bank 0."""
    bank, _, program = spec.rpartition(":")
    return (int(bank or 0, 0), int(program, 0))


def cmd_repeat(args: argparse.Namespace) -> int:
    from . import stability
    from .resets import Prober, named

    setup = [
        [0xC0 | args.channel, args.program & 0x7F],
        [0xB0 | args.channel, 7, 127],
        [0xB0 | args.channel, 11, 127],
        [0xB0 | args.channel, 91, args.reverb & 0x7F],
        [0xB0 | args.channel, 93, args.chorus & 0x7F],
    ]
    store = _store(args.save)

    with MidiLink(args.port) as link:
        report = midi_selftest(link, repeats=args.verify_reads, device_id=args.device_id)
        print(report)
        if not report.passed:
            print("\nSelftest failed. Not recording.")
            return 1

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

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "device_id": f"{args.device_id:02X}",
                    "method": "The same note was played and captured several times, and every "
                    "take after the first was aligned to it by cross correlation to a fraction "
                    "of a sample, scaled by its best fitting level, and subtracted. What is "
                    "left is reported next to the noise floor of the silence before the note, "
                    "raised 3 dB because two takes carry that noise independently. A residual "
                    "at the floor is the strongest claim this chain supports: not that the "
                    "unit repeats exactly, but that it repeats to everything the chain can see.",
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
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {path}")
    return 0


def _pair(dry_path: str, wet_path: str):
    """Load two takes and hand back the loudest channel of each, plus the rate."""
    from .takes import loudest, read

    dry, dry_rate = read(dry_path)
    wet, wet_rate = read(wet_path)
    if dry_rate != wet_rate:
        raise SystemExit(f"the two takes were captured at {dry_rate} and {wet_rate} Hz")
    return loudest(dry), loudest(wet), dry_rate


def cmd_motion(args: argparse.Namespace) -> int:
    """Say what an effect does over time, from a dry and a wet take of the same note."""
    from . import motion

    dry, wet, rate = _pair(args.dry, args.wet)
    found = motion.measure(
        dry,
        wet,
        rate,
        search_ms=(0.0, args.max_delay),
        rate_range=(args.min_rate, args.max_rate),
    )
    print(f"{args.dry} against {args.wet}, {rate} Hz")
    print(found.describe())

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "dry": str(args.dry),
                    "wet": str(args.wet),
                    "sample_rate": rate,
                    "searched_rate_hz": [args.min_rate, args.max_rate],
                    **found.to_json(),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {path}")
    return 0


def cmd_decay(args: argparse.Namespace) -> int:
    """Say how long an effect's tail takes to die, per octave band."""
    from . import decay as dec

    dry, wet, rate = _pair(args.dry, args.wet)
    tail, noise = dec.isolate_tail(dry, wet, rate, lead=args.lead)
    found = dec.measure(tail, rate, noise=noise)
    print(f"{args.dry} against {args.wet}, {rate} Hz")
    print(found.describe())

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "dry": str(args.dry),
                    "wet": str(args.wet),
                    "sample_rate": rate,
                    "lead_s": args.lead,
                    **found.to_json(),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {path}")
    return 0


def cmd_alias_scan(args: argparse.Namespace) -> int:
    from .aliases import Scanner, Snapshotter, cc, control_change, control_run, summarise

    regions = archive.regions(args.map, args.prefix)
    # A run that finds nothing cannot say whether the unit stores nothing or the
    # scan was broken, so one control change known to be stored is always sent.
    # Sent at both ends rather than once: a control that passes at the start says
    # the reading worked at the start, and a run has been seen to miss something
    # after that point.
    control = cc(args.channel, args.control_cc)
    with MidiLink(args.port) as link:
        print(f"MIDI: {link.ports.output_name}")
        report = midi_selftest(link, repeats=args.verify_reads, device_id=args.device_id)
        print(report)
        if not report.passed:
            print("\nSelftest failed. Not scanning.")
            return 1

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

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "device_id": f"{args.device_id:02X}",
                    "channel": args.channel + 1,
                    "kind": args.kind,
                    "positive_control": {
                        "controller": args.control_cc,
                        "sent": 2,
                        "detected": control_passes,
                        "at": control_hit.addresses if control_hit else [],
                        "why": "Sent before the first stimulus and after the last. Without it a "
                        "scan that finds nothing cannot be told from a scan that cannot find "
                        "anything; sent only once, it says nothing about the rest of the run.",
                    },
                    "left_changed_afterwards": {
                        "by_stimulus": scanner.residue,
                        "why": "Every stimulus ends on the value it started with, so anything "
                        "listed here is state one message carried into the next.",
                    },
                    "missed_by_the_first_pass": {
                        "stimuli": scanner.recovered,
                        "why": "Each stimulus that lands nothing is retried with its two values "
                        "swapped. Anything listed here is something a single pass would have "
                        "reported as absent.",
                    },
                    "kind_reached": {
                        "value": reached,
                        "why": "The control is a control change, so it cannot show that a "
                        "message of another kind arrived. Where this is false, every negative "
                        "in the run is about the path, not about the unit.",
                    },
                    "method": "Each stimulus was sent at its low value, snapshotted, sent at its "
                    "high value, snapshotted, and sent at its low value again. A byte is listed "
                    "only if it moved both times, to a different value each time.",
                    "region_prefix": args.prefix,
                    "regions_watched": len(regions),
                    "stimuli_sent": [s.label for s in stimuli],
                    "restless_addresses": sorted(restless),
                    "region_reads_failed": shot.unread,
                    "not_scanned": "Controllers 120 to 127 are channel mode messages. "
                    "Sending one resets the channel state every later attribution is measured "
                    "against, so they need a scan of their own.",
                    "rpn_parked": "RPN and NRPN were set to 7F 7F before the scan.",
                    "bank_latch_on_exit": latch,
                    "written_bytes_restored": restored,
                    "note": "A byte listed here followed the stimulus out and back. That says "
                    "where the value is kept, not that anything uses it.",
                    "attributed": [a.to_json() for a in found],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {path}")
    return 0 if bracketed and reached else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="soundings", description=__doc__)
    parser.add_argument("--port", help="substring of the MIDI port name")
    parser.add_argument("--device-id", type=lambda s: int(s, 0), default=roland.DEFAULT_DEVICE_ID)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("devices", help="list MIDI and audio devices").set_defaults(func=cmd_devices)

    p = sub.add_parser("selftest", help="prove the measurement path before trusting it")
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--ticks", type=int, default=10)
    p.add_argument("--audio", help="substring of the audio input device name")
    p.add_argument("--no-audio", action="store_true", help="MIDI checks only")
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser("read", help="read one address with RQ1")
    p.add_argument("address", help="three hex bytes, e.g. '40 01 30'")
    p.add_argument("size", type=lambda s: int(s, 0), nargs="?", default=1)
    p.add_argument("--timeout", type=float, default=1.0)
    p.set_defaults(func=cmd_read)

    sub.add_parser("identity", help="send an Identity Request").set_defaults(func=cmd_identity)

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
    p.add_argument(
        "--verify-reads",
        type=int,
        default=30,
        help="repeat reads in the selftest that gates the sweep",
    )
    p.add_argument(
        "--ceiling",
        type=lambda s: int(s, 0),
        default=64,
        help="largest size requested when bounding a region; a request far larger "
        "than any real region asks the machine for something no file would",
    )
    p.add_argument(
        "--low",
        default="00,30",
        help="comma-separated third bytes to try under each pair; a region need not begin at 00",
    )
    p.add_argument("--top-from", type=lambda s: int(s, 0), default=0x00)
    p.add_argument(
        "--top-to",
        type=lambda s: int(s, 0),
        default=None,
        help="restrict the top-byte range; default is the whole space",
    )
    p.add_argument("--out", help="write the result as JSON")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser(
        "write-probe",
        help="find what an address accepts by writing to it and reading it back",
    )
    p.add_argument(
        "regions",
        nargs="+",
        metavar="ADDR:COUNT",
        help="'40 01 30:24' probes 24 consecutive bytes from 40 01 30",
    )
    p.add_argument(
        "--settle",
        type=float,
        default=0.02,
        help="pause between a write and the read back that checks it",
    )
    p.add_argument("--verify-reads", type=int, default=20)
    p.add_argument("--out", help="write the result as JSON")
    p.set_defaults(func=cmd_write_probe)

    p = sub.add_parser(
        "tone-map",
        help="find which tones exist, by asking for each and seeing whether it was taken",
    )
    p.add_argument("--channel", type=int, default=0, help="zero based MIDI channel")
    p.add_argument(
        "--map-select",
        type=lambda s: int(s, 0),
        default=0,
        help="the CC32 value held for the whole survey",
    )
    p.add_argument(
        "--exhaustive",
        action="store_true",
        help="ask every bank for all 128 programs instead of sampling first; three times "
        "the cost and the only form with nothing to caveat",
    )
    p.add_argument("--banks-from", type=lambda s: int(s, 0), default=0)
    p.add_argument("--banks-to", type=lambda s: int(s, 0), default=127)
    p.add_argument(
        "--settle",
        type=float,
        default=0.04,
        help="pause after a program change before reading the part back; the tone was "
        "measured to reach the part block in a median 12.6 ms",
    )
    p.add_argument("--verify-reads", type=int, default=20)
    p.add_argument("--out", help="write the result as JSON")
    p.set_defaults(func=cmd_tone_map)

    p = sub.add_parser(
        "reset-probe",
        help="find what each reset restores, by breaking the state first",
    )
    p.add_argument(
        "--baseline",
        default="data/units/roland-sc8850-01/power-on-state.json",
        help="the power-on capture every reset is compared against",
    )
    p.add_argument("--map", default="data/units/roland-sc8850-01/address-map.json")
    p.add_argument(
        "--write-probe",
        default="data/units/roland-sc8850-01/write-probe.json",
        help="where the addresses that take any value are read from",
    )
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
    p.add_argument("--verify-reads", type=int, default=20)
    p.add_argument("--out", help="write the result as JSON")
    p.set_defaults(func=cmd_reset_probe)

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
    p.add_argument("--channel", type=int, default=0, help="zero based MIDI channel")
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
        type=lambda s: tuple(int(v, 0) for v in s.split(",")),
        default=(0x20, 0x60),
        help="for --kind address: the two values written; both must be inside what the "
        "address accepts, or a clamp answers them identically and reads as storing nothing",
    )
    p.add_argument(
        "--note",
        type=lambda s: int(s, 0),
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
    p.add_argument("--verify-reads", type=int, default=20)
    p.add_argument("--out", help="write the result as JSON")
    p.set_defaults(func=cmd_alias_scan)

    p = sub.add_parser(
        "repeat",
        help="find whether two takes of the same note can be compared, which every "
        "difference measurement rests on",
    )
    p.add_argument("--channel", type=int, default=0, help="zero based MIDI channel")
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
        type=_bank_program,
        nargs="+",
        default=[(0, 16), (0, 19), (0, 73), (0, 79), (0, 80), (8, 80), (8, 81)],
        help="programs to hold; most voices are not a frequency reference, so several "
        "are tried and the ones that wobble are reported as wobbling rather than read",
    )
    p.add_argument("--clock-note", type=int, default=69)
    p.add_argument("--clock-seconds", type=float, default=20.0)
    p.add_argument("--audio", help="substring of the audio input device name")
    p.add_argument(
        "--min-rise",
        type=float,
        default=12.0,
        help="dB the note must rise above the silence before it; below this the take holds "
        "no sound from the unit, which would otherwise read as the unit not repeating",
    )
    p.add_argument("--max-lead-in", type=float, default=-60.0, help=_LEAD_IN_HELP)
    p.add_argument("--verify-reads", type=int, default=20)
    p.add_argument("--save", help=_SAVE_HELP)
    p.add_argument("--out", help="write the result as JSON")
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
        type=lambda s: tuple(int(v, 0) for v in s.split(",")),
        default=(0, 127),
        help="the two settings compared; both must be inside what the parameter accepts, "
        "or a clamp answers them identically and reads as inaudible",
    )
    p.add_argument("--channel", type=int, default=0, help="zero based MIDI channel")
    p.add_argument(
        "--stimulus",
        nargs="+",
        default=list(_STIMULUS_DEFAULT),
        metavar="NAME",
        help="notes to ask the parameter under: "
        + ", ".join(_STIMULUS_NAMES)
        + ", plus 'broad' for the five that answer most parameters and 'all'. Audible under "
        "any is audible; a null carries the list of what was tried, because an inaudible "
        "result is as much a fact about the note as about the parameter",
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
    p.add_argument("--audio", help="substring of the audio input device name")
    p.add_argument("--min-rise", type=float, default=12.0)
    p.add_argument("--max-lead-in", type=float, default=-60.0, help=_LEAD_IN_HELP)
    p.add_argument("--verify-reads", type=int, default=20)
    p.add_argument("--save", help=_SAVE_HELP)
    p.add_argument("--out", help="write the result as JSON")
    p.set_defaults(func=cmd_contrast)

    p = sub.add_parser(
        "motion",
        help="say what an effect does over time -- its modulation rate, depth and "
        "shape -- from a dry and a wet take, with no machine attached",
    )
    p.add_argument("dry", help="a take with the effect off")
    p.add_argument("wet", help="the same note with the effect on")
    p.add_argument(
        "--max-delay",
        type=float,
        default=60.0,
        help="milliseconds of delay searched. A null is a fact about this range",
    )
    p.add_argument("--min-rate", type=float, default=0.05, help="slowest modulation searched, Hz")
    p.add_argument("--max-rate", type=float, default=20.0, help="fastest modulation searched, Hz")
    p.add_argument("--out", help="write the result as JSON")
    p.set_defaults(func=cmd_motion)

    p = sub.add_parser(
        "decay",
        help="say how long an effect's tail takes to die in each octave band, from a "
        "dry and a wet take, with no machine attached",
    )
    p.add_argument("dry", help="a take with the effect off")
    p.add_argument("wet", help="the same note with the effect on")
    p.add_argument(
        "--lead",
        type=float,
        default=0.5,
        help="seconds of silence at the head of the take; the per-band noise floor is "
        "measured in it, and it is what says where a tail stops being a tail",
    )
    p.add_argument("--out", help="write the result as JSON")
    p.set_defaults(func=cmd_decay)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
