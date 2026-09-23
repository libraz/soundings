"""Walk one address through a list of settings, recording a take at each.

`walk` writes the type and the routing, writes each `--prepare` state, then
steps `--slot` through `--settings`, writing and reading each one back before it
is recorded. Bypass and silence takes are recorded the same way, with
the effect taken out of the chain and left in it but silent.

Nothing here judges what it heard -- that is a reading stage's job, done later
from the takes this leaves on disk. `walk` only proves each write took before
recording what came after it, and records nothing it could not prove.
"""

from __future__ import annotations

import argparse
import time

from .. import perform, roland, stimuli
from ..resets import Prober, named
from ..takes import Store, unit_state
from . import options
from .session import prepared as prepare_state
from .session import verified_link

_STIMULUS_NAMES = tuple(stimuli.CATALOGUE)


def _settings(text: str) -> tuple[int, ...]:
    """'0,32,64,96,127' -- the bytes to walk, each written to --slot and read back."""
    values = tuple(int(v, 0) for v in text.split(","))
    if any(not 0 <= v <= 0x7F for v in values):
        raise argparse.ArgumentTypeError(f"a data byte is 0-127: {text!r}")
    return values


def register(sub) -> None:
    p = sub.add_parser(
        "walk",
        help="walk one address through a list of settings, writing and reading each "
        "one back before recording it",
    )
    options.add_subject(p, required=True, slot=True)
    options.add_channel(p)
    p.add_argument(
        "--settings",
        type=_settings,
        required=True,
        metavar="B,B,...",
        help="the bytes to walk, e.g. '0,32,64,96,127'. Spread across what the address "
        "accepts rather than its two ends, since a byte answering two settings tells "
        "the shape of nothing and this is the command that walks more than that",
    )
    p.add_argument(
        "--stimulus",
        required=True,
        metavar="NAME",
        help="the note this address is walked under, one of: " + ", ".join(_STIMULUS_NAMES),
    )
    p.add_argument(
        "--prepare",
        type=options.write_spec,
        action="append",
        default=[],
        metavar="ADDR=BYTES",
        help="further state --slot needs before it can be heard at all, written after "
        "the type and the routing and read back to prove it took: an address behind a "
        "second gate answers inaudible until that gate is open too, and that reads as "
        "a verdict on the address wearing the gate's name. Recorded with the result, "
        "since the verdict only holds in the state it was taken in",
    )
    p.add_argument("--takes", type=int, default=3, help="takes per setting")
    p.add_argument("--bypass", type=int, default=3, help="takes with the effect routed out")
    p.add_argument(
        "--silences",
        type=int,
        default=2,
        help="takes with the effect routed in and nothing played",
    )
    p.add_argument(
        "--settle", type=float, default=options.SETTLE_S, help="seconds after each write"
    )
    options.add_audio(p)
    options.add_verify_reads(p)
    p.add_argument("--save", required=True, help=options.SAVE_HELP)
    p.set_defaults(func=cmd_walk)


def cmd_walk(args: argparse.Namespace) -> int:
    """Write a type, its routing and any further state, then walk --slot and record."""
    try:
        (stim,) = stimuli.resolve([args.stimulus])
    except KeyError as exc:
        print(exc)
        return 1
    except ValueError:
        print(f"--stimulus names one voice to walk under, not a group: {args.stimulus!r}")
        return 1

    type_bytes = tuple(int(b, 16) for b in args.type.split())
    prepare = [("40 03 00", type_bytes), ("40 42 22", (0x01,)), *args.prepare]
    label = f"{args.type} at {args.slot}, {len(args.settings)} settings, {stim.name}"
    store = Store.open(args.save)

    def keep(recording, *, setting: str, index: int) -> None:
        store.keep(recording, stimulus=stim.name, setting=setting, take=index)
        manifest = store.close(
            **unit_state(prepare, settle_s=args.settle),
            address=args.slot,
            settings=list(args.settings),
            stimuli=[stim.to_json()],
            label=label,
        )
        print(f"    kept {len(store.entries)} takes under {manifest.parent}")

    def record(channel: int, notes, during) -> object | None:
        recording, _reopened = perform.record_notes_retrying(
            link,
            device=args.audio,
            channel=channel,
            notes=notes,
            during=during,
            seconds=stim.seconds,
            lead=stim.lead,
        )
        return recording if recording.healthy else None

    with verified_link(args, refusing="walking") as link:
        gs_reset = named("GS Reset", args.device_id)
        prober = Prober(link, baseline={}, device_id=args.device_id)
        prober.apply(gs_reset)

        if not prepare_state(link, prepare, device_id=args.device_id, settle=args.settle):
            return 1

        channel = stim.on(args.channel)
        for address, value in stim.writes:
            link.send(roland.dt1(address, [value], device_id=args.device_id))
        for message in (
            [0xC0 | channel, stim.program & 0x7F],
            [0xB0 | channel, 7, stim.volume],
            [0xB0 | channel, 11, 127],
            # The insertion effect alone: no system reverb or chorus on the takes.
            [0xB0 | channel, 91, 0],
            [0xB0 | channel, 93, 0],
        ):
            link.send(message)
        time.sleep(0.3)

        print(f"\n{label}")
        for byte in args.settings:
            if not prepare_state(
                link, [(args.slot, (byte,))], device_id=args.device_id, settle=args.settle
            ):
                return 1
            setting = f"v{byte:03d}"
            for index in range(args.takes):
                recording = record(channel, stim.played(), stim.during)
                if recording is None:
                    print(f"\n  lossy capture at {setting}, take {index}")
                    return 1
                keep(recording, setting=setting, index=index)

        if not prepare_state(
            link, [("40 42 22", (0x00,))], device_id=args.device_id, settle=args.settle
        ):
            return 1
        for index in range(args.bypass):
            recording = record(channel, stim.played(), stim.during)
            if recording is None:
                print(f"\n  lossy capture at out, take {index}")
                return 1
            keep(recording, setting="out", index=index)

        if not prepare_state(
            link, [("40 42 22", (0x01,))], device_id=args.device_id, settle=args.settle
        ):
            return 1
        for index in range(args.silences):
            recording = record(channel, (), ())
            if recording is None:
                print(f"\n  lossy capture at silence, take {index}")
                return 1
            keep(recording, setting="silence", index=index)

        prober.apply(gs_reset)

    return 0
