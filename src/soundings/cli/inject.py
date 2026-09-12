"""Send a signal into the machine's analogue input and measure what comes back.

The one command here needs an audio interface, and a MIDI port only when it is
asked to put the machine in a state first -- for the loopback pass, not even the
machine. It answers a different kind of question from everything else: not what
the machine plays, but what it does to a signal it is given.

Three runs, in order, and each is a precondition for reading the next:

1. `--loopback` with the cable going straight from the interface's output back to
   its input. That measures the converters, their filters and the cable, which
   are in series with everything else and would otherwise read as the machine.
2. Through the machine with no effect on the input. That answers whether the
   input reaches the output at all, at what latency and at what level. Until it
   does, nothing further is measurable.
3. Through the machine with an effect raised on the input, which is what
   `--prepare` is for. Comparing that with run 2 is what says whether the input
   is routed through the effects, which is the fact the whole approach depends on
   and which no manual page settles.

Run 3 is the one whose state has to be proved. Its two outcomes are a response
that differs from run 2 and a response that does not, and a preparation the
machine ignored produces the second while looking like the first was asked for.
"""

from __future__ import annotations

import argparse

from . import options, report


def register(sub) -> None:
    p = sub.add_parser(
        "transfer",
        help="play a swept sine into the machine's analogue input and measure what "
        "the path did to it, distortion separated from response",
    )
    options.add_audio(p)
    p.add_argument(
        "--out-channels",
        type=int,
        nargs="+",
        default=[3, 4],
        help="interface outputs the sweep is sent on, counting from 1",
    )
    p.add_argument(
        "--in-channels",
        type=int,
        nargs="+",
        default=[3, 4],
        help="interface inputs the return is captured on, counting from 1",
    )
    p.add_argument("--seconds", type=float, default=3.0, help="length of the sweep")
    p.add_argument("--low", type=float, default=20.0, help="lowest frequency swept")
    p.add_argument("--high", type=float, default=20000.0, help="highest frequency swept")
    p.add_argument(
        "--pad",
        type=float,
        default=2.0,
        help="silence recorded after the sweep, for the response to decay into. Too "
        "short and a reverb's tail wraps onto the start of its own impulse response",
    )
    p.add_argument(
        "--level",
        type=float,
        default=0.5,
        help="peak amplitude sent, 0 to 1. Half scale asks the input stage about its "
        "linear response; driving it to the rails measures its clipping instead, which "
        "is worth a run of its own at a stated level rather than by accident",
    )
    p.add_argument("--rate", type=int, default=48000, help="sample rate for both directions")
    p.add_argument(
        "--takes",
        type=int,
        default=1,
        help="sweeps to run. One sweep cannot say whether its own numbers repeat, and a "
        "disturbance that comes and goes then reads as a property of the path: measured "
        "here, a single sweep put one channel's distortion 30 dB away from where ten "
        "further sweeps put it, with nothing in the printout to say which was which",
    )
    p.add_argument(
        "--loopback",
        action="store_true",
        help="the cable goes straight from output back to input, with the machine not "
        "in the path. Says so in the result, so a reference cannot be mistaken for a "
        "measurement of the machine",
    )
    p.add_argument(
        "--reference",
        help="a WAV kept from a --loopback run, divided out of this one so what is "
        "left is the machine rather than the machine and the converters in series",
    )
    p.add_argument(
        "--prepare",
        type=options.write_spec,
        action="append",
        default=[],
        metavar="ADDR=BYTES",
        help="state to put the machine in before the sweep, written over MIDI and read "
        "back to prove it took: '40 03 00=02 01' selects an insertion effect. Recorded "
        "with the result, since a response is a measurement of the state it was taken in "
        "and nothing in the returned audio says which state that was",
    )
    p.add_argument(
        "--settle",
        type=float,
        default=0.4,
        help="seconds after each preparing write, before it is read back",
    )
    options.add_verify_reads(p)
    options.add_save(p)
    options.add_out(p)
    p.set_defaults(func=cmd_transfer)


def _sweep_for(args: argparse.Namespace):
    from .. import probe

    return probe.make_sweep(
        args.rate,
        seconds=args.seconds,
        low_hz=args.low,
        high_hz=args.high,
        pad=args.pad,
        amplitude=args.level,
    )


REPEATS_WITHIN_DB = 6.0
"""How far a channel's own numbers may sit apart across sweeps and still be read.

Measured on a working path: ten consecutive sweeps put the noise floor inside
3 dB of each other and the distortion inside 3 dB. A disturbance that came and
went put one channel's floor 38 dB apart across three sweeps and dragged its
distortion figure with it, always about 30 dB above whatever that sweep's floor
happened to be. The two cases are nowhere near each other, so the line between
them does not have to be a delicate one.
"""


PREPARED_CAVEAT = (
    "The sweep was run with the machine put in the state above, and each write was read back "
    "rather than assumed. A preparation that did not take leaves the machine as it was, so the "
    "sweep returns through a path with nothing raised on it -- which is exactly the result of a "
    "run made with no preparation at all, and reads as the input not reaching what was selected. "
    "That is the conclusion this command exists to reach, so it is the one an unverified state "
    "would manufacture."
)

NOT_WITH_LOOPBACK = (
    "--loopback takes the machine out of the path, so there is nothing there for --prepare to "
    "put in a state. A loopback run measures the converters and the cable, and a state written "
    "into a machine the signal never enters cannot change what comes back."
)


def cmd_transfer(args: argparse.Namespace) -> int:
    """Measure a path by sweeping it, and say what it did and what it broke."""
    import numpy as np

    from .. import probe, takes

    if args.prepare and args.loopback:
        raise SystemExit(NOT_WITH_LOOPBACK)

    sweep = _sweep_for(args)
    print(
        f"sweep {args.low:.0f} to {args.high:.0f} Hz over {args.seconds:.1f} s at "
        f"{args.level:.2f} full scale, out {args.out_channels} in {args.in_channels}"
        + (f", {args.takes} sweeps" if args.takes > 1 else "")
        + (", loopback (the machine is not in this path)" if args.loopback else "")
    )

    # Written before the sweep and not held open through it: the state stays in
    # the machine once the writes are acknowledged, and the port is one more thing
    # to have open while the interface is running.
    if args.prepare and not _put_in_state(args):
        return 1

    reference, reference_channel = None, None
    if args.reference:
        samples, rate = takes.read(args.reference)
        if rate != args.rate:
            raise SystemExit(f"the reference was captured at {rate} Hz, this run at {args.rate}")
        # Named rather than only taken. The sweep below is read from the channels
        # the invocation gives, so the one channel here that is chosen by loudness
        # is the only place this run could answer from an input nobody plugged
        # anything into, and a reference is divided out of every result.
        picked, reached = takes.channel_across(samples)
        reference_channel = {"read": picked, "reached_db": reached}
        reference = probe.deconvolve(takes.channel(samples, picked), sweep)

    store = takes.Store.open(args.save) if args.save else None
    captured = []
    for index in range(args.takes):
        recorded = probe.play_and_record(
            sweep,
            device=args.audio,
            output_channels=tuple(args.out_channels),
            input_channels=tuple(args.in_channels),
        )
        peak = float(np.abs(recorded).max())
        head = f"  sweep {index + 1}: " if args.takes > 1 else "  "
        if peak <= 0:
            print(f"{head}nothing came back.")
            print(
                "\n  Either the sweep is not reaching the input, or the wrong interface "
                "channels were named. A silent return is the same shape as a path that "
                "rejects the signal, and neither can be read as the other."
            )
            return 1
        print(f"{head}returned {recorded.shape[0]} frames, peak {20 * np.log10(peak):.1f} dBFS")
        captured.append(recorded)
        if store is not None:
            store.keep(
                _AsRecording(recorded, args.rate, args.audio or "default"),
                stimulus="sweep",
                setting="loopback" if args.loopback else "through",
                take=index,
            )

    results = [
        _read_channel(
            channel, [probe.deconvolve(r[:, column], sweep) for r in captured], reference, args
        )
        for column, channel in enumerate(args.in_channels)
    ]

    if store is not None:
        manifest = store.close(
            sweep=sweep.to_json(),
            level=args.level,
            out_channels=list(args.out_channels),
            in_channels=list(args.in_channels),
            loopback=args.loopback,
        )
        print(f"\nkept the return under {manifest.parent}")

    report.write_json(
        args.out,
        {
            "sweep": sweep.to_json(),
            "level": args.level,
            "takes": args.takes,
            "out_channels": list(args.out_channels),
            "in_channels": list(args.in_channels),
            "loopback": args.loopback,
            "reference": args.reference,
            **({"reference_channel": reference_channel} if reference_channel else {}),
            "prepared": [
                {"address": a, "bytes": " ".join(f"{v:02X}" for v in vs)} for a, vs in args.prepare
            ],
            **({"prepared_caveat": PREPARED_CAVEAT} if args.prepare else {}),
            "inputs": results,
        },
    )
    return 0


def _put_in_state(args: argparse.Namespace) -> bool:
    """Open the port, write the preparation, prove it took, and let the port go."""
    from .session import prepared, verified_link

    with verified_link(args, refusing="sweeping", show_port=True) as link:
        if not prepared(link, args.prepare, device_id=args.device_id, settle=args.settle):
            return False
    print(
        "prepared: "
        + ", ".join(f"{a}={' '.join(f'{v:02X}' for v in vs)}" for a, vs in args.prepare)
    )
    return True


def _read_channel(channel: int, found: list, reference, args: argparse.Namespace) -> dict:
    """Report one input, from however many sweeps were run through it.

    The reported response comes from the sweep whose floor is the median rather
    than from an average of the impulses: an average of a clean take and a
    disturbed one is a measurement of neither, while the median take is one the
    path actually produced.
    """
    import numpy as np

    from .. import probe

    floors = [f.noise_db for f in found]
    order = sorted(range(len(found)), key=lambda i: floors[i])
    middle = found[order[len(order) // 2]]
    spread = max(floors) - min(floors)

    print(f"\n  input {channel}:")
    if len(found) > 1:
        print(
            "    "
            + "  ".join(f"{f.noise_db:.1f}/{f.distortion_db:.1f}" for f in found)
            + "  floor/distortion per sweep, dB"
        )
        if spread > REPEATS_WITHIN_DB:
            print(
                f"    the floor moved {spread:.1f} dB across the sweeps, so nothing read "
                "off any one of them is a fact about the path. A distortion figure sits a "
                "fixed way above whatever floor its own sweep had, so it moved too."
            )
        else:
            print(f"    repeats: the floor sits inside {spread:.1f} dB across the sweeps")
    print(middle.describe())

    entry = {
        "channel": channel,
        **middle.to_json(),
        "takes": len(found),
        "noise_db_each": [round(f.noise_db, 2) for f in found],
        "distortion_db_each": [
            None if not np.isfinite(f.distortion_db) else round(f.distortion_db, 2) for f in found
        ],
        "floor_spread_db": round(spread, 2),
        "repeats": bool(spread <= REPEATS_WITHIN_DB),
    }
    if len(found) > 1 and spread > REPEATS_WITHIN_DB:
        entry["caveat"] = (
            "The floor moved more than this measurement can absorb across sweeps of the "
            "same path, so these numbers describe the disturbance rather than the path. "
            "A distortion figure tracks its own sweep's floor and moved with it."
        )

    if middle.usable:
        # `around`, not `linear`: a band-limited impulse is symmetric about
        # its peak and the half before it is the band limit rather than the
        # path, so cutting there costs the bottom of the band a spectrum it
        # never gets back. `divide_out` already windows that way.
        impulse = middle.around()
        if reference is not None and reference.usable:
            impulse = probe.divide_out(middle, reference, length=min(1.0, args.pad))
            print("    with the measurement chain divided out:")
        shape = probe.octave_levels(impulse, args.rate)
        print(
            "    "
            + "  ".join(f"{c:.0f}Hz {v:+.1f}" for c, v in shape.items() if np.isfinite(v))
            + "  dB relative to the 1 kHz band"
        )
        entry["octave_levels_db"] = {
            str(int(c)): None if not np.isfinite(v) else round(v, 2) for c, v in shape.items()
        }
        entry["chain_divided_out"] = reference is not None and reference.usable
    return entry


class _AsRecording:
    """The few fields `takes.Store` reads, for a return that was not a note capture."""

    def __init__(self, samples, sample_rate: int, device: str) -> None:
        self.samples = samples
        self.sample_rate = sample_rate
        self.device = device
        self.overflows = 0
        self.open_seconds = 0.0

    @property
    def seconds(self) -> float:
        return self.samples.shape[0] / self.sample_rate
