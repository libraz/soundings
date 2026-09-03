"""Send a signal into the machine's analogue input and measure what comes back.

The one command here needs an audio interface and nothing else -- no MIDI port,
and for the loopback pass not even the machine. It answers a different kind of
question from everything else: not what the machine plays, but what it does to a
signal it is given.

Three runs, in order, and each is a precondition for reading the next:

1. `--loopback` with the cable going straight from the interface's output back to
   its input. That measures the converters, their filters and the cable, which
   are in series with everything else and would otherwise read as the machine.
2. Through the machine with no effect on the input. That answers whether the
   input reaches the output at all, at what latency and at what level. Until it
   does, nothing further is measurable.
3. Through the machine with an effect raised on the input. Comparing that with
   run 2 is what says whether the input is routed through the effects, which is
   the fact the whole approach depends on and which no manual page settles.
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


def cmd_transfer(args: argparse.Namespace) -> int:
    """Measure a path by sweeping it, and say what it did and what it broke."""
    import numpy as np

    from .. import probe, takes

    sweep = _sweep_for(args)
    print(
        f"sweep {args.low:.0f} to {args.high:.0f} Hz over {args.seconds:.1f} s at "
        f"{args.level:.2f} full scale, out {args.out_channels} in {args.in_channels}"
        + (", loopback (the machine is not in this path)" if args.loopback else "")
    )

    recorded = probe.play_and_record(
        sweep,
        device=args.audio,
        output_channels=tuple(args.out_channels),
        input_channels=tuple(args.in_channels),
    )
    peak = float(np.abs(recorded).max())
    print(f"  returned {recorded.shape[0]} frames, peak {20 * np.log10(peak):.1f} dBFS")
    if peak <= 0:
        print(
            "\n  Nothing came back. Either the sweep is not reaching the input, or the "
            "wrong interface channels were named. A silent return is the same shape as "
            "a path that rejects the signal, and neither can be read as the other."
        )
        return 1

    reference = None
    if args.reference:
        samples, rate = takes.read(args.reference)
        if rate != args.rate:
            raise SystemExit(f"the reference was captured at {rate} Hz, this run at {args.rate}")
        reference = probe.deconvolve(takes.loudest(samples), sweep)

    results = []
    for column, channel in enumerate(args.in_channels):
        found = probe.deconvolve(recorded[:, column], sweep)
        print(f"\n  input {channel}:")
        print(found.describe())
        entry = {"channel": channel, **found.to_json()}

        if found.usable:
            impulse = found.linear
            if reference is not None and reference.usable:
                impulse = probe.divide_out(found, reference, length=min(1.0, args.pad))
                print("    with the measurement chain divided out:")
            shape = probe.octave_levels(impulse, args.rate)
            print(
                "    "
                + "  ".join(
                    f"{c:.0f}Hz {v:+.1f}" for c, v in shape.items() if np.isfinite(v)
                )
                + "  dB relative to the 1 kHz band"
            )
            entry["octave_levels_db"] = {
                str(int(c)): None if not np.isfinite(v) else round(v, 2)
                for c, v in shape.items()
            }
            entry["chain_divided_out"] = reference is not None and reference.usable
        results.append(entry)

    if args.save:
        store = takes.Store.open(args.save)
        store.keep(
            _AsRecording(recorded, args.rate, args.audio or "default"),
            stimulus="sweep",
            setting="loopback" if args.loopback else "through",
            take=0,
        )
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
            "out_channels": list(args.out_channels),
            "in_channels": list(args.in_channels),
            "loopback": args.loopback,
            "reference": args.reference,
            "inputs": results,
        },
    )
    return 0


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
