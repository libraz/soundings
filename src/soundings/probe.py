"""Measuring a path by sending a known signal through it, rather than a note.

Everything else here asks the machine to play something and measures what comes
out. That reaches the effects only through a voice, and a voice is a poor probe:
it is harmonic, so it excites a few frequencies and leaves the rest unmeasured;
it decays, so the effect's own tail is entangled with the note's; and it starts
when a MIDI message arrives, which is milliseconds of scatter.

With a signal fed into the machine's analogue input, none of that applies. The
input is a known signal of the measurer's choosing, so the effect can be asked
about every frequency at once and its response recovered outright.

**A swept sine, not noise or an impulse.** An impulse puts all its energy in one
sample and needs the input driven far too hard to clear the noise; noise needs
averaging to get anywhere. A sine sweeping exponentially in frequency spreads the
same energy over seconds, and it does one thing neither of the others can: when
it is deconvolved, the linear response lands at one time and **each order of
harmonic distortion lands at its own earlier time**, separated rather than folded
into the result. That matters here because some of these effects are deliberately
nonlinear -- an overdrive, a distortion, a compressor -- and the difference
between "this is what the effect does" and "this is what the effect does plus
what it broke" is otherwise invisible.

**The sweep measures its own latency.** The distance from the start of the
deconvolved result to the linear peak is the round trip through the converters,
the machine and back, to a fraction of a sample. Nothing has to be triggered on a
clock, so the open-latency problem that anchors every note capture here does not
arise at all.

**A loopback pass is not optional.** The measured response is the machine's
effect *and* the converter pair's own response, in series. Running the same sweep
from output straight back to input measures the second on its own, and dividing
it out is what makes a result a fact about the machine. Without it, the
converters' anti-aliasing filters read as the effect rolling off the top octave.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Sweep:
    """An exponential sine sweep and the filter that collapses it to an impulse."""

    signal: np.ndarray
    inverse: np.ndarray
    sample_rate: int
    seconds: float
    low_hz: float
    high_hz: float
    pad: float
    """Silence after the sweep, for the response to decay into before the take ends."""

    @property
    def played(self) -> np.ndarray:
        """What actually goes out: the sweep followed by the silence."""
        return np.concatenate([self.signal, np.zeros(int(self.pad * self.sample_rate))])

    @property
    def octaves(self) -> float:
        return float(np.log2(self.high_hz / self.low_hz))

    def harmonic_offset(self, order: int) -> float:
        """Seconds before the linear peak that harmonic `order` arrives.

        An exponential sweep reaches twice a frequency a fixed time later
        whatever the frequency, so the second harmonic of every part of the sweep
        is early by the same amount. That constant is what separates the orders.
        """
        return self.seconds * np.log(order) / np.log(self.high_hz / self.low_hz)

    def to_json(self) -> dict:
        return {
            "seconds": self.seconds,
            "low_hz": self.low_hz,
            "high_hz": self.high_hz,
            "pad_s": self.pad,
            "sample_rate": self.sample_rate,
        }


def make_sweep(
    sample_rate: int,
    *,
    seconds: float = 3.0,
    low_hz: float = 20.0,
    high_hz: float = 20000.0,
    pad: float = 2.0,
    fade: float = 0.02,
    amplitude: float = 0.5,
) -> Sweep:
    """An exponential sweep from `low_hz` to `high_hz`, with its inverse filter.

    `amplitude` is half scale by default. The input stage is being asked about
    its linear response, and driving it to the rails measures its clipping
    instead -- which is worth measuring, but as a separate run at a stated level
    rather than by accident.
    """
    high_hz = min(high_hz, sample_rate / 2.0 * 0.98)
    n = int(seconds * sample_rate)
    t = np.arange(n) / sample_rate
    ratio = np.log(high_hz / low_hz)
    phase = 2.0 * np.pi * low_hz * seconds / ratio * (np.exp(t / seconds * ratio) - 1.0)
    signal = amplitude * np.sin(phase)

    # Without a fade the sweep starts and ends on a step, which rings across the
    # whole band and lands on top of the response being measured.
    edge = max(1, int(fade * sample_rate))
    window = np.hanning(2 * edge)
    signal[:edge] *= window[:edge]
    signal[-edge:] *= window[edge:]

    # Time-reverse and undo the sweep's own pink tilt: an exponential sweep
    # spends longer in each low octave than each high one, so the reversal alone
    # would deconvolve to an impulse with 3 dB per octave of slope on it.
    inverse = signal[::-1] * np.exp(-t / seconds * ratio)
    return Sweep(
        signal=signal,
        inverse=inverse,
        sample_rate=sample_rate,
        seconds=seconds,
        low_hz=low_hz,
        high_hz=high_hz,
        pad=pad,
    )


@dataclass
class Harmonic:
    order: int
    level_db: float
    """Peak of this order relative to the linear peak. -inf when nothing was there."""

    arrived_ms: float
    """How far before the linear peak it landed. Fixed by the sweep, not measured."""


@dataclass
class Response:
    """What a path did to the sweep: its impulse response and what it distorted."""

    impulse: np.ndarray
    """The whole deconvolved result, distortion products and all, in arrival order."""

    peak: int
    """Index of the linear response's arrival."""

    origin: int
    """Where a path that does nothing would put the peak.

    Not zero. Deconvolving a sweep of length T puts the linear response at T,
    because that is where the sweep and its time-reversed inverse line up. A
    latency read straight off the peak would be the sweep's own length plus the
    path's, which for a three second sweep is three seconds of nonsense.
    """

    sample_rate: int
    harmonics: list[Harmonic]
    noise_db: float
    """The floor the impulse sits on, from the stretch before any distortion arrives."""

    @property
    def latency_ms(self) -> float:
        """Round trip through the converters, the machine and back."""
        return (self.peak - self.origin) / self.sample_rate * 1000.0

    @property
    def linear(self) -> np.ndarray:
        """The linear impulse response alone, from its arrival onwards.

        This is the thing worth having: it can be handed to `decay` directly,
        which sidesteps the note entirely. A tail measured through a played note
        is the note convolved with the effect and falls at the slower of the two;
        an impulse response has no note in it.

        Deliberately from the peak, discarding what sits before it. A sweep
        covering 40 Hz to 16 kHz deconvolves to an impulse band-limited to that
        range, and a band-limited impulse is symmetric about its peak -- the
        earlier half is the band limit, not the path, and no decay begins in it.
        Reading a spectrum needs that half back; use `around` for that.
        """
        return self.impulse[self.peak :]

    def around(self, *, before: float = 0.02, after: float = 0.1) -> np.ndarray:
        """A window centred on the arrival, for work in the frequency domain.

        Cutting at the peak throws away the pre-ringing the sweep's own band
        limit puts there, and a spectrum taken of the remaining half is wrong by
        tens of dB at the bottom of the band, where the ringing is longest. It
        measured 21 dB at 100 Hz on a path whose answer was known.
        """
        lo = max(0, self.peak - int(before * self.sample_rate))
        hi = min(self.impulse.size, self.peak + int(after * self.sample_rate))
        return self.impulse[lo:hi]

    @property
    def distortion_db(self) -> float:
        """All harmonic orders together, relative to the linear response."""
        power = sum(10 ** (h.level_db / 10.0) for h in self.harmonics if np.isfinite(h.level_db))
        return 10.0 * np.log10(power) if power > 0 else float("-inf")

    @property
    def usable(self) -> bool:
        return bool(np.isfinite(self.noise_db) and self.noise_db < -20.0)

    def describe(self) -> str:
        lines = [
            f"  latency   {self.latency_ms:.3f} ms round trip",
            f"  floor     {self.noise_db:.1f} dB under the peak",
        ]
        if self.harmonics:
            worst = max(self.harmonics, key=lambda h: h.level_db)
            lines.append(
                f"  harmonics {self.distortion_db:.1f} dB total, worst is order "
                f"{worst.order} at {worst.level_db:.1f} dB"
            )
        else:
            lines.append("  harmonics none above the floor")
        if not self.usable:
            lines.append(
                "  => the impulse does not stand clear of the noise. Nothing measured "
                "from it means anything; raise the send level or lengthen the sweep"
            )
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "latency_ms": round(self.latency_ms, 4),
            "noise_db": round(self.noise_db, 2),
            "distortion_db": None
            if not np.isfinite(self.distortion_db)
            else round(self.distortion_db, 2),
            "harmonics": [
                {
                    "order": h.order,
                    "level_db": None if not np.isfinite(h.level_db) else round(h.level_db, 2),
                    "arrived_ms_before_linear": round(h.arrived_ms, 3),
                }
                for h in self.harmonics
            ],
            "usable": self.usable,
            "method": "An exponential sine sweep was played into the path and the recording "
            "deconvolved with the sweep's inverse filter. The linear response arrives at one "
            "time and each order of harmonic distortion at its own earlier time, so a "
            "nonlinear effect's response is separated from what it distorted rather than "
            "folded into it. The arrival of the linear peak is the round trip latency.",
        }


def deconvolve(recorded: np.ndarray, sweep: Sweep, *, orders: int = 5) -> Response:
    """Collapse a recording of the sweep back to the path's impulse response."""
    recorded = np.asarray(recorded, dtype=np.float64)
    size = 1 << int(np.ceil(np.log2(recorded.size + sweep.inverse.size)))
    impulse = np.fft.irfft(
        np.fft.rfft(recorded, size) * np.fft.rfft(sweep.inverse, size), size
    )[: recorded.size + sweep.inverse.size - 1]

    # Normalise so a path that does nothing gives a peak of one. The inverse
    # filter's own scaling depends on the sweep's length and range, so it is
    # measured here rather than derived.
    unity = np.fft.irfft(
        np.fft.rfft(sweep.signal, size) * np.fft.rfft(sweep.inverse, size), size
    )
    scale = float(np.abs(unity).max())
    origin = int(np.argmax(np.abs(unity)))
    if scale > 0:
        impulse = impulse / scale

    peak = int(np.argmax(np.abs(impulse)))
    linear_level = float(np.abs(impulse[peak]))
    if linear_level <= 0:
        return Response(
            impulse=impulse,
            peak=peak,
            origin=origin,
            sample_rate=sweep.sample_rate,
            harmonics=[],
            noise_db=float("nan"),
        )

    harmonics = []
    for order in range(2, orders + 1):
        offset = sweep.harmonic_offset(order)
        centre = peak - int(offset * sweep.sample_rate)
        # Half the way to each neighbouring order, so no window holds two of them.
        before = sweep.harmonic_offset(order + 1)
        after = sweep.harmonic_offset(order - 1) if order > 2 else 0.0
        lo = centre - int((before - offset) / 2.0 * sweep.sample_rate)
        hi = centre + int((offset - after) / 2.0 * sweep.sample_rate)
        lo, hi = max(0, lo), min(impulse.size, hi)
        if hi - lo < 8:
            continue
        level = float(np.abs(impulse[lo:hi]).max())
        harmonics.append(
            Harmonic(
                order=order,
                level_db=20.0 * np.log10(level / linear_level) if level > 0 else float("-inf"),
                arrived_ms=offset * 1000.0,
            )
        )

    # The floor is read before the highest order arrives, which is the only
    # stretch of the result holding neither the response nor its distortion.
    quiet = peak - int(sweep.harmonic_offset(orders + 1) * sweep.sample_rate)
    floor = impulse[: max(quiet, 1)]
    noise = float(np.sqrt(np.mean(np.square(floor)))) if floor.size else 0.0
    return Response(
        impulse=impulse,
        peak=peak,
        origin=origin,
        sample_rate=sweep.sample_rate,
        harmonics=harmonics,
        noise_db=20.0 * np.log10(noise / linear_level) if noise > 0 else float("-inf"),
    )


def divide_out(measured: Response, reference: Response, *, length: float = 1.0) -> np.ndarray:
    """Remove the measurement chain's own response from a measured one.

    `reference` is the loopback pass -- the same sweep sent from the output
    straight back to the input, with the machine not in the path at all. What
    comes back from that is the converters, their anti-aliasing filters and the
    cable, which are in series with everything else measured through them and
    would otherwise read as the effect rolling off the top octave.

    Regularised rather than divided outright: the loopback response is near zero
    outside the converters' passband, and dividing by it there amplifies nothing
    but noise into a result that looks like enormous ultrasonic gain.

    **The result is returned about its own peak, not from sample zero.** A ratio
    of two responses is not causal -- it rings both before and after -- and its
    low frequencies ring longest. Slicing from zero drops what wrapped to the far
    end of the transform, which is exactly that low-frequency energy: measured
    20 dB of error at 100 Hz on a filter whose answer was known, tapering to
    nothing by a few kHz, which reads as the machine having no bass.
    """
    lead = int(0.02 * measured.sample_rate)
    a = measured.around(before=0.02, after=length)
    b = reference.around(before=0.02, after=length)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    size = 1 << int(np.ceil(np.log2(n * 4)))
    numerator = np.fft.rfft(a, size)
    denominator = np.fft.rfft(b, size)
    power = np.abs(denominator) ** 2
    floor = power.max() * 1e-6 if power.max() > 0 else 1e-30
    whole = np.fft.irfft(numerator * np.conj(denominator) / np.maximum(power, floor), size)
    return np.roll(whole, lead - int(np.argmax(np.abs(whole))))[:n]


def octave_levels(
    impulse: np.ndarray,
    sample_rate: int,
    *,
    reference_hz: float = 1000.0,
) -> dict[float, float]:
    """Level per octave band, in dB relative to the band holding `reference_hz`.

    Relative rather than absolute, because the absolute level of a deconvolved
    impulse depends on the send level and the input trim, neither of which is a
    property of the path. What the path did to the *shape* survives that.
    """
    from .decay import OCTAVE_CENTRES, band_limit

    levels = {}
    for centre in OCTAVE_CENTRES:
        band = band_limit(np.asarray(impulse, dtype=np.float64), sample_rate, centre)
        levels[centre] = float(np.sqrt(np.mean(np.square(band))))
    anchor = min(levels, key=lambda c: abs(np.log(c / reference_hz)))
    base = levels[anchor]
    if base <= 0:
        return dict.fromkeys(levels, float("-inf"))
    return {c: 20.0 * np.log10(v / base) if v > 0 else float("-inf") for c, v in levels.items()}


def play_and_record(
    sweep: Sweep,
    *,
    device: str | None,
    output_channels: tuple[int, ...],
    input_channels: tuple[int, ...],
) -> np.ndarray:
    """Send the sweep and capture the return on one device, sample-locked.

    One device for both directions is what makes the latency a fixed number
    rather than an estimate: two devices run on two clocks and drift apart over
    the length of a sweep.
    """
    import sounddevice as sd

    from .capture import resolve_device

    index = resolve_device(device)
    played = sweep.played
    block = np.zeros((played.size, len(output_channels)))
    for column in range(len(output_channels)):
        block[:, column] = played

    recorded = sd.playrec(
        block,
        samplerate=sweep.sample_rate,
        device=index,
        input_mapping=list(input_channels),
        output_mapping=list(output_channels),
        blocking=True,
    )
    return np.asarray(recorded, dtype=np.float64)


__all__ = [
    "Harmonic",
    "Response",
    "Sweep",
    "deconvolve",
    "divide_out",
    "make_sweep",
    "play_and_record",
]
