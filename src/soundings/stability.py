"""Whether two takes of the same thing can be compared at all.

Identifying an algorithm from its input and output rests on a step nothing has
established yet: that playing the same thing twice produces the same samples
twice. Averaging to get under the noise floor needs it, and so does subtracting
one configuration from another to see what a parameter changed. If takes do not
repeat, every such difference is dominated by whatever failed to repeat, and the
result reads as an effect.

Three things stand between two takes and a subtraction, and each is measured
here rather than assumed.

**They do not start at the same sample.** The note is triggered over MIDI from a
general-purpose machine, so the trigger scatters by milliseconds -- hundreds of
samples, far more than any effect being looked for. Takes are therefore aligned
by cross-correlation before being compared, to a fraction of a sample, since a
half-sample offset alone leaves a residual that rises with frequency and would
be read as a treble difference.

**They need not be at the same level**, and a gain difference of a tenth of a dB
leaves a residual 39 dB down that has nothing to do with the signal. The
best-fitting scalar gain is divided out and then reported, so a level difference
shows up as itself instead of hiding inside the residual.

**The residual cannot beat the noise.** Two takes each carrying independent
noise differ by that noise even if the unit is perfectly deterministic, so a
residual is only meaningful next to the floor it could not have gone below. The
floor is measured from the silence before the note in the same take, and what is
reported is the distance between the two. Zero distance is the strongest claim
this chain can support: not "the unit repeats exactly" but "the unit repeats to
everything this chain can see".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Comparison:
    """One take measured against a reference take of the same stimulus."""

    delay_samples: float
    """How much later the take is than the reference. Fractional; MIDI trigger scatter."""

    correlation: float
    """Normalised peak, 1.0 for the same waveform. Low means the alignment is meaningless."""

    gain_db: float
    """Best-fitting level difference, take relative to reference."""

    residual_db: float
    """What is left after aligning and levelling, relative to the reference. Lower is more alike."""

    floor_db: float
    """What two independent takes of a perfectly repeatable unit would leave, from the silence."""

    sample_rate: int

    @property
    def headroom_db(self) -> float:
        """Residual above the floor. 0 means as repeatable as this chain can see."""
        return self.residual_db - self.floor_db

    @property
    def unrepeatable_db(self) -> float:
        """The residual with the noise taken back out, relative to the signal.

        The residual holds two independent things added in power: the noise,
        which would be there for a perfectly deterministic unit, and whatever
        actually failed to repeat. Subtracting the first leaves the second, which
        is the only part that is a fact about the machine. -inf means the
        residual was entirely accounted for by noise.
        """
        excess = 10 ** (self.headroom_db / 10.0) - 1.0
        if excess <= 0:
            return float("-inf")
        return self.floor_db + 10.0 * np.log10(excess)

    @property
    def delay_ms(self) -> float:
        return self.delay_samples / self.sample_rate * 1000.0

    def to_json(self) -> dict:
        return {
            "delay_samples": round(self.delay_samples, 3),
            "correlation": round(self.correlation, 6),
            "gain_db": round(self.gain_db, 4),
            "residual_db": round(self.residual_db, 2),
            "noise_floor_db": round(self.floor_db, 2),
            "residual_above_floor_db": round(self.headroom_db, 2),
            "unrepeatable_db": None
            if self.unrepeatable_db == float("-inf")
            else round(self.unrepeatable_db, 2),
        }


def cross_correlate(reference: np.ndarray, take: np.ndarray) -> tuple[float, float]:
    """Return how many samples later `take` is than `reference`, and the normalised peak.

    The lag is interpolated between samples from the curvature at the peak. An
    integer lag is not enough: a half-sample error is a phase ramp across the
    band, which subtracts as a rising treble residual and reads as a filter
    difference rather than as a timing error.
    """
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(take, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    size = 1 << int(np.ceil(np.log2(a.size + b.size)))
    spectrum = np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size))
    correlation = np.fft.irfft(spectrum, size)

    peak = int(np.argmax(correlation))
    energy = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if energy <= 0:
        return 0.0, 0.0

    # The correlation is circular, so a peak in the upper half is a negative lag.
    left = correlation[(peak - 1) % size]
    right = correlation[(peak + 1) % size]
    centre = correlation[peak]
    denominator = left - 2.0 * centre + right
    offset = 0.0 if denominator == 0 else 0.5 * (left - right) / denominator
    lag = peak - size if peak > size // 2 else peak
    return -(lag + offset), float(centre / energy)


def shift(signal: np.ndarray, samples: float) -> np.ndarray:
    """Delay a signal by a fractional number of samples, band-limited.

    Circular, so the caller has to discard a guard band at both ends; every user
    here does. Interpolating in the time domain instead would low-pass the copy
    being subtracted, which is exactly the kind of difference being looked for.
    """
    n = signal.size
    spectrum = np.fft.rfft(signal)
    bins = np.fft.rfftfreq(n, d=1.0)
    return np.fft.irfft(spectrum * np.exp(-2j * np.pi * bins * samples), n=n)


def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal)))) if signal.size else 0.0


def _db(value: float, reference: float) -> float:
    if reference <= 0:
        return float("nan")
    if value <= 0:
        return -np.inf
    return 20.0 * np.log10(value / reference)


def noise_floor(signal: np.ndarray, sample_rate: int, *, before: float) -> float:
    """RMS of the lead-in silence, as dB relative to the whole signal's RMS.

    Measured in the same take rather than from a separate silent recording, so
    it carries whatever the unit and the converter were doing at the time.
    """
    lead = signal[: int(before * sample_rate)]
    if lead.size < sample_rate // 100:
        return float("nan")
    return _db(_rms(lead), _rms(signal))


def loudest_lead_in(signals: list[np.ndarray], sample_rate: int, *, before: float) -> float:
    """The loudest lead-in among a set of takes, in dBFS.

    The lead-in is what the noise floor is measured from, and everything here is
    judged against that floor. Anything sounding during it -- the tail of the
    take before, or another process driving the same unit -- raises it, which
    raises the yardstick, which makes a real difference read as noise. None of
    the other checks can see that: the capture is the right length, dropped no
    samples, and the note still rises clear of a lead-in that is merely louder
    than it should be.

    **Absolute, not relative to the take.** Contamination raises the lead-in in
    absolute terms. A quiet stimulus lowers the note instead and leaves the
    lead-in sitting on the converter's own floor, which is not a fault and must
    not be refused: measured against the take, a velocity 30 note reads the same
    as a contaminated one.
    """
    levels = []
    for signal in signals:
        lead = signal[: int(before * sample_rate)]
        if lead.size < sample_rate // 100:
            continue
        levels.append(_db(_rms(lead), 1.0))
    return max(levels) if levels else float("nan")


def signal_over_silence(signal: np.ndarray, sample_rate: int, *, before: float) -> float:
    """How far the take rises above its own lead-in, in dB.

    A capture from the wrong input device is silent, healthy and the right
    length, and every take of it is as unlike every other as noise is. That
    presents as the unit failing to repeat rather than as nothing being
    recorded, so it has to be excluded before a residual is believed.
    """
    lead = int(before * sample_rate)
    quiet, loud = signal[:lead], signal[lead:]
    if quiet.size < sample_rate // 100 or loud.size == 0:
        return float("nan")
    return _db(_rms(loud), _rms(quiet))


def compare(
    reference: np.ndarray,
    take: np.ndarray,
    sample_rate: int,
    *,
    silence_before: float = 0.3,
    guard: int = 256,
) -> Comparison:
    """Align a take to a reference, level it, subtract, and report what is left."""
    reference = np.asarray(reference, dtype=np.float64)
    take = np.asarray(take, dtype=np.float64)
    length = min(reference.size, take.size)
    reference, take = reference[:length], take[:length]

    delay, correlation = cross_correlate(reference, take)
    aligned = shift(take, -delay)

    edge = guard + int(np.ceil(abs(delay)))
    a, b = reference[edge:-edge], aligned[edge:-edge]

    denominator = float(np.dot(b, b))
    # The factor that scales the take up to the reference; the take's own level
    # relative to the reference is its reciprocal, which is what gets reported.
    gain = float(np.dot(a, b)) / denominator if denominator > 0 else 0.0
    residual = a - gain * b

    return Comparison(
        delay_samples=delay,
        correlation=correlation,
        gain_db=-_db(gain, 1.0) if gain > 0 else float("-inf"),
        residual_db=_db(_rms(residual), _rms(a)),
        # Two takes each carry this noise independently, so their difference
        # holds twice its power -- 3 dB more than one take's floor.
        floor_db=noise_floor(reference, sample_rate, before=silence_before) + 3.01,
        sample_rate=sample_rate,
    )


@dataclass
class ToneFit:
    frequency: float
    wobble_cycles: float
    """RMS departure of the phase from a straight line. A steady tone sits near zero.

    This is what separates a frequency from a number. Vibrato, detuned layers
    beating against each other and a drifting oscillator all still yield a
    confident-looking slope, and only the departure from the line says the tone
    was never steady enough for that slope to mean a frequency.
    """

    seconds: float

    @property
    def steady(self) -> bool:
        return self.wobble_cycles < 0.05

    def to_json(self) -> dict:
        return {
            "frequency_hz": round(self.frequency, 6),
            "phase_wobble_cycles": round(self.wobble_cycles, 5),
            "seconds_fitted": round(self.seconds, 3),
            "steady": self.steady,
        }


def tone_frequency(
    signal: np.ndarray,
    sample_rate: int,
    *,
    expected: float,
    span: float = 0.15,
    skip: float = 0.5,
    trim: float = 0.2,
) -> ToneFit | None:
    """Frequency of a sustained tone, from the slope of its unwrapped phase.

    Precise enough to read the clock. A peak-bin estimate resolves the sample
    rate over the length of the window; a phase slope resolves it over the number
    of cycles, which for half a minute of a mid-range note is parts per million.

    The band around `expected` is isolated first, because the phase of a signal
    with partials in it is not the phase of its fundamental. Returns None when
    nothing is there to measure.
    """
    signal = np.asarray(signal, dtype=np.float64)
    start = int(skip * sample_rate)
    stop = signal.size - int(trim * sample_rate)
    if stop - start < sample_rate:
        return None
    window = signal[start:stop]

    n = window.size
    spectrum = np.fft.rfft(window)
    bins = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    band = (bins > expected * (1 - span)) & (bins < expected * (1 + span))
    if not band.any() or not np.abs(spectrum[band]).any():
        return None

    # Zero everything outside the band and take the analytic signal of what is
    # left, so the unwrapped phase belongs to the fundamental alone.
    analytic = np.zeros(n, dtype=complex)
    analytic[: spectrum.size] = np.where(band, spectrum, 0.0) * 2.0
    narrow = np.fft.ifft(analytic)

    magnitude = np.abs(narrow)
    if magnitude.max() <= 0:
        return None
    # Fit only where the tone is actually present; a decayed tail is noise phase.
    strong = magnitude > magnitude.max() * 0.3
    index = np.flatnonzero(strong)
    if index.size < sample_rate // 2:
        return None
    first, last = index[0], index[-1]

    phase = np.unwrap(np.angle(narrow[first : last + 1]))
    times = np.arange(phase.size) / sample_rate
    slope, intercept = np.polyfit(times, phase, 1)
    wobble = float(np.sqrt(np.mean(np.square(phase - (slope * times + intercept)))))
    return ToneFit(
        frequency=float(slope) / (2.0 * np.pi),
        wobble_cycles=wobble / (2.0 * np.pi),
        seconds=float(phase.size) / sample_rate,
    )


def summarise(comparisons: list[Comparison], *, label: str) -> str:
    if not comparisons:
        return f"{label}: nothing to compare"
    residuals = [c.residual_db for c in comparisons]
    headrooms = [c.headroom_db for c in comparisons]
    delays = [c.delay_ms for c in comparisons]
    lines = [
        f"{label}: {len(comparisons)} takes against the first",
        f"  residual   {min(residuals):.1f} to {max(residuals):.1f} dB below the signal",
        f"  floor      {comparisons[0].floor_db:.1f} dB (two takes' worth of noise)",
        f"  above it   {min(headrooms):.1f} to {max(headrooms):.1f} dB",
        f"  trigger    {min(delays):+.2f} to {max(delays):+.2f} ms of scatter",
        f"  level      {min(c.gain_db for c in comparisons):+.3f} to "
        f"{max(c.gain_db for c in comparisons):+.3f} dB",
        f"  alignment  correlation {min(c.correlation for c in comparisons):.4f} at worst",
    ]
    worst = max(headrooms)
    apart = max(c.unrepeatable_db for c in comparisons)
    if worst < 1.0:
        lines.append("  => repeats to everything this chain can see; takes can be averaged")
    elif worst < 10.0:
        lines.append(
            f"  => what does not repeat is {apart:.0f} dB down, at the noise; averaging works"
        )
    elif worst < 25.0:
        lines.append(f"  => what does not repeat is {apart:.0f} dB down, above the noise")
    else:
        lines.append(f"  => does not repeat; {apart:.0f} dB down is most of the signal")
    return "\n".join(lines)


__all__ = [
    "Comparison",
    "ToneFit",
    "compare",
    "cross_correlate",
    "noise_floor",
    "shift",
    "summarise",
    "tone_frequency",
]
