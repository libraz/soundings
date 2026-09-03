"""What an effect does over time, which one comparison of two takes cannot see.

`audible` answers whether a parameter changed the sound. That is a yes or no, and
identifying an algorithm needs the next question: what the effect is doing. For a
whole family of them -- chorus, flanger, phaser, tremolo, auto-pan, rotary -- the
answer is a motion, and a motion is invisible to any measurement that reduces a
take to one number. Two takes of a chorus differ enormously and agree on nothing,
which is exactly what `audible` reports, and it says nothing about the chorus.

The move is to stop treating the wet take as a signal and start treating it as a
*copy of the dry take that has been moved around*. Then the measurement is the
motion itself: how far the copy is delayed, at each moment.

**The direct path is subtracted first.** A send-return effect returns its output
added to the untouched part, so the wet take is dry plus effect. Correlating that
against the dry take finds the dry path sitting at zero, not the effect, and the
delay reads as nothing. Fitting and removing the best scalar copy of the dry take
leaves the return alone, which is what gets tracked. The two are separable
because a moving delay decorrelates its output from its input; a static effect
does not separate this way, and does not need to, having no motion to find.

**A track is built from independent frames, deliberately.** Following the peak
nearest the last one -- slew limiting, the obvious way to keep a track tidy --
turns noise into a smooth random walk, and a smooth random walk fits an
oscillator. Every frame here picks its own peak with no memory of the frame
before, so a track that comes out periodic came out periodic on the evidence.

**A periodic input makes the delay ambiguous by its own period.** A held note at
261 Hz repeats every 3.8 ms, so a delay of 3.8 ms correlates as well as no delay
at all, and a chorus swinging further than half a period wraps. This is measured
rather than assumed: the dry take's own autocorrelation says how periodic it is
and at what spacing, and the result carries both so a wrapped track is not read
as a shallow one.

**A null is a fact about the search.** No line in the delay track's spectrum
means no periodic motion was found within the rates and depths looked in, and
those bounds travel with the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LINE_ABOVE_DB = 10.0
"""How far a spectral line must stand above the rest of the searched band to count.

The track of a real modulator is nearly a sinusoid and stands tens of dB clear.
The failure this guards against is the opposite one: a delay track built from
frames that found nothing has a spectrum too, and its largest bin is a line by
construction. Ten dB over the median of the band is roughly where a track of pure
noise stops producing one.
"""

EXPLAINS_THE_SERIES = 0.7
"""How much of a track a sinusoid at the found rate has to account for.

`shape_error` is the RMS of what is left over divided by the fitted amplitude, so
a sinusoid whose own RMS is amplitude over root two accounts for more than it
leaves below about 0.7. Any oscillator shape passes: a triangle leaves 0.12, a
square 0.44, a sawtooth 0.62.

It is the gate the line test alone cannot be. A signal's level rattles far more
than a delay track does, and the largest bin of a rattle stands ten or fifteen dB
over the median of the band as readily as a tremolo does. Both a static reverb
and a chorus were reported as having a level modulation on that evidence, at
shape errors of 6.8 and 23.9 -- lines that a sinusoid explained essentially none
of.
"""

FRAME_CONFIDENCE = 0.2
"""Normalised correlation a frame must reach before its delay is believed.

Below this the frame is dropped rather than contributing a peak picked out of
noise. Frames are dropped, not interpolated, so the gap is visible.
"""


@dataclass
class DelayTrack:
    """How far the effect's return lags the dry signal, frame by frame."""

    times: np.ndarray
    """Frame start times in seconds, from the beginning of the compared region."""

    delay_samples: np.ndarray
    """Delay per frame; NaN where the frame did not clear `FRAME_CONFIDENCE`."""

    confidence: np.ndarray
    """Normalised correlation at each frame's peak."""

    sample_rate: int
    hop: float
    searched_ms: tuple[float, float]

    ambiguity_ms: float
    """Spacing of the dry signal's own repetition, or NaN if it does not repeat."""

    periodicity: float
    """How strongly the dry signal repeats, 0 to 1. Near 1 means the track can wrap."""

    align_samples: float
    """Trigger scatter removed before tracking; delays are relative to the dry path."""

    @property
    def frame_rate(self) -> float:
        return 1.0 / self.hop

    @property
    def usable(self) -> np.ndarray:
        return ~np.isnan(self.delay_samples)

    @property
    def coverage(self) -> float:
        """Fraction of frames that found anything. A low value makes any fit suspect."""
        return float(self.usable.mean()) if self.delay_samples.size else 0.0

    @property
    def excursion_ms(self) -> float:
        """Peak-to-peak swing of the track, before any oscillator is fitted."""
        found = self.delay_samples[self.usable]
        if found.size < 2:
            return float("nan")
        return float(found.max() - found.min()) / self.sample_rate * 1000.0

    @property
    def wraps(self) -> bool:
        """Whether the swing is large enough for the input's own period to fold it."""
        if np.isnan(self.ambiguity_ms) or np.isnan(self.excursion_ms):
            return False
        return bool(self.periodicity > 0.5 and self.excursion_ms > self.ambiguity_ms / 2.0)

    def to_json(self) -> dict:
        return {
            "frames": int(self.delay_samples.size),
            "frames_found": int(self.usable.sum()),
            "coverage": round(self.coverage, 3),
            "frame_rate_hz": round(self.frame_rate, 2),
            "searched_ms": [round(v, 2) for v in self.searched_ms],
            "excursion_ms": None if np.isnan(self.excursion_ms) else round(self.excursion_ms, 3),
            "input_repeats_every_ms": None
            if np.isnan(self.ambiguity_ms)
            else round(self.ambiguity_ms, 3),
            "input_periodicity": round(self.periodicity, 3),
            "may_have_wrapped": self.wraps,
            "trigger_scatter_ms": round(self.align_samples / self.sample_rate * 1000.0, 3),
        }


@dataclass
class LfoFit:
    """A periodic motion found in a track, and how far it is from being a sine."""

    rate_hz: float
    depth: float
    unit: str
    """What `depth` is peak-to-peak in: milliseconds of delay, or dB of level."""

    centre: float
    """The value the motion swings about, in the same unit."""

    line_above_db: float
    """How far the line stands above the rest of the searched band."""

    shape_error: float
    """RMS of what a sinusoid at this rate could not explain, over its amplitude.

    Near zero is a sine. A triangle leaves about 0.12 in its own harmonics, which
    is what separates the two without needing to see the waveform. Anything much
    larger means the line is real but the motion is not a simple oscillation.
    """

    resolution_hz: float
    """Bin spacing of the track's spectrum. The rate is not known better than this."""

    @property
    def sinusoidal(self) -> bool:
        return self.shape_error < 0.05

    def describe(self) -> str:
        shape = "sine" if self.sinusoidal else f"not a sine (shape error {self.shape_error:.2f})"
        return (
            f"{self.rate_hz:.3f} Hz +/- {self.resolution_hz:.3f}, "
            f"{self.depth:.3f} {self.unit} peak to peak about {self.centre:.3f}, "
            f"{shape}, line {self.line_above_db:.0f} dB clear"
        )

    def to_json(self) -> dict:
        return {
            "rate_hz": round(self.rate_hz, 4),
            "rate_resolution_hz": round(self.resolution_hz, 4),
            "depth_peak_to_peak": round(self.depth, 4),
            "depth_unit": self.unit,
            "centre": round(self.centre, 4),
            "line_above_band_db": round(self.line_above_db, 1),
            "shape_error": round(self.shape_error, 4),
            "sinusoidal": self.sinusoidal,
        }


def _normalised_lags(reference: np.ndarray, take: np.ndarray, span: int) -> np.ndarray:
    """Correlation of `take` against `reference` at every lag from 0 to `span`.

    `take` must be `span` samples longer than `reference`, so that the reference
    window overlaps it completely at every lag. A shorter one would overlap less
    and less as the lag grows, which biases the peak toward no delay at all --
    exactly the answer the measurement exists to disprove.
    """
    n = reference.size
    a = reference - reference.mean()
    b = take - take.mean()
    size = 1 << int(np.ceil(np.log2(b.size + n)))
    correlation = np.fft.irfft(np.fft.rfft(b, size) * np.conj(np.fft.rfft(a, size)), size)

    power = np.concatenate(([0.0], np.cumsum(b * b)))
    overlap = power[n : n + span + 1] - power[: span + 1]
    energy = float((a * a).sum())
    if energy <= 0:
        return np.zeros(span + 1)
    return correlation[: span + 1] / np.sqrt(energy * np.maximum(overlap, 1e-30))


def _peak(curve: np.ndarray, lo: int) -> tuple[float, float]:
    """Interpolated position and height of the largest value at or after `lo`."""
    if curve.size <= lo:
        return float("nan"), 0.0
    index = lo + int(np.argmax(curve[lo:]))
    height = float(curve[index])
    if index <= 0 or index >= curve.size - 1:
        return float(index), height
    left, centre, right = curve[index - 1], curve[index], curve[index + 1]
    denominator = left - 2.0 * centre + right
    offset = 0.0 if denominator == 0 else 0.5 * (left - right) / denominator
    return float(index) + float(offset), height


def periodicity(
    signal: np.ndarray, sample_rate: int, *, lowest_hz: float = 40.0
) -> tuple[float, float]:
    """Spacing and strength of a signal's own repetition, from its autocorrelation.

    This is what says whether a delay measured against this signal is unique. A
    struck note is strongly periodic and folds any delay into one period; a
    cymbal or a noise burst is not, and measures a delay outright.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size < sample_rate // 10:
        return float("nan"), 0.0
    a = signal - signal.mean()
    size = 1 << int(np.ceil(np.log2(2 * a.size)))
    spectrum = np.fft.rfft(a, size)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum), size)[: a.size]
    if correlation[0] <= 0:
        return float("nan"), 0.0
    correlation = correlation / correlation[0]

    limit = min(int(sample_rate / lowest_hz), correlation.size - 2)
    # Walk down the flank of the zero-lag peak first: it is the largest value
    # there is, and searching from lag zero would return it every time.
    start = 1
    while start < limit and correlation[start] < correlation[start - 1]:
        start += 1
    lag, height = _peak(correlation[: limit + 1], start)
    if np.isnan(lag) or height <= 0 or lag <= 0:
        return float("nan"), 0.0
    return lag / sample_rate * 1000.0, height


def track_delay(
    dry: np.ndarray,
    wet: np.ndarray,
    sample_rate: int,
    *,
    window: float = 0.010,
    hop: float = 0.005,
    search_ms: tuple[float, float] = (0.0, 60.0),
    subtract_direct: bool = True,
) -> DelayTrack:
    """Measure, frame by frame, how far the wet take's return lags the dry take.

    The two takes are first aligned as wholes, so the MIDI trigger scatter --
    milliseconds, far larger than a chorus depth -- is removed and every delay
    reported is the effect's. That alignment locks onto whatever dominates, which
    is the direct path when one is present; with an effect returned entirely wet
    it locks onto the effect's own mean delay instead, and the track is then a
    motion about an unknown centre rather than an absolute delay.

    **The window is short because the delay moves inside it.** A modulator
    sweeping a few milliseconds at a few hertz shifts its output by a fraction of
    a millisecond over any window long enough to be comfortable, and a delay that
    moves during a frame smears that frame's correlation peak until nothing
    clears the confidence threshold. Measured on a 4 ms swing at 1.3 Hz: an 85 ms
    window found 38 percent of frames, a 10 ms window found all of them. The
    price is correlation gain, so a noisier take or a slower modulator can afford
    a longer one.
    """
    dry = np.asarray(dry, dtype=np.float64)
    wet = np.asarray(wet, dtype=np.float64)
    length = min(dry.size, wet.size)
    dry, wet = dry[:length], wet[:length]

    from .stability import cross_correlate, shift

    align, _ = cross_correlate(dry, wet)
    wet = shift(wet, -align)

    guard = 256 + int(np.ceil(abs(align)))
    dry, wet = dry[guard:-guard], wet[guard:-guard]

    if subtract_direct:
        denominator = float(np.dot(dry, dry))
        gain = float(np.dot(wet, dry)) / denominator if denominator > 0 else 0.0
        target = wet - gain * dry
    else:
        target = wet

    n = max(16, int(window * sample_rate))
    step = max(1, int(hop * sample_rate))
    lo = int(search_ms[0] / 1000.0 * sample_rate)
    span = int(search_ms[1] / 1000.0 * sample_rate)

    times, delays, confidences = [], [], []
    start = 0
    while start + n + span <= dry.size:
        curve = _normalised_lags(dry[start : start + n], target[start : start + n + span], span)
        lag, height = _peak(curve, lo)
        times.append(start / sample_rate)
        confidences.append(height)
        delays.append(lag if height >= FRAME_CONFIDENCE else float("nan"))
        start += step

    spacing, strength = periodicity(dry, sample_rate)
    return DelayTrack(
        times=np.array(times),
        delay_samples=np.array(delays),
        confidence=np.array(confidences),
        sample_rate=sample_rate,
        hop=hop,
        searched_ms=search_ms,
        ambiguity_ms=spacing,
        periodicity=strength,
        align_samples=align,
    )


def _find_line(
    series: np.ndarray,
    frame_rate: float,
    rate_range: tuple[float, float],
) -> tuple[float, float, float] | None:
    """Rate, height over the band, and bin spacing of the strongest line in a series."""
    n = series.size
    if n < 8:
        return None
    series = series - series.mean()
    # A slow drift between takes is a ramp across the whole track, which would
    # otherwise pile into the lowest bins and outrank a real line.
    index = np.arange(n)
    slope = np.polyfit(index, series, 1)
    series = series - np.polyval(slope, index)
    if not np.any(series):
        return None

    spectrum = np.abs(np.fft.rfft(series * np.hanning(n)))
    bins = np.fft.rfftfreq(n, d=1.0 / frame_rate)
    # Nothing below three bins. A series with a bend in it -- a note's envelope
    # decaying and then flattening onto the noise, which no straight line
    # removes -- has all its leftover energy there, and it presents as a very
    # slow, very deep, very convincing oscillation. The cost is that a modulator
    # slower than three cycles across the take cannot be told from a trend, and
    # is not reported. That is a fact about the take's length.
    band = (bins >= max(rate_range[0], 3.0 * frame_rate / n)) & (bins <= rate_range[1])
    if band.sum() < 4:
        return None

    within = np.where(band, spectrum, 0.0)
    peak = int(np.argmax(within))
    background = float(np.median(spectrum[band]))
    if background <= 0 or spectrum[peak] <= 0:
        return None
    above = 20.0 * np.log10(spectrum[peak] / background)

    resolution = float(bins[1] - bins[0])
    if 0 < peak < spectrum.size - 1:
        left, centre, right = spectrum[peak - 1], spectrum[peak], spectrum[peak + 1]
        denominator = left - 2.0 * centre + right
        offset = 0.0 if denominator == 0 else 0.5 * (left - right) / denominator
    else:
        offset = 0.0
    return float(bins[peak]) + offset * resolution, above, resolution


def _fit_sinusoid(series: np.ndarray, times: np.ndarray, rate: float) -> tuple[float, float, float]:
    """Amplitude, centre and relative RMS of what a sinusoid at `rate` leaves behind."""
    design = np.column_stack(
        [np.ones(times.size), np.cos(2 * np.pi * rate * times), np.sin(2 * np.pi * rate * times)]
    )
    coefficients, *_ = np.linalg.lstsq(design, series, rcond=None)
    amplitude = float(np.hypot(coefficients[1], coefficients[2]))
    residual = series - design @ coefficients
    error = float(np.sqrt(np.mean(residual**2)) / amplitude) if amplitude > 0 else float("inf")
    return amplitude, float(coefficients[0]), error


def fit_lfo(
    track: DelayTrack,
    *,
    rate_range: tuple[float, float] = (0.05, 20.0),
    line_above_db: float = LINE_ABOVE_DB,
    min_coverage: float = 0.5,
) -> LfoFit | None:
    """Find a periodic motion in a delay track, or report that there was none.

    Frames that found nothing are carried across by interpolation rather than
    dropped, because dropping them would leave an unevenly sampled series that a
    spectrum cannot be taken of. `DelayTrack.coverage` is what says how much of
    the answer that interpolation supplied.
    """
    if track.coverage < min_coverage or track.delay_samples.size < 8:
        return None
    found = np.flatnonzero(track.usable)
    # Only the interior is carried across. Extrapolating past the first and last
    # frame that found anything would fill the take's silent lead-in with a flat
    # stretch as long as the lead-in is, which is not a motion the effect made
    # and which the sinusoid then has to account for.
    first, last = found[0], found[-1]
    times = track.times[first : last + 1]
    delays = track.delay_samples[first : last + 1]
    inside = ~np.isnan(delays)
    if inside.sum() < 8:
        return None
    series = np.interp(times, times[inside], delays[inside])

    line = _find_line(series, track.frame_rate, rate_range)
    if line is None:
        return None
    rate, above, resolution = line
    if above < line_above_db or rate <= 0:
        return None

    amplitude, centre, error = _fit_sinusoid(series - series.mean(), times, rate)
    if error > EXPLAINS_THE_SERIES:
        return None
    to_ms = 1000.0 / track.sample_rate
    return LfoFit(
        rate_hz=rate,
        depth=2.0 * amplitude * to_ms,
        unit="ms",
        centre=(float(np.mean(series)) + centre) * to_ms,
        line_above_db=above,
        shape_error=error,
        resolution_hz=resolution,
    )


def level_lfo(
    signal: np.ndarray,
    sample_rate: int,
    *,
    hop: float = 0.005,
    rate_range: tuple[float, float] = (0.05, 20.0),
    line_above_db: float = LINE_ABOVE_DB,
) -> LfoFit | None:
    """Find a periodic motion in a signal's level, for effects that move no delay.

    Tremolo, auto-pan and the amplitude half of a rotary speaker leave the timing
    alone, so a delay track sees nothing and reports it honestly. This looks in
    the other place. The envelope is taken in dB, where a note's own decay is a
    straight line and comes out in the detrending; on a linear envelope the decay
    is a curve that no straight line removes, and its remains outrank the motion.
    """
    signal = np.asarray(signal, dtype=np.float64)
    step = max(1, int(hop * sample_rate))
    frames = signal.size // step
    if frames < 8:
        return None
    blocks = signal[: frames * step].reshape(frames, step)
    power = np.mean(blocks**2, axis=1)
    alive = power > power.max() * 1e-8
    if alive.sum() < 8:
        return None
    envelope = 10.0 * np.log10(np.maximum(power, power.max() * 1e-8))
    times = np.arange(frames) * hop

    line = _find_line(envelope[alive], 1.0 / hop, rate_range)
    if line is None:
        return None
    rate, above, resolution = line
    if above < line_above_db or rate <= 0:
        return None

    amplitude, centre, error = _fit_sinusoid(
        envelope[alive] - envelope[alive].mean(), times[alive], rate
    )
    if error > EXPLAINS_THE_SERIES:
        return None
    return LfoFit(
        rate_hz=rate,
        depth=2.0 * amplitude,
        unit="dB",
        centre=float(np.mean(envelope[alive])) + centre,
        line_above_db=above,
        shape_error=error,
        resolution_hz=resolution,
    )


@dataclass
class Motion:
    """Everything found about how one effect moves, including that it does not."""

    track: DelayTrack
    delay: LfoFit | None
    level: LfoFit | None

    @property
    def delay_answered(self) -> bool:
        """Whether anything this delay track says is worth reading.

        Not once it has wrapped. A swing wider than half the input's own period
        correlates as well at the wrong lag as the right one, and the damage runs
        both ways: the line can vanish, which reads as an effect standing still,
        or one can survive at the wrong place. Measured on a chorus folded into a
        3.8 ms period, the tracker returned 1.80 Hz for a true 0.90 and 46 ms of
        depth for a true 12, and the shape gate passed it. A fit is no safer here
        than a null, so both are withheld.
        """
        return not self.track.wraps

    @property
    def moves(self) -> bool:
        return (self.delay is not None and self.delay_answered) or self.level is not None

    @property
    def answered(self) -> bool:
        """Whether "nothing moves" is something this pair of takes can support."""
        return self.moves or self.delay_answered

    def describe(self) -> str:
        lines = [
            f"track: {self.track.usable.sum()}/{self.track.delay_samples.size} frames found "
            f"({self.track.coverage * 100:.0f}%), swing {self.track.excursion_ms:.2f} ms"
        ]
        if self.track.wraps:
            lines.append(
                f"  the input repeats every {self.track.ambiguity_ms:.2f} ms and the swing "
                "is wider than half of that, so the delay may have folded"
            )
        if self.delay is not None and self.delay_answered:
            lines.append(f"  delay: {self.delay.describe()}")
        elif self.delay is not None:
            # Shown rather than dropped: the number exists, and a reader who is
            # told it was withheld can go and get a source that would confirm it.
            lines.append(f"  delay: withheld -- the wrapped track fits {self.delay.describe()}")
        elif self.delay_answered:
            lines.append("  delay: no periodic motion")
        else:
            lines.append("  delay: not asked -- the track wrapped, so nothing it says holds")
        lines.append(
            f"  level: {self.level.describe()}" if self.level else "  level: no periodic motion"
        )
        if self.moves:
            pass
        elif self.answered:
            lines.append(
                "  => nothing moves. An effect that is audible and does not move here has "
                "no free-running modulator, which is what makes its response measurable "
                "by averaging"
            )
        else:
            lines.append(
                "  => no answer. The delay could not be read at all, so this is not the "
                "effect standing still; it is these takes being unable to say. A source "
                "with no period of its own is what would answer it"
            )
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "track": self.track.to_json(),
            "delay_lfo": self.delay.to_json() if self.delay else None,
            "level_lfo": self.level.to_json() if self.level else None,
            "moves": self.moves,
            "delay_answered": self.delay_answered,
            "conclusive": self.answered,
            "method": "The wet take is aligned to the dry take as a whole, the best scalar "
            "copy of the dry take is subtracted to leave the effect's return alone, and the "
            "return's delay against the dry signal is measured in independent frames. A "
            "periodic motion is a line in that track's spectrum standing clear of the rest "
            "of the searched band. Level is searched the same way, for effects that move no "
            "delay. A null names the rates and delays that were looked in, and is withheld "
            "for the delay when the input's own period folded the track, since a search that "
            "could not have found a line reports the same emptiness as one that looked.",
        }


def measure(
    dry: np.ndarray,
    wet: np.ndarray,
    sample_rate: int,
    *,
    search_ms: tuple[float, float] = (0.0, 60.0),
    rate_range: tuple[float, float] = (0.05, 20.0),
) -> Motion:
    """Track an effect's motion in both delay and level, and report both nulls."""
    track = track_delay(dry, wet, sample_rate, search_ms=search_ms)
    return Motion(
        track=track,
        delay=fit_lfo(track, rate_range=rate_range),
        level=level_lfo(wet, sample_rate, rate_range=rate_range),
    )


__all__ = [
    "EXPLAINS_THE_SERIES",
    "FRAME_CONFIDENCE",
    "LINE_ABOVE_DB",
    "DelayTrack",
    "LfoFit",
    "Motion",
    "fit_lfo",
    "level_lfo",
    "measure",
    "periodicity",
    "track_delay",
]
