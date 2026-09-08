"""How long a tail takes to die, per band, for the effects that have no motion.

`motion` covers the effects that move. The other half of the family -- reverb,
delay, gated and early-reflection programs -- hold still, and holding still is
what makes them measurable a different way: an effect with no free-running
modulator is a fixed system, so its return is a function of its input alone and
its decay is a property rather than a moment.

The decay is measured per octave band because that is where the design lives. A
reverb whose bands all decay together is a delay network with flat feedback; one
whose top decays faster has damping in the loop, and how much faster says how
much. A single broadband decay time averages that away into a number no design
can be recovered from.

**Backward integration, not the envelope.** A decaying tail is noisy: its
instantaneous level rattles ten dB about the trend, and a line fitted through it
lands wherever the rattle happened to fall. Integrating the energy from the end
backwards gives the curve the rattle would average to, which is what the fit
actually wants. The cost is that the integration accumulates the noise floor too,
and a curve dominated by accumulated noise flattens into a shelf that fits as a
very long decay -- so the tail is truncated where it reaches the floor, and the
floor's own contribution is removed before integrating.

**A decay time means nothing without its straightness.** A tail that is two
decays in sequence, or one that ran into the noise, still yields a slope and
therefore still yields a number. What separates those from a real decay is how
far the curve departed from the line, so that is measured and reported next to
it, and a band that could not be fitted says so instead of returning a time.

**The note has to die faster than the room, or the room is not what is measured.**
Isolating the return by subtracting the dry take gives the effect's output over
the whole take rather than only after note-off, which is the point of doing it --
but that output is the note convolved with the effect, and a convolution of two
decays falls at the slower of the two. A one second reverb on a note whose own
tail runs to 1.7 seconds measures 1.7 seconds, correctly and uselessly. The
stimulus for a decay is therefore the shortest note the machine has, and a result
close to the note's own decay is a result about the note.

**A residual has two floors and the lead-in only measures one of them.** The
silence at the head of the take says how much interface noise is under the
return. It says nothing about how much of the return is the subtraction's own
error, which lives under the note rather than before it and is the larger of the
two by far. Two takes of one setting subtracted from each other measure it
directly, and that is the yardstick every other measurement here is judged
against. Without it, a type that did nothing at all still produces eight decay
times: measured on this unit, one type's return stood 24.9 dB over the lead-in
noise at 63 Hz and 6.9 dB over the subtraction floor, and reported an 11 second
decay in a two second tail. A genuine return on the same run cleared that floor
by thirty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OCTAVE_CENTRES = (63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)

STRAIGHT_ENOUGH_DB = 1.5
"""RMS departure from the fitted line, above which the slope is not a decay time.

The number a curved fit returns is not wrong so much as meaningless: a tail made
of two decays in sequence has no single time, and one that ran into the noise
bends into a shelf. Both fit a line and both come back curved.
"""

CLEARS_THE_FLOOR_DB = 6.0
"""How far a band's return has to stand over the subtraction floor to be a return.

The same margin every comparison here takes, for the same reason: what two takes
of one setting differ by is what a difference between two settings has to beat.
"""

NOT_OVER_THE_FLOOR = (
    "the return did not stand clear of what subtracting two takes of one setting leaves"
)

NO_FLOOR_TAKEN = (
    "No pair of takes of one setting was given, so the subtraction floor was not measured and "
    "no band could be judged against it. The lead-in silence bounds the interface noise only; "
    "the error the subtraction itself leaves lives under the note, is the larger of the two, "
    "and is what a type that did nothing at all produces a full set of decay times out of."
)


@dataclass
class BandDecay:
    """One octave band's decay, and whether it was a decay at all."""

    centre_hz: float
    seconds: float
    """Time to fall 60 dB, extrapolated from the range that was fitted."""

    fitted_db: tuple[float, float]
    """The range of the integrated curve the line was fitted over."""

    curvature_db: float
    """RMS departure of the curve from the line. Above `STRAIGHT_ENOUGH_DB`, not a decay."""

    snr_db: float
    """How far the band's start sat above the noise. Bounds how far the fit could reach."""

    scatter: float = float("nan")
    """Upper bound on the relative spread of this time, from the band's own width.

    A decay time is estimated from a finite number of independent samples, and a
    narrow band over a short fit holds few: the count is the bandwidth times the
    fitted duration, and the spread goes as the inverse square root of it.

    **It bounds the spread, not any one reading.** Over 24 realisations of a known
    1.2 s decay the observed spread per band ran between half and two thirds of
    this figure -- 14 percent at 63 Hz down to 1 percent at 8 kHz -- so a single
    band landing outside it is ordinary. The bottom octave also sits a few percent
    low, its filter ringing being a real fraction of a short fit.

    It matters because the bands are compared with each other. A top octave
    reading 20 percent faster than the bottom is damping at 8 kHz and noise at
    63 Hz, and nothing but this number says which.
    """

    halves_s: tuple[float, float] | None = None
    """The same band fitted over the first half of the drop and over the second.

    One decay time is one number standing for a whole fall, and the fall is not
    always one slope. `curvature_db` already says how far the curve departed from
    the line, but a departure in dB is not a quantity anybody downstream can use:
    it bounds the error without naming what the error is. These are the two rates
    the departure is made of, fitted over -5..-15 and over -15..-25 dB.

    Measured across the bands this unit has: on the ones fitted and published,
    the two rates part company by more than the fit's own scatter in three out of
    five, at a median ratio of 1.6; on the ones refused for curving they part
    company in the same proportion at a median of 3.8. So the curvature threshold
    sorts these bands by how far apart the two rates are and not by whether they
    are two -- which is why a refused band and a published one are the same kind
    of thing, and why the published time needs these beside it.

    None where either half held too few points of the curve to fit a line to.
    """

    reason: str = ""
    """Why the band never reached a fit. Empty when one was reached, fit or not."""

    over_the_floor_db: float = float("nan")
    """How far this band's return stood over the subtraction floor.

    NaN when no pair of takes of one setting was given, which is not the same as
    zero: it says the floor was never measured, and `NO_FLOOR_TAKEN` carries that
    on the tail so an unjudged band is not read as one that passed.
    """

    @property
    def measured(self) -> bool:
        if not np.isnan(self.over_the_floor_db) and self.over_the_floor_db < CLEARS_THE_FLOOR_DB:
            return False
        return not self.reason and self.curvature_db <= STRAIGHT_ENOUGH_DB

    @property
    def why(self) -> str:
        """Why the band has no time, whichever of the two ways it has none.

        A band that never reached a fit carries its own sentence; one that was
        fitted and came back bent carries none, and the refusal is the curvature
        against the threshold. Read from one place because both are refusals and
        a record that spells out only the first publishes the second as a
        refusal with no reason beside a time it just said not to use.
        """
        if not np.isnan(self.over_the_floor_db) and self.over_the_floor_db < CLEARS_THE_FLOOR_DB:
            return f"{NOT_OVER_THE_FLOOR}, by {self.over_the_floor_db:.1f} dB"
        if self.reason:
            return self.reason
        if self.curvature_db > STRAIGHT_ENOUGH_DB:
            return f"curved by {self.curvature_db:.2f} dB, not one decay"
        return ""

    def to_json(self) -> dict:
        return {
            "centre_hz": self.centre_hz,
            # Dropped on either refusal, kept only where the band was measured.
            # A time beside `measured: false` is read as a time.
            "rt60_s": None if not self.measured else round(self.seconds, 4),
            "fitted_over_db": list(self.fitted_db),
            # Kept wherever a line was fitted, which is where it is the evidence
            # for the curvature refusal. On the routes that return before the fit
            # -- no fit reached, or the floor not cleared -- there is no number,
            # and writing the NaN that stands for one is not JSON any strict
            # reader accepts.
            "curvature_db": None if np.isnan(self.curvature_db) else round(self.curvature_db, 3),
            # Kept on a refused band as well as a published one. A band refused
            # for curving is a band whose two rates are furthest apart, which is
            # the one place they are most worth having, and dropping them there
            # would leave the refusal saying only that something was wrong.
            "halves_s": (
                None
                if self.halves_s is None
                else {
                    "over_minus_5_to_minus_15_db": round(self.halves_s[0], 4),
                    "over_minus_15_to_minus_25_db": round(self.halves_s[1], 4),
                }
            ),
            "snr_db": None if np.isnan(self.snr_db) else round(self.snr_db, 1),
            "over_the_subtraction_floor_db": (
                None if np.isnan(self.over_the_floor_db) else round(self.over_the_floor_db, 1)
            ),
            "scatter_bound": None if np.isnan(self.scatter) else round(self.scatter, 3),
            "measured": self.measured,
            "reason": self.why,
        }


@dataclass
class Tail:
    bands: list[BandDecay]
    sample_rate: int

    @property
    def floor_taken(self) -> bool:
        """Whether a pair of takes of one setting was given to measure the floor."""
        return any(not np.isnan(b.over_the_floor_db) for b in self.bands)

    @property
    def measured(self) -> list[BandDecay]:
        return [b for b in self.bands if b.measured]

    @property
    def damping(self) -> float:
        """Ratio of the lowest measured band's time to the highest's.

        Above one, the top decays faster than the bottom, which is damping in the
        loop. Near one, the network is flat. NaN when the two ends were not both
        measured, since a ratio of one measurement is not a ratio.
        """
        found = self.measured
        if len(found) < 2:
            return float("nan")
        return found[0].seconds / found[-1].seconds if found[-1].seconds > 0 else float("nan")

    def describe(self) -> str:
        lines = []
        for band in self.bands:
            if band.measured:
                lines.append(
                    f"  {band.centre_hz:>6.0f} Hz  {band.seconds:6.3f} s "
                    f"+/- {band.scatter * 100:.0f}% at most  "
                    f"(curvature {band.curvature_db:.2f} dB, {band.snr_db:.0f} dB over the noise)"
                )
            else:
                lines.append(f"  {band.centre_hz:>6.0f} Hz  --      {band.why}")
        if not self.floor_taken:
            lines.append(f"  !! {NO_FLOOR_TAKEN}")
        if not self.measured:
            lines.append("  => no band held still long enough above its own noise to be fitted")
        elif not np.isnan(self.damping):
            lines.append(
                f"  => {self.measured[0].centre_hz:.0f} Hz decays "
                f"{self.damping:.2f}x the {self.measured[-1].centre_hz:.0f} Hz time"
            )
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "bands": [b.to_json() for b in self.bands],
            "damping_low_over_high": None if np.isnan(self.damping) else round(self.damping, 3),
            "subtraction_floor_measured": self.floor_taken,
            "why_no_subtraction_floor": None if self.floor_taken else NO_FLOOR_TAKEN,
            "method": "Each octave band's energy is integrated from the end of the usable "
            "tail backwards, after the noise floor's own contribution has been taken out and "
            "the tail truncated where it reaches that floor. A straight line is fitted to the "
            "integrated curve and extrapolated to 60 dB. The RMS departure from that line is "
            "reported with it, because a tail that is two decays in sequence, or one that ran "
            "into the noise, yields a slope as readily as a real decay does.",
        }


def band_limit(signal: np.ndarray, sample_rate: int, centre: float) -> np.ndarray:
    """Keep one octave about `centre`, with a filter rather than a mask.

    Zeroing the spectrum outside the band is the obvious way and the wrong one
    here. A rectangular mask rings in time as 1/t, so the loud onset of a note
    smears across the whole buffer -- and it is circular, which wraps that onset
    round onto the quiet end. Both land where the tail is quietest, hold the
    curve up, and read as a longer decay. A fourth-order band-pass run forwards
    and backwards rings for a few cycles and stays put.
    """
    from scipy import signal as dsp

    nyquist = sample_rate / 2.0
    root_two = np.sqrt(2.0)
    low, high = centre / root_two, min(centre * root_two, nyquist * 0.99)
    if low >= high:
        return np.zeros_like(signal)
    sos = dsp.butter(4, [low / nyquist, high / nyquist], btype="band", output="sos")
    padlen = 3 * (sos.shape[0] * 2)
    if signal.size <= padlen:
        return np.zeros_like(signal)
    return dsp.sosfiltfilt(sos, signal)


def _fit(curve_db: np.ndarray, times: np.ndarray, upper: float, lower: float):
    """Slope and curvature of the straight part of an integrated decay."""
    window = (curve_db <= upper) & (curve_db >= lower)
    if window.sum() < 8:
        return None
    t, y = times[window], curve_db[window]
    slope, intercept = np.polyfit(t, y, 1)
    if slope >= 0:
        return None
    curvature = float(np.sqrt(np.mean((y - (slope * t + intercept)) ** 2)))
    return -60.0 / float(slope), curvature


def band_decay(
    tail: np.ndarray,
    sample_rate: int,
    centre: float,
    *,
    noise: np.ndarray,
    floor: np.ndarray | None = None,
) -> BandDecay:
    """Decay time of one octave band of a tail, against the noise and the floor.

    `noise` is a stretch of the same take with nothing sounding in it -- the
    silence before the note. It sets both what gets subtracted from the energy
    and where the tail stops being a tail.

    `floor` is what subtracting two takes of one setting leaves, which is the
    error the subtraction makes under the note rather than the noise before it.
    Where it is absent the band is fitted and not judged, and the tail says so.
    """
    root_two = np.sqrt(2.0)
    band = band_limit(np.asarray(tail, dtype=np.float64), sample_rate, centre)
    quiet = band_limit(np.asarray(noise, dtype=np.float64), sample_rate, centre)
    floor_power = float(np.mean(quiet**2))
    energy = band**2
    peak = float(energy.max()) if energy.size else 0.0
    snr = 10.0 * np.log10(peak / floor_power) if floor_power > 0 and peak > 0 else float("nan")

    over = float("nan")
    if floor is not None:
        left = band_limit(np.asarray(floor, dtype=np.float64), sample_rate, centre)
        # Both as RMS over the whole residual, which is the quantity the floor is
        # a floor for: the peak sees the onset, where an alignment error puts most
        # of what the subtraction failed to remove.
        left_rms = float(np.sqrt(np.mean(left**2)))
        band_rms = float(np.sqrt(np.mean(band**2)))
        over = 20.0 * np.log10(band_rms / left_rms) if left_rms > 0 and band_rms > 0 else -120.0

    blank = BandDecay(
        centre_hz=centre,
        seconds=float("nan"),
        fitted_db=(0.0, 0.0),
        curvature_db=float("nan"),
        snr_db=snr,
        over_the_floor_db=over,
    )
    if not np.isnan(over) and over < CLEARS_THE_FLOOR_DB:
        return blank
    if not np.isfinite(snr) or snr < 20.0:
        blank.reason = "never rose 20 dB over the noise in this band"
        return blank

    # Truncate where the tail reaches the floor. Integrating past that point
    # accumulates noise, which flattens the curve into a shelf and fits as a very
    # long decay -- the one failure that turns a dead band into a big number.
    step = max(1, sample_rate // 200)
    smoothed = np.convolve(energy, np.ones(step) / step, mode="same")
    above = np.flatnonzero(smoothed > floor_power * 10.0)
    if above.size < sample_rate // 50:
        blank.reason = "too little of the tail stood clear of the noise"
        return blank
    usable = energy[above[0] : above[-1]] - floor_power

    integrated = np.cumsum(np.maximum(usable, 0.0)[::-1])[::-1]
    if integrated[0] <= 0:
        blank.reason = "nothing left after the noise was taken out"
        return blank
    # The integral reaches exactly zero at its last sample by construction.
    curve = 10.0 * np.log10(np.maximum(integrated / integrated[0], 1e-30))
    times = np.arange(curve.size) / sample_rate

    # Both halves of the drop, fitted apart, before either whole-range fit is
    # tried. Independent of which range the fit below settles on, since they are
    # the parts the departure from a line is made of rather than a refinement of
    # whichever line got fitted.
    early = _fit(curve, times, -5.0, -15.0)
    late = _fit(curve, times, -15.0, -25.0)
    halves = (early[0], late[0]) if early is not None and late is not None else None
    blank.halves_s = halves

    for upper, lower in ((-5.0, -25.0), (-5.0, -15.0)):
        fitted = _fit(curve, times, upper, lower)
        if fitted is not None:
            seconds, curvature = fitted
            # An octave is centre * (sqrt2 - 1/sqrt2) wide; the fit lasted the
            # part of the 60 dB it actually covered.
            samples = centre * (root_two - 1.0 / root_two) * (upper - lower) / 60.0 * seconds
            return BandDecay(
                centre_hz=centre,
                seconds=seconds,
                fitted_db=(upper, lower),
                halves_s=halves,
                curvature_db=curvature,
                snr_db=snr,
                scatter=1.0 / np.sqrt(samples) if samples > 0 else float("nan"),
                over_the_floor_db=over,
            )
    blank.reason = "the curve never fell far enough to fit a line to"
    return blank


def measure(
    tail: np.ndarray,
    sample_rate: int,
    *,
    noise: np.ndarray,
    floor: np.ndarray | None = None,
    centres: tuple[float, ...] = OCTAVE_CENTRES,
) -> Tail:
    """Decay times of a tail across the octave bands, each with its own verdict."""
    return Tail(
        bands=[band_decay(tail, sample_rate, c, noise=noise, floor=floor) for c in centres],
        sample_rate=sample_rate,
    )


def isolate_tail(
    dry: np.ndarray,
    wet: np.ndarray,
    sample_rate: int,
    *,
    lead: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The effect's return alone, and a sample of the noise, from a dry and a wet take.

    Subtracting the dry take is what makes this a measurement of the effect and
    not of the note: a reverb tail measured from the wet take alone is the note's
    own decay for as long as the note is sounding, and only becomes the reverb
    once the note has gone. Removing the dry path gives the whole take instead of
    its last part.

    **The lead-in is read short of what was asked for.** Aligning the two takes
    trims a guard band off the front, which slides the note's onset back toward
    the head; a floor measured over the full lead-in then catches it, and a few
    milliseconds of onset in a third of a second of silence raises that floor by
    twenty dB. The band-pass ringing spreads it earlier still. Four fifths is the
    same margin the capture commands take, for the same reason.
    """
    from .stability import isolate

    apart = isolate(dry, wet)
    lead_samples = int(lead * 0.8 * sample_rate)
    return apart.residual, apart.residual[:lead_samples]


def subtraction_floor(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """What the subtraction leaves when nothing changed between the two takes.

    Two takes of one setting, put through the same align-level-subtract the
    effect's return is isolated by. What comes out is the error that step makes
    -- alignment to a sample, a gain fitted over the whole take, the unit's own
    run to run scatter -- and a return that does not stand over it by
    `CLEARS_THE_FLOOR_DB` is that error rather than an effect.
    """
    from .stability import isolate

    return isolate(first, second).residual


__all__ = [
    "CLEARS_THE_FLOOR_DB",
    "NOT_OVER_THE_FLOOR",
    "NO_FLOOR_TAKEN",
    "OCTAVE_CENTRES",
    "STRAIGHT_ENOUGH_DB",
    "BandDecay",
    "Tail",
    "band_decay",
    "band_limit",
    "isolate_tail",
    "measure",
    "subtraction_floor",
]
