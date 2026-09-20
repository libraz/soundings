"""Reading a modulator out of an effect's output alone, with no dry reference.

The delay tracker in `motion` needs a dry take to correlate against, and the
reference has to be the same strike: another take of the same note, and a twin
part struck alongside it, both fail the tracker's own control at every depth
while the same waveform passes it down to 0.375 ms. That is a constraint on the
reference, not on the effect, and it stops every type the repeatability sort
already called moving from being measured any further.

A held tone does not need one. A swept delay added to a steady tone writes the
sweep into the tone itself -- as a frequency shift proportional to how fast the
delay is changing, and as a comb moving through the partials. Both are readable
from the output on its own, so the reference the tracker cannot have is not
needed here.

**Two readings live here and they answer different questions.** `agreed_rate`
puts each partial's own period to a vote, which rejects a comb's notch train --
whose spacing depends on the partial's own frequency and so cannot win a vote --
and returns one figure. `common` sums the partials' level spectra instead and
returns the lines they share, which reports two modulators as two lines where a
vote returns one answer and can land between them. A type with a modulator per
stage needs the second; a type with one is read more robustly by the first. Both
report what they were held to rather than only what they found.

Run this module to see the controls: a modulation of known rate and depth put
into a synthetic tone, and the same tone with nothing done to it, which has to
come back empty.
"""

from __future__ import annotations

import numpy as np

from . import motion

FASTEST_HZ = 20.0
"""The top of the band searched for a modulator, which sets the decimation."""

CYCLES_WANTED = 2
"""How many cycles of the slowest candidate have to fit in the take.

The floor of the band is not a constant: it is whatever the take in hand can
carry this many cycles of. Held at a constant 0.05 Hz instead -- the figure an
eight second take happens to support -- it sat exactly on the answer of a sixty
second take, and the search pinned beside it and reported 0.11 Hz for a modulator
running at 0.05. A long take was being searched with a band sized for a short one.

Two cycles is thin, and it is deliberately what the earlier readings were held to
rather than a tightening applied after the fact. Types measured near 0.36 Hz in
eight second takes sit just above the floor two cycles gives, so raising it here
would retract those readings on no evidence about them. What each reading was
actually held to is returned beside it as `slowest_measurable_hz`, and a rate
within a few per cent of that figure wants a longer take before it is used.
"""


def strongest_partial(signal: np.ndarray, rate: int) -> float:
    """The frequency holding the most energy, which the demodulation rides on."""
    window = np.hanning(signal.size)
    spectrum = np.abs(np.fft.rfft(signal * window))
    return float(np.fft.rfftfreq(signal.size, 1.0 / rate)[int(np.argmax(spectrum))])


def demodulate(signal: np.ndarray, rate: int, centre_hz: float, width_hz: float = 60.0):
    """Instantaneous frequency and amplitude of one partial, as series in time."""
    from scipy.signal import butter, hilbert, sosfiltfilt

    nyquist = rate / 2.0
    low = max(centre_hz - width_hz, 1.0) / nyquist
    high = min(centre_hz + width_hz, nyquist * 0.99) / nyquist
    sos = butter(4, [low, high], btype="band", output="sos")
    band = sosfiltfilt(sos, np.asarray(signal, dtype=np.float64))
    analytic = hilbert(band)
    phase = np.unwrap(np.angle(analytic))
    frequency = np.diff(phase) / (2.0 * np.pi) * rate
    return frequency, np.abs(analytic)[:-1]


def line(series: np.ndarray, rate: float, label: str) -> dict | None:
    """The period of a series, and the shape of one cycle of it.

    Found by autocorrelation rather than by the strongest spectral line. A swept
    delay drags comb notches across the partial, so the series is periodic at the
    modulator's rate while being nothing like a sinusoid at it -- and the tallest
    line in its spectrum is then a harmonic. Three synthetic chorus settings were
    read as the 20th, 6th and 42nd harmonic of their own rate that way.

    The shape is the point of the fold, not a by-product: a sine and a triangle
    of the same rate and depth are the same measurement until one cycle is looked
    at, and which of the two an effect uses is exactly the sort of thing the
    identification work is after.
    """
    series = np.asarray(series, dtype=np.float64)
    series = series - series.mean()
    spread = float(np.std(series))
    if spread <= 0:
        return None
    # Down to a rate that still resolves the fastest modulation looked for,
    # before the autocorrelation. Taken at the capture rate this is quadratic in
    # a third of a million samples, which is tens of seconds a take and turned a
    # sixty-take re-read into something that ran out of its own time limit. The
    # band searched tops out at 20 Hz, so everything above a few hundred is
    # detail the answer cannot use.
    step = max(1, int(rate / (FASTEST_HZ * 20.0)))
    if step > 1:
        whole = (series.size // step) * step
        if whole < step * 8:
            return None
        series = series[:whole].reshape(-1, step).mean(axis=1)
        rate = rate / step
        spread = float(np.std(series))
        if spread <= 0:
            return None
    # The longest period looked for is set by the take, not by a constant: as
    # many samples as leave CYCLES_WANTED of it inside what was captured.
    longest, shortest = series.size // CYCLES_WANTED, max(2, int(rate / FASTEST_HZ))
    if longest <= shortest:
        return None
    whole = np.correlate(series, series, mode="full")[series.size - 1 :]
    whole = whole / whole[0]
    # Start past the first dip rather than at the shortest lag asked for. The
    # curve falls away from lag zero on any signal at all, so a search that
    # begins inside that fall returns its own starting point -- which is how a
    # shallow sweep at 0.4 Hz was read as 20 Hz, the edge of the band.
    falling = np.where(np.diff(whole[: max(longest, shortest + 2)]) > 0)[0]
    begin = max(shortest, int(falling[0]) + 1 if falling.size else shortest)
    if begin >= longest:
        return None
    window = whole[begin:longest]
    # A peak, not merely the largest value in the window: the largest can sit on
    # the boundary, where it is a statement about where the window was cut.
    interior = np.where((window[1:-1] >= window[:-2]) & (window[1:-1] >= window[2:]))[0] + 1
    if interior.size == 0:
        return None
    # The shortest peak that is nearly as tall as the tallest, not the tallest.
    # An autocorrelation of anything periodic peaks at every multiple of the
    # period, and those peaks are often the taller ones -- a 15 Hz modulation was
    # read as 5 Hz that way, its own third multiple. Taking the first peak that
    # comes close chooses the period over its multiples without needing to know
    # which it is looking at.
    tallest = float(np.max(window[interior]))
    if tallest <= 0:
        # Every peak in the band is a negative correlation, which is not a period
        # -- it is a series that never comes back to itself. Reported as no
        # period rather than as the least negative one, which is where the search
        # crashed on the first take that had none.
        return None
    close = interior[window[interior] >= tallest * 0.9]
    lag = int(close[0]) + begin
    strength = float(whole[lag])
    # The peak sits between samples, and after the decimation above a lag is
    # coarse: at the top of the band it is a few tens of samples, so rounding it
    # moved a 7.500 Hz modulation to 7.547. A parabola through the peak and its
    # two neighbours puts it back, and costs nothing.
    refined = float(lag)
    if 0 < lag < whole.size - 1:
        left, middle, right = whole[lag - 1], whole[lag], whole[lag + 1]
        bend = left - 2.0 * middle + right
        if bend != 0:
            refined = lag + 0.5 * (left - right) / bend
    # Fold at the period and average, which is the cycle's own shape. A series
    # that is not periodic averages towards its mean, and the flatness of what
    # comes out is what says so.
    cycles = series.size // lag
    folded = series[: cycles * lag].reshape(cycles, lag).mean(axis=0) if cycles >= 2 else None
    shape = None
    if folded is not None:
        phase = np.arange(lag) / lag
        sine = np.sin(2.0 * np.pi * phase)
        cosine = np.cos(2.0 * np.pi * phase)
        first = np.hypot(float(np.dot(folded, sine)), float(np.dot(folded, cosine))) * 2.0 / lag
        rest = float(np.sqrt(max(np.mean(folded**2) - first**2 / 2.0, 0.0)))
        shape = {
            "cycles_averaged": int(cycles),
            "peak_to_peak": round(float(folded.max() - folded.min()), 6),
            "fundamental": round(first, 6),
            "above_the_fundamental": round(rest, 6),
            "not_a_sine_by": round(rest / first, 4) if first > 0 else None,
            "notches": notches_per_cycle(folded),
            "cycle": [round(float(v), 6) for v in folded[:: max(1, lag // 64)]],
        }
    return {
        "what": label,
        "rate_hz": round(rate / refined, 4),
        "repeats_at_that_period": round(strength, 4),
        "series_spread": round(spread, 6),
        # What a null here is a null about. The period cannot be longer than the
        # part of the take it is looked for in, so a modulation slower than this
        # is outside the take rather than absent from the unit -- an injected
        # 0.15 Hz sweep, one cycle of which does not fit in seven seconds, came
        # back as 0.28 Hz and nothing in the reading said why. A rate returned
        # within a few per cent of this figure is standing on the edge of the
        # search and should be asked again of a longer take before it is used.
        "slowest_measurable_hz": round(rate / longest, 4),
        "shape": shape,
    }


def partials(
    signal: np.ndarray, rate: int, how_many: int = 4, apart_hz: float = 40.0
) -> list[float]:
    """The strongest frequencies in the signal, kept apart from one another.

    More than one, because a rate read off a single partial is one measurement
    and a rate the partials agree on is several. Where they disagree that is
    itself the reading -- an effect putting energy at frequencies the input never
    had is not a delay of the input, and no amount of tracking one partial says
    so.
    """
    signal = np.asarray(signal, dtype=np.float64)
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size)))
    frequencies = np.fft.rfftfreq(signal.size, 1.0 / rate)
    kept: list[float] = []
    for index in np.argsort(spectrum)[::-1]:
        candidate = float(frequencies[index])
        if candidate < 40.0:
            continue
        if all(abs(candidate - held) > apart_hz for held in kept):
            kept.append(candidate)
        if len(kept) >= how_many:
            break
    return kept


def read_partials(
    signal: np.ndarray, rate: int, *, how_many: int = 4, trim_s: float = 0.05
) -> list[dict]:
    """One reading per strong partial, so the rate can be corroborated or not."""
    signal = np.asarray(signal, dtype=np.float64)
    guard = int(trim_s * rate)
    if signal.size > 2 * guard:
        signal = signal[guard:-guard]
    out = []
    for centre in partials(signal, rate, how_many):
        frequency, amplitude = demodulate(signal, rate, centre)
        edge = int(0.15 * rate)
        amplitude = amplitude[edge:-edge]
        in_db = 20.0 * np.log10(np.maximum(amplitude, amplitude.max() * 1e-6))
        out.append({"partial_hz": round(centre, 2), "level_swing": line(in_db, rate, "level, dB")})
    return out


def in_agreement(rates: list[float], within: float) -> list[list[float]]:
    """The readings cut into groups that agree, each reading in exactly one group.

    A tolerance taken around each reading in turn is not a partition and it is not
    a measure of a group either. It is not a partition because a reading sitting
    between two others falls inside both neighbourhoods and is counted into both,
    so which group is called the largest follows the order the partials arrived in
    rather than the take. And it is not a measure of a group because it is taken
    from whichever reading the loop is holding: four readings spanning nearly six
    per cent were counted as agreeing within five, because the one they were all
    measured from sat in the middle of them.

    A group here is kept while its own span stays inside the tolerance on its own
    mean, which is the thing the tolerance was meant to say. The groups are built
    by joining the two neighbours that make the narrowest group first and stopping
    when no join fits, rather than by walking the sorted readings once and
    extending: a single walk starting from the lowest reading joins an outlier to
    the bottom of a real cluster before it reaches the rest of that cluster, and
    splits three readings that agree into two pairs that do not. Every reading
    lands in exactly one group and the result does not depend on the order the
    partials arrived in.

    With four readings and a tolerance this wide, two groupings can both fit and
    the narrower one is taken. That is a choice the readings do not settle, and it
    moves a median rather than a count: what the count is for -- whether a rate was
    agreed and by how many -- is the same under either.
    """
    groups = [[value] for value in sorted(rates)]
    while len(groups) > 1:
        narrowest = None
        for index in range(len(groups) - 1):
            joined = groups[index] + groups[index + 1]
            span = joined[-1] - joined[0]
            if span <= (sum(joined) / len(joined)) * within:
                if narrowest is None or span < narrowest[0]:
                    narrowest = (span, index)
        if narrowest is None:
            break
        _span, index = narrowest
        groups[index : index + 2] = [groups[index] + groups[index + 1]]
    return groups


def agreed_rate(readings: list[dict], within: float = 0.05) -> dict:
    """The rate the partials agree on, and how many of them did.

    A rate two partials of the same take give independently is much harder to
    account for as the reading's own than one partial's is. Disagreement is
    returned rather than resolved: it is the signature of a type that is not
    delaying its input.

    That holds for a take whose partials scatter, and it has to hold for the take
    whose partials split evenly between two rates as well. Two of four at one rate
    and two of four at another is not a reading of either: the take says both and
    the count says neither, so no rate is named and the two are published as what
    it split between. Often the pair stand in a small whole ratio, which is a
    demodulator locking a partial onto every second or third cycle -- and reading
    the higher of them as the rate on that account would be putting a model inside
    a record, which is the one thing a record here does not do. The rates are all
    published either way, so nothing is lost by declining to choose.

    A take where nothing reaches two is left naming the rate it named. That is not
    the same case and the difference is the gate downstream: a reading fewer than
    half the partials found is refused by name, so a lone partial's rate is already
    marked as no agreement and is already out of evidence. An even split is the one
    shape that passes the gate while having nothing behind it.
    """
    rates = [r["level_swing"]["rate_hz"] for r in readings if r["level_swing"]]
    if not rates:
        return {"rate_hz": None, "agreeing": 0, "of": len(readings), "rates": []}
    groups = in_agreement(rates, within)
    most = max(len(group) for group in groups)
    largest = [group for group in groups if len(group) == most]
    published = {"agreeing": most, "of": len(readings),
                 "rates": [round(r, 4) for r in rates]}
    if most == 1:
        # Nothing agreed with anything, so the published rate is one partial's and
        # `agreeing` says so. Which partial is the one the take was read from
        # first, rather than the lowest rate it returned: the reading carries no
        # information either way, and the take's own order is the one a reader can
        # follow back to a partial.
        return {"rate_hz": round(float(rates[0]), 4), **published}
    if len(largest) > 1:
        return {
            "rate_hz": None,
            **published,
            "split_between_hz": sorted(
                round(float(np.median(group)), 4) for group in largest
            ),
        }
    return {"rate_hz": round(float(np.median(largest[0])), 4), **published}


def arrivals(bypassed: list[float], routed: list[float], apart_hz: float = 40.0) -> list[float]:
    """Partials the routed take holds and the bypassed one does not.

    Energy at a frequency the input never carried is not something a delay can
    produce, however it is swept. Reported as frequencies rather than as a name.
    """
    return [f for f in routed if all(abs(f - held) > apart_hz for held in bypassed)]


SHAPE_IS_NOT_READABLE_HERE = """\
Fitting a sine and a triangle to the folded level cycle does not say which one
was swept. Delays swept by each, at one partial, with everything else held:

    depth 0.5 ms (1 notch)   sine injected -> triangle    triangle -> triangle
    depth 1.0 ms (2 notches) sine injected -> sine        triangle -> sine
    depth 3.0 ms (6 notches) sine injected -> triangle    triangle -> triangle

The answer tracks the depth and not the shape, and both candidates leave between
0.54 and 0.99 of the cycle unaccounted for at every setting. The reason is that
the level series is the comb's response to the delay rather than the delay: each
notch passage is a dip, so the cycle is a train of dips whose count follows the
excursion and whose arrangement carries the shape only through a warp nothing
here inverts.

The rate and the excursion are not affected -- both were recovered on these same
signals. What this closes is the shape, on this route. Reading it would mean
recovering the delay itself from the times the notches arrive, since consecutive
notches stand one period of the partial apart in delay, which turns their
arrival times into samples of the sweep. That is not done here.
"""


def notches_per_cycle(folded: np.ndarray) -> int:
    """Deep minima in one cycle of the level series.

    A delay swept past a partial drags the comb's notches through it, and each
    passage is one minimum. The count is what carries the excursion: the delay
    has to change by one period of the partial for the next notch to arrive, so
    the number of them across a cycle divides straight into a depth.

    Counted against the cycle's own range rather than an absolute level, since
    the depth of a notch depends on how evenly the two paths are mixed and that
    is not known here.
    """
    if folded.size < 8:
        return 0
    span = float(folded.max() - folded.min())
    if span <= 0:
        return 0
    deep = folded < folded.min() + 0.25 * span
    # Runs of adjacent samples below the line are one notch, not several.
    edges = np.diff(deep.astype(int))
    starts = int(np.sum(edges == 1)) + (1 if deep[0] else 0)
    return starts


def depth_from_notches(notches: int, partial_hz: float) -> float | None:
    """Peak-to-peak delay excursion, in ms, implied by that many notch passages.

    One notch arrives for every period of the partial the delay moves through,
    and a cycle of the modulator crosses the excursion twice.
    """
    if notches < 1 or partial_hz <= 0:
        return None
    return notches / (2.0 * partial_hz) * 1000.0


def read(signal: np.ndarray, rate: int, *, trim_s: float = 0.2) -> dict:
    """Rate, and what the two routes say about it, from the output alone."""
    signal = np.asarray(signal, dtype=np.float64)
    guard = int(trim_s * rate)
    signal = signal[guard:-guard] if signal.size > 2 * guard else signal
    centre = strongest_partial(signal, rate)
    frequency, amplitude = demodulate(signal, rate, centre)
    # The edges of the band-pass and of the analytic signal both ring, and the
    # ring is slow enough to fit a modulator. Cut it rather than let it be read
    # as one.
    edge = int(0.15 * rate)
    frequency, amplitude = frequency[edge:-edge], amplitude[edge:-edge]
    in_db = 20.0 * np.log10(np.maximum(amplitude, amplitude.max() * 1e-6))
    level = line(in_db, rate, "instantaneous level, dB")
    found = {
        "partial_hz": round(centre, 3),
        "frequency_swing": line(frequency, rate, "instantaneous frequency, Hz"),
        "level_swing": level,
    }
    if level is not None and level["shape"] is not None:
        found["delay_excursion_ms"] = depth_from_notches(level["shape"]["notches"], centre)
    return found



READ_AT_HZ = 400.0
"""What the level series are decimated to. Twenty times the fastest rate looked for."""

BAND_HZ = (0.30, 20.0)
"""Where a modulator is looked for. The floor is what a seven second take carries."""

APART = 0.05
"""How far apart two peaks have to sit to be counted as two, as a fraction."""



LEVEL, FREQUENCY = "level", "frequency"
"""The two series a modulator can be read off, named so a caller says which.

They put their energy in different harmonics of the same rate, so a line plain in
one can be buried in the other, and which is which is not predictable from the
type. Read whichever the question needs, and where a rate is in doubt read both.

**Neither breaks the tie the other cannot.** A sweep that crosses a cancellation
on the way up and again on the way down puts every event at half the period, and
the fundamental leaves *both* series: the pull on a partial follows where the
notch sits relative to it, not which way the sweep is going. A flanger with no
loop does exactly this, which is the printed centre of its feedback byte, which is
where every run that is not asking about the loop parks it. What tells a rate from
twice it there is a sweep of that neighbouring byte, not a second series.
"""


def spectrum(
    signal: np.ndarray, rate: int, how_many: int = 4, which: str = LEVEL
) -> tuple[np.ndarray, np.ndarray, float]:
    """The summed spectrum of the partials' series, its frequency axis, and the
    width one line occupies on it.

    Each partial is normalised before the sum so that a loud partial does not
    decide the answer on its own -- the question is which period they share, not
    which of them is loudest.

    The width is returned because the transform is zero-padded: the grid is far
    finer than the take can resolve, so a caller cutting a line out by counting
    bins would cut a hundredth of it. What sets the width is the length of the
    series, and it is the same for every line on the axis.
    """
    signal = np.asarray(signal, dtype=np.float64)
    step = max(1, int(rate / READ_AT_HZ))
    srate = rate / step
    total, width = None, 0.0
    for centre in partials(signal, rate, how_many):
        swing, amplitude = demodulate(signal, rate, centre)
        edge = int(0.15 * rate)
        swing, amplitude = swing[edge:-edge], amplitude[edge:-edge]
        series = (swing if which == FREQUENCY
                  else 20.0 * np.log10(np.maximum(amplitude, amplitude.max() * 1e-6)))
        whole = (series.size // step) * step
        thinned = series[:whole].reshape(-1, step).mean(axis=1)
        width = srate / thinned.size
        thinned = (thinned - thinned.mean()) * np.hanning(thinned.size)
        magnitude = np.abs(np.fft.rfft(thinned, n=1 << 16))
        peak = magnitude.max()
        if peak > 0:
            magnitude = magnitude / peak
        total = magnitude if total is None else total + magnitude
    if total is None:
        return np.empty(0), np.empty(0), 0.0
    freq = np.fft.rfftfreq(1 << 16, 1.0 / srate)
    inside = (freq >= BAND_HZ[0]) & (freq <= BAND_HZ[1])
    return freq[inside], total[inside] / total.max(), width


NEARBY = 1.6
"""How far either side of a frequency its floor is read, as a ratio.

A modulation spectrum falls steeply across this band, so a median over the whole
of it sits below the roughness at the bottom and above it at the top -- which
reports a line wherever the question is asked low and hides one wherever it is
asked high. Read off one partial that put a control at 469 times its floor where
a local window put it at 1.6. Summed over four partials and normalised the two
floors come out close, so this is insurance on the single-partial case rather
than a correction to every reading.
"""

CUT_WIDTHS = 3
"""How many line widths either side of a frequency are the line rather than floor."""


def over_the_floor(
    freq: np.ndarray, power: np.ndarray, width: float, hz: float, lines=()
) -> float | None:
    """How far the spectrum stands at `hz` above the roughness beside it.

    The roughness is read from a window around `hz` rather than from the band, and
    every line the take is expected to carry is cut out of it first -- so a tall
    line does not raise the bar it is itself judged against.

    `None` where there is no roughness left to measure: at the bottom of the band
    a line is wider than the gap to its neighbours, so the window is entirely cut
    away and the question cannot be asked of this take. Returned rather than
    reported as nothing, which is the answer a caller would act on.
    """
    if freq.size == 0 or width <= 0 or hz <= 0:
        return None
    cut = CUT_WIDTHS * width
    keep = (freq >= hz / NEARBY) & (freq <= hz * NEARBY)
    for line in [*lines, hz]:
        if line > 0:
            keep &= np.abs(freq - line) > cut
    within = np.abs(freq - hz) <= cut
    if not keep.any() or not within.any():
        return None
    floor = float(np.median(power[keep]))
    return float(power[within].max() / floor) if floor > 0 else None


def _rounded(value: float | None) -> float | None:
    """A height rounded for publication, keeping `None` as the absence it is."""
    return None if value is None else round(value, 2)


def common(
    signal: np.ndarray, rate: int, how_many: int = 5, which: str = LEVEL
) -> list[dict]:
    """The strongest lines the partials share, tallest first.

    Each line carries two heights, because they answer different questions. How
    tall it stands against the tallest line says which of the lines here matters;
    how far it stands out of the roughness beside it says whether it is a line at
    all. A reading that only has the first cannot tell a small line from none.
    """
    freq, power, width = spectrum(signal, rate, which=which)
    if freq.size == 0:
        return []
    peaks = [
        i
        for i in range(1, power.size - 1)
        if power[i] > power[i - 1] and power[i] > power[i + 1]
    ]
    peaks.sort(key=lambda i: -power[i])
    found: list[dict] = []
    for i in peaks:
        if all(abs(freq[i] - held["hz"]) > APART * freq[i] for held in found):
            found.append({
                "hz": round(float(freq[i]), 4),
                "of_the_tallest": round(float(power[i]), 3),
                "over_the_floor": _rounded(
                    over_the_floor(freq, power, width, float(freq[i]))),
            })
        if len(found) == how_many:
            break
    return found


STANDS_AT = 0.15
"""How tall a subharmonic has to stand, against the tallest line, to be believed."""

DIVIDE_BY = 12
"""The deepest subharmonic looked at. A swept comb's level series was seen to put
its ninth harmonic above its own fundamental, so the search has to reach past that."""


def fundamental(signal: np.ndarray, rate: int) -> dict | None:
    """The rate, taken as the lowest line the strong ones are multiples of.

    The tallest line is not the rate and taking it for the rate is the mistake
    this function exists to undo. A swept comb drags notches across a partial, so
    its level over one modulator cycle is a train of dips rather than a swell, and
    a train of dips puts most of its energy in harmonics. Injected at 0.90 Hz into
    the recorded voice, the tallest shared line came back at 2.6978 -- the third --
    with the fundamental third-tallest; at 2.05 Hz the tallest was again the third.

    So the tallest line is taken as a multiple of the answer rather than as the
    answer, and each of its divisors is tested: the answer is the smallest one that
    is itself a peak of some substance and that accounts for the other strong lines.
    """
    freq, power, _ = spectrum(signal, rate)
    if freq.size == 0:
        return None
    tallest = freq[int(np.argmax(power))]

    def stands(at: float) -> float:
        inside = np.abs(freq - at) <= APART * at
        return float(power[inside].max()) if inside.any() else 0.0

    # Scored on how many of the strong lines are whole multiples of the candidate,
    # not on how much energy sits along its ladder. Dividing the tallest line by
    # one more than it should be leaves a ladder whose rungs are mostly empty while
    # still landing on the tallest itself, and an energy score does not notice that:
    # injected at 0.90 Hz, the fourth of the tallest was preferred to the third,
    # which explained every strong line while the fourth explained two of five.
    strong = [(line["hz"], line["of_the_tallest"]) for line in common(signal, rate, 8)]
    scored: list[tuple[float, float, int]] = []
    for divisor in range(1, DIVIDE_BY + 1):
        candidate = tallest / divisor
        if candidate < BAND_HZ[0] or stands(candidate) < STANDS_AT:
            continue
        score = 0.0
        for line, height in strong:
            order = line / candidate
            if order >= 0.9 and abs(order - round(order)) <= APART * order:
                score += height
        scored.append((round(score, 4), float(candidate), divisor))
    if not scored:
        return {"rate_hz": round(float(tallest), 4), "divided_by": 1, "explains": 0.0}
    # Best score wins; a tie goes to the least division, so nothing is halved for free.
    score, candidate, divisor = max(scored, key=lambda row: (row[0], row[1]))
    return {
        "rate_hz": round(candidate, 4),
        "divided_by": divisor,
        "explains": round(score / max(sum(h for _, h in strong), 1e-9), 3),
        "stands": round(stands(candidate), 3),
    }


def height_at(
    signal: np.ndarray, rate: int, hz: float, which: str = LEVEL, lines=()
) -> float | None:
    """How tall the shared spectrum stands at one frequency, over its own floor.

    A ratio to the roughness rather than to the tallest line, because the question
    this answers is whether a line is there at all -- and the tallest line moves
    from take to take while the roughness does not. The frequency is the caller's:
    this is the reading to use where a rate is already known from somewhere else
    and what is wanted is whether this take carries it, which is the one question
    a period picker cannot be asked.

    `lines` are the other frequencies the take is expected to carry, so that a
    neighbouring harmonic is not counted as roughness.
    """
    freq, power, width = spectrum(signal, rate, which=which)
    return over_the_floor(freq, power, width, hz, lines)



def synthetic_check() -> None:
    """A chorus of known rate and depth, to show the reading is not vacuous."""
    rate = 48000
    seconds = 8.0
    for partial_hz, lfo_hz, depth_ms, centre_ms in (
        (440.0, 0.900, 5.0, 18.0),
        (261.6, 3.100, 2.0, 12.0),
        (880.0, 0.400, 8.0, 20.0),
    ):
        time = np.arange(int(seconds * rate)) / rate
        dry = np.sin(2.0 * np.pi * partial_hz * time)
        wet = dry + motion.modulated_copy(
            dry, rate, rate_hz=lfo_hz, depth_ms=depth_ms, centre_ms=centre_ms
        )
        found = read(wet, rate)
        print(f"  {partial_hz:6.1f} Hz tone, LFO {lfo_hz:5.3f} Hz depth {depth_ms} ms")
        for key in ("frequency_swing", "level_swing"):
            print(f"      {key.split('_')[0]:9s} {show(found[key])}")

    # And the same tone with nothing done to it, which must produce no cycle
    # worth reporting. The reading has to be able to come back empty.
    time = np.arange(int(seconds * rate)) / rate
    plain = np.sin(2.0 * np.pi * 440.0 * time)
    found = read(plain, rate)
    print("  440.0 Hz tone, nothing applied  (a shape here would be the reading's own)")
    for key in ("frequency_swing", "level_swing"):
        print(f"      {key.split('_')[0]:9s} {show(found[key])}")


def show(value: dict | None) -> str:
    if value is None:
        return "no period"
    shape = value["shape"]
    tail = (
        "no fold"
        if shape is None
        else f"pk-pk {shape['peak_to_peak']:9.4f}  not-a-sine {shape['not_a_sine_by']}"
    )
    return (
        f"{value['rate_hz']:6.3f} Hz  repeats {value['repeats_at_that_period']:5.3f}  "
        f"spread {value['series_spread']:9.4f}  {tail}"
    )




if __name__ == "__main__":
    print("what the reading finds when it is given a modulation it was told about,")
    print("and what it finds when there is none:")
    synthetic_check()
