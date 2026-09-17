"""What a held tone's partials do while an effect's modulator runs under them.

The delay tracker beside this reads one delay between a bypassed take and a
routed one, which needs the two takes to be the same note and needs the delay to
be a delay. This reads the routed take on its own, partial by partial, and asks
three separate things of it. They are kept separate because each of them is blind
to something the next one sees.

**Where the modulator is.** A projection is walked across a grid of rates rather
than a line being fitted to a series, so nothing here counts cycles: a matched
filter at a named rate works on a take holding two of them, where a line does
not. What it costs is resolution -- two rates closer together than one over the
take's length are not separated -- and that is reported rather than hidden.

**What kind of thing is moving.** A swept delay turns a partial's phase in
proportion to that partial's own frequency, an all-pass section turns them all by
the same angle, and a level modulation turns none of them. Reading the phase and
the level apart tells the three apart, and where the level moves it tells a comb
being swept from the whole voice rising and falling together: a partial near a
notch swings far more than one on a peak, and one envelope over the voice cannot
produce that.

**How far it moves.** Two routes, complementary rather than competing, and which
of them answers is itself a reading. Where the effect puts out the delayed path
and no direct one there is no comb, and the excursion comes off the phase --
mixing a partial down by its own frequency leaves `-2 pi f D(t)`, so the phase
carries the delay itself. Where a direct path is mixed in, the composite's phase
turns by far less and not in proportion -- measured on this unit's own carrier, a
fifth wet turns a two millisecond sweep into five thousandths -- and what the mix
produces instead is the comb, which is modelled forward.

Nothing here is told what any byte was set to. A rate is either where the take's
own projection peaks or what the comb model settles on from there, and the two are
reported apart because they are not the same reading.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import rates

TOP_HZ = 8000.0
"""Where the reading stops. Above it the carrier's orders are into the noise on
every voice tried, and a partial read out of the noise contributes a phase that
is not the effect's."""

WITHIN_DB = 40.0
"""How far under the loudest partial an order may sit and still be read."""

DROP = 4800
"""Samples cut from each end of a phase series before it is used. The boxcar runs
off the take there and the phase turns over on nothing."""

READ_AT_HZ = 200.0
"""What a phase series is thinned to. The boxcar has already smoothed away
everything above the partial's own spacing, so past this every sample is a copy
of its neighbour."""

LEVEL_AT_HZ = 400.0
"""What a level series is thinned to. Higher than the phase, because a notch
passage is the fastest thing in a level series and it is fast: at the widest
excursion read here a partial crosses one in a few milliseconds."""


@dataclass(frozen=True)
class Partials:
    """One take's partials, each with its phase and its level over time.

    `phase` and `level_db` share `at`. `kept` is which of the orders asked for
    cleared the threshold, and it is passed between takes on purpose: an effect
    that adds sidebands raises its own take's quiet orders over the threshold and
    brings a different set of partials into the answer, so two takes compared
    against each other stop being comparable unless one of them decides.
    """

    freqs_hz: np.ndarray
    phase: np.ndarray
    level_db: np.ndarray
    at: np.ndarray
    kept: np.ndarray
    carrier_hz: float

    @property
    def orders(self) -> int:
        return int(self.freqs_hz.size)

    @property
    def sounded_s(self) -> float:
        return float(self.at[-1] - self.at[0]) if self.at.size > 1 else 0.0

    @property
    def rates_are_separated_by(self) -> float:
        """How far apart two rates must be before this take can tell them apart.

        One over the length read, which is the width of a projection's own main
        lobe. A peak placed inside this of another is not a second modulator.
        """
        return 1.0 / self.sounded_s if self.sounded_s > 0 else float("inf")


def _one(signal: np.ndarray, rate: int, f: float, window: int, step: int):
    """One partial's phase, detrended, and its level in dB, on the same base.

    The delay is in the phase and not in its derivative. Differentiating to get a
    frequency first costs the reading everything -- a derivative multiplies the
    noise by the rate it is taken at, and at the slowest entry of this unit's rate
    table the frequency deviation of a two millisecond sweep is a third of a
    millihertz.

    One partial at a time deliberately. Holding every partial's complex envelope
    at once is the natural way to write it and costs tens of gigabytes on the
    dense low carriers this reading was developed against.
    """
    t = np.arange(signal.size, dtype=np.float64) / rate
    mixed = signal * np.exp(-2j * np.pi * f * t)
    pad = window // 2
    padded = np.concatenate((np.zeros(pad, dtype=complex), mixed, np.zeros(window, dtype=complex)))
    run = np.concatenate(([0.0 + 0.0j], np.cumsum(padded)))
    z = (run[window : window + signal.size] - run[: signal.size]) / window
    size = np.abs(z)[DROP:-DROP:step]
    phase = np.unwrap(np.angle(z))[DROP:-DROP:step]
    at = np.arange(phase.size, dtype=np.float64) * step / rate
    # A straight line comes out of the phase first: the partial is not exactly
    # where it was asked for, which tilts it, and the centre delay is a constant
    # in it. Neither is what the modulator did.
    phase = phase - np.polyval(np.polyfit(at, phase, 1), at)
    level = 20.0 * np.log10(np.maximum(size, size.max() * 1e-6))
    return phase, level - level.mean(), at, float(np.median(size))


def read(
    body: np.ndarray,
    rate: int,
    *,
    carrier_hz: float,
    top_hz: float = TOP_HZ,
    within_db: float = WITHIN_DB,
    keep: np.ndarray | None = None,
) -> Partials | None:
    """Every order of the carrier that clears the threshold, read one at a time.

    The boxcar is one period of the carrier long, so its nulls land on the
    neighbouring orders and each partial is read without them.
    """
    body = np.asarray(body, dtype=np.float64)
    freqs = np.array([carrier_hz * k for k in range(1, int(top_hz // carrier_hz) + 1)])
    if not freqs.size or body.size <= 2 * DROP:
        return None
    window = int(round(rate / carrier_hz))
    step = max(1, rate // int(READ_AT_HZ))
    said = [_one(body, rate, f, window, step) for f in freqs]
    sizes = np.array([row[3] for row in said])
    if keep is None:
        keep = sizes > sizes.max() * 10.0 ** (-within_db / 20.0)
    if not keep.any():
        return None
    phase = np.vstack([row[0] for row in said])[keep]
    level = np.vstack([row[1] for row in said])[keep]
    return Partials(
        freqs_hz=freqs[keep],
        phase=phase,
        level_db=level,
        at=said[0][2],
        kept=keep,
        carrier_hz=carrier_hz,
    )


def levels(
    body: np.ndarray, rate: int, *, carrier_hz: float, count: int, periods: int | None = None
) -> list[float] | None:
    """Each of the first `count` orders of the carrier as its own standing level.

    The same demodulation as `_one` above and a different reading, because the
    question is different and the difference is in one number -- how long the boxcar
    after the mixer is.

    `_one` reads a partial that is **moving**, so its boxcar is one carrier period
    long: short enough to follow a modulator, and with its nulls on the neighbouring
    orders. What that costs is a main lobe a whole fundamental wide, so everything
    within one fundamental of an order is read as part of it. On a carrier a
    nonlinearity has been fed that is not nothing -- products between two partials
    land between the orders, and a lobe that wide collects them.

    A standing level needs no time resolution at all, so the boxcar is the whole
    read stretch: a matched filter for a steady partial, and the narrowest lobe the
    take can give. Windowed rather than square, so that a fundamental measured a
    little off does not read every order through a sidelobe of the first. What it
    assumes is that the partial is steady over the stretch -- one that drifts turns
    under the filter and reads low -- and what says it was is the run's own repeats
    of one setting, which is a figure the caller has and this does not.

    `periods` reads it over a boxcar that many carrier periods long instead, and is
    here for asking whether the two answers agree rather than for producing a
    record: a reading that changes with the width of its filter is reporting what
    sits beside the orders as well as the orders.

    `None` where the stretch is too short to hold the reading.
    """
    body = np.asarray(body, dtype=np.float64)
    if carrier_hz <= 0 or body.size < 2:
        return None
    if periods is not None:
        if body.size <= 2 * DROP:
            return None
        window = int(round(periods * rate / carrier_hz))
        step = max(1, rate // int(LEVEL_AT_HZ))
        return [_one(body, rate, carrier_hz * k, window, step)[3] for k in range(1, count + 1)]
    at = np.arange(body.size, dtype=np.float64) / rate
    shaped = body * np.hanning(body.size)
    weight = float(np.sum(np.hanning(body.size)))
    if weight <= 0:
        return None
    return [
        float(np.abs(np.sum(shaped * np.exp(-2j * np.pi * carrier_hz * k * at))) / weight)
        for k in range(1, count + 1)
    ]


def project(series: np.ndarray, at: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """One size per grid rate, averaged over the partials, in the series' units.

    The sizes are averaged as they come and not normalised per partial. A swept
    delay swings a partial's phase in proportion to that partial's own frequency,
    so the partials carrying the most of it already weigh the most -- normalising
    first would let the quietest partial argue as loudly as the one the effect
    actually moved.
    """
    angle = 2.0 * np.pi * np.outer(grid, at)
    size = np.hypot(series @ np.cos(angle).T, series @ np.sin(angle).T) * 2.0 / at.size
    return size.mean(axis=0)


NOTHING_AT_ALL = 1e-9
"""What counts as nothing, as a proportion of the largest thing on the grid.

Relative and not absolute. An absolute floor under a division is a number in the
series' own units, and the units here depend on the material: a guard of a
million-millionth is far below a real take's projection and far *above* a
synthetic one's, where it silently turned a ratio of one into a hundredth.
"""


def _over(size: float, floor: np.ndarray, index: int) -> float:
    """How many times over the bypassed take's own projection, or one where it is nothing."""
    tiny = float(floor.max()) * NOTHING_AT_ALL
    under = float(floor[index])
    if under <= tiny:
        return 1.0 if float(size) <= tiny else float("inf")
    return float(size) / under


def peak(series: np.ndarray, at: np.ndarray, grid: np.ndarray, floor: np.ndarray) -> dict:
    """Where this take stands furthest over the one with nothing in the path.

    The comparison is against the same projection of the bypassed take, rate by
    rate, rather than against the middle of this take's own grid. A held tone is
    not steady: this unit's carrier drifts enough on its own to put a peak
    seventeen times its own grid median at half a hertz, which is a fact about the
    voice and would be read as a modulator by any statistic that had nothing to
    subtract it with.
    """
    size = project(series, at, grid)
    # What this take added, not how many times over the bypassed one it is. A
    # ratio picks whichever rate the bypassed take happens to be quietest at, and
    # the bypassed take is quietest where nothing is: an injected sweep at
    # 0.45 Hz came back as 2.31 that way, every time, on material the difference
    # places exactly.
    added = np.maximum(size - floor, 0.0)
    chosen = int(added.argmax())
    return {
        "hz": round(float(grid[chosen]), 4),
        "stands_over_bypassed": round(_over(size[chosen], floor, chosen), 3),
        "added": round(float(added[chosen]), 6),
        "size": round(float(size[chosen]), 6),
    }


def height_at(
    series: np.ndarray, at: np.ndarray, grid: np.ndarray, floor: np.ndarray, hz: float
) -> dict:
    """The same reading taken at one named rate rather than where it is largest."""
    size = project(series, at, grid)
    index = int(np.argmin(np.abs(grid - hz)))
    return {
        "hz": round(float(grid[index]), 4),
        "stands_over_bypassed": round(_over(size[index], floor, index), 3),
        "size": round(float(size[index]), 6),
    }


def _left_over(swing: np.ndarray, model: np.ndarray) -> float:
    """What a shape leaves of the swing after its best scaling, as a fraction."""
    gain = float(swing @ model / max(float(model @ model), 1e-18))
    left = float(np.sum((swing - gain * model) ** 2))
    return round(left / max(float(np.sum(swing**2)), 1e-18), 4)


WRONG_SHAPE_ON_A_CLEAN_ONE = 0.30
"""What the flat-turn model leaves on a fully wet injected sweep, which the delay
model explains exactly. A shape leaving more than the wrong model leaves on a
clean case has not been separated from anything, whichever of the two it is."""


def excursion(held: Partials, hz: float) -> dict:
    """The delay excursion each partial's phase swing implies, and its shape.

    The swing is reported per partial rather than as one fitted number, so the
    answer is a set of numbers that either agree or do not. A swept delay makes
    them agree, because the phase swing it causes is proportional to the partial's
    own frequency; anything turning every partial by the same angle makes the
    implied excursions fall as one over the order and disagree by more than their
    own median.
    """
    angle = 2.0 * np.pi * hz * held.at
    swing = np.hypot(held.phase @ np.cos(angle), held.phase @ np.sin(angle)) * 2.0 / held.at.size
    implied = swing / (np.pi * held.freqs_hz) * 1000.0
    as_a_delay = _left_over(swing, held.freqs_hz)
    as_a_flat_turn = _left_over(swing, np.ones(held.orders))
    best, other = min(as_a_delay, as_a_flat_turn), max(as_a_delay, as_a_flat_turn)
    told = None
    if best <= WRONG_SHAPE_ON_A_CLEAN_ONE and best <= 0.5 * other:
        told = "a swept delay" if as_a_delay < as_a_flat_turn else "the same angle at every partial"
    spread = float(np.percentile(implied, 84) - np.percentile(implied, 16))
    median = float(np.median(implied))
    return {
        "excursion_ms": round(median, 4),
        "excursion_spread_ms": round(spread, 4),
        "the_partials_agree": bool(spread < median),
        "as_a_delay": as_a_delay,
        "as_a_flat_turn": as_a_flat_turn,
        "what_the_phase_looks_like": told,
        "phase_swing_rad": [round(float(v), 5) for v in swing],
    }


def together(held: Partials) -> dict | None:
    """Whether the partials' levels move as one voice or as a comb over them.

    A comb puts a notch somewhere and sweeps it, so a partial sitting near the
    notch swings far more than one on a peak and the partials disagree in shape.
    One envelope over the whole voice moves every partial by the same decibels at
    the same moment, and cannot produce a disagreement.

    This is the gate the forward comb fit cannot supply for itself. A plain level
    modulation of the depth this unit's types show is fitted as a comb explaining
    more of its own series than any real comb does, so how well the fit explains
    the series is a gate against the wrong rate and none at all against the wrong
    mechanism.
    """
    if held.orders < 2:
        return None
    swings = [float(np.sqrt(np.mean(row**2))) for row in held.level_db]
    pairs = []
    for i in range(held.orders):
        for j in range(i + 1, held.orders):
            a, b = held.level_db[i], held.level_db[j]
            scale = float(np.std(a) * np.std(b))
            pairs.append(float(np.dot(a, b) / a.size / scale) if scale > 0 else 0.0)
    return {
        "swings_db": [round(s, 4) for s in swings],
        "swing_ratio": round(max(swings) / min(swings), 3) if min(swings) > 0 else None,
        "lowest_agreement": round(min(pairs), 4),
        "mean_agreement": round(float(np.mean(pairs)), 4),
    }


SHAPES = ("sine", "triangle")

SWING_STEP_RAD = 0.15
"""How finely the coarse scan walks the comb's phase swing.

Not a resolution, a requirement. The count of notches a swing produces is about
two swing over pi, so a swing changed by pi/2 moves the outermost notch by half a
period of the partial and the fit falls off the minimum entirely.
"""

DEEPEST_MIX = 0.95
"""How evenly the two paths may be mixed before the fit is sitting in a hole.

At a mix of one the comb's null is infinitely deep, and a model parked on that
null turns an arbitrarily small excursion into an arbitrarily large swing in dB.
The fit finds that branch and stays there. Capped rather than forbidden, because
a two path comb really can be mixed evenly; what the cap removes is the branch
where the excursion stops mattering.
"""

WIDEST_SWING_RAD = 70.0
"""The most comb phase swing scanned for, about 25 ms at 440 Hz."""

REFINE_FROM = 12
"""How many of the coarse scan's best candidates are refined. The surface has a
minimum wherever the notches line up, so the deepest one is often beside the one
the coarse grid happened to sample best."""

REFINE_PER_SEED = 3
"""How many of each seed's own best candidates are refined, whatever the rest did.

The coarse scan cannot rank seeds against each other. It holds the mix at three
values and the phase at sixteen, so a seed whose minimum needs a mix between two
of them scores badly coarsely and well once refined -- and pooling every seed's
rows into one list then refining the best of the pool leaves such a seed with no
refinement at all.

What it is worth is small and measured rather than assumed. One reading came back
explaining 0.077 of its series at a rate the take does not run at, with the right
rate sitting among its own seeds and inside its own search window, and refining
that seed returned 0.88 of the series without being told anything. Three other
readings that do cross the line the moment the rate is named do not cross it this
way, so a seed refined too few times is one cause of a fit landing wrong and not
the only one.
"""

SWING_SHOWS_DB = 0.30
"""Level swing, in dB RMS, below which there is nothing to fit a comb to. A tone
with no modulation on it fits every candidate perfectly, because they all reduce
to a flat line."""


def wave(shape: str, turns: np.ndarray) -> np.ndarray:
    """One cycle of a modulator, over turns of its own cycle, in [-1, 1]."""
    turns = turns % 1.0
    if shape == "sine":
        return np.sin(2.0 * np.pi * turns)
    if shape == "triangle":
        return 4.0 * np.abs(turns - np.floor(turns + 0.5)) - 1.0
    raise ValueError(shape)


def _comb_level(swings: np.ndarray, middle: float, mix: float, shaped: np.ndarray) -> np.ndarray:
    """The level a comb of that mix shows while its phase is swept like that."""
    level = 10.0 * np.log10(
        np.maximum(1.0 + mix * mix + 2.0 * mix * np.cos(middle + np.outer(swings, shaped)), 1e-9)
    )
    return level - level.mean(axis=1, keepdims=True)


def _residual(observed: np.ndarray, model: np.ndarray) -> float:
    """RMS left after the model is scaled to the observation as well as it can be.

    The scale is granted rather than fixed. The band the level is read through is
    narrow, so it keeps only the sidebands falling inside it and the swing that
    comes out is shallower than the comb's own by an amount this route does not
    model. Granting it to every candidate keeps that from deciding between them.
    """
    energy = float(np.dot(model, model))
    if energy <= 0:
        return float(np.sqrt(np.mean(observed**2)))
    scale = float(np.dot(observed, model)) / energy
    return float(np.sqrt(np.mean((observed - scale * model) ** 2)))


NEARLY_AS_WELL = 0.05
"""How much less of the series a fit may explain and still be an answer too.

The surface has a minimum wherever the notches line up, so asking whether another
minimum exists always returns yes and says nothing. What a reader needs is the
range of excursions that explain the series about as well as the winner, because
that is the width of what the reading actually settled. Five hundredths of the
series' own spread: below the difference between the two modulator shapes this
fit already declines to choose between at four tenths.

Not a gate. A width is reported and the winner is still the winner; a reader who
wants the number alone reads the number, and one who wants to know whether it was
determined reads the width beside it.
"""

THE_RATE_IS_SOUGHT_WITHIN = 0.1
"""How far either side of its seed the fitted rate may travel, as a proportion.

The rate is fitted here rather than taken from the projection, because the model
is far sharper in the rate than the projection that seeds it: a fiftieth of a
hertz out at half a hertz, this fit explains 0.16 of a series it explains 0.91 of
when placed. The projection cannot do better -- a level series with several
notches dragged past a partial in one cycle puts its largest peak on a multiple of
the rate, not on the rate -- so what it supplies is a neighbourhood and the model
settles the value inside it.

A tenth rather than wide enough to walk from one multiple of the rate to another.
The surface is multimodal in the rate, and an optimiser asked to cross a factor of
three in it stays where it started -- measured, not assumed. Reaching across is not
what the fit is for: the submultiples are enumerated as seeds, so what is left to
fit is the last few per cent the projection's grid cannot place.
"""


def fit_one(observed: np.ndarray, at: np.ndarray, seeds: tuple[float, ...], shape: str) -> dict:
    """Best fit of one shape: coarse over the whole space, then refined with the rate free.

    `seeds` are rates the projections put the modulator near. Each is scanned
    coarsely, and then two sets of candidates are refined: the best few of the
    pool, and the best few of every seed on its own. The second set is what keeps
    a seed from being crowded out of the pool by another seed's rows, which the
    coarse scan is not fine enough in the mix to rank fairly.
    """
    from scipy.optimize import minimize

    swings = np.arange(SWING_STEP_RAD, WIDEST_SWING_RAD, SWING_STEP_RAD)
    middles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    mixes = (0.4, 0.7, DEEPEST_MIX)
    starts = np.linspace(0.0, 1.0, 16, endpoint=False)

    found: list[tuple[float, ...]] = []
    for hz in sorted(set(seeds)):
        for start in starts:
            shaped = wave(shape, at * hz + start)
            for mix in mixes:
                for middle in middles:
                    level = _comb_level(swings, middle, mix, shaped)
                    along = level @ observed
                    energy = np.einsum("st,st->s", level, level)
                    left = float(np.dot(observed, observed)) - np.where(
                        energy > 0, along**2 / np.maximum(energy, 1e-30), 0.0
                    )
                    left = np.sqrt(np.maximum(left, 0.0) / observed.size)
                    where = int(np.argmin(left))
                    found.append((float(left[where]), float(swings[where]), middle, mix, start, hz))
    found.sort(key=lambda row: row[0])

    def costing(seed: float):
        """The cost this row is refined under, bounded around its own seed.

        Per seed and not over the whole list: the surface is multimodal in the
        rate, so a row starting on one multiple that was allowed to travel to
        another would be reporting a fit nobody seeded and nobody scanned.
        """
        low = seed * (1.0 - THE_RATE_IS_SOUGHT_WITHIN)
        high = seed * (1.0 + THE_RATE_IS_SOUGHT_WITHIN)

        def cost(free: np.ndarray) -> float:
            swing, middle, mix, start, hz = free
            if not (0.05 <= swing <= 200.0) or not (0.02 <= mix <= DEEPEST_MIX):
                return 1e6
            if not (low <= hz <= high):
                return 1e6
            model = _comb_level(np.array([swing]), middle, mix, wave(shape, at * hz + start))[0]
            return _residual(observed, model)

        return cost

    taken: dict[float, int] = {}
    rows = list(found[:REFINE_FROM])
    for row in found:
        if taken.get(row[5], 0) >= REFINE_PER_SEED:
            continue
        taken[row[5]] = taken.get(row[5], 0) + 1
        if row not in rows:
            rows.append(row)

    best = (found[0][0], np.array(found[0][1:], dtype=np.float64))
    settled: list[tuple[float, np.ndarray]] = []
    for row in rows:
        outcome = minimize(
            costing(row[5]),
            np.array(row[1:], dtype=np.float64),
            method="Nelder-Mead",
            options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 4000},
        )
        settled.append((float(outcome.fun), outcome.x))
        if outcome.fun < best[0]:
            best = (float(outcome.fun), outcome.x)
    swing, middle, mix, start, hz = best[1]
    return {
        "shape": shape,
        "left_over_db": round(best[0], 4),
        "settled_at": [(float(left), float(x[0])) for left, x in settled],
        "comb_phase_swing_rad": round(float(swing), 4),
        "mix": round(float(mix), 4),
        "starts_at_turn": round(float(start % 1.0), 4),
        "rate_hz": round(float(hz), 5),
    }


MARGIN_NEEDED = 0.40
"""How far apart two shapes' residuals must be, as a share of the level swing,
before one of them is named. Not a confidence: a line drawn where injected sweeps
of a known shape stopped being named correctly. A shallow sweep falls below it
because a sine and a triangle of the same small excursion really do write nearly
the same level series, so no shape may be inferred from a shape not being
offered."""


def comb_excursion_ms(swing_rad: float, partial_hz: float) -> float:
    """Peak-to-peak delay behind that much comb phase swing, in ms.

    The comb's phase is `2 pi f tau`, so a swing of one radian either side of the
    middle is an excursion of `2 / (2 pi f)` seconds peak to peak. This is the
    delay a two path comb would need to move its notches this far; on a type
    whose notches come from an all-pass section rather than from a delay line it
    is an equivalent and not a delay, and which of those a type is is not
    something a level series can say.
    """
    return 2.0 * swing_rad / (2.0 * np.pi * partial_hz) * 1000.0


def swept(
    signal: np.ndarray, rate: int, centre_ms: float, excursion_ms: float, hz: float, mix: float
) -> np.ndarray:
    """`(1-m) x + m x(t - D(t))`, for injecting a sweep of known size into a take.

    At a mix of one there is no comb, only the pitch, which is the case the phase
    route answers and the comb route must refuse. Below one the two paths beat and
    the comb appears. Both are needed as controls, because which of the two
    routes answers a type is one of the things being read.
    """
    n = np.arange(signal.size, dtype=np.float64)
    delay_ms = centre_ms + 0.5 * excursion_ms * np.sin(2.0 * np.pi * hz * n / rate)
    back = n - delay_ms / 1000.0 * rate
    clipped = np.clip(back, 0, signal.size - 2)
    low = np.floor(clipped).astype(np.int64)
    frac = clipped - low
    tapped = signal[low] * (1.0 - frac) + signal[low + 1] * frac
    tapped[back < 1.0] = 0.0
    return (1.0 - mix) * signal + mix * tapped


def comb(body: np.ndarray, rate: int, *seeds: float) -> dict:
    """The comb fitted forward on the strongest partial, near the rates it is given."""
    centre = rates.strongest_partial(body, rate)
    _, amplitude = rates.demodulate(body, rate, centre)
    edge = int(0.15 * rate)
    amplitude = amplitude[edge:-edge]
    in_db = 20.0 * np.log10(np.maximum(amplitude, amplitude.max() * 1e-6))
    step = max(1, int(rate / LEVEL_AT_HZ))
    whole = (in_db.size // step) * step
    level = in_db[:whole].reshape(-1, step).mean(axis=1)
    level = level - level.mean()
    at = np.arange(level.size, dtype=np.float64) / (rate / step)
    spread = float(np.sqrt(np.mean(level**2)))
    out = {
        "partial_hz": round(centre, 2),
        "seeded_at_hz": [round(float(h), 4) for h in seeds],
        "series_spread_db": round(spread, 4),
    }
    if spread < SWING_SHOWS_DB:
        return {
            **out,
            "excursion_ms": None,
            "why": "no level swing to fit a comb to, so nothing here is being swept "
            "against a direct path this reading can see",
        }
    fits = sorted(
        (fit_one(level, at, seeds, shape) for shape in SHAPES), key=lambda f: f["left_over_db"]
    )
    closer, other = fits
    apart = abs(closer["left_over_db"] - other["left_over_db"]) / spread
    explains = round(1.0 - closer["left_over_db"] / spread, 4)
    close = [
        comb_excursion_ms(swing, centre)
        for left, swing in closer["settled_at"]
        if left <= closer["left_over_db"] + NEARLY_AS_WELL * spread
    ]
    return {
        **out,
        "hz": closer["rate_hz"],
        "excursion_ms": round(comb_excursion_ms(closer["comb_phase_swing_rad"], centre), 4),
        "excursions_that_explain_it_about_as_well_ms": [
            round(min(close), 4),
            round(max(close), 4),
        ],
        "how_many_of_those": len(close),
        "comb_phase_swing_rad": closer["comb_phase_swing_rad"],
        "mix": closer["mix"],
        "left_over_db": {f["shape"]: f["left_over_db"] for f in fits},
        "explains": explains,
        "apart_by": round(apart, 4),
        "closer_to": None if apart < MARGIN_NEEDED else closer["shape"],
    }
