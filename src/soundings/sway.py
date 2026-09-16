"""A periodic modulation of how loud, and of how far to each side, inside one take.

`motion` measures a modulated *delay* and it is the only modulation reader this
harness had. Its positive control injects a swing in milliseconds, so on a type
whose modulator moves a level or a pan rather than a delay the control cannot pass
and never could: ten runs on two such types returned `control_failed` with takes
that swing by twenty-six decibels. The null was correct and the question was
unasked.

What is read here is the take's own level over time and the take's own difference
between channels over time. Both are quantities a free-running modulator does not
scramble between takes -- its phase differs on every note struck, its **rate,
depth and shape do not** -- which is the reason this is read inside one take, the
same reason `vibrato` is.

Reading both and reporting both is the point. A modulator that moves the mean and
leaves the difference alone, and one that moves the difference and leaves the mean
alone, are different stages of a part; asking only one of them would answer for
half the types and return a silent null for the other half.

**The level is read twice, in amplitude and in decibels, because nothing here knows
which the modulator works in.** A gain swung linearly and read logarithmically comes
back warped, and a warp is not symmetric, so it puts a second harmonic into a cycle
that has none -- measured here, a cycle whose second harmonic is a twentieth of its
first in amplitude reads as a fifth of it in decibels, on the same take. Which of
the two comes back the cleaner is the reading that says which domain the modulator
is applied in, and that is a finding rather than a setting to be chosen in advance.

**What is reported is the cycle, not a name for it.** Once a rate is found the
track is folded at that period and averaged, which gives one cycle of whatever the
modulator is, and the record carries that cycle together with the size of each
harmonic in it and how much of the cycle was spent rising. Which printed name that
cycle answers to is a comparison between this unit and a page, and it is made
where such comparisons are made.

**The frames are gated on the floor.** A level read out of silence is the noise's
own level, and it does not decay, so an untracked tail reads as a modulation
flattening out. Every frame has to stand over the floor measured in the take's own
lead-in.

**A null needs a control.** A modulation this cannot recover from one it put there
itself is not evidence the unit applied none. `control` multiplies the take by a
known swing and reports which depths came back, in both quantities.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

METHOD = (
    "The take's own level and its own difference between channels were tracked through a "
    "single take and each track was searched for a periodic component. Neither can be measured "
    "by comparing two takes: a free-running modulator is at a different phase every time the "
    "note is struck, so two takes with it on disagree about nearly everything. Its rate, its "
    "depth and the shape of its cycle are the same on every take even though its phase is not, "
    "so all three are read inside one. Where a rate was found the track was folded at that "
    "period and the cycles averaged, and what is reported is that cycle."
)

WHY_THE_TRACKS = (
    "Three tracks of one take, reported together, none of them the reading. The mean of the two "
    "channels and the difference between them, because a modulator moving the mean and leaving "
    "the difference alone and one moving the difference and leaving the mean alone are different "
    "stages, and asking only one would return a silent null for every type carrying the other -- "
    "and where a modulator moves the difference, the mean moves with it by however much the two "
    "channels fail to sum flat, so both are reported for both and the two depths are what say "
    "which quantity is driven. The mean twice, in amplitude and in decibels, because nothing "
    "here knows which of the two the modulator works in: a gain swung linearly and read "
    "logarithmically comes back warped, asymmetrically, which puts a second harmonic into a "
    "cycle that has none. Which reading comes back the cleaner says which domain the modulator "
    "is applied in."
)

WHY_CONTROL = (
    "A modulation this cannot recover from a take it was put into by hand is not evidence that "
    "the unit applied none, and nothing in the output separates the two. So every run multiplies "
    "one of its own takes by a known swing and reports which depths came back, in each of the "
    "two quantities. It bounds the run in both directions and neither alone: a null is worth "
    "reading only over the depths the control recovered, and a rate found at a depth shallower "
    "than the shallowest of those is a fit this was never shown able to make on this material."
)

WHY_CHANNEL = (
    "Both channels of the interface the unit is on, named rather than chosen. Every other stage "
    "here reduces a take to one channel; this one cannot, because half of what it reads is the "
    "difference between two. The pair is the pair that reached highest across the run, for the "
    "same reason a single channel is chosen once: an interface carries inputs the unit is not "
    "on, they are not silent, and a level track of an idle preamp finds no modulation -- which "
    "is what a setting with none also returns."
)

WHY_FLOOR_GATE = (
    "Frames were dropped for holding no signal. A level read out of silence is the noise's own "
    "level and it does not decay, so a tail tracked past the floor reads as a modulation "
    "flattening out -- and a struck note spends most of a take on its way there."
)

SHALLOWER_THAN_THE_CONTROL = (
    "These rows report a modulation shallower than the shallowest depth the control recovered, "
    "so this was never shown able to find one that small on this material. They are named rather "
    "than removed, because the rows are what the search returned and a row deleted for being "
    "unsupported leaves a setting looking as though nothing was found in it."
)

WHY_THE_CYCLE = (
    "One cycle of whatever was modulating, got by folding the track at the rate found and "
    "averaging the cycles. It is reported as a curve, as the size of each harmonic in it "
    "relative to the first, and as the fraction of the cycle spent going up. No name is put on "
    "it here: which printed shape a cycle answers to is a comparison between this unit and a "
    "page, and this record is only one of the two."
)

WHY_GOING_UP = (
    "How much of the cycle the curve spends climbing, as a fraction, read off the sign of its "
    "own slope. It is what separates two shapes the harmonics cannot: a cycle that rises slowly "
    "and drops at once, and one that jumps up and falls away slowly, hold every harmonic in the "
    "same proportion and differ only in which way round they are. Where a cycle is flat over "
    "most of its length the figure is read off whatever moves it there and means little, which "
    "is stated rather than left for a reader to infer from a number near a half."
)

FRAME_S = 0.02
"""Short enough that the fastest modulation searched is well inside the track's own
Nyquist, long enough to average over a held note's partials beating against each
other -- which is tens of hertz and out of the search band either way."""

HOP_S = 0.005

FRAME_OVER_FLOOR_DB = 12.0
"""How far over the take's own lead-in a frame must stand to be tracked."""

SEARCH_HZ = (0.3, 15.0)
"""Where a modulation is looked for. A null is a fact about this range."""

TREND_ORDER = 3
"""Order of the trend taken out before a track is searched.

A struck note decays, and a decay is most of what a level track holds. It does not
decay along a straight line either, so the same cubic `vibrato` needed on a pitch
track is needed here -- with the difference that here the trend is the larger part
of the signal rather than the smaller.
"""

MIN_CYCLES = 3.0
"""Cycles a track must hold before a rate is claimed. One cycle of a slow
oscillation and one bend of a decay are the same curve."""

CYCLE_POINTS = 64
"""Points the folded cycle is reported on. A power of two, and far more than the
harmonics the record reports, so nothing in the curve is an artefact of the grid."""

HARMONICS = 6
"""Harmonics reported beside the fundamental. The shapes a page of this era prints
are told apart by the first three; six is two octaves of margin over that."""

CONTROL_DEPTHS_DB = (24.0, 12.0, 6.0, 3.0, 1.5, 0.75)
CONTROL_RATE_HZ = 3.7


def _db(x: np.ndarray | float) -> np.ndarray | float:
    return 20.0 * np.log10(np.maximum(np.asarray(x, dtype=float), 1e-15))


@dataclass
class Track:
    """One quantity's modulation, and the cycle it turned out to have."""

    name: str = ""
    measured_in: str = "dB"
    rate_hz: float = float("nan")
    depth: float = float("nan")
    cycle: list[float] = field(default_factory=list)
    harmonics: list[float] = field(default_factory=list)
    going_up_fraction: float = float("nan")
    cycles_folded: int = 0
    not_found_because: str = ""

    @property
    def found(self) -> bool:
        return bool(not np.isnan(self.rate_hz))

    def describe(self) -> str:
        if not self.found:
            return f"{self.name}: no periodic component"
        harmonics = ", ".join(f"h{i + 2} {v:.3f}" for i, v in enumerate(self.harmonics[:3]))
        return (
            f"{self.name}: {self.rate_hz:.3f} Hz, {self.depth:.3f} {self.measured_in} peak to "
            f"peak, up {self.going_up_fraction:.3f} of the cycle, {harmonics}"
        )

    def to_json(self) -> dict:
        return {
            "measured_in": self.measured_in,
            "rate_hz": None if np.isnan(self.rate_hz) else round(self.rate_hz, 4),
            "depth": None if np.isnan(self.depth) else round(self.depth, 4),
            "cycle": [round(v, 5) for v in self.cycle],
            "harmonics_of_the_first": [round(v, 5) for v in self.harmonics],
            "going_up_fraction": (
                None if np.isnan(self.going_up_fraction) else round(self.going_up_fraction, 4)
            ),
            "cycles_folded": self.cycles_folded,
            "not_found_because": self.not_found_because or None,
        }


@dataclass
class Swing:
    """What one take's level and channel difference did over time."""

    level: Track = field(default_factory=lambda: Track("level", "of the mean"))
    level_in_db: Track = field(default_factory=lambda: Track("level in dB", "dB"))
    balance: Track = field(default_factory=lambda: Track("balance", "dB"))
    tracked_frames: int = 0
    frames_below_floor: int = 0
    floor_db: float = float("nan")
    searched_hz: tuple[float, float] = SEARCH_HZ
    notes: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.level.found or self.balance.found

    def describe(self) -> str:
        if not self.found:
            return (
                f"no periodic modulation of level or balance between {self.searched_hz[0]} "
                f"and {self.searched_hz[1]} Hz. {self.tracked_frames} frames tracked, "
                f"{self.frames_below_floor} under the floor"
            )
        return f"{self.level.describe()}; {self.balance.describe()}"

    def to_json(self) -> dict:
        return {
            "level": self.level.to_json(),
            "level_in_db": self.level_in_db.to_json(),
            "balance": self.balance.to_json(),
            "searched_hz": list(self.searched_hz),
            "tracked_frames": self.tracked_frames,
            "frames_below_floor": self.frames_below_floor,
            "noise_floor_db": None if np.isnan(self.floor_db) else round(self.floor_db, 2),
            "notes": self.notes,
        }


def tracks(
    pair: np.ndarray, rate: int, *, lead_s: float = 0.6
) -> tuple[dict[str, np.ndarray], Swing]:
    """The take's level in amplitude, its level in decibels, and its channel difference.

    Returned with the frames that held no signal marked, rather than removed: a
    gap in a track is not the same as a shorter track, and the search has to know
    where the gaps are before it fills them.
    """
    found = Swing()
    size, hop = int(FRAME_S * rate), int(HOP_S * rate)
    lead = int(lead_s * 0.8 * rate)
    quiet = pair[:lead] if lead > size else pair[:size]
    found.floor_db = float(np.mean(_db(np.sqrt((quiet.astype(float) ** 2).mean(0)))))
    gate = found.floor_db + FRAME_OVER_FLOOR_DB

    start = int(lead_s * rate)
    count = max(0, (len(pair) - size - start) // hop)
    out = {name: np.full(count, np.nan) for name in ("level", "level_in_db", "balance")}
    for index in range(count):
        frame = pair[start + index * hop : start + index * hop + size].astype(float)
        amplitude = np.sqrt((frame * frame).mean(0))
        sides = _db(amplitude)
        if float(np.mean(sides)) < gate:
            found.frames_below_floor += 1
            continue
        out["level"][index] = float(np.mean(amplitude))  # scaled to its own median below
        out["level_in_db"][index] = float(np.mean(sides))
        out["balance"][index] = float(sides[0] - sides[1])
    found.tracked_frames = int(np.count_nonzero(~np.isnan(out["level"])))
    return out, found


def _fill(series: np.ndarray) -> np.ndarray | None:
    """A track with its gaps interpolated, or None where too little was tracked."""
    usable = ~np.isnan(series)
    if int(np.count_nonzero(usable)) < 32:
        return None
    return np.interp(np.arange(len(series)), np.flatnonzero(usable), series[usable])


WHY_AT_THE_EDGE = (
    "The strongest component of this track sat in the lowest or the highest bin the search "
    "covers, which is not a peak: it is whatever leans into the band from outside it -- the "
    "take's own decay below, the frame-to-frame noise of the track above. The whole reading is "
    "refused rather than the one bin, because the next strongest bin on a track with nothing "
    "modulating it returns a depth at a rate inside the band and nothing in the numbers would "
    "say where it came from. A row refused this way bounds nothing outside `searched_hz`."
)


def _detrend(series: np.ndarray) -> np.ndarray:
    axis = np.arange(len(series))
    return series - np.polyval(np.polyfit(axis, series, TREND_ORDER), axis)


def _between_the_bins(spectrum: np.ndarray, k: int, step_hz: float) -> float:
    """How far past bin `k` the peak really sits, by a fit through its neighbours.

    A rate read as the bin it landed in is wrong by up to half a bin, and what that
    costs is not the rate -- it is the cycle. Folding `c` cycles on a period that is
    out by a fraction `e` smears the average by `c * e` of a cycle, so a reading
    made longer to fold more cycles gets worse rather than better at exactly the
    point it was made longer for. Measured on a sawtooth at 1 Hz over sixteen
    cycles: the bin was 2.5 per cent low, the fold came back forty per cent as deep
    with its rise and fall averaged away, and the same track read at the
    interpolated rate returned the sawtooth.

    A quadratic through the peak and its two neighbours, which is what a windowed
    magnitude spectrum is locally shaped like.
    """
    a, b, c = float(spectrum[k - 1]), float(spectrum[k]), float(spectrum[k + 1])
    curve = a - 2.0 * b + c
    if curve >= 0.0:
        return 0.0
    shift = 0.5 * (a - c) / curve
    # A peak more than half a bin from the bin it was found in is not this peak.
    return float(np.clip(shift, -0.5, 0.5)) * step_hz


def _rate_of(series: np.ndarray, hop_hz: float, search: tuple[float, float]):
    """The strongest periodic component of a detrended track, and why there is none."""
    n = len(series)
    spectrum = np.abs(np.fft.rfft(series * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / hop_hz)
    lowest = max(search[0], MIN_CYCLES * hop_hz / n)
    band = (freqs >= lowest) & (freqs <= search[1])
    if not band.any():
        return float("nan"), WHY_FLOOR_GATE
    index = int(np.argmax(spectrum[band]))
    # A maximum at either edge of the band is not a peak, it is what leans into
    # the search from outside it -- and the whole track is what is refused, not
    # that one bin. At the bottom it is a decay, and reporting it would put every
    # note's own settling at the slowest rate the search allows. At the top it is
    # the track's own frame-to-frame noise, and reporting it returns one figure
    # for every take in a run, which reads as a rate the byte does not move
    # rather than as no reading. Taking the next strongest bin instead would be
    # worse than either: on a track with no modulation in it that returns a large
    # depth at a rate inside the band, with nothing in the numbers saying so.
    if index in (0, int(band.sum()) - 1):
        return float("nan"), WHY_AT_THE_EDGE
    here = int(np.flatnonzero(band)[index])
    return float(freqs[here]) + _between_the_bins(
        spectrum, here, float(freqs[1] - freqs[0])
    ), ""


def _fold(series: np.ndarray, hop_hz: float, rate_hz: float) -> tuple[np.ndarray, int]:
    """One averaged cycle of the track, on a fixed grid, and how many went into it."""
    period = hop_hz / rate_hz
    cycles = int(len(series) // period)
    if cycles < 2:
        return np.array([]), 0
    grid = np.arange(CYCLE_POINTS) / CYCLE_POINTS
    stacked = np.stack(
        [
            np.interp((k + grid) * period, np.arange(len(series)), series)
            for k in range(cycles)
        ]
    )
    return stacked.mean(0), cycles


def _harmonics(cycle: np.ndarray) -> list[float]:
    """Each harmonic's size relative to the first, which no phase can move."""
    spectrum = np.abs(np.fft.rfft(cycle))
    if len(spectrum) < 2 or spectrum[1] <= 0:
        return []
    return [float(spectrum[k] / spectrum[1]) for k in range(2, min(HARMONICS + 2, len(spectrum)))]


def _going_up(cycle: np.ndarray) -> float:
    """How much of the cycle the curve spends climbing, off the sign of its slope.

    Where the fold started does not enter it, because the cycle is closed before
    the slope is taken. It is the one reading here that separates a curve from its
    own mirror image: two cycles differing only in which way round they run hold
    every harmonic in the same proportion, and a reading of magnitudes cannot see
    the difference at all.
    """
    if len(cycle) < 4:
        return float("nan")
    closed = np.concatenate([cycle, cycle[:1]])
    return float(np.mean(np.diff(closed) > 0))


def _read(series: np.ndarray, out: Track, hop_hz: float, search, least: float) -> Track:
    filled = _fill(series)
    if filled is None:
        return out
    detrended = _detrend(filled)
    rate_hz, why = _rate_of(detrended, hop_hz, search)
    if np.isnan(rate_hz):
        out.not_found_because = why
        return out
    cycle, cycles = _fold(detrended, hop_hz, rate_hz)
    if cycles < 2:
        return out
    depth = float(cycle.max() - cycle.min())
    if depth < least:
        return out
    out.rate_hz, out.depth, out.cycles_folded = rate_hz, depth, cycles
    out.cycle = [float(v) for v in cycle]
    out.harmonics = _harmonics(cycle)
    out.going_up_fraction = _going_up(cycle)
    return out


def measure(
    pair: np.ndarray,
    rate: int,
    *,
    lead_s: float = 0.6,
    search_hz: tuple[float, float] = SEARCH_HZ,
    least_db: float = 0.5,
) -> Swing:
    """Say whether the take's level or its channel difference was modulated."""
    series, found = tracks(pair, rate, lead_s=lead_s)
    found.searched_hz = search_hz
    if found.tracked_frames < 64:
        found.notes.append(WHY_FLOOR_GATE)
        return found
    hop_hz = 1.0 / HOP_S
    # The amplitude track is carried as a fraction of its own middle rather than
    # in whatever units the converter wrote. A take is as loud as the gain it was
    # recorded at, and a depth in those units is not comparable with the next
    # take's, let alone with the decibel reading beside it. Divided through, the
    # same bar means the same thing on every take and the two readings of the one
    # quantity can be put side by side.
    middle = float(np.nanmedian(series["level"]))
    fraction = series["level"] / middle if middle > 0 else series["level"]
    least_fraction = 10.0 ** (least_db / 20.0) - 1.0
    found.level = _read(fraction, found.level, hop_hz, search_hz, least_fraction)
    found.level_in_db = _read(
        series["level_in_db"], found.level_in_db, hop_hz, search_hz, least_db
    )
    found.balance = _read(series["balance"], found.balance, hop_hz, search_hz, least_db)
    return found


def _with_swing(pair: np.ndarray, rate: int, *, hz: float, db: float, panned: bool) -> np.ndarray:
    """The take multiplied by a known swing, in the mean or in the difference."""
    t = np.arange(len(pair)) / rate
    wave = np.sin(2 * np.pi * hz * t)[:, None]
    gain = 10.0 ** ((db / 2.0) * wave / 20.0)
    if not panned:
        return pair.astype(float) * gain
    return pair.astype(float) * np.concatenate([gain, 1.0 / gain], axis=1)


def control(
    pair: np.ndarray,
    rate: int,
    *,
    lead_s: float = 0.6,
    search_hz: tuple[float, float] = SEARCH_HZ,
    within_hz: float = 0.3,
) -> dict:
    """Put a known swing into this take and report which depths came back.

    Always, not only when the answer is a null: knowing this works on this
    material is what says a recovered rate is the unit's and not the search's.
    Once for each quantity, because a run that could read one of them and not the
    other would otherwise publish a null for the one it could not.
    """
    got: dict[str, list[float]] = {"level": [], "balance": []}
    for panned, key in ((False, "level"), (True, "balance")):
        for depth in CONTROL_DEPTHS_DB:
            found = measure(
                _with_swing(pair, rate, hz=CONTROL_RATE_HZ, db=depth, panned=panned),
                rate,
                lead_s=lead_s,
                search_hz=search_hz,
            )
            # The level control is counted recovered if either reading of the
            # level found it. They are two readings of one quantity, and a
            # control that demanded both would bound the run by the worse of them.
            track = found.balance if panned else found.level
            other = None if panned else found.level_in_db
            near = [
                t for t in (track, other)
                if t is not None and t.found and abs(t.rate_hz - CONTROL_RATE_HZ) <= within_hz
            ]
            if near:
                got[key].append(depth)
    return {
        "injected_rate_hz": CONTROL_RATE_HZ,
        "depths_tried_db": list(CONTROL_DEPTHS_DB),
        "depths_recovered_db": got,
        "shallowest_recovered_db": {
            key: (min(values) if values else None) for key, values in got.items()
        },
        "why": WHY_CONTROL,
    }


def record(
    by_setting: dict[str, list[Swing]],
    *,
    takes: str,
    searched_hz: tuple[float, float],
    control: dict | None,
    control_taken_from: str | None,
    channel: dict | None = None,
) -> dict:
    """The record a run of this leaves: its rows, and what the control does not reach.

    Assembled here rather than at the call site because the limitations only exist
    once the rows and the control are both in hand, and a record that put them
    together by hand would leave them out of the next one.
    """
    floors = {} if control is None else (control.get("shallowest_recovered_db") or {})
    unsupported = [
        {
            "setting": setting,
            "take": index,
            "quantity": name,
            "depth_db": round(track.depth, 3),
        }
        for setting, rows in by_setting.items()
        for index, found in enumerate(rows)
        for name, track in (
            ("level_in_db", found.level_in_db),
            ("balance", found.balance),
        )
        if track.found
        and floors.get(name.split("_in_")[0]) is not None
        and track.depth < floors[name.split("_in_")[0]]
    ]
    return {
        "takes": takes,
        "method": METHOD,
        "why_the_tracks": WHY_THE_TRACKS,
        "why_the_cycle": WHY_THE_CYCLE,
        "why_the_going_up_fraction": WHY_GOING_UP,
        "why_the_floor_gate": WHY_FLOOR_GATE,
        "searched_hz": list(searched_hz),
        "cycle_points": CYCLE_POINTS,
        **({"channel": channel} if channel is not None else {}),
        "control": control,
        "control_taken_from": control_taken_from,
        "shallower_than_the_control_recovered": {
            "rows": unsupported,
            "why": SHALLOWER_THAN_THE_CONTROL,
        },
        "by_setting": {
            setting: [found.to_json() for found in rows] for setting, rows in by_setting.items()
        },
    }


__all__ = [
    "CONTROL_DEPTHS_DB",
    "CONTROL_RATE_HZ",
    "CYCLE_POINTS",
    "HARMONICS",
    "METHOD",
    "SEARCH_HZ",
    "WHY_AT_THE_EDGE",
    "SHALLOWER_THAN_THE_CONTROL",
    "WHY_CHANNEL",
    "WHY_CONTROL",
    "WHY_FLOOR_GATE",
    "WHY_GOING_UP",
    "WHY_THE_CYCLE",
    "WHY_THE_TRACKS",
    "Swing",
    "Track",
    "control",
    "measure",
    "record",
    "tracks",
]
