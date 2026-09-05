"""A periodic modulation of pitch, which no comparison of two takes can measure.

`motion` measures a modulated *delay* and `audible` measures a difference between
two takes. Neither can say anything about a vibrato, and the reason is the one
`audible` already records: a free-running modulator is at a different phase every
time the note is struck, so two takes with it on disagree about nearly
everything. The yardstick swallows the change, and every parameter asked while
the modulator runs comes back inconclusive -- measured on this unit, takes of one
setting went from 0.2 dB apart to 10.6 with modulation switched on.

So a vibrato has to be measured *inside one take* rather than between two, and
what is measured is the note's own frequency over time. That is a quantity the
modulator does not scramble: its phase differs between takes, its **rate and
depth do not**.

**The frames have to be gated on the floor.** The same defect the delay tracker
had: a note decays into silence, and a frequency estimated from silence is the
noise's own peak reported with full confidence. Every frame is required to stand
over the floor measured in the take's own lead-in.

**A null needs a control.** A rate this cannot recover from a vibrato it put
there itself is not evidence that the unit has none, and there is no way to tell
the two apart from the output. `control` resamples the take with a known
modulation and reports which depths came back.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

METHOD = (
    "The note's own frequency was tracked through a single take and the track was searched for "
    "a periodic component. A vibrato cannot be measured by comparing two takes at all: a "
    "free-running modulator is at a different phase every time the note is struck, so the two "
    "disagree about nearly everything and the yardstick swallows any change. Its rate and depth "
    "are the same on every take even though its phase is not, so both are read inside one."
)

WHY_CONTROL = (
    "A modulation the tracker cannot recover from a take it was put into by hand is not evidence "
    "that the unit applied none, and nothing in the output separates the two. So every run puts "
    "a known modulation into one of its own takes and reports which depths came back. It bounds "
    "the run in both directions and neither alone: a null is worth reading only over the depths "
    "the control recovered, and a rate found at a depth shallower than the shallowest of those "
    "is a fit the tracker was never shown able to make on this material."
)

WHY_ONE_SETTING_CARRIES_THE_CONTROL = (
    "The control was injected into one setting's take rather than into each, and the setting is "
    "named beside this. It cannot be otherwise: a take that already carries a modulation ends up "
    "with two in the track, the search finds the unit's own, and the control reads as having "
    "failed -- measured here, a control injected into the takes with the vibrato on recovered "
    "nothing at any depth. The cost is that the depth reached here bounds the other settings only "
    "as far as their takes resemble this one, and the frames each row stood over the floor with "
    "are reported so that a reader can see where they do not."
)

SHALLOWER_THAN_THE_CONTROL = (
    "These rows report a modulation shallower than the shallowest depth the control recovered, so "
    "the tracker was never shown able to find one that small on this material. They are named "
    "rather than removed, because the rows are what the search returned and a row deleted for "
    "being unsupported leaves a setting looking as though nothing was found in it. What they are "
    "not is evidence that the unit modulated anything."
)

WHY_FLOOR_GATE = (
    "Frames were dropped for holding no signal as well as for finding no pitch. A frequency "
    "estimated from silence is the noise's own strongest period reported as a note, and a "
    "struck note spends most of a take decaying towards it."
)

FRAME_S = 0.04
"""Long enough for two periods of the lowest note asked here, short enough for 15 Hz."""

HOP_S = 0.005

FRAME_OVER_FLOOR_DB = 12.0
"""How far over the take's own lead-in a frame must stand to be tracked."""

SEARCH_HZ = (0.5, 15.0)
"""Where a vibrato is looked for. A null is a fact about this range."""

TREND_ORDER = 3
"""Order of the trend taken out before the track is searched.

A struck note settles in pitch as it decays and does not do it along a straight
line. Measured on a piano note with the vibrato depth at zero, a linear fit left
356 cents of curve behind, which read as a modulation at the bottom of the
search -- deeper than anything the unit's own vibrato produces.
"""

MIN_CYCLES = 3.0
"""Cycles the track must hold before a rate is claimed.

A trend that bends -- a note settling in pitch as it decays -- fits a slow
oscillation over one cycle as well as a real one does, and there is nothing in a
single cycle to tell them apart.
"""

CONTROL_DEPTHS_CENTS = (100.0, 50.0, 25.0, 12.0, 6.0, 3.0)
CONTROL_RATE_HZ = 5.3


@dataclass
class Wobble:
    """What one take's pitch did over time."""

    rate_hz: float = float("nan")
    depth_cents: float = float("nan")
    f0_hz: float = float("nan")
    tracked_frames: int = 0
    frames_below_floor: int = 0
    frames_without_pitch: int = 0
    floor_db: float = float("nan")
    searched_hz: tuple[float, float] = SEARCH_HZ
    notes: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(not np.isnan(self.rate_hz))

    def describe(self) -> str:
        if not self.found:
            return (
                f"no periodic pitch modulation between {self.searched_hz[0]} and "
                f"{self.searched_hz[1]} Hz. {self.tracked_frames} frames tracked, "
                f"{self.frames_below_floor} under the floor, "
                f"{self.frames_without_pitch} with no pitch in them"
            )
        return (
            f"pitch modulated at {self.rate_hz:.2f} Hz, {self.depth_cents:.1f} cents peak to "
            f"peak, around {self.f0_hz:.2f} Hz. {self.tracked_frames} frames tracked"
        )

    def to_json(self) -> dict:
        return {
            "rate_hz": None if np.isnan(self.rate_hz) else round(self.rate_hz, 3),
            "depth_cents": None if np.isnan(self.depth_cents) else round(self.depth_cents, 2),
            "f0_hz": None if np.isnan(self.f0_hz) else round(self.f0_hz, 3),
            "searched_hz": list(self.searched_hz),
            "tracked_frames": self.tracked_frames,
            "frames_below_floor": self.frames_below_floor,
            "frames_without_pitch": self.frames_without_pitch,
            "noise_floor_db": None if np.isnan(self.floor_db) else round(self.floor_db, 2),
            "notes": self.notes,
        }


def _db(x: float) -> float:
    return float(20.0 * np.log10(x + 1e-15))


def _rms_db(frame: np.ndarray) -> float:
    return _db(float(np.sqrt(np.mean(frame * frame))))


def _frame_pitch(frame: np.ndarray, rate: int, span: tuple[int, int]) -> float:
    """One frame's period, by autocorrelation with a parabolic peak.

    Autocorrelation rather than the phase of a bandpassed analytic signal: a
    struck note has partials either side of whatever band is chosen, and their
    beating walks the phase far enough to swamp a few cents of vibrato.
    """
    low, high = span
    x = frame - frame.mean()
    corr = np.correlate(x, x, mode="full")[len(x) - 1 :]
    if corr[0] <= 0:
        return float("nan")
    window = corr[low:high]
    if len(window) < 3:
        return float("nan")
    k = int(np.argmax(window))
    if k == 0 or k == len(window) - 1:
        return float("nan")
    a, b, c = window[k - 1], window[k], window[k + 1]
    denom = a - 2 * b + c
    offset = 0.0 if denom == 0 else 0.5 * (a - c) / denom
    return rate / (low + k + offset)


def track(
    x: np.ndarray,
    rate: int,
    *,
    lead_s: float = 0.6,
    low_hz: float = 60.0,
    high_hz: float = 2000.0,
) -> tuple[np.ndarray, Wobble]:
    """The note's frequency through the take, and what was dropped getting it."""
    found = Wobble()
    lead = int(lead_s * 0.8 * rate)
    found.floor_db = _rms_db(x[:lead]) if lead > rate // 100 else float("nan")
    gate = float("nan") if np.isnan(found.floor_db) else found.floor_db + FRAME_OVER_FLOOR_DB

    size, hop = int(FRAME_S * rate), int(HOP_S * rate)
    span = (max(1, int(rate / high_hz)), min(size - 1, int(rate / low_hz)))
    series = []
    for start in range(int(lead_s * rate), len(x) - size, hop):
        frame = x[start : start + size]
        if not np.isnan(gate) and _rms_db(frame) < gate:
            found.frames_below_floor += 1
            series.append(float("nan"))
            continue
        f0 = _frame_pitch(frame, rate, span)
        if np.isnan(f0):
            found.frames_without_pitch += 1
        series.append(f0)
    out = np.array(series, dtype=float)
    found.tracked_frames = int(np.count_nonzero(~np.isnan(out)))
    return out, found


def _rate_and_depth(cents: np.ndarray, hop_hz: float, search: tuple[float, float]):
    """The strongest periodic component of a pitch track, and how deep it is.

    The track is detrended first: a struck note settles in pitch as it decays,
    and a ramp puts its whole energy at the bottom of the search where it reads
    as a very slow, very deep modulation.
    """
    n = len(cents)
    if n < 8:
        return float("nan"), float("nan")
    # Cubic rather than linear. A struck note does not settle in pitch along a
    # straight line, and a linear fit leaves the curve behind: measured on a
    # piano note with the vibrato depth at zero, what was left read as 356 cents
    # of modulation at the bottom of the search.
    index_axis = np.arange(n)
    detrended = cents - np.polyval(np.polyfit(index_axis, cents, TREND_ORDER), index_axis)
    spectrum = np.abs(np.fft.rfft(detrended * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / hop_hz)
    # A rate needs enough of the track to have held it. One cycle of a slow
    # oscillation and one bend of a trend are the same curve.
    lowest = max(search[0], MIN_CYCLES * hop_hz / n)
    band = (freqs >= lowest) & (freqs <= search[1])
    if not band.any():
        return float("nan"), float("nan")
    index = int(np.argmax(spectrum[band]))
    # A maximum at the edge of the band is not a peak, it is whatever is left of
    # a trend leaning out of the search. The same reading that survives a cubic
    # detrend still lands there, and reporting the edge would put every note's
    # own settling at the slowest rate the search allows.
    if index == 0:
        return float("nan"), float("nan")
    rate = float(freqs[band][index])
    # Depth from the track itself at that rate, by projection, rather than from
    # the spectrum's own scale, which the window and the length both move.
    t = np.arange(n) / hop_hz
    basis = np.stack([np.cos(2 * np.pi * rate * t), np.sin(2 * np.pi * rate * t)])
    amplitude = float(np.hypot(*(basis @ detrended))) * 2.0 / n
    return rate, amplitude * 2.0


def measure(
    x: np.ndarray,
    rate: int,
    *,
    lead_s: float = 0.6,
    search_hz: tuple[float, float] = SEARCH_HZ,
    least_cents: float = 2.0,
) -> Wobble:
    """Say whether the note's pitch was modulated, and at what rate and depth."""
    series, found = track(x, rate, lead_s=lead_s)
    found.searched_hz = search_hz
    usable = ~np.isnan(series)
    if found.tracked_frames < 16:
        found.notes.append(WHY_FLOOR_GATE)
        return found
    filled = np.interp(np.arange(len(series)), np.flatnonzero(usable), series[usable])
    found.f0_hz = float(np.median(filled))
    cents = 1200.0 * np.log2(np.clip(filled, 1e-6, None) / found.f0_hz)
    rate_hz, depth = _rate_and_depth(cents, 1.0 / HOP_S, search_hz)
    if np.isnan(rate_hz) or depth < least_cents:
        found.notes.append(WHY_FLOOR_GATE)
        return found
    found.rate_hz, found.depth_cents = rate_hz, depth
    return found


def _with_vibrato(x: np.ndarray, rate: int, *, hz: float, cents: float) -> np.ndarray:
    """The take resampled so its pitch swings by a known depth at a known rate."""
    t = np.arange(len(x)) / rate
    ratio = 2.0 ** ((cents / 2.0) * np.sin(2 * np.pi * hz * t) / 1200.0)
    warped = np.cumsum(ratio) / rate
    return np.interp(warped, t, x)


def control(
    x: np.ndarray,
    rate: int,
    *,
    lead_s: float = 0.6,
    search_hz: tuple[float, float] = SEARCH_HZ,
    within_hz: float = 0.5,
) -> dict:
    """Put a known modulation into this take and report which depths came back.

    Always, not only when the answer is a null: knowing the tracker works on this
    material is what says a recovered rate is the unit's and not the search's.
    """
    recovered = []
    for depth in CONTROL_DEPTHS_CENTS:
        found = measure(
            _with_vibrato(x, rate, hz=CONTROL_RATE_HZ, cents=depth),
            rate,
            lead_s=lead_s,
            search_hz=search_hz,
        )
        if found.found and abs(found.rate_hz - CONTROL_RATE_HZ) <= within_hz:
            recovered.append(depth)
    return {
        "injected_rate_hz": CONTROL_RATE_HZ,
        "depths_tried_cents": list(CONTROL_DEPTHS_CENTS),
        "depths_recovered_cents": recovered,
        "shallowest_recovered_cents": min(recovered) if recovered else None,
        "why": WHY_CONTROL,
    }


def record(
    by_setting: dict[str, list[Wobble]],
    *,
    takes: str,
    searched_hz: tuple[float, float],
    control: dict | None,
    control_taken_from: str | None,
) -> dict:
    """The record a vibrato run leaves: its rows, and what the control does not reach.

    Assembled here rather than at the call site because the two limitations only
    exist once the rows and the control are both in hand, and a record that put
    them together by hand would leave them out of the next one.

    A control taken from one setting bounds every row, and one row can fall
    outside it in a direction the control's own prose used to leave unsaid.
    """
    floor = None if control is None else control.get("shallowest_recovered_cents")
    unsupported = [
        # Rounded as the row it points at is, so the same depth does not appear
        # twice in one record at two precisions and read as two measurements.
        {"setting": setting, "take": index, "depth_cents": round(found.depth_cents, 2)}
        for setting, rows in by_setting.items()
        for index, found in enumerate(rows)
        if found.found and floor is not None and found.depth_cents < floor
    ]
    return {
        "takes": takes,
        "method": METHOD,
        "searched_hz": list(searched_hz),
        "control": control,
        "control_taken_from": control_taken_from,
        **(
            {"why_one_setting_carries_the_control": WHY_ONE_SETTING_CARRIES_THE_CONTROL}
            if control_taken_from is not None
            else {}
        ),
        "shallower_than_the_control_recovered": {
            "rows": unsupported,
            "why": SHALLOWER_THAN_THE_CONTROL,
        },
        "by_setting": {
            setting: [found.to_json() for found in rows] for setting, rows in by_setting.items()
        },
    }


__all__ = [
    "CONTROL_DEPTHS_CENTS",
    "CONTROL_RATE_HZ",
    "METHOD",
    "SEARCH_HZ",
    "SHALLOWER_THAN_THE_CONTROL",
    "WHY_CONTROL",
    "WHY_FLOOR_GATE",
    "WHY_ONE_SETTING_CARRIES_THE_CONTROL",
    "Wobble",
    "control",
    "measure",
    "record",
    "track",
]
