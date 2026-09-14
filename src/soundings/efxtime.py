"""Read what one insertion effect's delay slot put between a sound and its copy.

The quantity is a time, and the whole of this stage is that it is read off one
take rather than two. Two takes cannot carry it. Each take begins with a note-on
sent over a serial bus, and the scatter between two of them is milliseconds --
the same size as one step of a byte that covers half a second in a hundred and
twenty-eight places. A routed take correlated against a bypassed one measures the
bus and reports it as the effect.

One take carries it where the output holds both paths at once. A copy added to
its own source puts a ripple in the log of the spectrum whose period in frequency
is `1/D`, so the spectrum of that log -- the cepstrum -- has a peak at quefrency
`D`. Nothing in that needs the stimulus to be harmonic, and a noisy one is better
than a tonal one rather than worse: what the reading wants is a source whose own
log spectrum is smooth, and a held note's is a comb of its own.

**The even rahmonics are negative and that is the signature.** Expanding the log
gives `sum_n (-1)^(n+1) r^n cos(2 pi f n D) / n`, so the peak at `D` is positive,
the one at `2D` negative, the one at `3D` positive again. A reading that finds its
strongest neighbours at three and five times where it put the delay has found a
two-path comb; one that finds them at two and three times has found something
else, and the record publishes them so the difference can be seen rather than
taken on trust.

**What the reading cannot do.** It returns where a copy is, not how loud it is,
and it returns each copy separately -- so a feedback path appears as further peaks
at multiples of the delay and is not distinguished here from the rahmonics of a
single one. It returns nothing at all when there is no copy: at the bottom of a
printed range beginning at zero the peak is the carrier's own roughness, and that
is what the admission rule below is for. And it is bounded below by the frame the
cepstrum is taken over and above by the same frame, which is stated in the record
as the range that was searched rather than left to be inferred from the numbers
that came back.

**The floor is the run's own repeats.** A setting taken more than once gives how
far apart the reading put the same state twice, and that is the only figure that
says what a difference between two settings has to clear. Where no setting was
taken twice the record says so; it does not substitute the quefrency step, which
is the grid the answer is quantised to and not a measurement of anything.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import takes

VALUE = "value"
"""The named group a `--setting` pattern has to capture."""

QUESTION = (
    "Where one insertion effect's delay slot put the copy it returns, at each "
    "setting of its byte, in milliseconds."
)

METHOD = (
    "One take per setting, with the effect's balance at the point that carries both "
    "the direct path and the return, so the output holds a sound and its copy at "
    "once. The cepstrum of the held part of the take is averaged over every frame "
    "that fits in it, the same curve taken with the part routed past the effect is "
    "subtracted, and the strongest peak of what is left is the delay. Two further "
    "peaks are reported beside it. The take is read from one channel named for the "
    "whole run, and a setting taken more than once gives the run its floor."
)

LIMITS = (
    "The reading places a copy and does not weigh one. How much of the output the "
    "return is does not come out of a cepstral peak -- the peak's height moves with "
    "it, but it moves with the stimulus and with the take's own roughness too, so "
    "the height is published as how far the peak stood over that roughness and is "
    "not a proportion of anything. "
    "A second tap and a feedback path both appear as further peaks at multiples of "
    "the first, which is where a single copy's own rahmonics appear as well, so this "
    "stage does not tell them apart and does not try. What it publishes is where the "
    "peaks are. "
    "The range searched is bounded at both ends by the frame: below by where a "
    "quefrency is shorter than the analysis can resolve, above by half the frame. A "
    "delay outside it is not reported as absent, it is not reported at all, and the "
    "searched range is in this record so that a reader can see which of the two a "
    "missing setting is."
)

NOT_HERE = (
    "No curve and no table. Which shape these times lie on as the byte rises, and "
    "whether the steps between them are a stored table or a formula, is a fit across "
    "settings and types and is not made here. The printed range for this address is "
    "in documents/, under the same address, and is evidence about a page rather than "
    "about this unit; the two are not joined."
)

WHY_CHANNEL = (
    "The channel every figure in this record was read from, and the highest each "
    "channel of the interface reached while the takes with the effect out were "
    "sounding. Chosen once for the run rather than per take: the interface carries "
    "inputs the unit is not on, those inputs are not silent, and a take whose output "
    "falls below one of them is read from that input instead -- which returns a full "
    "plausible cepstrum of something else. `loudest_elsewhere` names every take whose "
    "own loudest channel is not the one used, because a reading that changed channel "
    "is exactly what the figures cannot say on their own."
)

WHY_EFFECT_OUT = (
    "The same chain with the part routed past the effect, which is what every "
    "reading below is taken against. It does two things and they are different. It "
    "is subtracted, so whatever the stimulus, the converters and the room put into a "
    "log spectrum is taken out of each reading rather than being read as a copy. And "
    "its own strongest peak is published, because a chain that puts a peak of its own "
    "somewhere is a chain that could hand a reading a delay the effect never made -- "
    "so a reader can see whether any setting landed near it."
)

WHY_ADMITTED = (
    "A peak is read as a delay only where it stood over the carrier's own roughness "
    "by the margin an injected comb was measured to need. Below that the strongest "
    "peak of a curve is wherever the roughness happened to be highest, which is a "
    "number that looks exactly like a short delay and is not one. The settings that "
    "fail it are listed with what they returned rather than dropped: a byte whose "
    "bottom settings return nothing is a reading about the bottom of the range, and a "
    "record that simply omitted them would read as a shorter sweep."
)

WHY_FLOOR = (
    "How far apart the reading put one setting's own repeats, which is what a "
    "difference between two settings has to clear to be a difference. Null where no "
    "setting was taken twice -- in which case this run measured no floor, and the "
    "quefrency step below is the grid the answers are quantised to and not a "
    "substitute for one."
)

WHY_RAHMONICS = (
    "The two next strongest peaks after the one read as the delay. A single copy puts "
    "its own at odd multiples of where it sits and not at even ones, because the even "
    "terms of the expanded log are negative; a second tap or a feedback path puts one "
    "at every multiple. Published so that which of those a reading found can be seen "
    "rather than assumed, and so that a first peak sitting at half of where the "
    "structure really is shows as a row whose neighbours do not fit."
)

WHY_HELD = (
    "Every address written before the takes were made, beside the one that was swept. "
    "A time read out of a cepstrum is a time through the whole chain, so what else "
    "was in the chain decides what the reading is of -- a feedback path returns "
    "further copies, a damping filter shapes them, and a balance that carries only "
    "the return leaves nothing for it to beat against and no peak at all."
)

STANDS_OUT = 6.0
"""How far a peak has to stand over the carrier's own roughness to be a delay.

Not chosen here. It is read off injection controls made on this archive's own
stimuli: every injected comb that came back within a twentieth of a millisecond
stood at or above this, and the best peak of every carrier with nothing in its
path stood below it. A bar set from the numbers this run produced would be the
gate tuned after the result, which is the one thing it may not be.
"""


def _cepstrum(
    body: np.ndarray, rate: int, *, frame: int, hop: int, searched_ms: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, int]:
    """The cepstrum of one take, averaged over every frame that fits in it.

    Averaged rather than taken once over the whole take. A delay that does not move
    puts its ripple in the same place in every frame, so the peak adds across them
    while the carrier's own roughness does not -- and with the modulator of a moving
    type stopped, or on a type that has none, that is the whole of what this reading
    buys over a single transform.
    """
    window = np.hanning(frame)
    starts = np.arange(0, max(body.size - frame, 1), hop)
    quefrency_ms = np.arange(frame // 2 + 1) / rate * 1000.0
    low, high = searched_ms
    band = (quefrency_ms >= low) & (quefrency_ms <= high)
    total = np.zeros(int(band.sum()))
    for start in starts:
        chunk = body[start : start + frame]
        if chunk.size < frame:
            break
        spectrum = np.fft.rfft(chunk * window)
        log_magnitude = np.log(np.abs(spectrum) + 1e-20)
        whole = np.fft.irfft(log_magnitude - log_magnitude.mean())
        total += whole[: frame // 2 + 1][band]
    return total / max(starts.size, 1), quefrency_ms[band], int(starts.size)


def _peaks(
    curve: np.ndarray, quefrency_ms: np.ndarray, *, how_many: int, apart_ms: float
) -> tuple[list[tuple[float, float]], float]:
    """The strongest separated peaks, each with how far it stood over the roughness.

    The roughness is the spread of the whole searched band rather than an assumed
    floor, so a row states its peak against the curve it was found on. Peaks are
    taken one at a time and a window either side of each is set aside, because the
    top of a broad peak is several quefrencies wide and a reading that did not do
    this would return one peak three times.
    """
    spread = float(np.std(curve))
    found: list[tuple[float, float]] = []
    taken = np.zeros_like(curve, dtype=bool)
    for _ in range(how_many):
        usable = ~taken
        if not usable.any():
            break
        at = int(np.argmax(np.where(usable, curve, -np.inf)))
        found.append((float(quefrency_ms[at]), float(curve[at]) / max(spread, 1e-12)))
        taken |= np.abs(quefrency_ms - quefrency_ms[at]) < apart_ms
    return found, spread


def _body(samples: np.ndarray, rate: int, *, index: int, lead_s: float, trim_s: float,
          hold_s: float | None) -> np.ndarray:
    one = takes.channel(samples, index)
    seconds = one.size / rate
    hold = hold_s if hold_s is not None else seconds - lead_s - trim_s
    first = int((lead_s + trim_s) * rate)
    last = int((lead_s + hold - trim_s) * rate)
    return np.asarray(one[first:last], dtype=np.float64)


def _matched(pattern, listed: dict, files: list[str]) -> list[tuple[str, dict, object]]:
    out = []
    for name in files:
        entry = listed.get(name, {})
        source, found = takes.named_by(pattern, entry, name)
        if found is not None:
            out.append((name, entry, (source, found)))
    return out


def read_directory(
    where: str | Path,
    *,
    type_id: str,
    address: str,
    setting: str,
    control: str,
    stimulus: str | None = None,
    held: list[dict] | None = None,
    channel: int | None = None,
    frame: int = 65536,
    hop: int = 16384,
    searched_ms: tuple[float, float] = (0.4, 620.0),
    apart_ms: float = 1.0,
    lead_s: float = 0.6,
    trim_s: float = 0.5,
    hold_s: float | None = None,
    progress=None,
) -> dict:
    """Every take under `where` whose setting matches, read into one record.

    Two patterns rather than four. There is no reference setting: a time is read
    absolutely and not as a deviation, so nothing here is reported against a flat
    take, and the run's floor comes from settings that were taken twice rather than
    from repeats of one state that is not in the sweep. What the takes with the
    effect out give is a curve to subtract and a statement about the chain, which
    is the control and not a reference.

    A take neither pattern names is counted rather than dropped: a pattern that
    matches nothing and a directory that holds nothing produce the same empty
    record otherwise, and they are different mistakes.
    """
    where = Path(where)
    listed, files = takes.listing(where)
    swept = takes.capturing(setting, VALUE)
    routed_past = re.compile(control)

    outs = _matched(routed_past, listed, files)
    if not outs:
        raise ValueError(f"no take under {where} matched the control {control!r}")

    # The channel from the takes with the effect out, which are the ones the unit
    # is certainly sounding in whatever the swept byte did to the level.
    reached, control_levels = takes.channel_reaching(where, [name for name, _, _ in outs])
    used = reached if channel is None else int(channel)
    elsewhere: list[str] = []

    def curve_of(name: str) -> tuple[np.ndarray, np.ndarray, int]:
        samples, rate = takes.read(where / name)
        own = int(np.argmax(takes.channel_levels(samples)))
        if own != used and name not in elsewhere:
            elsewhere.append(name)
        body = _body(samples, rate, index=used, lead_s=lead_s, trim_s=trim_s, hold_s=hold_s)
        return _cepstrum(body, rate, frame=frame, hop=hop, searched_ms=searched_ms)

    stacked, quefrency_ms, frames = None, None, 0
    for name, _, _ in outs:
        got, quefrency_ms, frames = curve_of(name)
        stacked = got if stacked is None else stacked + got
    floor_curve = stacked / len(outs)
    out_peaks, _ = _peaks(floor_curve, quefrency_ms, how_many=1, apart_ms=apart_ms)
    out_at, out_stands = out_peaks[0]

    claimed = {name for name, _, _ in outs}
    readings: list[dict] = []
    for name, _entry, (source, found) in _matched(swept, listed, files):
        claimed.add(name)
        got, _, _ = curve_of(name)
        peaks, spread = _peaks(got - floor_curve, quefrency_ms, how_many=3, apart_ms=apart_ms)
        (at, stands), rest = peaks[0], peaks[1:]
        reading = {
            VALUE: int(found[VALUE]),
            "take": name,
            "ms": round(at, 4),
            "stands": round(stands, 2),
            "admitted": bool(stands >= STANDS_OUT),
            "also_ms": [round(q, 4) for q, _ in rest],
            "also_stands": [round(s, 2) for _, s in rest],
            "roughness": round(spread, 8),
            "named_by": source,
        }
        readings.append(reading)
        if progress:
            progress(reading)

    readings.sort(key=lambda r: (r[VALUE], r["take"]))
    admitted = [r for r in readings if r["admitted"]]

    # The floor, from any setting the run took more than once. Only admitted rows
    # can give one: two settings that both returned the roughness are two readings
    # of the roughness, and how far apart they landed says nothing about how well
    # a delay repeats.
    seen: dict[int, list[float]] = {}
    for row in admitted:
        seen.setdefault(row[VALUE], []).append(row["ms"])
    spreads = [max(v) - min(v) for v in seen.values() if len(v) > 1]
    floor_ms = round(float(max(spreads)), 4) if spreads else None

    step_ms = float(quefrency_ms[1] - quefrency_ms[0]) if quefrency_ms.size > 1 else 0.0
    record = {
        "question": QUESTION,
        "type": type_id,
        "address": address,
        "method": METHOD,
        "limits": LIMITS,
        "not_in_this_record": NOT_HERE,
        "searched_ms": [round(v, 4) for v in searched_ms],
        "frame": frame,
        "hop": hop,
        "frames_averaged": frames,
        "quefrency_step_ms": round(step_ms, 6),
        "peaks_apart_ms": apart_ms,
        "stands_out": STANDS_OUT,
        "why_admitted": WHY_ADMITTED,
        "floor_ms": floor_ms,
        "settings_taken_twice": sorted(v for v, got in seen.items() if len(got) > 1),
        "why_floor": WHY_FLOOR,
        "why_also": WHY_RAHMONICS,
        "channel": {
            "read": used,
            "chosen_by": "given" if channel is not None else (
                "loudest in the takes with the effect out"
            ),
            "reference_db": control_levels,
            "loudest_elsewhere": sorted(elsewhere),
            "why": WHY_CHANNEL,
        },
        "with_the_effect_out": {
            "takes": sorted(name for name, _, _ in outs),
            "ms": round(out_at, 4),
            "stands": round(out_stands, 2),
            "would_be_read_as_a_delay": bool(out_stands >= STANDS_OUT),
            "why": WHY_EFFECT_OUT,
        },
        "held": held or [],
        "why_held": WHY_HELD,
        "takes_from": str(where),
        "manifest": takes.manifest_note(listed, files),
        "settings_asked": sorted({r[VALUE] for r in readings}),
        "settings_admitted": sorted({r[VALUE] for r in admitted}),
        "readings": readings,
        "takes_not_matching": takes.not_matching(
            [str(listed.get(name, {}).get("setting") or name)
             for name in files if name not in claimed]
        ),
    }
    if stimulus:
        record["stimulus"] = stimulus
    return record


__all__ = [
    "LIMITS",
    "METHOD",
    "NOT_HERE",
    "QUESTION",
    "STANDS_OUT",
    "VALUE",
    "read_directory",
]
