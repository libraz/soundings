"""What an effect did to the phase of what went through it, band by band.

A band energy says how much of each band came out and nothing about when. Two
filters that pass the same energy in every band and answer a step differently are
the same measurement under `efx-bands` and different sounds, and telling them
apart is what a rendering has to get right before a recording and a rendered
buffer can be held against each other at all.

**The excitation is the machine's own voice, and that decides the method.** A
signal put into the analogue input does not reach the insertion effects on the
unit this was written against, so a phase response cannot be taken from a known
stimulus through a known path. What is left is two takes of one note -- the effect
flat, then the effect doing something -- and the phase between them. That works
only where the two takes are the same waveform, and whether they are is a
measurement rather than an assumption: a stimulus whose spectrum repeats exactly
and whose samples do not gives a perfect `efx-bands` record and no phase at all.

**So coherence is published beside every figure, and it is the control.** A phase
in a band where the two takes carry different signals is the angle between two
unrelated things: it is a number, it is stable enough to look like a reading, and
it means nothing. A record here states the coherence of each band and the rule
for reading it rather than quietly dropping the bands that failed, because which
bands a stimulus reaches is a fact about the stimulus worth keeping.

**Two takes of one note do not start at the same sample**, so what is measured is
the phase after an alignment this module chose, and a term proportional to
frequency belongs to that alignment rather than to the effect. The delay taken out
is published for the same reason: a reader who wants it back can have it. And the
two clocks -- the unit's and the converter's -- run at slightly different rates, so
the take is lined up once by whole samples and each block afterwards by a fraction
of one, and how far that moved across the take is reported.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .efxbands import THIRD_OCTAVES

BLOCK_S = 1.0
"""How long a block is aligned and transformed as one.

Short enough that two clocks a few tens of parts per million apart have not moved
a sample inside it, long enough to hold a few dozen transforms. Measured on the
unit this was written against: aligning once for an eight second take returns a
coherence of 0.06 at 1 kHz, and aligning a second at a time returns 0.71 from the
same takes.
"""

WINDOW = 2048
"""Samples per transform inside a block. At 48 kHz a little over forty milliseconds."""

READABLE = 0.5
"""The coherence below which a phase is not read as a phase.

Stated rather than applied: every band is published with its own coherence beside
it, and this is the line this project draws when it reads one of these records
back. Half is not a threshold the mathematics produces -- it is where a band holds
as much of the other take as of its own noise, and a reader who wants a stricter
line has the figures to draw it.
"""

WHY_COHERENCE = (
    "How much of the two takes is the same signal, band by band, on a scale where one "
    "is the same waveform and zero is two unrelated ones. It is the control on every "
    "phase in this record and not a quality note beside them: a phase measured between "
    "two takes that do not carry the same signal in a band is the angle between two "
    "unrelated things, and it is as steady and as plausible as a real one. Bands below "
    "the stated line are published rather than dropped, because which bands a stimulus "
    "carries into a measurement is a fact about the stimulus and belongs in the record "
    "that the stimulus produced."
)

WHY_ALIGNED = (
    "Where each block of the take was lined up, in samples, counting from the first "
    "sample of the dry take. The take is lined up once by whole samples and each block "
    "afterwards by a fraction of one, because the unit's clock and the converter's do "
    "not run at the same rate: how far the lag moved from the first block to the last "
    "is `drift_samples`, and over a take of a few seconds that is tens of parts per "
    "million. Choosing whole samples again per block would be the defect it looks like "
    "the cure for -- where the effect has turned the phase near a quarter turn the "
    "correlation is nearly as large a sample either side of the truth, so noise chooses "
    "between them, neighbouring blocks choose differently, and summing them subtracts "
    "half a turn at the top of the range from itself."
)

WHY_DELAY = (
    "The delay taken out of every figure beside `phase_deg_less_delay`, in samples, and "
    "how many of the transform's bins it was fitted over. An unknown delay between two "
    "takes is a straight line through the origin in frequency and so is a delay the "
    "effect itself put there: the two cannot be separated by this measurement, and "
    "removing the line removes both. What survives it is the part of the phase that is "
    "not a delay, which is the part a level and a band energy do not carry either. "
    "`phase_deg` beside it is the same figure with the line still in, so a reader who "
    "wants it back has it. The fit is made on the transform's own bins rather than on "
    "the bands because a lag of a few samples turns the phase through whole turns "
    "between one band and the next and through a fraction of one between neighbouring "
    "bins, and it is weighted by coherence so that bands the stimulus did not reach do "
    "not decide where the line runs."
)

LIMITS = (
    "A phase is reported per band and a band is a width: where the phase turns quickly "
    "inside one band the figures either side of the turn are averages over a turning "
    "quantity and the coherence of that band falls, which is what says so. The "
    "measurement is between two takes of one chain, so everything the two takes share "
    "-- the converter, the cable, the unit's output stage -- cancels and is not in "
    "these figures; what is in them is the difference the swept byte made. Nothing "
    "here is a delay, a group delay or an order: those are a fit across bands and the "
    "fit is not made in this archive. "
    "`bins` is how many of the transform's own bins the band held, and at the bottom of "
    "a third octave set that is one: the figure there is an average over the take's "
    "blocks and not over frequency at all, which is not a worse measurement but is a "
    "different one, and a band of one bin cannot report that the phase turned inside it."
)

CONTROL_FAILED = (
    "The control pair returned a phase of its own outside the bound stated beside it, "
    "so this record's figures cannot be told from what the method returns when nothing "
    "was done. Nothing above is a finding about the effect."
)


def align(first: np.ndarray, second: np.ndarray) -> tuple[int, float]:
    """The lag that lines `second` up with `first`, and the correlation there.

    Measured over whatever it is given rather than over a whole take: a stimulus
    that is noise correlates with itself at lags of seconds, so a search over an
    eight second take answers with one of those instead of with the offset between
    two recordings.

    **Whole samples only, and once for the take.** A lag chosen again per block by
    this estimator is the defect it looks like the cure for: where the effect has
    turned the phase by something near a quarter turn the correlation is nearly as
    large a sample either side of the truth, so noise chooses between them and
    neighbouring blocks choose differently. What each block is turned by instead is
    the line fitted to its own cross spectrum.
    """
    size = 1 << int(np.ceil(np.log2(first.size + second.size)))
    corr = np.fft.irfft(
        np.fft.rfft(first, n=size) * np.conj(np.fft.rfft(second, n=size)), n=size
    )
    corr = np.concatenate([corr[-(size // 2) :], corr[: size // 2]])
    scale = float(np.sqrt(np.sum(first**2) * np.sum(second**2))) or 1.0
    found = int(np.argmax(np.abs(corr)))
    return found - size // 2, float(abs(corr[found]) / scale)


def _wrapped(radians: float) -> float:
    """An angle in degrees, put back inside half a turn either way."""
    return round(float(np.degrees(np.angle(np.exp(1j * radians)))), 2)


SMOOTH = 17
"""Bins the coherence is averaged over before it is asked where the takes agree.

One bin's coherence is one draw. The question the average answers is where the
agreement holds rather than where it happened to, and an odd number so the average
stays centred on the bin it is asked about.
"""


def _agreeing(inside, coherent):
    """The longest stretch of bins where the two takes agree, out of those in range.

    **A line has to be fitted where there is a line.** Unwrapping runs through
    every bin it is given, and a bin whose coherence is nothing has a phase that is
    nothing, so a range carried past the point the takes stop agreeing unwraps a
    random walk and weighs it into the fit. Left in, it moved a block's lag by tens
    of samples between one block and the next, on takes whose clocks are tens of
    parts per million apart and cannot move it by more than one.
    """
    held = np.convolve(
        np.clip(coherent[inside], 0.0, 1.0), np.ones(SMOOTH) / SMOOTH, mode="same"
    )
    best_at = best_run = at = run = 0
    for i, agreeing in enumerate(held >= READABLE):
        if not agreeing:
            run = 0
            continue
        at = at if run else i
        run += 1
        if run > best_run:
            best_at, best_run = at, run
    # Nowhere is not everywhere. A run too short to fit a line over is answered
    # with no line rather than with one fitted to the whole range, which is a line
    # through a random walk and a delay through nothing.
    return inside[best_at : best_at + best_run] if best_run >= 8 else inside[:0]


def _line(freq, across, coherent, low: float, high: float):
    """The straight line the phase runs along, fitted where the takes agree.

    An unknown delay between two takes is exactly a line through the origin in
    frequency, so it cannot be separated from a delay the effect itself put there.
    The line is fitted on the transform's own bins rather than on the bands: a lag
    of a few samples turns the phase through many whole turns between one band and
    the next and cannot be unwrapped from them, and turns it by a fraction of one
    between neighbouring bins.
    """
    inside = np.flatnonzero((freq >= low) & (freq <= high) & (freq > 0))
    if inside.size < 8:
        return None, 0
    inside = _agreeing(inside, coherent)
    if inside.size < 8:
        return None, 0
    spot = freq[inside]
    turned = np.unwrap(np.angle(across[inside]))
    weight = np.clip(coherent[inside], 0.0, 1.0)
    if float(weight.sum()) <= 0:
        return None, 0
    total = float(weight.sum())
    mean_x = float((weight * spot).sum()) / total
    mean_y = float((weight * turned).sum()) / total
    spread = float((weight * (spot - mean_x) ** 2).sum())
    if spread <= 0:
        return None, 0
    together = float((weight * (spot - mean_x) * (turned - mean_y)).sum())
    return together / spread, int(inside.sum())


def _spectra(
    dry: np.ndarray,
    wet: np.ndarray,
    rate: int,
    *,
    block_s: float,
    window: int,
    low: float,
    high: float,
):
    """Cross and auto spectra, summed over every transform of every aligned block.

    **The take is lined up once by whole samples and each block afterwards by a
    fraction of one.** Choosing an integer lag per block looks like the same thing
    and is not: where the effect has turned the phase by something near a quarter
    turn the correlation is nearly as large a sample either side of the truth, so
    noise chooses between them, neighbouring blocks choose differently, and
    summing them subtracts half a turn at the top of the range from itself. What
    each block is turned by instead is the line fitted to its own cross spectrum,
    which has no whole samples in it to jump between.
    """
    have = min(dry.size, wet.size)
    # A block longer than the takes is one block over the whole of them rather than
    # no block at all: asking for a longer one is how a reader asks what the
    # measurement returns when the drift is left in, and returning nothing answers
    # that question with silence.
    block = min(max(int(block_s * rate), window * 2), have)
    coarse, _ = align(dry[:block], wet[:block])
    wet = np.roll(wet[:have], coarse)
    dry = dry[:have]
    trim = abs(coarse) + 1
    dry, wet = dry[trim : have - trim], wet[trim : have - trim]
    have = dry.size
    block = min(block, have)

    freq = np.fft.rfftfreq(window, 1.0 / rate)
    across = np.zeros(window // 2 + 1, dtype=complex)
    of_dry = np.zeros(window // 2 + 1)
    of_wet = np.zeros(window // 2 + 1)
    hop = np.hanning(window)
    turns: list[float] = []
    transforms = 0
    for start in range(0, have - block + 1, block):
        here_across = np.zeros(window // 2 + 1, dtype=complex)
        here_dry = np.zeros(window // 2 + 1)
        here_wet = np.zeros(window // 2 + 1)
        for at in range(start, start + block - window, window // 2):
            fd = np.fft.rfft(dry[at : at + window] * hop)
            fw = np.fft.rfft(wet[at : at + window] * hop)
            here_across += fw * np.conj(fd)
            here_dry += np.abs(fd) ** 2
            here_wet += np.abs(fw) ** 2
            transforms += 1
        coherent = np.abs(here_across) ** 2 / np.maximum(here_dry * here_wet, 1e-30)
        slope, _ = _line(freq, here_across, coherent, low, high)
        turns.append(0.0 if slope is None else slope)
        across += here_across * np.exp(-1j * turns[-1] * freq)
        of_dry += here_dry
        of_wet += here_wet
    return freq, across, of_dry, of_wet, coarse, turns, transforms


def measure(
    dry: np.ndarray,
    wet: np.ndarray,
    rate: int,
    *,
    bands=THIRD_OCTAVES,
    width_octaves: float = 1 / 3,
    block_s: float = BLOCK_S,
    window: int = WINDOW,
) -> dict:
    """The phase the wet take leads the dry one by, and the coherence of each band."""
    edge = 2 ** (width_octaves / 2)
    low, high = bands[0] / edge, bands[-1] * edge
    freq, across, of_dry, of_wet, coarse, turns, transforms = _spectra(
        dry, wet, rate, block_s=block_s, window=window, low=low, high=high
    )
    coherent = np.abs(across) ** 2 / np.maximum(of_dry * of_wet, 1e-30)
    # Each block was already turned by its own line, so what is left of a line
    # here is what those lines did not agree on. The delay this record says it
    # took out is the whole of it: the coarse lag, plus the mean of the lines.
    left, bins = _line(freq, across, coherent, low, high)
    slope = None if left is None else left + float(np.mean(turns))
    delay = (
        None
        if slope is None
        else round(float(coarse) - (slope / (2 * np.pi) * rate), 4)
    )
    lags = [round(float(coarse) - turn / (2 * np.pi) * rate, 3) for turn in turns]
    readings = []
    for centre in bands:
        first = int(np.searchsorted(freq, centre / edge, side="left"))
        last = int(np.searchsorted(freq, centre * edge, side="left"))
        if last <= first:
            readings.append(
                {
                    "hz": centre,
                    "phase_deg": None,
                    "phase_deg_less_delay": None,
                    "coherence": None,
                    "bins": 0,
                }
            )
            continue
        summed = across[first:last].sum()
        power = float(of_dry[first:last].sum() * of_wet[first:last].sum())
        angle = float(np.angle(summed))
        readings.append(
            {
                "hz": centre,
                # As measured after the whole-sample alignment and no further: the
                # blocks' own lines are put back so that what this column carries
                # is the line the column beside it had taken out.
                "phase_deg": _wrapped(angle + float(np.mean(turns)) * centre),
                "phase_deg_less_delay": None if left is None else _wrapped(angle - left * centre),
                "coherence": round(float(abs(summed) ** 2 / max(power, 1e-30)), 3),
                "bins": last - first,
            }
        )
    return {
        "bands_hz": list(bands),
        "band_width_octaves": round(width_octaves, 6),
        "block_s": block_s,
        "window_samples": window,
        "transforms": transforms,
        "aligned_by": {
            "lag_samples": lags,
            "drift_samples": round(lags[-1] - lags[0], 3) if len(lags) > 1 else None,
            "why": WHY_ALIGNED,
        },
        "less_a_delay_of": {
            "samples": delay,
            "fitted_over_bins": bins,
            "why": WHY_DELAY,
        },
        "readable_above": READABLE,
        "why_coherence": WHY_COHERENCE,
        "readings": readings,
    }


def _largest(readings) -> tuple[float | None, float | None]:
    return max(
        (
            (abs(r["phase_deg_less_delay"]), r["hz"])
            for r in readings
            if r["phase_deg_less_delay"] is not None
            and (r["coherence"] or 0.0) >= READABLE
        ),
        default=(None, None),
    )


def control(pairs, rate: int, **how) -> dict:
    """The same measurement between takes of ONE setting, over every pair of them.

    What the method returns when the answer is known to be nothing. A phase read
    between two repeats of one setting is the alignment's and the stimulus's, so
    the largest figure it returns in a band this record calls readable is the
    bound every figure in the record has to stand above to be the effect's. A
    negative without this is a claim that the method would have noticed.

    **Every pair of the repeats and not one of them.** One pair is one draw, and
    two repeats that happened to agree draw a bound that half the other pairs of
    the same setting would cross -- which is a bound that lets the run's own noise
    through as a reading. The largest of the pairs is what stands here, and each
    pair's own figure is kept beside it so a bound set by one outlying take can be
    seen to have been.
    """
    each = []
    for first, second in pairs:
        found = measure(first, second, rate, **how)
        worst = _largest(found["readings"])
        each.append(
            {
                "largest_deg": None if worst[0] is None else round(worst[0], 2),
                "largest_at_hz": worst[1],
                "readable_bands": sum(
                    1 for r in found["readings"] if (r["coherence"] or 0.0) >= READABLE
                ),
                "readings": found["readings"],
                "aligned_by": found["aligned_by"],
                "less_a_delay_of": found["less_a_delay_of"],
            }
        )
    standing = [p for p in each if p["largest_deg"] is not None]
    worst = max(standing, key=lambda p: p["largest_deg"], default=None)
    return {
        "largest_deg": None if worst is None else worst["largest_deg"],
        "largest_at_hz": None if worst is None else worst["largest_at_hz"],
        "readable_bands": max((p["readable_bands"] for p in each), default=0),
        "pairs": each,
        "why": "Takes of one setting read exactly as the pair above was, every pair of "
        "them. What they return is what the method returns for an effect that did "
        "nothing, and a figure above the largest of them is a figure and not a reading. "
        "Every pair rather than one because one pair is one draw: two repeats that "
        "happened to agree draw a bound the run's own noise would cross.",
    }


def reading_of(
    dry: str,
    wet: str,
    repeats: list[str],
    *,
    bands,
    width_octaves: float,
    block_s: float | None = None,
    lead_s: float,
    subject: dict,
    stimulus: str | None = None,
) -> dict:
    """Everything one pair of takes and its repeats make, assembled in one place.

    Here rather than in the command that calls it because this shape had two
    writers and they drifted. A driver assembling it beside the handler measured
    its control over every repeat and stored a command line naming a control set
    two takes short, so a replay of that line returned a different bound while
    every other field agreed -- a record that reads as reproducible and is not.
    One writer cannot disagree with itself.

    The control is read on the channel the pair was read on. A bound measured on
    the other leg of the unit bounds a comparison that was never made.
    """
    from . import takes as takestore

    dry_frames, wet_frames, rate, picked = takestore.read_pair(dry, wet)
    head = int(lead_s * rate)
    how = {"bands": bands, "width_octaves": width_octaves}
    if block_s is not None:
        how["block_s"] = block_s
    found = measure(dry_frames[head:], wet_frames[head:], rate, **how)

    vouched, pairs = None, list(combinations(repeats, 2))
    if pairs:
        loaded = []
        for first, second in pairs:
            a, b, control_rate, _ = takestore.read_pair(first, second, on=picked["read"])
            if control_rate != rate:
                raise ValueError(
                    f"a control pair is {control_rate} Hz and the pair is {rate} Hz"
                )
            loaded.append((a[head:], b[head:]))
        vouched = control(loaded, rate, **how)

    return {
        **subject,
        **({"stimulus": stimulus} if stimulus else {}),
        "takes_from": _one_directory(dry, wet, *repeats),
        "dry": str(dry),
        "wet": str(wet),
        # Named as the pairs they were, not as the takes they were drawn from.
        # `control.pairs` carries a figure per pair and no file names, so this is
        # the only place that says which two takes each of those figures is of.
        "control_from": [[str(a), str(b)] for a, b in pairs] or None,
        "sample_rate": rate,
        "channel": picked,
        "lead_s": lead_s,
        "limits": LIMITS,
        **found,
        **({"control": vouched} if vouched else {}),
        **(
            {"control_failed": CONTROL_FAILED}
            if vouched is not None and vouched["largest_deg"] is None
            else {}
        ),
        "conclusive": is_conclusive(found, vouched) if vouched else None,
    }


def _one_directory(*paths: str) -> str | None:
    """The directory the takes came from, where they all came from one.

    Left out rather than guessed at where they did not: a reading made across two
    directories has no single store behind it, and naming the first would say it
    had.
    """
    from pathlib import Path

    where = {str(Path(p).parent) for p in paths}
    return where.pop() if len(where) == 1 else None


def is_conclusive(found: dict, vouched: dict) -> bool:
    """Whether any band's phase stands above what the control returned."""
    bound = vouched.get("largest_deg")
    if bound is None:
        return False
    return any(
        r["phase_deg_less_delay"] is not None
        and (r["coherence"] or 0.0) >= READABLE
        and abs(r["phase_deg_less_delay"]) > bound
        for r in found["readings"]
    )


def describe(found: dict, vouched: dict | None = None) -> str:
    lines = []
    for reading in found["readings"]:
        if reading["phase_deg_less_delay"] is None:
            continue
        mark = " " if (reading["coherence"] or 0.0) >= READABLE else "?"
        lines.append(
            f"  {reading['hz']:>7.0f} Hz  {reading['phase_deg_less_delay']:+7.1f} deg"
            f"   coherence {reading['coherence']:.2f}{mark}"
        )
    readable = sum(
        1 for r in found["readings"] if (r["coherence"] or 0.0) >= READABLE
    )
    lines.append(
        f"  {readable} of {len(found['readings'])} bands carried the same signal in "
        f"both takes; the rest are printed with a ? and are not readings"
    )
    if vouched is not None:
        lines.append(
            "  control: two takes of one setting return "
            + (
                f"{vouched['largest_deg']:+.1f} deg at {vouched['largest_at_hz']:.0f} Hz"
                if vouched["largest_deg"] is not None
                else "no readable band at all, so nothing above is a finding"
            )
        )
    return "\n".join(lines)


__all__ = [
    "BLOCK_S",
    "CONTROL_FAILED",
    "LIMITS",
    "READABLE",
    "WHY_ALIGNED",
    "WHY_COHERENCE",
    "WHY_DELAY",
    "WINDOW",
    "align",
    "control",
    "describe",
    "is_conclusive",
    "measure",
]
