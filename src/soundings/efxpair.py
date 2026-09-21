"""What one insertion effect does between the unit's two outputs.

Every other reading in this archive is taken from one channel, which makes the second
output a thing records mention and nothing reads. The quantity that needs it is named
by a parked claim in its own words: two of its branches "want a stage this archive does
not have rather than a booking", and the takes both would be read out of are recorded.

**The control is read before the answer and decides whether there is one.** A lag
between two outputs is the output stage's only if it holds still across the take. A
flanger modulates its two sides against each other, so a number that swings with the
modulation is the modulator's whatever it is called -- and a reading whose own
within-take spread is the size of the thing being separated cannot separate it. The
spread is therefore published beside every lag and never folded into it.

**Two outputs that are not one signal have no lag to read.** Where the channels are
far apart in level and correlate poorly, the quieter one is a different signal rather
than a delayed copy, and a cross-correlation of the two still returns a number. Such a
setting is published saying so rather than with a figure nobody should use.

**The sign is stated as measured, not as derived.** The throwaway that established the
quantity carried a docstring with the sign inverted and the figures were unaffected,
because nothing in it depended on the direction. A published stage does, so the
convention is fixed here against a delay this module injects into a take and reads
back, and the recovery is published with every record.

Nothing here opens a MIDI port.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import phase as ph
from . import takes

BLOCK = 8192
"""Samples the lag is read over at a time, which is 171 ms at the capture's rate.

Short enough that a modulator running at a few hertz moves between blocks rather than
being averaged flat, which is what lets the control fire. Long enough that the
correlation has something to sit on: at this length the narrowest band the phase
reading uses still holds several cycles.
"""

PAIRED_WITHIN_DB = 30.0
"""How far apart the two outputs may sit and still be read as one signal.

Over this the quieter channel is not a delayed copy of the louder one, and a lag read
between them is a number about two different signals. Measured against the run's own
takes rather than an absolute: an effect that pans its wet path hard over is the case
this exists for, and it is refused by name instead of publishing a lag from it.
"""

CORRELATES_ABOVE = 0.5
"""The normalised correlation a block needs before its lag is a lag.

Under this the peak the interpolation refined is not distinguishable from the next one
along, so the sub-sample figure is precision on a number that has not been found.
"""

MOST_OF_THE_BLOCKS = 0.5
"""What fraction of a take's blocks must carry a lag before its median is one.

Not a bar on the answer but on what a median means. A median over blocks stands for
the take only if most of the take is in it; where a fifth of the blocks correlate and
four fifths do not, the median is a summary of the minority and reads exactly like a
summary of the take. The figure is a majority and not a tuned number -- it was not
moved after seeing which settings it admitted, and the settings it refuses are named
with their own numbers so a reader can disagree.
"""

WHY_THE_PAIR = (
    "The two channels are the pair the unit arrived on, taken from the interface's own "
    "pairing and not from which two were loudest. Which of the pair is the unit's first "
    "output is read from the output map published for this rig, because an interface "
    "presents inputs in pairs and says nothing about what was patched into them."
)

WHY_THE_SPREAD_IS_THE_CONTROL = (
    "A lag between two outputs is a property of the output stage only if it holds still "
    "while the effect runs. The take is cut into blocks and one lag is read per block, "
    "so a modulator that moves the two sides against each other shows as a spread "
    "within the take. Where that spread is the size of the difference being looked for, "
    "the difference has not been measured, and this is the figure that says so."
)

WHY_A_BYPASS_IS_READ_AGAINST_ANOTHER = (
    "The take with the part routed past the effect is read against a second such take "
    "rather than against its own floor. A reading of one take against itself returns "
    "nought by construction and would publish the stage's own arithmetic as the unit's "
    "null. Where the run recorded only one, no null is published and the record says so."
)


def _lag_of_block(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """How far `b` arrives after `a`, in samples, and how well the two correlate.

    Positive is `b` later. The peak of the cross-correlation sits at the negative of
    that, and the sign is turned here rather than left for a caller to remember --
    which is the mistake the throwaway made and did not notice.
    """
    a = a - a.mean()
    b = b - b.mean()
    size = 1 << (2 * len(a) - 1).bit_length()
    spectrum = np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size))
    whole = np.fft.irfft(spectrum, size)
    whole = np.concatenate([whole[-(len(a) - 1) :], whole[: len(a)]])
    at = int(np.argmax(whole))
    offset = 0.0
    if 0 < at < len(whole) - 1:
        left, middle, right = whole[at - 1], whole[at], whole[at + 1]
        bottom = left - 2 * middle + right
        offset = 0.5 * (left - right) / bottom if bottom else 0.0
    norm = float(np.sqrt((a * a).sum() * (b * b).sum()))
    peak = -(float(at - (len(a) - 1)) + offset)
    return peak, (float(whole[at] / norm) if norm else 0.0)


def _over_blocks(a: np.ndarray, b: np.ndarray, rate: int, block: int) -> tuple[list, list]:
    """One lag and one correlation per block, in microseconds."""
    lags, fits = [], []
    for start in range(0, max(0, len(a) - block), block // 2):
        lag, fit = _lag_of_block(a[start : start + block], b[start : start + block])
        lags.append(lag * 1e6 / rate)
        fits.append(fit)
    return lags, fits


def _recovers_an_injected_delay(a: np.ndarray, rate: int, block: int, *, by_us: float) -> dict:
    """Run the reading itself against a delay this module put in, and see what returns.

    **The check is the reading and not a simpler version of it.** A held note is
    periodic, so a cross-correlation taken over a whole take has a peak every period
    and picks whichever is largest -- read that way, a 50 us delay came back as -6769
    us at a correlation of 0.999, which is one period out and looks like a clean
    answer. The reading itself is taken block by block and summarised by a median, so
    the check is taken block by block and summarised by a median. What it measures is
    then what the record publishes, including the periodic ambiguity if there is one.

    A fractional delay is made in the frequency domain, which is the only way to shift
    by less than a sample without choosing an interpolator whose own error would be
    what the reading recovered.
    """
    shift = by_us * 1e-6 * rate
    spectrum = np.fft.rfft(a)
    turned = spectrum * np.exp(-2j * np.pi * np.arange(len(spectrum)) * shift / len(a))
    later = np.fft.irfft(turned, len(a))
    lags, fits = _over_blocks(a, later, rate, block)
    holding = [lag for lag, fit in zip(lags, fits, strict=True) if fit >= CORRELATES_ABOVE]
    got = float(np.median(holding)) if holding else None
    return {
        "injected_us": round(by_us, 3),
        "recovered_us": None if got is None else round(got, 3),
        "off_by_us": None if got is None else round(got - by_us, 3),
        "blocks": len(lags),
        "blocks_correlating": len(holding),
        "spread_us": (
            round(float(max(holding) - min(holding)), 3) if len(holding) > 1 else None
        ),
        "correlates": [round(min(fits), 3), round(max(fits), 3)] if fits else None,
        "why": (
            "A delay put into one channel by this module and read back by the same "
            "reading, block by block, exactly as a setting is read. It fixes the sign "
            "as measured rather than as derived, and it is what fails first if the "
            "block length or the interpolation stops resolving under a sample. Where "
            "`off_by_us` is large the take is periodic enough that the correlation has "
            "more than one peak to choose from, and every lag in the record is open to "
            "the same ambiguity by the same amount."
        ),
    }


def _read_take(
    frames: np.ndarray,
    rate: int,
    pair: tuple[int, int],
    *,
    first: int,
    last: int,
    block: int,
    bands,
    width_octaves: float,
) -> dict:
    """One take's lag, its spread within the take, and the phase band by band."""
    body = frames[first:last, list(pair)]
    lags, fits = _over_blocks(body[:, 0], body[:, 1], rate, block)
    holding = [lag for lag, fit in zip(lags, fits, strict=True) if fit >= CORRELATES_ABOVE]
    levels = takes.channel_levels(body)
    banded = ph.measure(
        body[:, 0], body[:, 1], rate, bands=bands, width_octaves=width_octaves
    )
    enough = len(holding) > MOST_OF_THE_BLOCKS * len(lags) if lags else False
    return {
        "blocks": len(lags),
        "blocks_correlating": len(holding),
        "most_of_the_blocks_carried_it": enough,
        "lag_us": round(float(np.median(holding)), 3) if enough else None,
        "median_of_the_minority_us": (
            None if enough or not holding else round(float(np.median(holding)), 3)
        ),
        "within_take_us": (
            round(float(max(holding) - min(holding)), 3) if len(holding) > 1 else None
        ),
        "correlates_over_every_block": (
            [round(min(fits), 3), round(max(fits), 3)] if fits else None
        ),
        "correlates": [round(min(fits), 3), round(max(fits), 3)] if fits else None,
        "apart_db": round(float(levels[0] - levels[1]), 2),
        "one_signal": abs(levels[0] - levels[1]) <= PAIRED_WITHIN_DB,
        "bands": banded["readings"],
        "less_a_delay_of": banded["less_a_delay_of"],
        "phase_aligned_by": banded["aligned_by"],
    }


def read_directory(
    root: str | Path,
    *,
    type_id: str,
    address: str | None,
    setting: str,
    bypassed: str | None = None,
    held: list[dict] | None = None,
    held_not_spelled_out: str | None = None,
    settled_s: float | None = None,
    lead_s: float = 0.6,
    trim_s: float = 0.2,
    hold_s: float = 8.0,
    block: int = BLOCK,
    bands=ph.THIRD_OCTAVES,
    width_octaves: float = 1 / 3,
    outputs: dict | None = None,
    progress=None,
) -> dict:
    """Read a take store and say what one effect did between the two outputs."""
    import json

    root = Path(root)
    manifest = json.loads((root / "takes-manifest.json").read_text())
    pattern = re.compile(setting)
    passing = re.compile(bypassed) if bypassed else None

    swept, aside, missed = [], [], []
    for entry in manifest["takes"]:
        name = entry.get("setting", entry["file"])
        if passing is not None and passing.search(name):
            aside.append(entry)
        elif (found := pattern.search(name)) is not None:
            swept.append((int(found.group("value")), entry))
        else:
            missed.append(name)

    if not swept:
        raise ValueError(f"no take under {root} matched {setting!r}")

    names = [entry["file"] for _, entry in swept] + [entry["file"] for entry in aside]
    pair, reached = takes.channel_pair_reaching(root, names)

    readings, checked = [], None
    for value, entry in sorted(swept, key=lambda row: row[0]):
        frames, rate = takes.read(root / entry["file"])
        first = int((lead_s + trim_s) * rate)
        last = int((lead_s + hold_s - trim_s) * rate)
        got = _read_take(
            frames, rate, pair, first=first, last=last, block=block,
            bands=bands, width_octaves=width_octaves,
        )
        if checked is None:
            checked = _recovers_an_injected_delay(
                frames[first:last, pair[0]], rate, block, by_us=50.0
            )
        readings.append({"value": value, "take": entry["file"], **got})
        if progress:
            progress(readings[-1])

    null = None
    if len(aside) > 1:
        frames, rate = takes.read(root / aside[0]["file"])
        other, _ = takes.read(root / aside[1]["file"])
        first = int((lead_s + trim_s) * rate)
        last = int((lead_s + hold_s - trim_s) * rate)
        span = min(last, len(frames), len(other))
        null = _read_take(
            np.concatenate(
                [frames[first:span, [pair[0]]], other[first:span, [pair[1]]]], axis=1
            ),
            rate, (0, 1), first=0, last=span - first, block=block,
            bands=bands, width_octaves=width_octaves,
        )

    return {
        "question": (
            "what one insertion effect does between the unit's two outputs, at each "
            "setting of one byte: how far the second arrives after the first, how far "
            "that moves within a take, and the phase between them band by band"
        ),
        "type": type_id,
        "address": address,
        "method": {
            "lag": (
                "Block cross-correlation between the two channels of one take, refined "
                "off the three points around the peak so a lag under the converter's own "
                "sample is a number. Positive is the second output arriving later."
            ),
            "the_control": WHY_THE_SPREAD_IS_THE_CONTROL,
            "the_null": WHY_A_BYPASS_IS_READ_AGAINST_ANOTHER,
            "block_samples": block,
            "correlates_above": CORRELATES_ABOVE,
            "paired_within_db": PAIRED_WITHIN_DB,
            "settled_s": settled_s,
            "lead_s": lead_s,
            "trim_s": trim_s,
            "hold_s": hold_s,
        },
        "recovers_an_injected_delay": checked,
        "channel": {
            "read": list(pair),
            "chosen_by": "the pair",
            "reached_db": reached,
            "named_by": outputs,
            "why": WHY_THE_PAIR,
        },
        "held": held or [],
        "why_held": held_not_spelled_out,
        "readings": readings,
        "settings_asked": sorted({row["value"] for row in readings}),
        "settings_admitted": sorted(
            {row["value"] for row in readings if row["lag_us"] is not None and row["one_signal"]}
        ),
        "settings_refused": sorted(
            {row["value"] for row in readings if row["lag_us"] is None or not row["one_signal"]}
        ),
        "why_refused": (
            "A setting is refused where the two channels are further apart in level than "
            "the run reads as one signal, or where fewer than most of a take's blocks "
            "carried a lag. Each refused reading keeps its own numbers -- the level "
            "difference, how many blocks correlated, and the median of those that did "
            "under `median_of_the_minority_us` -- so the refusal can be disagreed with."
        ),
        "floor": _floor(readings),
        "routed_past_the_effect": {
            "takes": [entry["file"] for entry in aside],
            "read_against": aside[1]["file"] if len(aside) > 1 else None,
            "why": WHY_A_BYPASS_IS_READ_AGAINST_ANOTHER,
            "with_nothing_in_its_path": null,
        },
        "limits": (
            "A lag is published only where the two outputs are one signal and the blocks "
            "correlate. Where they do not, the setting is named as refused rather than "
            "given a figure, because a cross-correlation of two different signals still "
            "returns a number. The within-take spread is the control and is never "
            "subtracted from the lag: a reading whose spread is the size of the "
            "difference being looked for has not separated it. The banded phase carries "
            "its own coherence per band, and a band under it is not a phase."
        ),
        "not_in_this_record": [
            "Which half of the pair is the unit's first output, unless `channel.named_by` "
            "carries it. The pairing is structural; the patching is a measurement and "
            "lives in the output map.",
            "Anything about what reaches two ears. These are two sockets.",
        ],
        "takes_from": str(root),
        "takes_not_matching": {"count": len(missed), "settings": missed},
    }


def _floor(readings: list[dict]) -> dict:
    """What the run's own repeats of one setting say a difference has to clear."""
    per_value: dict[int, list[float]] = {}
    for row in readings:
        if row["lag_us"] is not None:
            per_value.setdefault(row["value"], []).append(row["lag_us"])
    repeated = {value: got for value, got in per_value.items() if len(got) > 1}
    spreads = [max(got) - min(got) for got in repeated.values()]
    within = [row["within_take_us"] for row in readings if row["within_take_us"] is not None]
    return {
        "across_takes_us": round(max(spreads), 3) if spreads else None,
        "settings_taken_twice": sorted(repeated),
        "widest_within_a_take_us": round(max(within), 3) if within else None,
        "why": (
            "Two floors and they answer different questions. The first is how far one "
            "setting's own takes disagree, which is what a difference between settings "
            "has to clear. The second is how far one take disagrees with itself, which "
            "is what says whether the quantity is the output stage's at all."
        ),
    }
