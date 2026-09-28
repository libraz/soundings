"""The response a pair of note takes differ by, recovered from takes already saved.

A run strikes one stimulus several times at each of two settings of one byte: a
dry setting and a wet one. What this recovers is the stereo response `h` that,
convolved with the mean of the dry take's two channels, gives what the wet take
adds on each channel -- `wet_c = dry_c + h_c * (dry_L + dry_R) / 2`.

**Regularised deconvolution, averaged over pairs of takes.** The takes are
carried to the rate the graph models are drawn at and aligned to a fraction of a
sample, and `h` is the cross spectrum of what was added with the dry mono over
the dry mono's own auto spectrum plus one Tikhonov term, both summed over the
pairs. Nothing at or above `graph.CUT_HZ` is estimated.

**Where the wet takes sit against the dry ones is not in the takes.** A dry take
that is one panned voice, shifted by a fraction of a sample, differs from itself
by `pan_c` times its own derivative -- which is a response of the mono like any
other, so the deconvolution absorbs a misalignment instead of exposing it.
Correlating a wet take against a dry one is biased by the return's own
correlation with the voice. So dry takes are aligned among themselves, wet takes
among themselves, and the one lag between the two sets is the one that leaves no
dry derivative in the response within `ZERO_LAG_S` of zero.

**The dry take subtracted is never the one used as the input.** Two takes of one
setting differ by a little, and subtracting the input's own take would return
that difference, correlated with the input, as a spurious direct path.

**What the response is good to is measured, not assumed.** The takes are split
into two disjoint halves, each gives its own response, and their difference is
published beside the response, which is truncated where its own energy stops
clearing that difference. Each half's response is then asked to predict what the
other half's wet takes added.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import reproduce, takes
from .decay import OCTAVE_CENTRES
from .render import graph

RATE_HZ = 32000
"""The rate the response is recovered and published at: the rate graph models are drawn at."""

WINDOW_S = 0.02
"""The width the response's energy envelope is read in when it is truncated.

Twenty milliseconds is 640 samples on each of two channels at the published rate,
so the noise in one window is read to about half a decibel.
"""

BAND_CENTRES_HZ = OCTAVE_CENTRES
"""The octave bands the half-split floor and the input's level are read in."""

SIGNIFICANT = 4
"""Significant figures the published response is rounded to."""

ALIGN_STEP = 1e-4
"""Samples of lag below which one more alignment pass is not taken."""

ALIGN_PASSES = 10

ZERO_LAG_S = 0.001
"""How far either side of zero lag the response is read for a dry part left misaligned.

A wet take sitting a fraction of a sample off the dry ones leaves the dry's own
derivative in the response, within a few samples of zero. A millisecond holds that
and nothing of a return arriving later than it.
"""

QUESTION = (
    "What stereo response, convolved with the mean of the dry take's two channels, gives "
    "what the wet take adds on each channel."
)

METHOD = (
    "Every take was carried to 32000 Hz through the graph resampler. The dry takes were "
    "aligned to the first dry take, and the wet takes to the first wet take, by the cross "
    "correlation of the two channels' mean, interpolated to a fraction of a sample and "
    "repeated on its own shifted result until a pass moved it by less than 1e-4 samples; "
    "both channels were shifted by that one lag, with no level fitted. The wet set as a "
    "whole was then moved until the response held no component shaped as the dry's pan "
    "times its own derivative within 1 ms of zero lag. Within each half of the takes, "
    "every dry take was used as the input with every wet take, and the other dry take of "
    "that half subtracted from the wet one as the dry part. The response is the sum over "
    "the pairs of the cross spectrum of what was added with the input's mean, over the "
    "sum of the input's auto spectrum plus the Tikhonov term below, with nothing kept at "
    "or above the ceiling; the published response pools both halves."
)

LIMITS = (
    "The response is of the whole chain from the dry take's channel mean to the added "
    "part of the wet take, at the two settings asked and with everything else as held: "
    "whatever the wet setting changed beyond an addition linear in the dry mean is folded "
    "into it or left in the held-out residual, and nothing here tells the two apart. It is "
    "estimated only as far as the stimulus excited it: `input_over_regularisation_db` "
    "says per band how far the input stood above the Tikhonov term, and a band near or "
    "under zero there was pulled towards zero rather than measured. Nothing at or above "
    "`below_hz` is estimated. The takes end a fixed time after the note, so the part of "
    "the response later than the take's length after the note is not in them at all; "
    "`length_s` says where the response stopped clearing its own half-split noise, which "
    "is the only truncation made. Where the wet takes sit against the dry ones is fixed by "
    "taking the response to have nothing shaped as the dry's own derivative within "
    "`zero_lag_s` of zero: a return that did would have that part of it read as a "
    "misalignment and removed, and nothing in these takes can tell the two apart. "
    "`half_split` is the disagreement of two responses each "
    "made from half the takes; the published response averages both halves and was not "
    "itself compared with anything independent of them."
)

NOT_HERE = (
    "No claim about what produced the response -- which effect, which structure, which "
    "parameter table. This is a response recovered from takes, published with the "
    "disagreement it was recovered to."
)

WHY_CHANNEL = (
    "Which pair of channels every take was read from, as left and right, and the highest "
    "each channel of the interface reached across the takes read. An interface carries "
    "inputs the unit is not on, so the pair is named rather than picked per take."
)

WHY_ALIGNMENT = (
    "The lag each take was delayed by to sit on the reference, at 32000 Hz. The takes "
    "were each triggered on their own, so they start hundreds of samples apart. One pass "
    "of an interpolated correlation peak is biased by a few hundredths of a sample, which "
    "at the ceiling leaves the difference of two takes about 33 dB down, so the pass is "
    "repeated on its own shifted result until it stops moving. A wet take is not "
    "correlated against a dry one, because the return correlates with the voice and pulls "
    "the peak; the wet set is moved as one until the response holds no dry derivative "
    "within `zero_lag_s` of zero, `wet_set_left_samples` is what that last left, and "
    "`dry_pan` is each channel's gain on the dry takes' mean that the derivative is read "
    "with."
)

WHY_REGULARISATION = (
    "The Tikhonov term added to the input's auto spectrum at every bin. It is the input's "
    "own noise power per bin, read from the lead-in before the note of every input take "
    "and averaged below the ceiling, so a bin where the stimulus put no more than the "
    "noise is pulled towards zero instead of divided by noise. `value` is per input take, "
    "in squared sample units per bin of a transform the length of one take, and the sum "
    "over the pairs is what was added."
)

WHY_LENGTH = (
    "The response is published from `starts_at_samples`, which may be before zero lag, "
    "for `length_samples`. Walking outward from its loudest window, each way, it stops at "
    "the first window whose energy is less than twice the energy of the two halves' "
    "difference over four there: the pooled response carries a quarter of that difference "
    "as noise, so beyond that window the part of it that is not noise is smaller than the "
    "noise. Before zero, because a response cut at the ceiling rings on both sides of "
    "where it starts."
)

WHY_HALF_SPLIT = (
    "The energy of the difference between the two halves' responses, both cut to the "
    "published length, relative to the energy of their mean -- overall, and per octave "
    "band below the ceiling. This is what the response is good to: a feature of it "
    "smaller than this is not told apart from the difference of two sets of takes."
)

WHY_HELD_OUT = (
    "Each half's response, cut to the published length, convolved with the other half's "
    "dry inputs and subtracted from what that half's wet takes added, as energy relative "
    "to what was added. `takes_repeat_db` is the difference of that half's two dry takes "
    "on the same scale: a wet take's dry part is a different strike from the dry take "
    "subtracted from it, so no response predicts below roughly that."
)

WHY_LEVEL = (
    "`energy_db` is the sum of the squared response on the channel, the gain it has for a "
    "white input. `return_over_dry_db` is the energy of the response applied to each dry "
    "take's channel mean over that dry take's own energy on the channel, averaged over "
    "the dry takes."
)

WHY_WHOLE = (
    "The pooled response before truncation, over lags one take's length either side of "
    "zero, beside the noise the half-split difference puts in it over the same lags. The "
    "two meeting means nothing was recovered. `before_zero_db` is the part of the "
    "published response before zero lag, relative to all of it, and null where it starts "
    "at or after zero."
)

WHY_TAKES_END = (
    "The energy of each take's last window, in the two channels' mean, relative to its "
    "loudest window, averaged over the takes. A take that ends while the voice or its "
    "return is still sounding has lost the part of the response to that tail, and the "
    "deconvolution divides the loss back in wherever the input is thin; on a synthetic "
    "voice left 42 dB up at the end this alone held the recovery at -38 dB. Near the "
    "converter's own floor here means the takes held the whole event."
)

WHY_HELD = (
    "What else the run had written when it took these takes. The response is of the whole "
    "chain, so the same pair of settings taken with another byte moved is a response of a "
    "different chain, and nothing in the numbers says so."
)

WHY_HELD_FROM = (
    "Where `held` above was read from, rather than carried in by hand from another "
    "record's held block."
)


def _mono(pair: np.ndarray) -> np.ndarray:
    return pair.mean(axis=1)


def _db(num: float, den: float) -> float | None:
    if den <= 0.0 or num <= 0.0:
        return None
    return round(float(10.0 * np.log10(num / den)), 2)


def _aligned(pair: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, float, int]:
    """`pair` delayed onto `reference` by its channel mean, and the lag and passes taken."""
    mono = _mono(pair)
    n = reference.size
    size = reproduce.correlation_size(reference, mono)
    lag, passes = 0.0, 0
    while passes < ALIGN_PASSES:
        step = reproduce.lag_of(reference, reproduce.delayed(mono, lag, size)[:n], size)
        lag += step
        passes += 1
        if abs(step) < ALIGN_STEP:
            break
    out = np.stack(
        [reproduce.delayed(pair[:, c], lag, size)[:n] for c in range(2)], axis=1
    )
    return out, lag, passes


def _at_rate(pair: np.ndarray, rate: int) -> np.ndarray:
    return np.stack([graph.resampled(pair[:, c], rate, RATE_HZ) for c in range(2)], axis=1)


def _shifted(pair: np.ndarray, lag: float) -> np.ndarray:
    n = pair.shape[0]
    size = reproduce.correlation_size(pair[:, 0], pair[:, 0])
    return np.stack([reproduce.delayed(pair[:, c], lag, size)[:n] for c in range(2)], axis=1)


def _pan(dry: list[np.ndarray]) -> np.ndarray:
    """Each channel's gain on the dry takes' mean, over the dry takes: `dry_c = pan_c * mean`."""
    gains = [
        [float(np.dot(t[:, c], _mono(t)) / np.dot(_mono(t), _mono(t))) for c in range(2)]
        for t in dry
    ]
    return np.mean(np.array(gains), axis=0)


def _derivative(size: int, inband: np.ndarray) -> np.ndarray:
    """What delaying a unit impulse by one sample less than nothing adds, below the ceiling.

    The first-order term of `exp(-j w e) - 1` in `e`: a response recovered from wet
    takes sitting `e` samples late against the dry ones carries `e` of this, times the
    dry's own pan on each channel, at zero lag."""
    turns = np.fft.rfftfreq(size)
    spectrum = np.where(inband, -2j * np.pi * turns, 0.0)
    return np.fft.irfft(spectrum, size)


def _derivative_share(h: np.ndarray, derivative: np.ndarray, pan: np.ndarray) -> float:
    """How many samples late the wet set sits, read off the response within `ZERO_LAG_S`."""
    reach = int(round(ZERO_LAG_S * RATE_HZ))
    near = np.arange(-reach, reach + 1) % h.shape[0]
    basis = derivative[near]
    along = sum(pan[c] * float(np.dot(h[near, c], basis)) for c in range(2))
    power = sum(pan[c] ** 2 for c in range(2)) * float(np.dot(basis, basis))
    return along / power if power > 0 else 0.0


def _pairs(dry: list[np.ndarray], wet: list[np.ndarray]) -> list[tuple[int, int, int]]:
    """(input dry, subtracted dry, wet) for one half: the subtracted is never the input."""
    return [
        (k, m, j)
        for k in range(len(dry))
        for m in range(len(dry))
        if m != k
        for j in range(len(wet))
    ]


def _band_masks(freq: np.ndarray) -> list[np.ndarray]:
    root_two = np.sqrt(2.0)
    return [
        (freq >= c / root_two) & (freq < min(c * root_two, graph.CUT_HZ))
        for c in BAND_CENTRES_HZ
    ]


def _half(dry, wet, *, size, inband, lead):
    """The summed spectra of one half, and the per-input noise its Tikhonov term is made of."""
    n = dry[0].shape[0]
    sxx = np.zeros(size // 2 + 1)
    syx = np.zeros((size // 2 + 1, 2), dtype=complex)
    noise = []
    for k, m, j in _pairs(dry, wet):
        x = _mono(dry[k])
        big_x = np.fft.rfft(x, size)
        big_y = np.fft.rfft(wet[j] - dry[m], size, axis=0)
        sxx += np.abs(big_x) ** 2
        syx += big_y * np.conj(big_x)[:, None]
        quiet = x[lead[0] : lead[1]]
        noise.append(float(np.mean(np.abs(np.fft.rfft(quiet, size)[inband]) ** 2)) * n / quiet.size)
    return sxx, syx, noise


def _solved(sxx, syx, reg, inband, size) -> np.ndarray:
    spectrum = syx / (sxx + reg)[:, None]
    spectrum[~inband] = 0.0
    return np.fft.irfft(spectrum, size, axis=0)


def _energy(spectrum: np.ndarray) -> float:
    """Energy in the bins given, on the scale of `_db` ratios: only ever compared with another."""
    return float(np.sum(np.abs(spectrum) ** 2))


def _ended(pairs: list[np.ndarray], window: int) -> float | None:
    """The last window's energy relative to the loudest one, averaged over the takes, in dB."""
    ratios = []
    for pair in pairs:
        mono = _mono(pair)
        frames = mono[: (mono.size // window) * window].reshape(-1, window)
        energy = np.sum(frames**2, axis=1)
        if energy.max() > 0:
            ratios.append(energy[-1] / energy.max())
    return _db(float(np.mean(ratios)), 1.0) if ratios else None


def _lags(h: np.ndarray, start: int, stop: int) -> np.ndarray:
    """The lags `start` to `stop` of a circular response, negative lags from its end."""
    return h[np.arange(start, stop) % h.shape[0]]


def _placed(segment: np.ndarray, start: int, size: int) -> np.ndarray:
    """A span of lags put back into a circular response `size` long."""
    out = np.zeros((size, segment.shape[1]))
    out[np.arange(start, start + segment.shape[0]) % size] = segment
    return out


def _span(h: np.ndarray, difference: np.ndarray, *, n: int, window: int) -> tuple[int, int]:
    """The lags, on both sides of zero, the response clears its own noise over.

    Outward from its loudest window, each way, to the first window whose energy is less
    than twice what the halves' difference puts there. Outward rather than the last
    window to clear anywhere, because the deconvolution's noise is coloured and a
    single window far out swings past twice now and then. Both sides, because a
    response cut at the ceiling rings before its own zero as well as after it.
    """
    reach = (n // window) * window
    shape = (-1, window, h.shape[1])
    e_h = np.sum((_lags(h, -reach, reach) ** 2).reshape(shape), axis=(1, 2))
    e_n = np.sum((_lags(difference, -reach, reach) ** 2).reshape(shape), axis=(1, 2)) / 4.0
    clears = e_h >= 2.0 * e_n
    peak = int(np.argmax(e_h))
    if not clears[peak]:
        return 0, 0
    start, stop = peak, peak + 1
    while start > 0 and clears[start - 1]:
        start -= 1
    while stop < clears.size and clears[stop]:
        stop += 1
    return start * window - reach, stop * window - reach


def _rounded(h: np.ndarray) -> np.ndarray:
    return np.vectorize(lambda v: float(f"{v:.{SIGNIFICANT}g}"), otypes=[float])(h)


def recover(
    dry: list[np.ndarray], wet: list[np.ndarray], rate: int, *, lead_s: float
) -> dict:
    """The response the wet takes add over the dry ones, with its half-split floor.

    `dry` and `wet` are stereo takes, `(samples, 2)` at `rate`, in the order they
    are to be split in: the first half of each list is one half, the rest the other.
    Each half needs two dry takes and one wet one. `lead_s` is how long each take
    runs before the note, where the input's own noise is read.
    """
    if len(dry) < 4 or len(wet) < 2:
        raise ValueError(
            f"a half-split needs two dry takes and one wet one in each half; "
            f"given {len(dry)} dry and {len(wet)} wet"
        )
    n_in = min(t.shape[0] for t in [*dry, *wet])
    carried = [_at_rate(np.asarray(t[:n_in], dtype=float), rate) for t in [*dry, *wet]]
    raw_dry, raw_wet = carried[: len(dry)], carried[len(dry) :]
    n = raw_dry[0].shape[0]
    size = 1 << int(np.ceil(np.log2(2 * n)))
    freq = np.fft.rfftfreq(size, 1.0 / RATE_HZ)
    inband = freq < graph.CUT_HZ
    lead = (int(0.05 * RATE_HZ), int((lead_s - 0.05) * RATE_HZ))
    if lead[1] - lead[0] < int(0.1 * RATE_HZ):
        raise ValueError(f"a lead of {lead_s} s leaves too little before the note to read noise in")

    reference = _mono(raw_dry[0])
    dry32, dry_lags = [], []
    for take in raw_dry:
        out, lag, passes = _aligned(take, reference)
        dry32.append(out)
        dry_lags.append({"lag_samples": round(lag, 4), "passes": passes})
    pan = _pan(dry32)

    dh, wh = len(dry) // 2, len(wet) // 2
    split = {
        "dry": [list(range(dh)), list(range(dh, len(dry)))],
        "wet": [list(range(wh)), list(range(wh, len(wet)))],
    }
    # Wet takes are aligned to each other, which nothing biases since they share the
    # return; where the set sits against the dry takes is settled below.
    first, common, _ = _aligned(raw_wet[0], reference)
    among = [(0.0, 0)] + [
        (lag - common, passes)
        for _, lag, passes in (_aligned(take, _mono(first)) for take in raw_wet[1:])
    ]
    derivative = _derivative(size, inband)
    rounds, shared = 0, 0.0
    while True:
        wet32 = [
            _shifted(take, common + lag) for take, (lag, _) in zip(raw_wet, among, strict=True)
        ]
        halves = [(dry32[:dh], wet32[:wh]), (dry32[dh:], wet32[wh:])]
        spectra = [_half(d, w, size=size, inband=inband, lead=lead) for d, w in halves]
        per_input = float(np.mean([v for _, _, noise in spectra for v in noise]))
        regs = [per_input * len(noise) for _, _, noise in spectra]
        sxx_all = spectra[0][0] + spectra[1][0]
        h_all = _solved(sxx_all, spectra[0][1] + spectra[1][1], sum(regs), inband, size)
        rounds += 1
        shared = _derivative_share(h_all, derivative, pan)
        if abs(shared) < ALIGN_STEP or rounds >= ALIGN_PASSES:
            break
        common -= shared
    h_halves = [
        _solved(sxx, syx, reg, inband, size)
        for (sxx, syx, _), reg in zip(spectra, regs, strict=True)
    ]
    lags = dry_lags + [
        {"lag_samples": round(common + lag, 4), "passes": passes} for lag, passes in among
    ]

    window = int(round(WINDOW_S * RATE_HZ))
    difference = h_halves[0] - h_halves[1]
    start, stop = _span(h_all, difference, n=n, window=window)
    length = stop - start
    kept = _lags(h_all, start, stop)
    published = _rounded(kept)
    halves_cut = [_lags(h, start, stop) for h in h_halves]
    spec_halves = [np.fft.rfft(_placed(h, start, size), size, axis=0) for h in halves_cut]

    band_masks = _band_masks(freq)
    if length:
        gap = np.sum(np.abs(spec_halves[0] - spec_halves[1]) ** 2, axis=1)
        mean = np.sum(np.abs((spec_halves[0] + spec_halves[1]) / 2.0) ** 2, axis=1)
        overall = _db(float(gap[inband].sum()), float(mean[inband].sum()))
        bands = [_db(float(gap[b].sum()), float(mean[b].sum())) for b in band_masks]
    else:
        overall, bands = None, [None] * len(band_masks)

    held_out = []
    for est, test in ((0, 1), (1, 0)):
        d, w = halves[test]
        added = left = repeat = 0.0
        for k, m, j in _pairs(d, w):
            spec_y = np.fft.rfft(w[j] - d[m], size, axis=0)[inband]
            guess = (np.fft.rfft(_mono(d[k]), size)[:, None] * spec_halves[est])[inband]
            added += _energy(spec_y)
            left += _energy(spec_y - guess)
            repeat += _energy(np.fft.rfft(d[k] - d[m], size, axis=0)[inband])
        held_out.append(
            {
                "estimated_on": {key: split[key][est] for key in split},
                "predicting": {key: split[key][test] for key in split},
                "residual_db": _db(left, added),
                "takes_repeat_db": _db(repeat, added),
            }
        )

    level = {}
    spec_kept = np.fft.rfft(_placed(kept, start, size), size, axis=0)[inband]
    for c, side in enumerate(("left", "right")):
        back = sum(_energy(np.fft.rfft(_mono(t), size)[inband] * spec_kept[:, c]) for t in dry32)
        own = sum(_energy(np.fft.rfft(t[:, c], size)[inband]) for t in dry32)
        level[side] = {
            "energy_db": _db(float(np.sum(kept[:, c] ** 2)), 1.0),
            "return_over_dry_db": _db(back, own),
        }

    in_band_sxx = float(np.mean(sxx_all[inband]))
    return {
        "response": {
            "rate_hz": RATE_HZ,
            "below_hz": graph.CUT_HZ,
            "starts_at_samples": start,
            "length_samples": length,
            "length_s": round(length / RATE_HZ, 4),
            "window_s": WINDOW_S,
            "why_length": WHY_LENGTH,
            "significant_figures": SIGNIFICANT,
            "rounding_db": _db(float(np.sum((published - kept) ** 2)), float(np.sum(kept**2))),
            "left": published[:, 0].tolist(),
            "right": published[:, 1].tolist(),
        },
        "regularisation": {
            "value": per_input,
            "under_the_input_db": _db(sum(regs), in_band_sxx),
            "input_over_regularisation_db": [
                {"centre_hz": c, "db": _db(float(np.mean(sxx_all[b])), sum(regs))}
                for c, b in zip(BAND_CENTRES_HZ, band_masks, strict=True)
            ],
            "read_from_s": [round(lead[0] / RATE_HZ, 3), round(lead[1] / RATE_HZ, 3)],
            "why": WHY_REGULARISATION,
        },
        "half_split": {
            "split": split,
            "overall_db": overall,
            "bands": [
                {"centre_hz": c, "difference_db": v}
                for c, v in zip(BAND_CENTRES_HZ, bands, strict=True)
            ],
            "why": WHY_HALF_SPLIT,
        },
        "held_out": held_out,
        "level": level,
        "whole_estimate": {
            "energy_db": _db(float(np.sum(_lags(h_all, -n, n) ** 2)), 1.0),
            "noise_db": _db(float(np.sum(_lags(difference, -n, n) ** 2)) / 4.0, 1.0),
            "before_zero_db": _db(
                float(np.sum(kept[: max(-start, 0)] ** 2)), float(np.sum(kept**2))
            ),
            "why": WHY_WHOLE,
        },
        "takes_end": {
            "dry_db": _ended(dry32, window),
            "wet_db": _ended(wet32, window),
            "why": WHY_TAKES_END,
        },
        "alignment": {
            "reference": 0,
            "at_hz": RATE_HZ,
            "wet_set_rounds": rounds,
            "wet_set_left_samples": round(shared, 5),
            "zero_lag_s": ZERO_LAG_S,
            "dry_pan": [round(float(v), 4) for v in pan],
            "takes": lags,
            "why": WHY_ALIGNMENT,
        },
    }


def read_directory(
    where: str | Path,
    *,
    type_id: str | None,
    address: str | None = None,
    controller: int | None = None,
    dry: str,
    wet: str,
    stimulus: str | None = None,
    held: list[dict] | None = None,
    held_from: str | None = None,
    held_not_spelled_out: str | None = None,
    channels: tuple[int, int] | None = None,
    lead_s: float = 0.6,
) -> dict:
    """The dry and wet takes under `where`, read into one record.

    `dry` and `wet` are patterns over each take's setting; a take both name is
    refused, since it cannot be on both sides of the difference. Takes are split
    into halves in file-name order.
    """
    if (address is None) == (controller is None):
        raise ValueError("a record is about one address or one controller, not both or neither")
    where = Path(where)
    listed, files = takes.listing(where)
    dry_re, wet_re = re.compile(dry), re.compile(wet)
    dry_names, wet_names, skipped = [], [], []
    for name in files:
        entry = listed.get(name, {})
        _, is_dry = takes.named_by(dry_re, entry, name)
        _, is_wet = takes.named_by(wet_re, entry, name)
        if is_dry and is_wet:
            raise ValueError(f"{name} is named by both --dry and --wet")
        if is_dry:
            dry_names.append(name)
        elif is_wet:
            wet_names.append(name)
        else:
            skipped.append(str(entry.get("setting") or name))
    dry_names.sort()
    wet_names.sort()
    names = dry_names + wet_names
    if channels is None:
        picked, levels = takes.channel_pair_reaching(where, names)
        used = tuple(picked)
    else:
        _, levels = takes.channel_pair_reaching(where, names)
        used = tuple(channels)

    loaded, rate = [], None
    for name in names:
        samples, got = takes.read(where / name)
        if rate is not None and got != rate:
            raise ValueError(f"{name} is at {got} Hz and the takes before it at {rate} Hz")
        rate = got
        loaded.append(np.asarray(samples[:, list(used)], dtype=np.float64))
    found = recover(loaded[: len(dry_names)], loaded[len(dry_names) :], rate, lead_s=lead_s)

    def named(indices: list[int], pool: list[str]) -> list[str]:
        return [pool[i] for i in indices]

    split = found["half_split"]["split"]
    found["half_split"]["split"] = {
        "dry": [named(h, dry_names) for h in split["dry"]],
        "wet": [named(h, wet_names) for h in split["wet"]],
    }
    for row in found["held_out"]:
        for key in ("estimated_on", "predicting"):
            row[key] = {
                "dry": named(row[key]["dry"], dry_names),
                "wet": named(row[key]["wet"], wet_names),
            }
    found["alignment"]["reference"] = names[0]
    found["alignment"]["takes"] = [
        {"take": name, **lag} for name, lag in zip(names, found["alignment"]["takes"], strict=True)
    ]

    record = {
        "question": QUESTION,
        "type": type_id,
        **({"address": address} if address is not None else {"controller": controller}),
        "method": METHOD,
        "limits": LIMITS,
        "not_in_this_record": NOT_HERE,
        "channel": {
            "read": list(used),
            "as": ["left", "right"],
            "chosen_by": "given" if channels is not None else "the pair that reached highest",
            "reached_db": levels,
            "why": WHY_CHANNEL,
        },
        "dry": {"pattern": dry, "takes": dry_names},
        "wet": {"pattern": wet, "takes": wet_names},
        "lead_s": lead_s,
        "held": held or [],
        "why_held": WHY_HELD,
        **({"held_from": held_from, "why_held_from": WHY_HELD_FROM} if held_from else {}),
        **({"held_not_spelled_out": held_not_spelled_out} if held_not_spelled_out else {}),
        "takes_from": str(where),
        "manifest": takes.manifest_note(listed, files),
        "takes_not_matching": takes.not_matching(skipped),
        "why_level": WHY_LEVEL,
        "why_held_out": WHY_HELD_OUT,
        **found,
    }
    if stimulus:
        record["stimulus"] = stimulus
    return record


__all__ = [
    "BAND_CENTRES_HZ",
    "LIMITS",
    "METHOD",
    "NOT_HERE",
    "QUESTION",
    "RATE_HZ",
    "SIGNIFICANT",
    "WINDOW_S",
    "read_directory",
    "recover",
]
