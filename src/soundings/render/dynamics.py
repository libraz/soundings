"""The nodes that act on a signal's level: shapers, detectors and what they drive.

Each is drawn a span `[a, b)` at a time, like the linear nodes, with anything carried
across spans in its state. Only the envelope recurses, so only it runs a sample at a
time; the rest are whole-array operations.

An envelope's output is a level in decibels whatever domain it smoothed in, which is
what a gain computer reads and what a control map is written against. A gain
computer's output is a multiplier, which is what a vca applies.

An oversampled shaper filters with a linear-phase FIR and so reads ahead of the
sample it draws. Outside a loop that is the whole take at once and costs nothing;
inside one it has no order to be drawn in and is refused.
"""

from __future__ import annotations

import functools

import numpy as np
from scipy import signal

from .linear import Node

# ---------------------------------------------------------------- shaper

CURVES = ("points", "tanh", "hard", "cubic")

PASSBAND_OF_RATE = 0.4375
STOPBAND_DB = 120.0
"""The oversampling filter passes to 0.4375 of the model's rate and stops from half of it.

That passes everything a comparison reads (12 kHz of a 32 kHz model) and holds whatever
would fold back 120 dB down, 40 dB under the alias bar the node is tested to.
"""


@functools.cache
def _oversampling_fir(oversample: int) -> np.ndarray:
    width = (0.5 - PASSBAND_OF_RATE) * 2.0 / oversample
    taps, beta = signal.kaiserord(STOPBAND_DB, width)
    corner = (0.5 + PASSBAND_OF_RATE) / oversample
    return signal.firwin(taps | 1, corner, window=("kaiser", beta))


def _curve(node: dict, x: np.ndarray) -> np.ndarray:
    curve = node["curve"]
    if curve == "tanh":
        return np.tanh(x)
    if curve == "hard":
        return np.clip(x, -1.0, 1.0)
    if curve == "cubic":
        return np.where(np.abs(x) < 1.0, 1.5 * x - 0.5 * x**3, np.sign(x))
    points = sorted((float(i), float(o)) for i, o in node["points"])
    return np.interp(x, [p[0] for p in points], [p[1] for p in points])


def _shaper(node: dict, drawing, a: int, b: int) -> None:
    """A memoryless curve after `drive` decibels of gain, run `oversample` times faster.

    `tanh` and `hard` saturate at 1; `cubic` is `1.5x - 0.5x^3` up to 1 and flat
    beyond; `points` interpolates `[in, out]` pairs and holds its end values outside
    them. The rate is raised and lowered by polyphase resampling through one Kaiser
    FIR, so what the curve puts above the model's Nyquist is removed, not folded.
    """
    x = drawing.signal(node["input"])[a:b]
    gain = 10.0 ** (drawing.value(node["drive"], a, b) / 20.0)
    oversample = node["oversample"]
    out = drawing.output(node["id"])
    if oversample == 1:
        out[a:b] = _curve(node, gain * x)
        return
    if (a, b) != (0, drawing.size):
        raise ValueError(
            f"{node['id']} is oversampled, which reads ahead of the sample it draws; "
            "inside a loop that has no order to be drawn in"
        )
    fir = _oversampling_fir(oversample)
    raised = signal.resample_poly(x, oversample, 1, window=fir)
    # A control-driven drive is held across the samples the rate was raised by.
    gain = np.repeat(gain, oversample) if np.ndim(gain) else gain
    out[a:b] = signal.resample_poly(_curve(node, gain * raised), 1, oversample, window=fir)


def _is_count(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _check_shaper(node: dict, referenced) -> None:
    if node["curve"] not in CURVES:
        raise ValueError(f"{node['id']} has curve {node['curve']!r}, not one of {list(CURVES)}")
    if node["curve"] == "points" and not node.get("points"):
        raise ValueError(f"{node['id']} has curve `points` and no `points`")
    if not _is_count(node["oversample"]):
        raise ValueError(
            f"{node['id']} has oversample {node['oversample']!r}, not a whole 1 or more"
        )


# ---------------------------------------------------------------- envelope

DETECTORS = ("peak", "rms")
TOPOLOGIES = ("branching", "decoupled")
DOMAINS = ("linear", "log")


def _coefficient(ms, fs: float, size: int) -> list[float]:
    return np.broadcast_to(np.exp(-1000.0 / (np.asarray(ms) * fs)), (size,)).tolist()


def _envelope(node: dict, drawing, a: int, b: int) -> None:
    """A level detector, output in decibels.

    `peak` smooths `|x|` and `rms` smooths `x^2`; in the `log` domain `|x|` is taken to
    decibels first, clamped at `floor_db`, and smoothed there. Each pole has
    `alpha = exp(-1 / (tau fs))` and reaches 1 - 1/e of a step in `tau`. `branching`
    switches between the attack and release poles on whether the input is above the
    level; `decoupled` runs a release peak-hold into an attack pole, so a release
    takes about attack plus release (Giannoulis, Massberg and Reiss, JAES 2012).
    """
    x = drawing.signal(node["input"])[a:b]
    log = node["domain"] == "log"
    if log:
        floor = drawing.value(node["floor_db"], a, b)
        with np.errstate(divide="ignore"):
            level = np.maximum(20.0 * np.log10(np.abs(x)), floor)
    else:
        level = np.abs(x) if node["detector"] == "peak" else x * x
    size = b - a
    attack = _coefficient(drawing.value(node["attack_ms"], a, b), drawing.fs, size)
    release = _coefficient(drawing.value(node["release_ms"], a, b), drawing.fs, size)
    state = drawing.state(node["id"])
    start = float(np.min(floor)) if log else 0.0
    y, held = state.get("y", start), state.get("held", start)
    drawn = [0.0] * size
    if node["topology"] == "branching":
        for i, v in enumerate(level.tolist()):
            k = attack[i] if v > y else release[i]
            y = k * y + (1.0 - k) * v
            drawn[i] = y
    else:
        for i, v in enumerate(level.tolist()):
            held = max(v, release[i] * held + (1.0 - release[i]) * v)
            y = attack[i] * y + (1.0 - attack[i]) * held
            drawn[i] = y
    state["y"], state["held"] = y, held
    smoothed = np.asarray(drawn)
    if log:
        drawing.output(node["id"])[a:b] = smoothed
        return
    with np.errstate(divide="ignore"):
        scale = 20.0 if node["detector"] == "peak" else 10.0
        drawing.output(node["id"])[a:b] = scale * np.log10(smoothed)


def _check_envelope(node: dict, referenced) -> None:
    for key, allowed in (("detector", DETECTORS), ("topology", TOPOLOGIES), ("domain", DOMAINS)):
        if node[key] not in allowed:
            raise ValueError(f"{node['id']} has {key} {node[key]!r}, not one of {list(allowed)}")
    if node["domain"] == "log":
        if "floor_db" not in node:
            raise ValueError(
                f"{node['id']} has no `floor_db`, which a log-domain envelope requires"
            )
        if node["detector"] == "rms":
            raise ValueError(
                f"{node['id']} is an rms detector in the log domain, which smooths the same "
                "decibels a peak detector does; write it as `peak`"
            )


# ---------------------------------------------------------------- gain computer, vca


def _gain_computer(node: dict, drawing, a: int, b: int) -> None:
    """A level in decibels to a gain multiplier, with a quadratic knee.

    Below the knee the gain is 1; above it the output level rises `1/ratio` dB per dB
    over `threshold_db`; across the `knee_db` wide knee centred on the threshold the
    reduction is `(1/ratio - 1)(over + knee/2)^2 / (2 knee)`.
    """
    level = drawing.signal(node["input"])[a:b]
    threshold = drawing.value(node["threshold_db"], a, b)
    knee = drawing.value(node["knee_db"], a, b)
    slope = 1.0 / drawing.value(node["ratio"], a, b) - 1.0
    over = level - threshold
    # A silent level is -inf and a zero knee divides by zero, both in branches not taken.
    with np.errstate(invalid="ignore", divide="ignore"):
        reduction = np.where(
            2.0 * over <= -knee,
            0.0,
            np.where(
                2.0 * over >= knee, slope * over, slope * (over + knee / 2.0) ** 2 / (2.0 * knee)
            ),
        )
    drawing.output(node["id"])[a:b] = 10.0 ** (reduction / 20.0)


def _vca(node: dict, drawing, a: int, b: int) -> None:
    """The input times the control, sample by sample."""
    x = drawing.signal(node["input"])[a:b]
    drawing.output(node["id"])[a:b] = x * drawing.signal(node["control"])[a:b]


# ---------------------------------------------------------------- hold, quantize


def _hold(node: dict, drawing, a: int, b: int) -> None:
    """Sample and hold, taking the input at the first sample on or after each 1/rate_hz."""
    x = drawing.signal(node["input"])[a:b]
    rate = drawing.value(node["rate_hz"], a, b)
    state = drawing.state(node["id"])
    if np.ndim(rate) == 0:
        phase = rate * np.arange(a, b) / drawing.fs
    else:
        steps = np.concatenate([[state.get("phase", 0.0)], rate / drawing.fs])
        running = np.cumsum(steps)
        phase, state["phase"] = running[:-1], float(running[-1])
    tick = np.floor(phase)
    before = np.concatenate([[state.get("tick", -1.0)], tick[:-1]])
    taken = np.where(tick != before, np.arange(b - a), -1)
    latest = np.maximum.accumulate(taken)
    held = np.where(latest >= 0, x[np.maximum(latest, 0)], state.get("held", 0.0))
    state["tick"], state["held"] = float(tick[-1]), float(held[-1])
    drawing.output(node["id"])[a:b] = held


def _quantize(node: dict, drawing, a: int, b: int) -> None:
    """A `bits` word over full scale +-1: round to nearest, mid-tread, then saturate.

    Rounding adds half a step and truncates, so a half step rounds up; the word holds
    `-1` to `1 - 2^(1 - bits)`. Rounding rather than truncating, and saturating rather
    than wrapping, are candidate choices this node makes, not facts about a unit.
    """
    bits = node["bits"]
    step = 2.0 ** (1 - bits)
    x = drawing.signal(node["input"])[a:b]
    words = np.clip(np.floor(x / step + 0.5), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    drawing.output(node["id"])[a:b] = words * step


def _check_quantize(node: dict, referenced) -> None:
    if not _is_count(node["bits"]):
        raise ValueError(f"{node['id']} has bits {node['bits']!r}, not a whole 1 or more")


NODES = {
    "shaper": Node(_shaper, ("input", "curve", "drive", "oversample"), check=_check_shaper),
    "envelope": Node(
        _envelope,
        ("input", "detector", "topology", "domain", "attack_ms", "release_ms"),
        check=_check_envelope,
    ),
    "gain_computer": Node(_gain_computer, ("input", "threshold_db", "ratio", "knee_db")),
    "vca": Node(_vca, ("input", "control")),
    "hold": Node(_hold, ("input", "rate_hz")),
    "quantize": Node(_quantize, ("input", "bits"), check=_check_quantize),
}
