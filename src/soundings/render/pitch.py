"""The nodes that move a signal's pitch."""

from __future__ import annotations

import numpy as np

from . import linear
from .linear import REACH, Node

CROSSFADES = ("linear", "hann", "s_curve")


def _faded(shape: str, t: np.ndarray) -> np.ndarray:
    """A pointer's gain at `t`, 0 where it wraps and 1 half a window away from that."""
    if shape == "linear":
        return t
    if shape == "hann":
        return np.sin(np.pi * t / 2.0) ** 2
    return t * t * (3.0 - 2.0 * t)


def _pitch(node: dict, drawing, a: int, b: int) -> None:
    """Two read pointers on one delay line, half a window apart, crossfaded.

    Each pointer sweeps its delay as a sawtooth of period `T = window_ms`, by
    `(1 - ratio) T` per period, so what it reads moves at `ratio` times the input's
    rate; the sweep's nearest point is `REACH[interpolation]` samples back. The
    pointers read through `delay`'s own interpolators. A pointer's gain is `g(t)`,
    `t` its triangle position (0 at its wrap, 1 half a window later), and the other
    pointer's is `g(1 - t)`, so the two always sum to 1:

    - `linear`: `g(t) = t`.
    - `hann`: `g(t) = sin^2(pi t / 2)`.
    - `s_curve`: `g(t) = 3t^2 - 2t^3` (smoothstep), slow-fast-slow; a candidate for
      the S-shaped ramp, not a known shape.

    A sine at `f` comes out as lines at `f + k/T` only. With `u = f (1 - ratio) T`,
    the largest is at `f * ratio` when `u` is an even integer, and within `1/T` of it
    otherwise. Its envelope is `sqrt(1 - 4 g(t) g(1 - t) sin^2(pi u / 2))`: it repeats
    every half window, is 1 where one pointer holds alone, and dips to
    `|cos(pi u / 2)|` at every crossover whatever the shape; the shape sets only how
    long it stays down. At a ratio of 1 the output is the input `REACH` samples late.

    The sweep starts at the drawing's `lfo_phase`, in cycles of the window; half a
    cycle swaps the two pointers and so draws the same output as none.
    """
    ratio = drawing.value(node["ratio"], a, b)
    window = drawing.value(node["window_ms"], a, b) * drawing.fs / 1000.0
    excursion = (1.0 - ratio) * window
    nearest = REACH[node["interpolation"]] + max(-excursion, 0.0)
    phase = np.arange(a, b) / window + drawing.lfo_phase
    out = np.zeros(b - a)
    for name, offset in (("pointer-a", 0.0), ("pointer-b", 0.5)):
        swept = (phase + offset) % 1.0
        control = f"{node['id']}.{name}-ms"
        drawing.output(control)[a:b] = (nearest + excursion * swept) * 1000.0 / drawing.fs
        pointer = {
            "id": f"{node['id']}.{name}",
            "input": node["input"],
            "time_ms": {"value": 0.0},
            "interpolation": node["interpolation"],
            "modulated_by": {"control": control, "depth_ms": {"value": 1.0}},
        }
        linear._delay(pointer, drawing, a, b)
        t = 1.0 - np.abs(2.0 * swept - 1.0)
        out += _faded(node["crossfade"], t) * drawing.signal(pointer["id"])[a:b]
    drawing.output(node["id"])[a:b] = out


def _check_pitch(node: dict, referenced) -> None:
    if node["crossfade"] not in CROSSFADES:
        raise ValueError(
            f"{node['id']} crossfades by {node['crossfade']!r}, not one of {list(CROSSFADES)}"
        )
    if node["interpolation"] not in REACH:
        raise ValueError(
            f"{node['id']} interpolates by {node['interpolation']!r}, not one of {sorted(REACH)}"
        )
    for key in ("ratio", "window_ms"):
        if "control" in node[key]:
            raise ValueError(
                f"{node['id']} has its `{key}` driven by a control; the sweep holds one of each"
            )


NODES = {
    "pitch": Node(
        _pitch,
        ("input", "ratio", "window_ms", "crossfade", "interpolation"),
        check=_check_pitch,
    ),
}
