"""The linear nodes: gain, mix, pan, delay, lfo, section and a whole `lti` chain.

Every node is drawn a span `[a, b)` at a time into its own output, with whatever it
carries from one span to the next kept in its state, so one function serves both the
acyclic parts (one span, the whole take) and a loop (many short spans). A delay is
the only node that reads its input from before the span it draws, which is what lets
a loop close through it.

A section is the coefficients `reproduce.section_sos` returns, so a stage filtered
here and a stage read as a curve there are one set of numbers. An all-pass chain
with a loop has no sections; it is drawn as its loop, sample by sample, and held
against the formula `reproduce.response()` closes it with.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import signal

from .. import reproduce


@dataclass(frozen=True)
class Node:
    """One kind of node: how it is drawn and what a model has to give it.

    `render(node, drawing, a, b)` writes the node's output over `[a, b)`.
    `required` names the values the kind cannot be drawn without. `check(node,
    referenced)` refuses a node whose values the kind cannot draw. `lead(node,
    drawing)` is how many samples behind the sample being drawn the newest input it
    reads is, at the closest point of the take; only a kind with one can close a
    loop. `refers` is the key naming another model and the kind that model must be.
    """

    render: Callable
    required: tuple[str, ...]
    check: Callable | None = None
    lead: Callable | None = None
    ports: tuple[str, ...] = ("",)
    refers: tuple[str, str] | None = None


def _gain(node: dict, drawing, a: int, b: int) -> None:
    gain = drawing.value(node["gain"], a, b)
    if node["unit"] == "db":
        gain = 10.0 ** (gain / 20.0)
    drawing.output(node["id"])[a:b] = drawing.signal(node["input"])[a:b] * gain


def _check_gain(node: dict, referenced) -> None:
    if node["unit"] not in ("ratio", "db"):
        raise ValueError(f"{node['id']} has a gain in {node['unit']!r}, not `ratio` or `db`")


def _mix(node: dict, drawing, a: int, b: int) -> None:
    out = np.zeros(b - a)
    for ref in node["inputs"]:
        out += drawing.value(node["weights"][ref], a, b) * drawing.signal(ref)[a:b]
    drawing.output(node["id"])[a:b] = out


def _check_mix(node: dict, referenced) -> None:
    missing = [ref for ref in node["inputs"] if ref not in node["weights"]]
    if missing:
        raise ValueError(f"{node['id']} has no `weights` for {missing}")


def _sides(law: dict, position) -> tuple:
    """The two multipliers of a pan law at a position on its own byte's axis."""
    out = []
    for side in ("left", "right"):
        points = sorted((int(v), float(x)) for v, x in law["sides"][side])
        out.append(np.interp(position, [p[0] for p in points], [p[1] for p in points]))
    return out[0], out[1]


def _pan(node: dict, drawing, a: int, b: int) -> None:
    left, right = _sides(drawing.referenced[node["law"]], drawing.value(node["position"], a, b))
    x = drawing.signal(node["input"])[a:b]
    drawing.output(node["id"], "left")[a:b] = x * left
    drawing.output(node["id"], "right")[a:b] = x * right


def _check_pan(node: dict, referenced) -> None:
    if "sides" not in referenced:
        raise ValueError(f"{node['id']} names {node['law']}, which carries no `sides`")


# ---------------------------------------------------------------- delay

REACH = {"none": 0, "linear": 1, "allpass": 1, "lagrange3": 2}
"""How far an interpolator reads towards the present from the delay it was asked for."""


def _delay_samples(node: dict, drawing, a: int, b: int):
    time_ms = drawing.value(node["time_ms"], a, b)
    modulation = node.get("modulated_by")
    if modulation is not None:
        depth = drawing.value(modulation["depth_ms"], a, b)
        time_ms = time_ms + depth * drawing.signal(modulation["control"])[a:b]
    return np.broadcast_to(time_ms * drawing.fs / 1000.0, (b - a,))


def _taken(x: np.ndarray, at: np.ndarray) -> np.ndarray:
    inside = (at >= 0) & (at < x.size)
    return np.where(inside, x[np.clip(at, 0, x.size - 1)], 0.0)


def _delay(node: dict, drawing, a: int, b: int) -> None:
    """A delay line, its read position moving every sample when it is modulated.

    `none` truncates to the whole sample below. `linear` and `lagrange3` weight the
    samples either side (four for the cubic, one of them a sample nearer than the
    delay); `allpass` is the first-order all-pass `(eta + z^-1) / (1 + eta z^-1)`,
    `eta = (1 - f) / (1 + f)`, whose delay at DC is the fraction `f`.
    """
    x = drawing.signal(node["input"])
    delay = _delay_samples(node, drawing, a, b)
    whole = np.floor(delay).astype(np.int64)
    f = delay - whole
    base = np.arange(a, b) - whole
    kind = node["interpolation"]
    out = drawing.output(node["id"])
    if kind == "none":
        out[a:b] = _taken(x, base)
    elif kind == "linear":
        out[a:b] = (1.0 - f) * _taken(x, base) + f * _taken(x, base - 1)
    elif kind == "lagrange3":
        out[a:b] = (
            -f * (f - 1.0) * (f - 2.0) / 6.0 * _taken(x, base + 1)
            + (f + 1.0) * (f - 1.0) * (f - 2.0) / 2.0 * _taken(x, base)
            - (f + 1.0) * f * (f - 2.0) / 2.0 * _taken(x, base - 1)
            + (f + 1.0) * f * (f - 1.0) / 6.0 * _taken(x, base - 2)
        )
    else:
        state = drawing.state(node["id"])
        last = state.get("last", 0.0)
        eta = ((1.0 - f) / (1.0 + f)).tolist()
        near, far = _taken(x, base).tolist(), _taken(x, base - 1).tolist()
        drawn = [0.0] * (b - a)
        for i in range(b - a):
            last = far[i] + eta[i] * (near[i] - last)
            drawn[i] = last
        state["last"] = last
        out[a:b] = drawn


def _delay_lead(node: dict, drawing) -> float:
    delay = _delay_samples(node, drawing, 0, drawing.size)
    return float(np.min(delay)) - REACH[node["interpolation"]]


def _check_delay(node: dict, referenced) -> None:
    if node["interpolation"] not in REACH:
        raise ValueError(
            f"{node['id']} interpolates by {node['interpolation']!r}, not one of {sorted(REACH)}"
        )
    modulation = node.get("modulated_by")
    if modulation is not None and not {"control", "depth_ms"} <= set(modulation):
        raise ValueError(f"{node['id']} is modulated_by something with no `control` or `depth_ms`")


# ---------------------------------------------------------------- lfo

SHAPES = ("sine", "triangle", "square", "saw", "points")


def _shaped(node: dict, phase: np.ndarray) -> np.ndarray:
    """One cycle per unit of phase; sine, triangle and saw rise through zero at phase 0,
    and a square is high for the first half of the cycle."""
    shape = node["shape"]
    if shape == "sine":
        return np.sin(2 * np.pi * phase)
    if shape == "triangle":
        return 4.0 * np.abs((phase - 0.25) % 1.0 - 0.5) - 1.0
    if shape == "square":
        return np.where(phase % 1.0 < 0.5, 1.0, -1.0)
    if shape == "saw":
        return 2.0 * ((phase + 0.5) % 1.0) - 1.0
    points = sorted((float(p), float(v)) for p, v in node["points"])
    return np.interp(phase, [p[0] for p in points], [p[1] for p in points], period=1.0)


def _lfo(node: dict, drawing, a: int, b: int) -> None:
    """A control signal in [-1, 1] (a `points` shape spans its own values).

    Phase is in cycles: `phase_offset` plus the drawing's starting phase. A rate
    driven by a control is integrated sample by sample, and `slew_s` is the time
    constant a one-pole spends getting the rate to where the control puts it.
    """
    rate = drawing.value(node["rate_hz"], a, b)
    state = drawing.state(node["id"])
    if np.ndim(rate) == 0:
        start = drawing.value(node["phase_offset"], a, b) + drawing.lfo_phase
        phase = start + rate * np.arange(a, b) / drawing.fs
    else:
        if "phase" not in state:
            state["phase"] = drawing.value(node["phase_offset"], a, b) + drawing.lfo_phase
            state["rate"] = float(rate[0])
        slew = node.get("slew_s")
        seconds = 0.0 if slew is None else drawing.value(slew, a, b)
        share = 1.0 if seconds == 0 else 1.0 - math.exp(-1.0 / (seconds * drawing.fs))
        rate, _ = signal.lfilter(
            [share], [1.0, share - 1.0], rate, zi=[(1.0 - share) * state["rate"]]
        )
        state["rate"] = float(rate[-1])
        steps = np.concatenate([[state["phase"]], rate / drawing.fs])
        running = np.cumsum(steps)
        phase = running[:-1]
        state["phase"] = float(running[-1])
    drawing.output(node["id"])[a:b] = _shaped(node, phase)


def _check_lfo(node: dict, referenced) -> None:
    if node["shape"] not in SHAPES:
        raise ValueError(f"{node['id']} has shape {node['shape']!r}, not one of {list(SHAPES)}")
    if node["shape"] == "points" and not node.get("points"):
        raise ValueError(f"{node['id']} has shape `points` and no `points`")


# ---------------------------------------------------------------- section

STAGE_VALUES = {
    "shelf": ("side", "corner_hz", "gain_db"),
    "peaking": ("centre_hz", "q", "gain_db"),
    "pole": ("side", "corner_hz", "sections"),
    "gain": ("gain_db",),
    "allpass-chain": ("sections", "corner_hz", "mix"),
}
"""The stages a section can be, and the values each is built from."""

_NOT_A_STAGE_VALUE = ("id", "kind", "input", "stage")


def is_value(item) -> bool:
    """Whether a node's entry is one of its values: `value`, `byte`+`map` or `control`+`map`."""
    return isinstance(item, dict) and (
        "value" in item or "byte" in item or ("control" in item and "map" in item)
    )


def _stage_of(node: dict, values: dict) -> dict:
    stage = {"kind": node["stage"]}
    for key, item in node.items():
        if key not in _NOT_A_STAGE_VALUE:
            stage[key] = {"fixed": values[key]} if key in values else item
    return stage


def _section(node: dict, drawing, a: int, b: int) -> None:
    """One stage as second-order sections; a control-driven value rebuilds them per sample."""
    state = drawing.state(node["id"])
    values = {k: drawing.value(v, a, b) for k, v in node.items() if is_value(v)}
    x = drawing.signal(node["input"])[a:b]
    out = drawing.output(node["id"])
    if all(np.ndim(v) == 0 for v in values.values()):
        if "sos" not in state:
            state["sos"] = reproduce.section_sos(_stage_of(node, values), {}, drawing.fs)
            state["zi"] = np.zeros((len(state["sos"]), 2))
        out[a:b], state["zi"] = signal.sosfilt(state["sos"], x, zi=state["zi"])
        return
    for i in range(b - a):
        now = {k: (v[i] if np.ndim(v) else v) for k, v in values.items()}
        sos = reproduce.section_sos(_stage_of(node, now), {}, drawing.fs)
        zi = state.setdefault("zi", np.zeros((len(sos), 2)))
        if len(sos) != len(zi):
            raise ValueError(f"{node['id']} changes how many sections it has as its control moves")
        out[a + i : a + i + 1], state["zi"] = signal.sosfilt(sos, x[i : i + 1], zi=zi)


def _check_section(node: dict, referenced) -> None:
    stage = node["stage"]
    if stage not in STAGE_VALUES:
        raise ValueError(f"{node['id']} is a {stage!r} stage, not one of {sorted(STAGE_VALUES)}")
    for key in STAGE_VALUES[stage]:
        if key not in node:
            raise ValueError(f"{node['id']} has no `{key}`, which a {stage} section requires")
    if "feedback" in node:
        raise ValueError(
            f"{node['id']} is an all-pass chain with a loop, which has no sections: "
            "write the loop as nodes"
        )


# ---------------------------------------------------------------- lti


class _AllpassLoop:
    """An all-pass chain with a loop, drawn as the loop `reproduce._allpass_chain` closes.

    Closed around the chain, `u = x + g L(w)`, `w = A^k u`, output `x + mix A^(n-k) w`.
    Closed around the output, `u = x + g L(mix A^n u)`, output `u + mix A^n u`. `L` is
    the loop's delay and, when there is one, its one-pole high pass.
    """

    def __init__(self, stage: dict, bytes_now: dict[str, int], fs: float):
        def value(key):
            return reproduce._value(stage[key], bytes_now)

        corner = value("corner_hz")
        if stage.get("q") is None:
            t = math.tan(math.pi * corner / fs)
            c = (t - 1.0) / (t + 1.0)
            row = (c, 1.0, 0.0, c, 0.0)
        else:
            w0 = 2 * math.pi * corner / fs
            cos0, alpha = math.cos(w0), math.sin(w0) / (2.0 * value("q"))
            norm = 1.0 + alpha
            row = ((1 - alpha) / norm, -2 * cos0 / norm, 1.0, -2 * cos0 / norm, (1 - alpha) / norm)
        self.rows = [row] * int(stage["sections"])
        self.states = [[0.0, 0.0] for _ in self.rows]
        self.inner = int(stage.get("feedback_sections") or stage["sections"])
        self.mix = value("mix")
        self.feedback = value("feedback")
        self.around_output = stage.get("feedback_around", "the-chain") == "the-output"
        self.line = [0.0] * int(stage["feedback_delay"])
        self.at = 0
        corner_hp = stage.get("feedback_highpass_hz")
        self.blocked = None if corner_hp is None else math.tan(math.pi * corner_hp / fs)
        self.hp_in = self.hp_out = 0.0

    def _through(self, x: float, first: int, last: int) -> float:
        for row, state in zip(self.rows[first:last], self.states[first:last], strict=True):
            b0, b1, b2, a1, a2 = row
            y = b0 * x + state[0]
            state[0] = b1 * x - a1 * y + state[1]
            state[1] = b2 * x - a2 * y
            x = y
        return x

    def process(self, block: np.ndarray) -> np.ndarray:
        out = [0.0] * block.size
        sections = len(self.rows)
        for i, x in enumerate(block.tolist()):
            late = self.line[self.at]
            if self.blocked is not None:
                late, self.hp_in = (
                    (late - self.hp_in - (self.blocked - 1.0) * self.hp_out) / (1.0 + self.blocked),
                    late,
                )
                self.hp_out = late
            u = x + self.feedback * late
            if self.around_output:
                returned = self.mix * self._through(u, 0, sections)
                out[i] = u + returned
            else:
                returned = self._through(u, 0, self.inner)
                out[i] = x + self.mix * self._through(returned, self.inner, sections)
            self.line[self.at] = returned
            self.at = (self.at + 1) % len(self.line)
        return np.asarray(out)


class _Sections:
    def __init__(self, sos: np.ndarray):
        self.sos = sos
        self.zi = np.zeros((len(sos), 2))

    def process(self, block: np.ndarray) -> np.ndarray:
        out, self.zi = signal.sosfilt(self.sos, block, zi=self.zi)
        return out


def _has_a_loop(stage: dict) -> bool:
    return stage["kind"] == "allpass-chain" and "feedback" in stage


def _lti(node: dict, drawing, a: int, b: int) -> None:
    """A whole `lti` model's chain, stage after stage, at the drawing's bytes."""
    state = drawing.state(node["id"])
    if "stages" not in state:
        chain = drawing.referenced[node["model"]]["chain"]
        state["stages"] = [
            _AllpassLoop(stage, drawing.bytes_now, drawing.fs)
            if _has_a_loop(stage)
            else _Sections(reproduce.section_sos(stage, drawing.bytes_now, drawing.fs))
            for stage in chain
        ]
    x = drawing.signal(node["input"])[a:b]
    for stage in state["stages"]:
        x = stage.process(x)
    drawing.output(node["id"])[a:b] = x


def _check_lti(node: dict, referenced) -> None:
    for stage in referenced["chain"]:
        if _has_a_loop(stage) and int(stage.get("feedback_delay", 0)) < 1:
            raise ValueError(
                f"{node['id']} names {node['model']}, whose loop returns with no delay in "
                "it; that has no order to be drawn in and is held only as a response"
            )


NODES = {
    "gain": Node(_gain, ("input", "gain", "unit"), check=_check_gain),
    "mix": Node(_mix, ("inputs", "weights"), check=_check_mix),
    "pan": Node(
        _pan,
        ("input", "position", "law"),
        check=_check_pan,
        ports=("left", "right"),
        refers=("law", "pan"),
    ),
    "delay": Node(
        _delay, ("input", "time_ms", "interpolation"), check=_check_delay, lead=_delay_lead
    ),
    "lfo": Node(_lfo, ("shape", "rate_hz", "phase_offset"), check=_check_lfo),
    "section": Node(_section, ("input", "stage"), check=_check_section),
    "lti": Node(_lti, ("input", "model"), check=_check_lti, refers=("model", "lti")),
}
