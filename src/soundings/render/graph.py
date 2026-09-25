"""Loading a `graph` model and drawing it on a take, sample by sample where it must be.

The vocabulary is closed: a node kind is drawn only if one of `linear`, `dynamics` or
`pitch` lists it, and this module holds no knowledge of any kind beyond the entry that
list gives it. A model is refused at load for anything that would make it unreadable
rather than wrong: an unknown kind, a missing value, a name that reaches nothing, a
loop with no delay in it, a printed row it does not account for.

Drawing runs in the order the wiring allows. Everything outside a loop is one span,
the whole take. A loop -- a strongly connected component -- is drawn in blocks no
longer than the shortest distance any delay in it reads back, less its interpolator's
reach, so nothing in a block reads a sample the block has not drawn yet. A setting
that brings that distance under one sample has no order to be drawn in and is refused
as `Unrenderable`, by name; so is a byte value the model names no state for.

A take is carried to the model's rate and back through one FIR, and compared below
`CUT_HZ`, where that FIR leaves nothing behind.
"""

from __future__ import annotations

import functools
import json
import math
from pathlib import Path

import numpy as np
from scipy import signal

from .. import reproduce
from ..inferences import EFFECT_BLOCK
from . import dynamics, linear, pitch

NODES = linear.NODES | dynamics.NODES | pitch.NODES

CUT_HZ = 12000
"""The ceiling a drawn take and the unit's take are both cut at before they are compared.

A round trip 48 to 32 kHz and back through the FIR below leaves -69.9 dB under it on a
broadband take; the resampler's default window left -43 dB.
"""

RESAMPLING_TAPS = 1201
RESAMPLING_CORNER_HZ = 15000.0
RESAMPLING_WINDOW = ("kaiser", 12.0)

REQUIRED = ("model", "sample_rate_hz", "inputs", "outputs", "nodes", "rows")
ROW_STATES = ("bound", "fixed_at_power_on")


class Unrenderable(ValueError):
    """A setting under which a loop reads a sample it has not drawn yet."""


# ---------------------------------------------------------------- the wiring


def _controls(item) -> list[str]:
    """Every node a value or a modulation reads as a control, however deep."""
    found = []
    if isinstance(item, dict):
        if isinstance(item.get("control"), str):
            found.append(item["control"])
        for value in item.values():
            found.extend(_controls(value))
    elif isinstance(item, list):
        for value in item:
            found.extend(_controls(value))
    return found


def _audio(node: dict) -> list[str]:
    if "input" in node:
        return [node["input"]]
    return list(node.get("inputs", []))


def _read_bytes(item) -> set[str]:
    found = set()
    if isinstance(item, dict):
        if "byte" in item and "map" in item:
            found.add(item["byte"])
        for value in item.values():
            found |= _read_bytes(value)
    elif isinstance(item, list):
        for value in item:
            found |= _read_bytes(value)
    return found


def _owner(ref: str, ids: set[str]) -> str | None:
    """The node a reference reads from, or None for one of the graph's own inputs."""
    if ref in ids:
        return ref
    head = ref.split(".", 1)[0]
    return head if head in ids else None


def _edges(model: dict) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Each node's predecessors: all of them, and those it reads in the same sample."""
    ids = {node["id"] for node in model["nodes"]}
    every, immediate = {}, {}
    for node in model["nodes"]:
        delayed = NODES[node["kind"]].lead is not None
        audio = {_owner(ref, ids) for ref in _audio(node)} - {None}
        controls = {_owner(ref, ids) for ref in _controls(node)} - {None}
        every[node["id"]] = audio | controls
        immediate[node["id"]] = controls if delayed else audio | controls
    return every, immediate


def _components(nodes: list[str], edges: dict[str, set[str]]) -> list[list[str]]:
    """Strongly connected components, each after every component it reads from."""
    index, low, on_stack, stack, found = {}, {}, set(), [], []

    def visit(v: str) -> None:
        index[v] = low[v] = len(index)
        stack.append(v)
        on_stack.add(v)
        for w in sorted(edges[v]):
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            component = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                component.append(w)
                if w == v:
                    break
            found.append(component)

    for v in nodes:
        if v not in index:
            visit(v)
    return found


def _looped(component: list[str], edges: dict[str, set[str]]) -> bool:
    return len(component) > 1 or component[0] in edges[component[0]]


def _in_order(component: list[str], immediate: dict[str, set[str]]) -> list[str]:
    inside = set(component)
    ordered, done = [], set()

    def visit(v: str) -> None:
        done.add(v)
        for w in sorted(immediate[v] & inside):
            if w not in done:
                visit(w)
        ordered.append(v)

    for v in sorted(component):
        if v not in done:
            visit(v)
    return ordered


# ---------------------------------------------------------------- loading


def _printed(root: Path, unit: str, effect_type: str) -> set[str]:
    """Every address the unit's document prints for a type, parsed or read by hand."""
    meta = json.loads((root / "data" / "units" / unit / "meta.json").read_text())
    named = (meta.get("documents") or [None])[0]
    if named is None:
        raise ValueError(f"{unit} names no document, so its printed rows cannot be accounted for")
    folder = root / "documents" / named
    rows = list(json.loads((folder / "effect-list.json").read_text())["rows"])
    by_hand = folder / "by-hand.json"
    if by_hand.is_file():
        rows += json.loads(by_hand.read_text()).get("tables", {}).get("effect-list", [])
    msb, lsb = effect_type.split()
    return {
        f"{EFFECT_BLOCK} {row['address_lsb']}"
        for row in rows
        if row.get("msb") == msb and row.get("lsb") == lsb and "address_lsb" in row
    }


def _referenced(model: dict, models_dir: Path) -> dict[str, dict]:
    found = {}
    for node in model["nodes"]:
        refers = NODES[node["kind"]].refers
        if refers is None:
            continue
        key, kind = refers
        name = node[key]
        other = json.loads((models_dir / name).read_text())
        if other["model"].get("kind") != kind:
            raise ValueError(
                f"{node['id']} reads {name}, which is a {other['model'].get('kind')!r} model "
                f"and not a `{kind}` one"
            )
        rate = other.get("sample_rate_hz")
        if rate is not None and rate != model["sample_rate_hz"]:
            raise ValueError(
                f"{node['id']} reads {name}, drawn at {rate} Hz, into a graph drawn at "
                f"{model['sample_rate_hz']} Hz: the two sample_rate_hz have to agree"
            )
        found[name] = other
    return found


def addresses_read(model: dict, *, models_dir: Path) -> set[str]:
    """Every address a graph reads through a byte, in its own nodes or a model one names.

    A node that names another model -- an `lti` chain, a `pan` law -- reads whatever
    that model reads, so the row it binds is read by the graph. Nodes of a kind
    outside the vocabulary contribute only their own values.
    """
    read = _read_bytes(model["nodes"])
    for node in model["nodes"]:
        kind = NODES.get(node.get("kind"))
        if kind is not None and kind.refers is not None and node.get(kind.refers[0]):
            read |= _read_bytes(json.loads((Path(models_dir) / node[kind.refers[0]]).read_text()))
    return read


def _check_rows(model: dict, models_dir: Path, root: Path) -> None:
    read = addresses_read(model, models_dir=models_dir)
    rows = model["rows"]
    for address, state in rows.items():
        if state not in ROW_STATES:
            raise ValueError(f"rows gives {address} as {state!r}, not one of {list(ROW_STATES)}")
        if state == "bound" and address not in read:
            raise ValueError(f"rows marks {address} `bound` and no node reads it")
    for address in sorted(read):
        if rows.get(address) != "bound":
            raise ValueError(f"{address} is read by the graph and is not `bound` in rows")
    effect_type = model["model"]["type"]
    missing = _printed(root, model["model"]["unit_id"], effect_type) - set(rows)
    if missing:
        raise ValueError(f"rows omits {sorted(missing)}, which {effect_type} prints")


def load_graph(model: dict, *, models_dir: Path, root: Path) -> dict:
    """A `graph` model checked for everything that would stop it being drawn.

    `models_dir` is where the files an `lti` or `pan` node names are, and `root` the
    tree whose `data/units/<unit>/meta.json` names the document the printed rows are
    read from. What comes back is the model with the models its nodes name under
    `referenced`, keyed by file name.
    """
    for key in REQUIRED:
        if key not in model:
            raise ValueError(f"a graph model has no `{key}`")
    ids = [node["id"] for node in model["nodes"]]
    if len(set(ids)) != len(ids) or set(ids) & set(model["inputs"]):
        raise ValueError("a node id is used twice, or is also the name of an input")
    for node in model["nodes"]:
        if node["kind"] not in NODES:
            raise ValueError(
                f"{node['id']} is a {node['kind']!r}, which is not a node kind this "
                f"renderer draws ({sorted(NODES)})"
            )
        for key in NODES[node["kind"]].required:
            if key not in node:
                raise ValueError(f"{node['id']} has no `{key}`, which a {node['kind']} requires")
    referenced = _referenced(model, Path(models_dir))
    for node in model["nodes"]:
        kind = NODES[node["kind"]]
        if kind.check is not None:
            other = None if kind.refers is None else referenced[node[kind.refers[0]]]
            kind.check(node, other)

    names = set(model["inputs"])
    for node in model["nodes"]:
        ports = NODES[node["kind"]].ports
        names |= {node["id"]} if ports == ("",) else {f"{node['id']}.{p}" for p in ports}
    for node in model["nodes"]:
        for ref in _audio(node) + _controls(node):
            if ref not in names:
                raise ValueError(f"{node['id']} reads {ref!r}, which names nothing in this graph")
    for name, ref in model["outputs"].items():
        if ref not in names:
            raise ValueError(f"output {name} is {ref!r}, which names nothing in this graph")
    for node in model["nodes"]:
        _check_control_maps(node)

    _, immediate = _edges(model)
    for component in _components(ids, immediate):
        if _looped(component, immediate):
            raise ValueError(
                f"{sorted(component)} close a loop with no delay in it, which has no order "
                "to be drawn in"
            )
    _check_rows(model, Path(models_dir), Path(root))
    return {**model, "referenced": referenced}


def _check_control_maps(item) -> None:
    if isinstance(item, dict):
        if isinstance(item.get("control"), str) and "map" in item:
            if item["map"].get("kind") != "points":
                raise ValueError(
                    f"a value driven by {item['control']} maps it by "
                    f"{item['map'].get('kind')!r}; a control is mapped by `points`"
                )
        for value in item.values():
            _check_control_maps(value)
    elif isinstance(item, list):
        for value in item:
            _check_control_maps(value)


# ---------------------------------------------------------------- drawing


class _Drawing:
    """What a node sees while it is drawn: signals, its own state, and its values."""

    def __init__(self, model: dict, size: int, bytes_now: dict[str, int], lfo_phase: float):
        self.fs = float(model["sample_rate_hz"])
        self.size = size
        self.bytes_now = bytes_now
        self.lfo_phase = lfo_phase
        self.referenced = model["referenced"]
        self._signals: dict[str, np.ndarray] = {}
        self._states: dict[str, dict] = {}
        self._constants: dict[int, float] = {}

    def signal(self, ref: str) -> np.ndarray:
        return self._signals[ref]

    def output(self, node_id: str, port: str = "") -> np.ndarray:
        key = f"{node_id}.{port}" if port else node_id
        if key not in self._signals:
            self._signals[key] = np.zeros(self.size)
        return self._signals[key]

    def state(self, node_id: str) -> dict:
        return self._states.setdefault(node_id, {})

    def value(self, spec: dict, a: int, b: int):
        """A constant, a byte through its map, or a control through its map over `[a, b)`."""
        if "value" in spec:
            return float(spec["value"])
        if "byte" in spec:
            # A byte holds one value for the whole drawing; a loop asks every sample.
            if id(spec) not in self._constants:
                try:
                    self._constants[id(spec)] = reproduce._value(spec, self.bytes_now)
                except reproduce.NoStateNamed as refused:
                    raise Unrenderable(f"{spec['byte']}: {refused}") from refused
            return self._constants[id(spec)]
        points = sorted((float(c), float(v)) for c, v in spec["map"]["points"])
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        control = self._signals[spec["control"]][a:b]
        if spec["map"].get("log", False):
            return np.exp(np.interp(control, xs, np.log(ys)))
        return np.interp(control, xs, ys)


def _block(model: dict, component: list[str], drawing: _Drawing, max_block: int | None) -> int:
    nodes = {node["id"]: node for node in model["nodes"]}
    inside = set(component)
    closest, by = math.inf, None
    for node_id in component:
        node = nodes[node_id]
        if set(_controls(node)) & inside:
            raise ValueError(f"{node_id} is driven by a control drawn inside its own loop")
        lead = NODES[node["kind"]].lead
        if lead is None:
            continue
        reach = lead(node, drawing)
        if reach < closest:
            closest, by = reach, node_id
    if closest < 1.0:
        raise Unrenderable(
            f"{model['model'].get('id')}: {by} reads {closest:.3f} samples back beyond its "
            "interpolator's reach, under the one sample a loop needs"
        )
    size = int(math.floor(closest))
    return size if max_block is None else min(size, max_block)


def run(
    model: dict,
    inputs: dict[str, np.ndarray],
    bytes_now: dict[str, int],
    *,
    lfo_phase: float = 0.0,
    max_block: int | None = None,
) -> dict[str, np.ndarray]:
    """Draw a loaded graph at its own rate, returning each named output.

    `lfo_phase` is added to every oscillator's `phase_offset`, in cycles. `max_block`
    caps the block a loop is drawn in; at 1 a loop is drawn a sample at a time.
    """
    if set(inputs) != set(model["inputs"]):
        raise ValueError(f"the graph takes {model['inputs']}, and was given {sorted(inputs)}")
    size = len(next(iter(inputs.values())))
    drawing = _Drawing(model, size, bytes_now, lfo_phase)
    for name, samples in inputs.items():
        drawing._signals[name] = np.asarray(samples, dtype=float)
    nodes = {node["id"]: node for node in model["nodes"]}
    for node in model["nodes"]:
        for port in NODES[node["kind"]].ports:
            drawing.output(node["id"], port)

    every, immediate = _edges(model)
    for component in _components(list(nodes), every):
        if not _looped(component, every):
            node = nodes[component[0]]
            NODES[node["kind"]].render(node, drawing, 0, size)
            continue
        block = _block(model, component, drawing, max_block)
        order = [nodes[v] for v in _in_order(component, immediate)]
        for a in range(0, size, block):
            b = min(a + block, size)
            for node in order:
                NODES[node["kind"]].render(node, drawing, a, b)
    return {name: drawing.signal(ref).copy() for name, ref in model["outputs"].items()}


# ---------------------------------------------------------------- takes


@functools.cache
def _fir(upsampled_hz: int) -> np.ndarray:
    return signal.firwin(
        RESAMPLING_TAPS, RESAMPLING_CORNER_HZ, window=RESAMPLING_WINDOW, fs=upsampled_hz
    )


def resampled(samples: np.ndarray, rate_from: int, rate_to: int) -> np.ndarray:
    """One channel carried from one rate to another through the FIR above."""
    if rate_from == rate_to:
        return np.array(samples, dtype=float)
    common = math.gcd(rate_from, rate_to)
    up, down = rate_to // common, rate_from // common
    return signal.resample_poly(samples, up, down, window=_fir(rate_from * up))


def render_take(
    model: dict,
    take: np.ndarray,
    rate: int,
    bytes_now: dict[str, int],
    *,
    channels: tuple[int, int],
    lfo_phase: float = 0.0,
) -> np.ndarray:
    """A take drawn through a loaded graph, in the take's own shape and rate.

    The graph's inputs are read from `channels`, in order, and its outputs written
    back to the same channels; every other channel comes back silent.
    """
    fs = int(model["sample_rate_hz"])
    outputs = list(model["outputs"])
    if not len(model["inputs"]) == len(outputs) == len(channels):
        raise ValueError(
            f"the graph takes {len(model['inputs'])} inputs and gives {len(outputs)} "
            f"outputs, and was handed {len(channels)} channels"
        )
    inputs = {
        name: resampled(take[:, c], rate, fs)
        for name, c in zip(model["inputs"], channels, strict=True)
    }
    drawn = run(model, inputs, bytes_now, lfo_phase=lfo_phase)
    out = np.zeros(take.shape)
    for name, c in zip(outputs, channels, strict=True):
        out[:, c] = resampled(drawn[name], fs, rate)[: take.shape[0]]
    return out
