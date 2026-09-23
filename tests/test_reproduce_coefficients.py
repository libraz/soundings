"""The coefficients a feedback-free stage is built from are the response it has.

`section_sos` hands a time-domain renderer the same stage `response()` evaluates,
so the two must be one reading and not two. Checked on every feedback-free stage
of every `lti` model under `inferences/models/`, as the model is published.

Settings: the Cartesian product of bytes 0, 32, 64, 96 and 127 over the addresses
the stage reads, plus every byte the stage's own maps list (a `points` or `states`
key, a `window` end), one address at a time with the others at 64. The listed bytes
are where a map changes rule, which is where a claim's readings were taken.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from soundings import reproduce
from soundings.reproduce import section_sos

MODELS = Path(__file__).resolve().parent.parent / "inferences" / "models"
GRID = (0, 32, 64, 96, 127)
ROUNDING = 4 * np.finfo(float).eps


def _lti_models() -> list[tuple[str, dict]]:
    found = []
    for path in sorted(MODELS.glob("*.json")):
        model = json.loads(path.read_text())
        if model["model"]["kind"] == "lti":
            found.append((path.name, model))
    return found


def _params(stage: dict) -> list[dict]:
    return [v for v in stage.values() if isinstance(v, dict) and "byte" in v]


def _listed_bytes(spec: dict) -> set[int]:
    kind = spec["kind"]
    if kind == "points":
        return {int(v) for v, _ in spec["points"]}
    if kind == "states":
        return {int(k) for k in spec["values"] if k != "*"}
    if kind == "window":
        return {int(spec["low"]), int(spec["high"])}
    return set()


def _settings(stage: dict) -> list[dict[str, int]]:
    addresses = sorted({p["byte"] for p in _params(stage)})
    settings = [
        dict(zip(addresses, combo, strict=True))
        for combo in itertools.product(GRID, repeat=len(addresses))
    ]
    for param in _params(stage):
        for byte in sorted(_listed_bytes(param["map"])):
            setting = dict.fromkeys(addresses, 64)
            setting[param["byte"]] = byte
            settings.append(setting)
    return settings


def _condition(sos: np.ndarray, freq: np.ndarray, fs: float) -> np.ndarray:
    # Relative condition number of evaluating the sections: sum of |coeffs| / |poly| per polynomial.
    z1 = np.exp(-1j * 2 * np.pi * freq / fs)
    powers = np.stack([np.ones_like(z1), z1, z1 * z1])
    kappa = np.zeros(freq.shape)
    for row in sos:
        for coeffs in (row[:3], row[3:]):
            kappa += np.abs(coeffs).sum() / np.abs(coeffs @ powers)
    return kappa


def _feedback_free(stage: dict) -> bool:
    return not (stage["kind"] == "allpass-chain" and "feedback" in stage)


def _stages() -> list:
    cases = []
    for name, model in _lti_models():
        for i, stage in enumerate(model["chain"]):
            if _feedback_free(stage):
                cases.append(pytest.param(model, stage, id=f"{name}[{i}]"))
    return cases


def test_every_lti_model_has_a_feedback_free_stage_to_check():
    assert len(_stages()) > 100


@pytest.mark.parametrize(("model", "stage"), _stages())
def test_the_coefficients_are_the_response(model: dict, stage: dict):
    fs = float(model["sample_rate_hz"])
    freq = np.geomspace(10.0, 0.499 * fs, 400)
    alone = {"sample_rate_hz": model["sample_rate_hz"], "chain": [stage]}
    for bytes_now in _settings(stage):
        sos = section_sos(stage, bytes_now, fs)
        assert sos.ndim == 2 and sos.shape[1] == 6, sos.shape
        assert np.all(sos[:, 3] == 1.0)
        _, from_sos = signal.sosfreqz(sos, worN=freq, fs=fs)
        rendered = reproduce.response(alone, bytes_now, freq)
        bound = ROUNDING * _condition(sos, freq, fs) * np.abs(rendered)
        ratio = np.abs(from_sos - rendered) / bound
        assert ratio.max() <= 1.0, (bytes_now, float(ratio.max()))


def test_a_chain_with_a_loop_has_no_sections():
    """The loop's denominator would have to be factored, which moves the values."""
    stage = {
        "kind": "allpass-chain",
        "sections": 8,
        "corner_hz": {"fixed": 1000.0},
        "mix": {"fixed": 1.0},
        "feedback": {"fixed": 0.5},
    }
    with pytest.raises(ValueError):
        section_sos(stage, {}, 32000.0)
