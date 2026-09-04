"""Judging takes that were recorded in an earlier session.

The point of keeping takes is that a question raised after the session can be
answered without paying for the device again. Until this existed the verdict was
reachable only at capture time, so it could not be.

What these guard is that the offline path is the same judgement and not a weaker
one, and that it refuses rather than guesses when the directory cannot support
the comparison asked of it. No hardware: takes are written in the shape a --save
run leaves them.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from soundings import audible, rejudge
from soundings.takes import write

SR = 48000


def note(*, seed: int, gain: float = 1.0, seconds: float = 1.0, take: int = 0) -> np.ndarray:
    """A struck note with a silent head, so a lead-in exists to measure a floor in.

    Each take of it differs a little, as every take off a machine does. Identical
    takes would make the yardstick perfect, and a yardstick of minus infinity is
    cleared by anything -- which would let these pass without measuring.
    """
    n = int(seconds * SR)
    index = np.arange(n)
    body = np.random.default_rng(seed).standard_normal(n) * np.where(
        index < 0.2 * SR, 0.0, np.exp(-4.0 * index / SR)
    )
    jitter = np.random.default_rng(1000 + take).standard_normal(n) * 1e-3
    return (body + jitter) * gain


def a_run(root, *, settings: dict[str, list[np.ndarray]], stimulus: str = "struck") -> None:
    """Write one saved run: several takes at each of two settings, plus a manifest."""
    entries = []
    for setting, takes in settings.items():
        for index, signal in enumerate(takes):
            name = f"{stimulus}-{setting}-{index:02d}"
            write(root / name, signal[:, None], SR)
            entries.append(
                {
                    "file": f"{name}.wav",
                    "stimulus": stimulus,
                    "setting": setting,
                    "take": index,
                }
            )
    (root / "takes-manifest.json").write_text(
        json.dumps(
            {
                "label": "CC7 0 against 127",
                "controller": 7,
                "address": None,
                "takes": entries,
                "stimuli": [
                    {
                        "name": stimulus,
                        "program": 0,
                        "note": 60,
                        "velocity": 100,
                        "hold_s": 1.0,
                        "captured_s": 1.0,
                        "lead_s": 0.2,
                    }
                ],
            }
        )
    )


def test_a_saved_run_reaches_the_verdict_its_session_would_have(tmp_path):
    """The same comparator over the same takes with the same yardstick, so the
    numbers must be the session's and not merely the same conclusion."""
    quiet = [note(seed=1, gain=1.0, take=i) for i in range(4)]
    loud = [note(seed=1, gain=4.0, take=10 + i) for i in range(4)]
    a_run(tmp_path, settings={"0": quiet, "127": loud})

    overall, manifest = rejudge.judge_directory(tmp_path)
    live = audible.judge(quiet, loud, SR, label="x", stimulus="y", silence_before=0.16)

    assert overall.audible and overall.heard_by == ["struck"]
    assert overall.verdicts[0].to_json()["across_setting_level_db"] == pytest.approx(
        live.to_json()["across_setting_level_db"]
    )


def test_two_settings_that_sound_the_same_are_not_called_audible(tmp_path):
    """The offline path must be able to return a null, or a passing verdict from
    it says nothing."""
    takes = [note(seed=2, take=i) for i in range(4)]
    a_run(tmp_path, settings={"0": takes, "127": [t.copy() for t in takes]})

    overall, _ = rejudge.judge_directory(tmp_path)

    assert not overall.audible


def test_the_pair_compared_defaults_to_the_first_and_last_recorded(tmp_path):
    """A two-valued run leaves exactly one pair, and reading it out of the
    manifest is what keeps the command from being told what it can work out."""
    a_run(tmp_path, settings={"0": [note(seed=3)], "64": [note(seed=4)], "127": [note(seed=5)]})

    manifest = json.loads((tmp_path / "takes-manifest.json").read_text())

    assert rejudge.settings_in(manifest) == ["0", "64", "127"]


def test_a_setting_the_run_never_recorded_is_refused(tmp_path):
    """Silently falling back to a pair that was recorded would answer a question
    nobody asked, under the label of the one they did."""
    a_run(
        tmp_path,
        settings={
            "0": [note(seed=6, take=i) for i in range(2)],
            "127": [note(seed=6, take=5 + i) for i in range(2)],
        },
    )

    with pytest.raises(rejudge.NothingToJudge):
        rejudge.judge_directory(tmp_path, second="64")


def test_a_run_with_takes_at_one_setting_only_is_refused(tmp_path):
    """Half a comparison, and there is nothing to compare it against."""
    a_run(tmp_path, settings={"0": [note(seed=8, take=i) for i in range(3)]})

    with pytest.raises(rejudge.NothingToJudge):
        rejudge.judge_directory(tmp_path)


def test_each_stimulus_in_a_directory_is_judged_on_its_own(tmp_path):
    """A run asks a parameter under several notes and the verdict is their union,
    so pooling their takes would answer a question none of them asked."""
    a_run(
        tmp_path,
        settings={
            "0": [note(seed=9, take=i) for i in range(3)],
            "127": [note(seed=9, gain=4.0, take=5 + i) for i in range(3)],
        },
    )
    manifest = json.loads((tmp_path / "takes-manifest.json").read_text())
    for index in range(3):
        for setting, gain in (("0", 1.0), ("127", 1.0)):
            name = f"sustained-{setting}-{index:02d}"
            write(tmp_path / name, note(seed=20, gain=gain, take=30 + index)[:, None], SR)
            manifest["takes"].append(
                {"file": f"{name}.wav", "stimulus": "sustained", "setting": setting, "take": index}
            )
    manifest["stimuli"].append({"name": "sustained", "lead_s": 0.2})
    (tmp_path / "takes-manifest.json").write_text(json.dumps(manifest))

    overall, _ = rejudge.judge_directory(tmp_path)

    assert sorted(v.stimulus_name for v in overall.verdicts) == ["struck", "sustained"]
