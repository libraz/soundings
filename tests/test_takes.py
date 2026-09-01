"""Controls for keeping takes, whose failures are silent by construction.

A store that loses a channel, rescales on the way back, or writes a manifest that
does not name the file next to it fails without saying so: the WAVs are there,
they play, and the numbers computed from them are merely different. Since the
whole point is to be able to measure again months later with the machine gone,
there is nothing left to check them against by then.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest

from soundings import takes

SR = 48000


@dataclass
class FakeRecording:
    samples: np.ndarray
    sample_rate: int = SR
    device: str = "test input"
    overflows: int = 0
    open_seconds: float = 0.65

    @property
    def seconds(self) -> float:
        return self.samples.shape[0] / self.sample_rate


def stereo(seconds: float = 0.5, *, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            rng.standard_normal(int(seconds * SR)) * 0.2,
            rng.standard_normal(int(seconds * SR)) * 0.05,
        ],
        axis=1,
    )


def test_a_take_comes_back_sample_for_sample(tmp_path) -> None:
    signal = stereo()
    takes.write(tmp_path / "one", signal, SR)
    back, rate = takes.read(tmp_path / "one.wav")
    assert rate == SR
    assert back.shape == signal.shape
    assert np.abs(back - signal).max() < 1e-7


def test_both_channels_survive_rather_than_only_the_loud_one() -> None:
    """A stereo effect -- an auto-pan, the second half of a rotary -- is only in
    the difference between the channels, and a mono store deletes it."""
    signal = stereo()
    assert signal.shape[1] == 2
    assert takes.loudest(signal).shape == (signal.shape[0],)
    assert np.allclose(takes.loudest(signal), signal[:, 0])


def test_a_quiet_take_is_not_scaled_up_on_the_way_out(tmp_path) -> None:
    """Normalising would divide out the very level differences a verdict is made of."""
    signal = stereo() * 1e-4
    takes.write(tmp_path / "quiet", signal, SR)
    back, _ = takes.read(tmp_path / "quiet.wav")
    assert np.abs(back).max() == pytest.approx(np.abs(signal).max(), rel=1e-6)


def test_the_manifest_names_every_file_beside_it(tmp_path) -> None:
    store = takes.Store.open(tmp_path / "run")
    for value in (0, 127):
        for index in range(2):
            store.keep(
                FakeRecording(stereo(0.2, seed=value + index)),
                stimulus="struck",
                setting=str(value),
                take=index,
            )
    manifest = store.close(label="CC91 0 against 127", controller=91)

    written = json.loads(manifest.read_text())
    assert written["controller"] == 91
    assert len(written["takes"]) == 4
    for entry in written["takes"]:
        assert (manifest.parent / entry["file"]).exists()
        assert entry["channels"] == 2
        assert entry["sample_rate"] == SR


def test_a_setting_that_is_not_a_filename_still_makes_one(tmp_path) -> None:
    """An address is three hex bytes with spaces in it, and it names a take."""
    store = takes.Store.open(tmp_path / "run")
    path = store.keep(
        FakeRecording(stereo(0.2)), stimulus="struck", setting="40 01 30 = 7F", take=0
    )
    assert path.exists()
    assert " " not in path.name
