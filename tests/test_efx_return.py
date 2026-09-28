"""Controls for reading how much of a stereo take is not one L/R direction.

A covariance always has two eigenvalues, so a take with nothing decorrelated in
it still returns a smaller one -- what has to be checked is that it lands at the
numerical floor rather than announcing a share that is not there. And a share
that is there has to come back at the level it was put in at, not at whatever
the covariance estimate happens to scatter to.

The takes here are synthetic, built from independent noise streams so the answer
is known: a source panned any way at all is one direction, and an independent
stream added to each channel at a stated level is the only decorrelated power in
the take.
"""

from __future__ import annotations

import numpy as np
import pytest

from soundings import efxreturn, takes

SR = 48000


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def coherent_pair(seed: int, seconds: float = 2.0, gain_l: float = 0.7, gain_r: float = 0.3):
    """One source, panned, with nothing added to either channel independently."""
    rng = np.random.default_rng(seed)
    mono = rng.standard_normal(int(seconds * SR)) * 0.2
    return np.stack([mono * gain_l, mono * gain_r], axis=1)


def with_decorrelated(pair: np.ndarray, seed: int, level: float) -> np.ndarray:
    """The pair with an independent stream added to each channel at a stated RMS."""
    rng = np.random.default_rng(seed)
    added_l = rng.standard_normal(pair.shape[0])
    added_r = rng.standard_normal(pair.shape[0])
    added_l *= level / rms(added_l)
    added_r *= level / rms(added_r)
    out = pair.copy()
    out[:, 0] += added_l
    out[:, 1] += added_r
    return out


def test_a_fully_coherent_pair_reads_at_the_noise_floor() -> None:
    """Any pan of one source is one direction, whatever the two gains are."""
    for gain_l, gain_r in ((0.7, 0.3), (1.0, 0.0), (-0.4, 0.9), (0.5, 0.5)):
        pair = coherent_pair(1, gain_l=gain_l, gain_r=gain_r)
        found = efxreturn.measure(pair, SR)
        assert found["incoherent_db"] < found["level_db"] - 80.0
        assert found["frames"] > 0


def test_an_added_decorrelated_component_reads_at_the_level_it_was_put_in_at() -> None:
    """The smaller eigenvalue is exactly the independent power when both channels
    carry the same amount of it -- adding an isotropic term to a rank-one matrix
    shifts both eigenvalues up by that term, so the smaller one becomes it."""
    for level in (0.02, 0.05, 0.1):
        pair = with_decorrelated(coherent_pair(2), seed=3, level=level)
        found = efxreturn.measure(pair, SR)
        expected_db = 20.0 * np.log10(level)
        assert found["incoherent_db"] == pytest.approx(expected_db, abs=0.5)


def test_silence_reads_as_the_silence() -> None:
    """Nothing played returns nothing, on both figures, rather than a plausible number."""
    pair = np.zeros((int(2.0 * SR), 2))
    found = efxreturn.measure(pair, SR)
    assert found["level_db"] < -150.0
    assert found["incoherent_db"] < -150.0


def test_a_short_pan_gives_no_frames_rather_than_a_false_reading() -> None:
    """Shorter than one frame, there is nothing to pool and the reading says so."""
    pair = coherent_pair(4, seconds=0.01)
    found = efxreturn.measure(pair, SR)
    assert found["frames"] == 0
    assert found["incoherent_db"] is None
    assert found["level_db"] is None


# ---- the directory sweep


@pytest.fixture
def swept(tmp_path):
    """A run whose settings hold a known amount of decorrelated return, and one that holds none."""
    store = takes.Store.open(tmp_path / "returns")
    levels = {0: None, 64: 0.03, 127: 0.08}
    for value, level in levels.items():
        for take in range(2):
            pair = coherent_pair(10 + value + take, gain_l=0.6, gain_r=0.4)
            if level is not None:
                pair = with_decorrelated(pair, seed=100 + value + take, level=level)
            six = np.zeros((pair.shape[0], 4))
            six[:, 2:4] = pair
            store.keep(
                takes_recording(six),
                stimulus="struck_kit",
                setting=f"send-{value:03d}",
                take=take,
            )
    store.close(question="a run of known returns")
    return tmp_path / "returns"


def takes_recording(samples: np.ndarray):
    from dataclasses import dataclass

    @dataclass
    class FakeRecording:
        samples: np.ndarray
        sample_rate: int = SR
        device: str = "test input"
        overflows: int = 0
        open_seconds: float = 0.6

        @property
        def seconds(self) -> float:
            return self.samples.shape[0] / self.sample_rate

    return FakeRecording(samples)


def at(found: dict, value: int) -> dict:
    return next(r for r in found["readings"] if r[efxreturn.VALUE] == value)


def test_the_directory_sweep_reports_each_setting_and_its_own_floor(swept) -> None:
    found = efxreturn.read_directory(
        swept,
        type_id="01 26",
        address="40 03 17",
        setting=r"send-(?P<value>\d+)",
        channels=(2, 3),
        lead_s=0.0,
        trim_s=0.0,
        hold_s=2.0,
    )
    assert found["settings_asked"] == [0, 64, 127]
    assert at(found, 0)["incoherent_db"] < at(found, 127)["incoherent_db"]
    assert found["floor_incoherent_db"] is not None
    assert found["settings_taken_twice"] == [0, 64, 127]
    assert found["channel"]["read"] == [2, 3]
    assert found["type"] == "01 26"
    assert found["address"] == "40 03 17"


def test_a_setting_pattern_without_a_value_group_is_refused(swept) -> None:
    with pytest.raises(ValueError, match="value"):
        efxreturn.read_directory(
            swept, type_id="01 26", address="40 03 17", setting=r"send-\d+", channels=(2, 3)
        )


def test_address_and_controller_are_mutually_exclusive(swept) -> None:
    with pytest.raises(ValueError, match="address"):
        efxreturn.read_directory(
            swept,
            type_id="01 26",
            address="40 03 17",
            controller=91,
            setting=r"send-(?P<value>\d+)",
            channels=(2, 3),
        )


def test_held_from_names_where_held_was_read_and_says_so(swept) -> None:
    """A held block carried in from the capturing run's own report is marked as such."""
    found = efxreturn.read_directory(
        swept,
        type_id="01 26",
        address="40 03 17",
        setting=r"send-(?P<value>\d+)",
        channels=(2, 3),
        held=[{"address": "40 03 00", "bytes": "01 26"}, {"address": "40 42 22", "bytes": "01"}],
        held_from=".cache/efx-params-01-26/40-03-17.json",
    )
    assert found["held_from"] == ".cache/efx-params-01-26/40-03-17.json"
    assert found["why_held_from"]
    assert "GS Reset" in found["why_held_from"]


def test_no_held_from_is_omitted_rather_than_null(swept) -> None:
    found = efxreturn.read_directory(
        swept,
        type_id="01 26",
        address="40 03 17",
        setting=r"send-(?P<value>\d+)",
        channels=(2, 3),
    )
    assert "held_from" not in found
    assert "why_held_from" not in found
