"""Controls for reading a parameter as a profile rather than as a number.

Every failure this guards against reads as a measurement. A floor computed from
one take is zero, so every deviation clears it and a record of pure noise looks
like a record of a filter. A reference taken from the wrong takes shifts every
profile by a constant and the sweep still looks monotonic. A sweep pattern that
also matches the reference folds the flat takes into the readings and the curve
grows a value that never moved. None of these raise; they publish.

The takes here are synthetic, so the answer is known: a band given a known
attenuation has to come back at that band and at that figure, and a band given
nothing has to come back inside the floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from soundings import efxbands, takes

SR = 48000
SECONDS = 3.0
BANDS = (250, 1000, 4000)


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


def noise(seed: int, *, cut_at: float | None = None, by_db: float = 0.0) -> np.ndarray:
    """Noise, optionally with one third octave scaled by a known amount.

    Shaped in the frequency domain so the attenuation is exactly the figure asked
    for over exactly the band asked for -- the reading has to return that figure,
    and a reading that returns most of it is a reading with a leak in it.
    """
    rng = np.random.default_rng(seed)
    body = rng.standard_normal(int(SECONDS * SR)) * 0.1
    if cut_at is not None:
        spectrum = np.fft.rfft(body)
        freq = np.fft.rfftfreq(body.size, 1.0 / SR)
        inside = (freq >= cut_at / 2 ** (1 / 6)) & (freq < cut_at * 2 ** (1 / 6))
        spectrum[inside] *= 10.0 ** (by_db / 20.0)
        body = np.fft.irfft(spectrum, n=body.size)
    return np.stack([body, body * 0.5], axis=1)


@pytest.fixture
def directory(tmp_path):
    """A run's worth of takes: four repeats of flat, a control, and a sweep."""
    store = takes.Store.open(tmp_path / "run")
    for index in range(4):
        store.keep(FakeRecording(noise(index)), stimulus="held", setting=f"flat-{index:02d}", take=0)
    # The control and every swept take carry one of the reference takes' own
    # noise, so that the only difference between them is the band this shaped on
    # purpose. Independent noise would put the untouched bands on either side of a
    # floor drawn from four other draws, and the assertions below would pass or
    # fail on the seed rather than on the reading.
    store.keep(FakeRecording(noise(1)), stimulus="held", setting="bypassed-00", take=0)
    for value, by_db in ((0, -12.0), (64, 0.0), (127, 6.0)):
        store.keep(
            FakeRecording(noise(0, cut_at=1000, by_db=by_db)),
            stimulus="held",
            setting=f"04-{value:03d}",
            take=0,
        )
    store.close(question="a synthetic run")
    return tmp_path / "run"


def read(where, **extra):
    return efxbands.read_directory(
        where,
        type_id="01 00",
        address="40 03 04",
        setting=r"held-04-(?P<value>\d+)-00",
        reference=r"held-flat-\d+-00",
        control=r"held-bypassed-\d+-00",
        bands_hz=BANDS,
        hold_s=SECONDS - 1.2,
        **extra,
    )


def test_the_band_that_was_cut_is_the_band_that_comes_back(directory) -> None:
    found = read(directory)
    by_value = {r["value"]: r for r in found["readings"]}
    assert sorted(by_value) == [0, 64, 127]
    assert by_value[0]["largest_at_hz"] == 1000
    assert by_value[0]["largest_db"] == pytest.approx(-12.0, abs=0.5)
    assert by_value[127]["largest_at_hz"] == 1000
    assert by_value[127]["largest_db"] == pytest.approx(6.0, abs=0.5)


def test_a_setting_that_did_nothing_reports_nothing_rather_than_a_small_number(
    directory,
) -> None:
    """The floor is the point. Without it the same take reads as a tiny tilt."""
    found = read(directory)
    flat = next(r for r in found["readings"] if r["value"] == 64)
    assert flat["outside_the_floor_hz"] == []
    assert flat["largest_db"] is None
    assert all(edge > 0 for edge in found["reference"]["floor_db"])


def test_the_bands_that_were_not_touched_stay_inside_the_floor(directory) -> None:
    found = read(directory)
    cut = next(r for r in found["readings"] if r["value"] == 0)
    assert cut["outside_the_floor_hz"] == [1000]


def test_the_reference_takes_are_not_also_read_as_settings(directory) -> None:
    """A sweep pattern that reaches the flat takes grows a setting nobody swept."""
    found = read(directory)
    names = {r["take"] for r in found["readings"]}
    assert not names & set(found["reference"]["takes"])
    assert not names & set(found["control"]["takes"])


def test_the_control_is_read_against_the_same_reference(directory) -> None:
    found = read(directory)
    assert len(found["control"]["readings"]) == 1
    assert found["control"]["readings"][0]["outside_the_floor_hz"] == []


def test_a_take_no_pattern_named_is_counted_rather_than_dropped(directory) -> None:
    store = takes.Store.open(directory)
    store.keep(FakeRecording(noise(9)), stimulus="held", setting="strays-000", take=0)
    found = read(directory)
    assert found["takes_not_matching"]["count"] == 1
    assert any("stray" in name for name in found["takes_not_matching"]["settings"])


def test_a_reference_that_matches_nothing_refuses_rather_than_returns(directory) -> None:
    """Left to compute, an empty reference is a mean of nothing and a floor of zero."""
    with pytest.raises(ValueError, match="reference"):
        efxbands.read_directory(
            directory,
            type_id="01 00",
            address="40 03 04",
            setting=r"held-04-(?P<value>\d+)-00",
            reference=r"held-nothing-\d+-00",
            bands_hz=BANDS,
            hold_s=SECONDS - 1.2,
        )


def test_a_setting_pattern_without_the_value_group_refuses(directory) -> None:
    with pytest.raises(ValueError, match="value"):
        efxbands.read_directory(
            directory,
            type_id="01 00",
            address="40 03 04",
            setting=r"held-04-\d+-00",
            reference=r"held-flat-\d+-00",
            bands_hz=BANDS,
            hold_s=SECONDS - 1.2,
        )


def test_the_record_says_where_it_was_read_from_and_what_was_held(directory) -> None:
    found = read(directory, held=[{"address": "40 03 06", "bytes": "40"}])
    assert found["type"] == "01 00"
    assert found["address"] == "40 03 04"
    assert found["bands_hz"] == list(BANDS)
    assert found["held"] == [{"address": "40 03 06", "bytes": "40"}]
    assert str(directory) == found["takes_from"]
    assert found["manifest"]["files_present"] >= 8
