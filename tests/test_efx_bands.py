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
    # The same chain with nothing played, forty decibels down. A setting that
    # turns the output off lands here, and its bands are this shape.
    for index in range(2):
        store.keep(
            FakeRecording(noise(200 + index) * 0.01),
            stimulus="held",
            setting=f"silence-{index:02d}",
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
        **{
            "control": r"held-bypassed-\d+-00",
            "silence": r"held-silence-\d+-00",
            "bands_hz": BANDS,
            "hold_s": SECONDS - 1.2,
            **extra,
        },
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


def test_a_setting_that_turned_the_output_off_says_how_far_above_silence_it_was(
    directory,
) -> None:
    """The trap this control exists for.

    A level slot at its bottom byte records the room rather than the effect, and
    read as a deviation from the flat setting that is a large, ragged, frequency
    dependent profile -- which looks exactly like a reading. The number that tells
    the two apart is how far the take was above a take with nothing played.
    """
    store = takes.Store.open(directory)
    store.keep(
        FakeRecording(noise(7) * 0.01), stimulus="held", setting="04-999", take=0
    )
    found = read(directory)
    off = next(r for r in found["readings"] if r["value"] == 999)
    loud = next(r for r in found["readings"] if r["value"] == 127)
    assert off["above_the_silence_db"] == pytest.approx(0.0, abs=1.0)
    assert loud["above_the_silence_db"] > 30.0
    assert found["silence"]["heard_db"] < found["reference"]["heard_db"]


def test_the_floors_own_profile_is_published_rather_than_only_its_level(
    directory,
) -> None:
    """So a reading suspected of being the floor can be held against the floor."""
    found = read(directory)
    assert len(found["silence"]["takes"]) == 2
    assert len(found["silence"]["band_db"]) == len(BANDS)
    assert found["reference"]["above_the_silence_db"] > 30.0


def test_without_silence_takes_the_record_says_so_rather_than_guessing(directory) -> None:
    found = efxbands.read_directory(
        directory,
        type_id="01 00",
        address="40 03 04",
        setting=r"held-04-(?P<value>\d+)-00",
        reference=r"held-flat-\d+-00",
        bands_hz=BANDS,
        hold_s=SECONDS - 1.2,
    )
    assert found["silence"]["takes"] == []
    assert found["silence"]["band_db"] is None
    assert all(r["above_the_silence_db"] is None for r in found["readings"])


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


@pytest.fixture
def crossed(tmp_path):
    """A run on an interface whose second input carries something the unit is not.

    Channel 0 is the unit and channel 1 is another input, kept at a steady level
    the unit passes on its way down. One swept setting turns the unit's own output
    below it, which is where reading each take's loudest channel stops reading the
    unit and starts reading the other input.
    """
    other = np.random.default_rng(900).standard_normal(int(SECONDS * SR)) * 0.004

    def pair(unit: np.ndarray) -> np.ndarray:
        return np.stack([unit, other], axis=1)

    store = takes.Store.open(tmp_path / "run")
    for index in range(4):
        store.keep(FakeRecording(pair(noise(index)[:, 0])), stimulus="held",
                   setting=f"flat-{index:02d}", take=0)
    for value, scale in ((0, 0.0005), (64, 1.0), (127, 1.0)):
        store.keep(FakeRecording(pair(noise(0)[:, 0] * scale)), stimulus="held",
                   setting=f"16-{value:03d}", take=0)
    store.close(question="a synthetic run on a crowded interface")
    return tmp_path / "run"


def crossed_read(where, **extra):
    return efxbands.read_directory(
        where,
        type_id="01 00",
        address="40 03 16",
        setting=r"held-16-(?P<value>\d+)-00",
        reference=r"held-flat-\d+-00",
        bands_hz=BANDS,
        hold_s=SECONDS - 1.2,
        **extra,
    )


def test_the_record_names_the_channel_it_read_and_every_channel_it_saw(directory) -> None:
    found = read(directory)
    picked = found["channel"]
    assert picked["read"] == 0
    assert picked["chosen_by"] == "loudest in the reference takes"
    assert len(picked["reference_db"]) == 2
    # The fixture's second channel is half the first, which is 6 dB down.
    assert picked["reference_db"][0] - picked["reference_db"][1] == pytest.approx(6.0, abs=0.5)


def test_a_take_the_unit_went_quiet_in_is_still_read_from_the_units_channel(
    crossed,
) -> None:
    """The one that reads as a measurement. Reading each take's own loudest channel
    returns a full profile of the other input, which is not a number the byte
    produced and not a number anything in the record contradicts."""
    found = crossed_read(crossed)
    quiet = next(r for r in found["readings"] if r["value"] == 0)
    loud = next(r for r in found["readings"] if r["value"] == 127)
    # Read from the unit, the setting that turned it off is far below the others.
    assert quiet["heard_db"] < loud["heard_db"] - 40
    assert found["channel"]["read"] == 0


def test_a_take_whose_loudest_channel_is_elsewhere_is_named(crossed) -> None:
    found = crossed_read(crossed)
    astray = found["channel"]["loudest_elsewhere"]
    assert len(astray) == 1
    assert "16-000" in astray[0]


def test_a_channel_given_on_the_command_line_is_used_and_said_to_be(crossed) -> None:
    found = crossed_read(crossed, channel=1)
    assert found["channel"]["read"] == 1
    assert found["channel"]["chosen_by"] == "given"
    # Every reading is now of the other input, so the byte moves nothing.
    assert all(r["largest_db"] is None for r in found["readings"])


@pytest.fixture
def two_legs(tmp_path):
    """A run whose second channel is the unit's other output rather than an input.

    The fixture's takes carry the same shaping in both channels at half the level,
    so the second channel follows the parameter and disagrees with the first about
    nothing -- which is what a record has to be able to say, and what it cannot say
    from one channel.
    """
    store = takes.Store.open(tmp_path / "run")
    for index in range(4):
        store.keep(FakeRecording(noise(index)), stimulus="held", setting=f"flat-{index:02d}", take=0)
    for value, by_db in ((0, -12.0), (64, 0.0), (127, 6.0)):
        store.keep(
            FakeRecording(noise(0, cut_at=1000, by_db=by_db)),
            stimulus="held",
            setting=f"04-{value:03d}",
            take=0,
        )
    store.close(question="a synthetic run on two legs")
    return tmp_path / "run"


def test_the_second_channel_is_read_and_named(two_legs) -> None:
    found = read(two_legs, silence=None, control=None)
    beside = found["other_channel"]
    assert beside["read"] == 1
    assert len(beside["band_db"]) == len(BANDS)


def test_the_second_channel_follows_the_parameter_and_agrees_with_the_first(
    two_legs,
) -> None:
    """Both halves matter. That it moved says it carried the unit rather than an
    idle input; that it agrees says one channel was the whole answer here."""
    found = read(two_legs, silence=None, control=None)
    cut = next(r for r in found["readings"] if r["value"] == 0)
    assert cut["other_db"] == pytest.approx(-12.0, abs=0.5)
    assert abs(cut["apart_db"]) < 0.5


def test_a_second_channel_carrying_something_else_does_not_follow(crossed) -> None:
    """The control's own control. The other channel here is an unrelated input, so
    it stays put while the first swings, and the record shows exactly that."""
    found = crossed_read(crossed)
    assert found["other_channel"]["read"] == 1
    for reading in found["readings"]:
        assert abs(reading["other_db"]) < 1.0


def test_a_take_with_one_channel_reports_no_second_one(tmp_path) -> None:
    store = takes.Store.open(tmp_path / "run")
    mono = lambda seed: noise(seed)[:, :1]  # noqa: E731
    for index in range(4):
        store.keep(FakeRecording(mono(index)), stimulus="held", setting=f"flat-{index:02d}", take=0)
    for value in (0, 64):
        store.keep(FakeRecording(mono(0)), stimulus="held", setting=f"04-{value:03d}", take=0)
    store.close(question="a mono run")
    found = read(tmp_path / "run", silence=None, control=None)
    assert found["other_channel"]["read"] is None
    assert all("apart_db" not in r for r in found["readings"])
