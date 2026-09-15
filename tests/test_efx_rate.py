"""Controls for reading a rate off saved takes.

The failure these guard against is the one that publishes rather than raises. An
interface has inputs the unit is not plugged into, and they are not silent; a
take whose output falls below one of them answers from that input instead, and a
rate read off an idle preamp is a number with a take behind it, a level beside it
and nothing to mark it as somebody else's.

The takes here are synthetic, so the answer is known: a tone amplitude modulated
at a stated frequency has to come back at that frequency, and a take the run
silenced has to come back as silence rather than as the room.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from soundings import efxrate, takes

SR = 48000
SECONDS = 6.0
RATE_HZ = 3.0


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


def modulated(rate_hz: float, *, scale: float = 1.0) -> np.ndarray:
    """A tone whose level swings at `rate_hz`, which is the rate to be read back."""
    t = np.arange(int(SECONDS * SR)) / SR
    carrier = np.sin(2 * np.pi * 440.0 * t) + 0.5 * np.sin(2 * np.pi * 880.0 * t)
    return carrier * (0.6 + 0.4 * np.sin(2 * np.pi * rate_hz * t)) * 0.2 * scale


@pytest.fixture
def crowded(tmp_path):
    """A run on an interface whose second input carries something the unit is not.

    Channel 0 is the unit and channel 1 is another input at a steady level the
    unit passes on its way down, so the setting that silences the unit is the
    take where reading each take's own loudest channel stops reading the unit.
    """
    other = np.random.default_rng(7).standard_normal(int(SECONDS * SR)) * 0.002
    store = takes.Store.open(tmp_path / "run")
    for value, scale in ((0, 0.0002), (64, 1.0), (127, 1.0)):
        store.keep(
            FakeRecording(np.stack([modulated(RATE_HZ, scale=scale), other], axis=1)),
            stimulus="held",
            setting=f"07-{value:03d}",
            take=0,
        )
    store.close(question="a synthetic run on a crowded interface")
    return tmp_path / "run"


def read(where, **extra):
    return efxrate.read_directory(
        where,
        type_id="01 21",
        address="40 03 07",
        setting=r"held-07-(?P<value>\d+)-00",
        hold_s=SECONDS - 1.2,
        **extra,
    )


def test_the_rate_that_was_modulated_is_the_rate_that_comes_back(crowded) -> None:
    found = read(crowded)
    loud = next(r for r in found["readings"] if r["value"] == 127)
    assert loud["rate_hz"] == pytest.approx(RATE_HZ, abs=0.2)


def test_the_record_names_the_channel_it_read_and_what_each_one_reached(crowded) -> None:
    picked = read(crowded)["channel"]
    assert picked["read"] == 0
    assert picked["chosen_by"] == "highest across the takes read"
    assert len(picked["reached_db"]) == 2
    assert picked["reached_db"][0] > picked["reached_db"][1]


def test_a_take_the_unit_went_quiet_in_is_still_read_from_the_units_channel(
    crowded,
) -> None:
    """The one that reads as a measurement. Read from whichever channel is loudest
    in it, the silenced take returns the other input's level and whatever rate can
    be found in noise, and nothing in the record contradicts it."""
    found = read(crowded)
    quiet = next(r for r in found["readings"] if r["value"] == 0)
    loud = next(r for r in found["readings"] if r["value"] == 127)
    assert quiet["heard_db"] < loud["heard_db"] - 40


def test_a_take_whose_loudest_channel_is_elsewhere_is_named(crowded) -> None:
    astray = read(crowded)["channel"]["loudest_elsewhere"]
    assert len(astray) == 1
    assert "07-000" in astray[0]


def test_a_channel_given_on_the_command_line_is_used_and_said_to_be(crowded) -> None:
    found = read(crowded, channel=1)
    assert found["channel"]["read"] == 1
    assert found["channel"]["chosen_by"] == "given"


def test_the_channel_is_chosen_from_the_highest_reached_not_the_average(tmp_path) -> None:
    """A sweep that silences the unit in most of its settings still reads the unit.

    Averaged over the run the quiet channel wins, which is how a level sweep loses
    its own output to an input nobody plugged anything into.
    """
    other = np.random.default_rng(11).standard_normal(int(SECONDS * SR)) * 0.002
    store = takes.Store.open(tmp_path / "run")
    for value in range(6):
        scale = 1.0 if value == 5 else 0.0002
        store.keep(
            FakeRecording(np.stack([modulated(RATE_HZ, scale=scale), other], axis=1)),
            stimulus="held",
            setting=f"07-{value:03d}",
            take=0,
        )
    store.close(question="a sweep that is mostly off")
    found = read(tmp_path / "run")
    assert found["channel"]["read"] == 0
    assert len(found["channel"]["loudest_elsewhere"]) == 5


def test_a_setting_pattern_without_the_value_group_refuses(crowded) -> None:
    with pytest.raises(ValueError, match="value"):
        efxrate.read_directory(
            crowded, type_id="01 21", address="40 03 07", setting=r"held-07-\d+-00"
        )


@pytest.fixture
def approached(tmp_path):
    """One byte written twice, from a reset and from the other end of the range.

    Two takes of value 64, modulated at different rates, which is the shape a
    record of one byte to one rate cannot hold: both rows say 64 and they disagree.
    """
    store = takes.Store.open(tmp_path / "run")
    for came_from, value, rate_hz in (
        ("rest", 64, RATE_HZ),
        ("127", 64, RATE_HZ * 1.5),
        ("rest", 127, RATE_HZ * 2),
    ):
        store.keep(
            FakeRecording(np.stack([modulated(rate_hz), np.zeros(int(SECONDS * SR))], 1)),
            stimulus="held",
            setting=f"07-from-{came_from}-to-{value:03d}",
            take=0,
        )
    store.close(question="one byte approached from two places")
    return tmp_path / "run"


def from_either_end(where):
    return efxrate.read_directory(
        where,
        type_id="01 22",
        address="40 03 03",
        setting=r"held-07-from-(?P<from>rest|\d+)-to-(?P<value>\d+)-00",
        hold_s=SECONDS - 1.2,
    )


def test_where_a_byte_was_written_from_lands_in_the_reading(approached) -> None:
    rows = from_either_end(approached)["readings"]
    assert [(r["value"], r["came_from"]) for r in rows] == [
        (64, "127"),
        (64, "rest"),
        (127, "rest"),
    ]


def test_two_takes_of_one_byte_keep_their_own_rates(approached) -> None:
    """The reading the record exists to be able to hold at all."""
    rows = [r for r in from_either_end(approached)["readings"] if r["value"] == 64]
    by_where = {r["came_from"]: r["rate_hz"] for r in rows}
    assert by_where["rest"] == pytest.approx(RATE_HZ, abs=0.2)
    assert by_where["127"] == pytest.approx(RATE_HZ * 1.5, abs=0.2)


def test_the_record_says_what_came_from_means_only_when_it_carries_one(
    approached, crowded
) -> None:
    assert "why_came_from" in from_either_end(approached)
    assert "why_came_from" not in read(crowded)
    assert all("came_from" not in r for r in read(crowded)["readings"])
