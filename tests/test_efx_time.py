"""Controls for reading a delay as a time rather than as whatever peaked highest.

Every failure this guards against reads as a measurement. A cepstrum always has a
strongest peak, so a take with no copy in it returns a delay -- a short, plausible,
repeatable one. A control curve left unsubtracted hands every reading whatever the
stimulus itself put in the log spectrum. A floor drawn from settings that returned
nothing says the reading is exact. A search whose top is the printed end of the
range reports a byte that ran past it as having stopped there. None of these
raise; they publish.

The takes here are synthetic, so the answer is known: a copy put in at a stated
delay has to come back at that delay, and a take with none has to be refused.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from soundings import efxtime, takes

SR = 48000
SECONDS = 6.0
LEAD = 0.6
HOLD = 5.0


@dataclass
class FakeRecording:
    samples: np.ndarray
    sample_rate: int = SR
    device: str = "test input"
    overflows: int = 0
    open_seconds: float = LEAD

    @property
    def seconds(self) -> float:
        return self.samples.shape[0] / self.sample_rate


def carrier(seed: int) -> np.ndarray:
    """Broadband noise, which is what this reading wants of a stimulus.

    A held note would be the wrong control here rather than a harder one: its own
    log spectrum is a comb, so it puts peaks in the cepstrum at its period and
    every multiple, and a test built on it would be measuring the mask that keeps
    those out instead of the reading.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(SECONDS * SR)) * 0.1


def with_a_copy(body: np.ndarray, delay_ms: float, ratio: float = 0.8) -> np.ndarray:
    """The same signal with one copy of itself added at a stated delay."""
    late = int(round(delay_ms / 1000.0 * SR))
    out = body.copy()
    out[late:] += ratio * body[: body.size - late]
    return out


def stereo(body: np.ndarray) -> np.ndarray:
    return np.stack([body, body * 0.5], axis=1)


DELAYS = {0: None, 16: 3.0, 64: 12.0, 127: 40.0}
"""What each setting had put into it. `None` is a take with no copy at all, which
is what the bottom of a printed range beginning at zero returns."""


@pytest.fixture
def swept(tmp_path):
    """A run whose settings hold known delays, and one that holds none."""
    store = takes.Store.open(tmp_path / "delays")
    for index in range(2):
        store.keep(
            FakeRecording(stereo(carrier(100 + index))),
            stimulus="held",
            setting=f"out-{index:02d}",
            take=0,
        )
    for value, delay in DELAYS.items():
        body = carrier(value)
        store.keep(
            FakeRecording(stereo(body if delay is None else with_a_copy(body, delay))),
            stimulus="held",
            setting=f"delay-{value:03d}",
            take=0,
        )
    # One setting taken again, which is the only thing that gives the run a floor.
    again = carrier(64)
    store.keep(
        FakeRecording(stereo(with_a_copy(again, DELAYS[64]))),
        stimulus="held",
        setting="again-064",
        take=0,
    )
    store.close(question="a run of known delays")
    return tmp_path / "delays"


def read(where, **extra) -> dict:
    asked = {
        "type_id": "01 50",
        "address": "40 03 03",
        "setting": r"(?:delay|again)-(?P<value>\d+)",
        "control": "out-",
        "frame": 16384,
        "hop": 4096,
        "searched_ms": (0.4, 80.0),
        "channel": 0,
        "lead_s": LEAD,
        "hold_s": HOLD,
    }
    asked.update(extra)
    return efxtime.read_directory(where, **asked)


def at(found: dict, value: int) -> dict:
    return next(r for r in found["readings"] if r[efxtime.VALUE] == value)


def test_a_copy_comes_back_at_the_delay_it_was_put_in_at(swept) -> None:
    """The whole stage in one assertion: a stated delay has to be returned."""
    found = read(swept)
    for value, delay in DELAYS.items():
        if delay is None:
            continue
        assert at(found, value)["ms"] == pytest.approx(delay, abs=0.05)
        assert at(found, value)["admitted"]


def test_a_take_with_no_copy_in_it_is_refused_rather_than_given_a_delay(swept) -> None:
    """A cepstrum always peaks somewhere, so the absence has to be a verdict.

    Without the admission rule this row returns a short delay with nothing wrong
    with its shape, and a byte whose printed range begins at zero then publishes a
    table whose first entries are the roughness of the room.
    """
    found = read(swept)
    assert not at(found, 0)["admitted"]
    assert 0 not in found["settings_admitted"]
    assert 0 in found["settings_asked"]
    assert found["why_admitted"]


def test_the_even_rahmonic_is_not_the_one_that_comes_back(swept) -> None:
    """A single copy puts its neighbours at odd multiples, not at every multiple.

    The even terms of the expanded log are negative, so a reading whose next peaks
    sit at three and five times where it put the delay has found a two-path comb.
    One that found them at two and four has found something else, and the record
    publishes them for exactly that reason.

    Asked of the setting whose third and fifth multiples both fit inside the
    search. Past its top a rahmonic is not absent from the unit, it is outside the
    reading, and a row read there reports the roughness where its neighbours would
    have been.
    """
    row = at(read(swept), 64)
    ratios = sorted(q / row["ms"] for q in row["also_ms"])
    assert ratios == pytest.approx([3.0, 5.0], abs=0.05)


def test_the_floor_comes_from_a_setting_taken_twice_and_says_so_when_it_cannot(
    swept, tmp_path
) -> None:
    """A run that repeated nothing measured no floor, and must not invent one.

    The quefrency step is the grid the answers land on and is available whether or
    not anything was repeated, so reporting it as the floor would hand every later
    comparison a sensitivity nobody measured.
    """
    found = read(swept)
    assert found["floor_ms"] is not None
    assert found["settings_taken_twice"] == [64]

    store = takes.Store.open(tmp_path / "once")
    for index in range(2):
        store.keep(
            FakeRecording(stereo(carrier(200 + index))),
            stimulus="held",
            setting=f"out-{index:02d}",
            take=0,
        )
    store.keep(
        FakeRecording(stereo(with_a_copy(carrier(1), 12.0))),
        stimulus="held",
        setting="delay-064",
        take=0,
    )
    store.close(question="a run that repeated nothing")
    once = read(tmp_path / "once")
    assert once["floor_ms"] is None
    assert once["settings_taken_twice"] == []
    assert once["quefrency_step_ms"] > 0
    assert once["why_floor"]


def test_the_chain_with_the_effect_out_is_published_and_not_only_subtracted(
    swept,
) -> None:
    """It answers two questions and the record has to carry both.

    Subtracted, it takes the stimulus out of every reading. Published, it says
    whether the chain puts a peak of its own anywhere -- because one that does could
    hand a setting a delay the effect never made, and nothing in that row would say
    so.

    Whether it would be read as a delay is decided on this curve after the same
    subtraction every setting gets, not before it. Before it, the source's own shape
    sits at the bottom of the search whatever is in the path, so a verdict taken
    there is true of every run and says nothing about any of them.
    """
    out = read(swept)["with_the_effect_out"]
    assert len(out["takes"]) == 2
    assert out["would_be_read_as_a_delay"] == out["with_nothing_in_its_path"]["admitted"]
    assert out["why"]


def test_a_delay_past_the_search_is_absent_rather_than_pinned_to_the_edge(
    swept,
) -> None:
    """Searching only as far as the printed end would report a longer byte as it.

    A byte that runs further than the page says is a finding about the unit. Read
    against a search that stops at the printed end it comes back sitting exactly on
    it, which is the answer the page already gave and looks like agreement.
    """
    found = read(swept, searched_ms=(0.4, 20.0))
    assert found["searched_ms"] == [0.4, 20.0]
    assert at(found, 127)["ms"] != pytest.approx(40.0, abs=0.05)
    assert at(found, 127)["ms"] <= 20.0


def test_the_record_says_what_was_held_and_where_the_takes_came_from(swept) -> None:
    """A time read from a cepstrum is a time through the whole chain."""
    held = [{"address": "40 03 05", "bytes": "40"}]
    found = read(swept, held=held, stimulus="noise, 5 s")
    assert found["held"] == held
    assert found["why_held"]
    assert found["stimulus"] == "noise, 5 s"
    assert found["takes_from"].endswith("delays")
    assert found["type"] == "01 50"
    assert found["address"] == "40 03 03"


def test_a_run_with_no_take_of_the_effect_out_is_refused(swept, tmp_path) -> None:
    """Not optional here. Without it every reading carries the stimulus's own shape."""
    with pytest.raises(ValueError, match="control"):
        read(swept, control="nothing-matches-this")


def test_a_setting_pattern_without_a_value_group_is_refused(swept) -> None:
    """Which byte a take was made at is the one thing the pattern has to yield."""
    with pytest.raises(ValueError, match="value"):
        read(swept, setting=r"delay-\d+")


def test_takes_matching_nothing_are_counted_rather_than_dropped(swept) -> None:
    """A pattern that matches nothing and a directory that holds nothing differ."""
    found = read(swept, setting=r"delay-(?P<value>0+)$")
    assert found["takes_not_matching"]["count"] > 0


# ---- what the reading does at its own bottom


def test_a_delay_under_the_bottom_of_the_search_shows_as_one_in_its_neighbours(
    swept,
) -> None:
    """It comes back at three times itself, and the only thing that says so is the gap.

    The fundamental is outside the search while its odd rahmonics are inside, so the
    strongest peak sits at `3D` and stands where a delay stands. A true delay has
    nothing between itself and three times itself; this row has `5D/3` and `7D/3`
    sitting there, which is one and two thirds and two and a third of where it was
    read. Both are nearer the peak than a delay's first rahmonic, so a window set in
    milliseconds rather than in quefrencies closes over them and the row becomes
    indistinguishable from a real one.
    """
    under = {"searched_ms": (5.0, 80.0)}   # above the delay, below three times it
    row = at(read(swept, **under), 16)
    assert row["ms"] == pytest.approx(3 * DELAYS[16], abs=0.05)
    ratios = sorted(q / row["ms"] for q in row["also_ms"])
    assert ratios == pytest.approx([5 / 3, 7 / 3], abs=0.05)

    # The same row with the window set as a time instead. It reaches past `5D` and
    # `7D` without reaching `9D`, so the nearest thing left beside the peak is the
    # ninth rahmonic at three times it -- which is where a real delay's first
    # neighbour sits. Nothing in the row is then wrong on its face.
    hidden = at(read(swept, **under, apart_ms=5 * DELAYS[16]), 16)
    assert hidden["ms"] == row["ms"]
    assert min(q / hidden["ms"] for q in hidden["also_ms"]) == pytest.approx(3.0, abs=0.05)


def test_the_search_and_the_peak_window_default_to_what_the_transform_resolves(
    swept,
) -> None:
    """Neither is a time chosen for the reading, and a record says which was used.

    A bottom set above what the run could resolve is the whole of the failure above,
    so the default cannot be a round number of milliseconds: on a byte covering three
    orders of magnitude it would be under the first setting and over the last.
    """
    found = efxtime.read_directory(
        swept,
        type_id="01 50",
        address="40 03 03",
        setting=r"(?:delay|again)-(?P<value>\d+)",
        control="out-",
        frame=16384,
        hop=4096,
        channel=0,
        lead_s=LEAD,
        hold_s=HOLD,
    )
    resolved = efxtime.APART_STEPS * 1000.0 / SR
    assert found["quefrency_step_ms"] == pytest.approx(1000.0 / SR, rel=1e-4)
    assert found["searched_ms"][0] == pytest.approx(resolved, rel=1e-4)
    assert found["peaks_apart_ms"] == pytest.approx(resolved, rel=1e-4)


def test_a_take_with_nothing_in_its_path_is_read_and_not_only_subtracted(swept) -> None:
    """The sensitivity a refused setting rests on, measured on the run's own stimulus.

    A bottom setting returning nothing says nothing unless a take that certainly
    holds no copy also returns nothing under the same reading. Read against the
    average it would be read against itself, so it is read against the other take.
    """
    null = read(swept)["with_the_effect_out"]["with_nothing_in_its_path"]
    assert null["take"] != null["against"]
    assert not null["admitted"]
    assert null["stands"] < efxtime.STANDS_OUT


def test_a_run_with_one_take_of_the_effect_out_says_it_measured_no_null(
    swept, tmp_path
) -> None:
    """One take has nothing to be read against, and an invented null is worse than none."""
    store = takes.Store.open(tmp_path / "lonely")
    store.keep(
        FakeRecording(stereo(carrier(300))), stimulus="held", setting="out-00", take=0
    )
    store.keep(
        FakeRecording(stereo(with_a_copy(carrier(2), 12.0))),
        stimulus="held",
        setting="delay-064",
        take=0,
    )
    store.close(question="a run with one control take")
    out = read(tmp_path / "lonely")["with_the_effect_out"]
    assert out["with_nothing_in_its_path"] is None
    assert out["would_be_read_as_a_delay"] is None
    assert out["why_nothing_in_its_path"]
