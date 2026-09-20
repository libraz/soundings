"""Controls for reading how far a delay swings rather than whatever the level did.

Every failure this guards against reads as a measurement. A held note is never
perfectly steady, so a take with nothing moving in it still hands a projection a
peak and a fit a number. A level modulation with no delay anywhere in it is fitted
as a comb explaining more of its own series than a real comb does. A floor drawn
from settings that named no excursion says the reading is exact. A refused setting
dropped instead of published turns the bottom of a depth byte into a shorter
sweep. None of these raise; they publish.

The takes here are synthetic, so the answer is known: a sweep put in at a stated
size has to come back at that size, and a take with none has to be refused.

The forward comb fit is the expensive part of the stage by three orders -- the
demodulation is under a tenth of a second and the fit is tens -- so the run every
assertion here is made against is built once for the module. Sizes are set by what
the fit costs: it scales with the length of the level series and with the number of
rates the projections seed it at, and neither has anything to do with what is being
tested.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from soundings import efxexcursion, partials, takes

SR = 16000
CARRIER_HZ = 200.0
SECONDS = 3.5
LEAD = 0.6
HOLD = 2.5
AT_HZ = 2.0

MIX = 0.25
"""How much of the output is the delayed path in the takes made here.

Not an even mix, and that is the point rather than a convenience. The forward fit
was calibrated on injected combs and its one measured failure corner is a shallow
sweep mixed evenly -- there a millisecond came back half again too large and one
span closed around the wrong answer. A control placed in that corner would be
testing the corner, and it would be failed by a reading that is working exactly as
the stage publishes it to work.
"""


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
    """A held note, which is the stimulus this reading is taken on.

    Harmonic rather than broadband, because the whole reading is a demodulation of
    the carrier's own orders: noise has none and would test nothing. A little drift
    is put on it so the take is not steadier than a real one -- a perfectly steady
    tone makes every gate here trivially passable.
    """
    rng = np.random.default_rng(seed)
    n = np.arange(int(SECONDS * SR), dtype=np.float64)
    drift = 1.0 + 0.01 * np.sin(2.0 * np.pi * 0.11 * n / SR + rng.random())
    out = np.zeros_like(n)
    for order in range(1, 9):
        out += (1.0 / order) * np.sin(2.0 * np.pi * CARRIER_HZ * order * n / SR)
    return 0.1 * out * drift


def with_a_sweep(body: np.ndarray, excursion_ms: float, mix: float = MIX) -> np.ndarray:
    return partials.swept(body, SR, 3.0 + excursion_ms / 2.0, excursion_ms, AT_HZ, mix)


def stereo(body: np.ndarray) -> np.ndarray:
    return np.stack([body, body * 0.5], axis=1)


SWEEPS = {0: None, 64: 5.0}
"""What each setting had put into it, peak to peak in milliseconds. `None` is a
setting with no sweep at all, which is what the bottom of a depth byte returns."""


def read(where, **extra) -> dict:
    asked = {
        "type_id": "01 23",
        "address": "40 03 07",
        "setting": r"depth-(?P<value>\d+)",
        "carrier_hz": CARRIER_HZ,
        "channel": 0,
        "lead_s": LEAD,
        "hold_s": HOLD,
        "grid_hz": (1.00, 6.00),
        "step_hz": 0.05,
    }
    asked.update(extra)
    return efxexcursion.read_directory(where, **asked)


@pytest.fixture(scope="module")
def swept(tmp_path_factory):
    """A run whose settings hold known excursions, and one that holds none."""
    where = tmp_path_factory.mktemp("swept") / "depth"
    store = takes.Store.open(where)
    # Two of them. One control gives the run a floor to read against and no way to
    # say what that floor returns on material it cannot find anything in.
    for take in range(2):
        store.keep(
            FakeRecording(stereo(carrier(1 + take))),
            stimulus="held",
            setting="bypassed",
            take=take,
        )
    for value, size in SWEEPS.items():
        body = carrier(value)
        store.keep(
            FakeRecording(stereo(body if size is None else with_a_sweep(body, size))),
            stimulus="held",
            setting=f"depth-{value:03d}",
            take=0,
        )
    # One setting taken again, which is the only thing that gives the run a floor.
    store.keep(
        FakeRecording(stereo(with_a_sweep(carrier(640), SWEEPS[64]))),
        stimulus="held",
        setting="depth-064",
        take=1,
    )
    store.close(question="a run of known excursions")
    return where


@pytest.fixture(scope="module")
def found(swept) -> dict:
    return read(swept)


def rows(found: dict, value: int) -> list[dict]:
    return [r for r in found["readings"] if r[efxexcursion.VALUE] == value]


def test_a_sweep_comes_back_at_the_size_it_was_put_in_at(found) -> None:
    """The whole stage in one assertion: a stated excursion has to be returned."""
    for value, size in SWEEPS.items():
        if size is None:
            continue
        for got in rows(found, value):
            assert got["admitted"], f"setting {value} named no excursion"
            assert got["excursion_ms"] == pytest.approx(size, rel=0.2)


def test_a_setting_with_no_sweep_in_it_names_no_excursion(found) -> None:
    """A take with nothing moving must be refused rather than given a small number.

    The failure this catches is the one that looks like a result: a held note
    drifts, a projection finds a peak in the drift, and a fit hands that peak a
    number in milliseconds. Published, it reads as a depth byte that does
    something at its bottom setting.
    """
    assert not rows(found, 0)[0]["admitted"]


def test_the_null_is_one_control_against_another(found) -> None:
    """The sensitivity every refusal rests on, measured rather than assumed.

    A bottom setting returning nothing means nothing unless a take that certainly
    holds no sweep also returns nothing here -- and that take has to be read
    against a *different* control. Read against its own floor the comparison is
    one by construction and refuses by construction, which publishes a guaranteed
    refusal as a measured one.
    """
    null = found["routed_past_the_effect"]["with_nothing_in_its_path"]
    assert null is not None
    assert null["take"] != null["against"]
    assert not null["admitted"]


def test_a_run_with_one_control_measures_no_null(tmp_path) -> None:
    """Null rather than the control compared with itself.

    This is the failure the field exists to prevent: a record that filled it from
    the only control it had would report a refusal that could not have come out
    any other way, and nothing in the number would say so.
    """
    store = takes.Store.open(tmp_path / "lonely")
    store.keep(
        FakeRecording(stereo(carrier(9))), stimulus="held", setting="bypassed", take=0
    )
    store.keep(
        FakeRecording(stereo(with_a_sweep(carrier(10), 5.0))),
        stimulus="held",
        setting="depth-064",
        take=0,
    )
    store.close(question="a run with one control")
    aside = read(tmp_path / "lonely")["routed_past_the_effect"]
    assert aside["with_nothing_in_its_path"] is None


def test_a_refused_setting_is_published_with_what_it_returned(found) -> None:
    """A refusal is a reading about the bottom of the range, not an absence."""
    assert 0 in found["settings_asked"]
    assert 0 not in found["settings_admitted"]
    refused = rows(found, 0)[0]
    assert refused["read_at_hz"] is not None
    assert "the_level_swing_is" in refused


def test_the_floor_comes_from_the_settings_taken_twice(found) -> None:
    """Two takes of one state are what a difference between two settings clears."""
    floor = found["floor"]
    assert floor["settings_taken_twice"] == [64]
    assert floor["ms"] is not None
    assert floor["as_a_fraction"] is not None


def test_a_run_that_repeated_nothing_measures_no_floor() -> None:
    """Null rather than a substitute: a grid step is not a measurement of anything.

    Asked of the rule rather than of another eight seconds of fitting. What decides
    a floor is which rows a gate admitted and how many takes of one setting are
    among them, and both are in the rows by the time the rule sees them.
    """
    alone = [{"value": 16, "excursion_ms": 1.0}, {"value": 64, "excursion_ms": 3.0}]
    assert efxexcursion._floor_of(alone, "excursion_ms")["ms"] is None
    assert efxexcursion._floor_of(alone, "excursion_ms")["settings_taken_twice"] == []


def test_a_floor_is_not_taken_over_rows_the_gate_refused() -> None:
    """Two refusals are not two readings of a quantity.

    The failure is silent and flattering: a byte whose bottom settings return
    nothing gets those nothings read as repeats, and the run publishes a floor of
    zero beside readings that never repeated at all.
    """
    refused = [{"value": 0, "excursion_ms": None}, {"value": 0, "excursion_ms": None}]
    assert efxexcursion._floor_of(refused, "excursion_ms")["ms"] is None


def test_a_level_modulation_with_no_delay_in_it_is_not_a_swing(tmp_path) -> None:
    """The mechanism a forward comb fit cannot refuse on its own.

    A plain envelope over the whole voice is fitted as a comb explaining more of
    its series than any real comb here does. What separates them is not the fit:
    a comb sweeps a notch, so partials either side of it move against each other,
    while one envelope moves every partial together.
    """
    store = takes.Store.open(tmp_path / "shaken")
    store.keep(
        FakeRecording(stereo(carrier(2))), stimulus="held", setting="bypassed", take=0
    )
    body = carrier(3)
    n = np.arange(body.size, dtype=np.float64)
    shaken = body * (1.0 + 0.55 * np.sin(2.0 * np.pi * AT_HZ * n / SR))
    store.keep(FakeRecording(stereo(shaken)), stimulus="held", setting="depth-064", take=0)
    store.close(question="a level swing with no delay in it")

    got = rows(read(tmp_path / "shaken"), 64)[0]
    assert got["the_level_swing_is"] == "the whole voice"
    assert not got["admitted"]


def test_takes_the_pattern_does_not_name_are_counted(swept) -> None:
    """A pattern matching nothing and a directory holding nothing are different.

    Both leave a record with no readings in it, so the one that had takes to read
    has to say how many it looked at.
    """
    found = read(swept, setting=r"nothing-(?P<value>\d+)-00")
    assert found["readings"] == []
    # Every take but the two controls, which are not settings the pattern missed.
    assert found["takes_not_matching"]["count"] == len(SWEEPS) + 1
    assert len(found["routed_past_the_effect"]["takes"]) == 2


def test_a_directory_with_no_control_is_refused(tmp_path) -> None:
    """Which orders are read is decided by the bypassed take and by nothing else."""
    store = takes.Store.open(tmp_path / "lone")
    store.keep(FakeRecording(stereo(carrier(4))), stimulus="held", setting="depth-064", take=0)
    store.close(question="a run with no control")
    with pytest.raises(ValueError, match="control"):
        read(tmp_path / "lone")


def test_the_record_states_what_it_does_not_hold(found) -> None:
    """No curve, no table, no verdict -- and the record says so in its own words."""
    assert found["question"] == efxexcursion.QUESTION
    assert found["not_in_this_record"] == efxexcursion.NOT_HERE
    assert found["method"] == efxexcursion.METHOD
    assert found["held"] == []
