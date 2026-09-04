"""What a parameter did between the channels, which a comparison in one cannot see.

Two failures are guarded. A parameter that moves signal between the channels is
invisible to every other measurement here, which works in one channel by design;
and it does not read as nothing there, it reads as the unit failing to repeat
itself, which is a verdict about the machine wearing the parameter's name.

Takes are synthesised rather than recorded: what is under test is the reading.
"""

from __future__ import annotations

import json

import numpy as np

from soundings import balance
from soundings.takes import write

RATE = 8000
LEAD = 0.6


def take(path, *, left_db: float, right_db: float, seed: int) -> None:
    """A take with a note at a given level in each of two channels of four.

    Four channels because the interface has more inputs than the unit uses, and
    which two carry it is something the reading has to work out rather than be
    told.
    """
    rng = np.random.default_rng(seed)
    n = int(RATE * 2.0)
    lead = int(RATE * LEAD)
    note = np.zeros(n)
    note[lead:] = np.sin(2 * np.pi * 220 * np.arange(n - lead) / RATE)
    frames = rng.normal(0, 1e-5, (n, 4))
    frames[:, 2] += note * 10 ** (left_db / 20)
    frames[:, 3] += note * 10 ** (right_db / 20)
    write(path, frames, RATE)


def saved(tmp_path, settings: dict[str, list[tuple[float, float]]], stimulus: str = "struck"):
    """A directory shaped like one a --save run leaves."""
    takes = []
    seed = 0
    for setting, levels in settings.items():
        for index, (left, right) in enumerate(levels):
            name = f"{stimulus}-{setting}-{index:02d}.wav"
            seed += 1
            take(tmp_path / name, left_db=left, right_db=right, seed=seed)
            takes.append({"stimulus": stimulus, "setting": setting, "take": index, "file": name})
    (tmp_path / "takes-manifest.json").write_text(
        json.dumps({"stimuli": [{"name": stimulus, "lead_s": LEAD}], "takes": takes})
    )
    return tmp_path


def test_a_balance_that_lands_somewhere_new_each_take_is_reported(tmp_path) -> None:
    """Measured on this unit's part panpot at 0: the two channels land at
    -54.5/-49.3, -47.3/-62.9, -52.2/-50.6 and -63.7/-47.4 over four takes, while
    at 127 they sit at -58.6 and -48.4 on every one of them."""
    root = saved(
        tmp_path,
        {
            "127": [(-20.0, -10.0)] * 4,
            "0": [(-16.0, -11.0), (-8.0, -24.0), (-14.0, -12.0), (-25.0, -9.0)],
        },
    )

    (found,) = balance.measure(root)

    assert found.did_not_repeat
    assert found.while_the_total_stayed


def test_a_fixed_pan_at_both_settings_shows_nothing(tmp_path) -> None:
    """The guard has to have an outside. A chain with an imbalance of its own
    would otherwise read as a parameter that pans."""
    root = saved(tmp_path, {"0": [(-20.0, -10.0)] * 4, "127": [(-20.0, -10.0)] * 4})

    (found,) = balance.measure(root)

    assert not found.did_not_repeat and not found.moved_between_settings


def test_a_pan_that_moved_between_the_settings_is_reported(tmp_path) -> None:
    """The other of the two facts, and the one a level read in either channel
    alone would report as a level change."""
    root = saved(tmp_path, {"0": [(-10.0, -20.0)] * 4, "127": [(-20.0, -10.0)] * 4})

    (found,) = balance.measure(root)

    assert found.moved_between_settings and not found.did_not_repeat


def test_a_level_change_reaching_both_channels_is_not_called_a_pan(tmp_path) -> None:
    """A balance that holds while the pair's total moves is a level change, and
    neither channel on its own can tell the two apart."""
    root = saved(tmp_path, {"0": [(-30.0, -20.0)] * 4, "127": [(-20.0, -10.0)] * 4})

    (found,) = balance.measure(root)

    assert not found.moved_between_settings
    assert not found.while_the_total_stayed


def test_the_channels_are_chosen_from_the_setting_that_sounded(tmp_path) -> None:
    """A parameter that silences its part at one of its two values leaves the
    other setting's takes holding nothing but noise. Choosing the pair from those
    picks whichever input carried the most of it and then refuses the run for
    being mono -- measured, that was one of forty-six addresses in a part block,
    silenced at the value the plan asks first."""
    root = saved(
        tmp_path,
        {"127": [(-300.0, -300.0)] * 4, "0": [(-20.0, -10.0)] * 4},
    )

    found = balance.measure(root)

    assert len(found) == 1
    assert set(found[0].channels) == {2, 3}


def test_a_mono_source_is_passed_over_rather_than_given_a_number(tmp_path) -> None:
    """What a parameter does between two channels cannot be asked of one, and a
    balance computed against a channel holding only noise would report the
    noise's own wander as a pan."""
    root = saved(tmp_path, {"0": [(-20.0, -300.0)] * 4, "127": [(-20.0, -300.0)] * 4})

    assert balance.measure(root) == []


def test_the_yardstick_is_the_steadier_setting_not_the_wider(tmp_path) -> None:
    """The wider one is what is being asked about, so taking it into the
    yardstick would be measuring it against itself."""
    root = saved(
        tmp_path,
        {
            "0": [(-20.0, -10.0)] * 4,
            "127": [(-16.0, -11.0), (-8.0, -24.0), (-14.0, -12.0), (-25.0, -9.0)],
        },
    )

    (found,) = balance.measure(root)

    assert found.yardstick_db < 1.0


def test_every_number_survives_the_json_round_trip(tmp_path) -> None:
    """The record is the archive, so a figure the JSON drops did not happen."""
    root = saved(tmp_path, {"0": [(-10.0, -20.0)] * 4, "127": [(-20.0, -10.0)] * 4})

    written = balance.measure(root)[0].to_json()

    assert written["moved_between_settings"]
    assert [len(s["balance_db"]) for s in written["by_setting"]] == [4, 4]
    assert written["channels"] == [3, 2] or written["channels"] == [2, 3]
