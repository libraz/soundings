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


def take(path, *, left_db: float, right_db: float, seed: int, idle_db: float = -100.0) -> None:
    """A take with a note at a given level in each of two channels of four.

    Four channels because the interface has more inputs than the unit uses, and
    which two carry it is something the reading has to work out rather than be
    told. `idle_db` is what the two the unit is *not* on hold, through the lead as
    well as the body: an input nothing is plugged into is not silent.
    """
    rng = np.random.default_rng(seed)
    n = int(RATE * 2.0)
    lead = int(RATE * LEAD)
    note = np.zeros(n)
    note[lead:] = np.sin(2 * np.pi * 220 * np.arange(n - lead) / RATE)
    frames = rng.normal(0, 1e-5, (n, 4))
    frames[:, 0] += rng.normal(0, 1.0, n) * 10 ** (idle_db / 20)
    frames[:, 1] += rng.normal(0, 1.0, n) * 10 ** ((idle_db - 5.0) / 20)
    frames[:, 2] += note * 10 ** (left_db / 20)
    frames[:, 3] += note * 10 ** (right_db / 20)
    write(path, frames, RATE)


def saved(
    tmp_path,
    settings: dict[str, list[tuple[float, float]]],
    stimulus: str = "struck",
    idle_db: float = -100.0,
):
    """A directory shaped like one a --save run leaves."""
    takes = []
    seed = 0
    for setting, levels in settings.items():
        for index, (left, right) in enumerate(levels):
            name = f"{stimulus}-{setting}-{index:02d}.wav"
            seed += 1
            take(tmp_path / name, left_db=left, right_db=right, seed=seed, idle_db=idle_db)
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


def test_a_mono_source_is_given_no_number_and_says_why(tmp_path) -> None:
    """What a parameter does between two channels cannot be asked of one, and a
    balance computed against a channel holding only noise would report the
    noise's own wander as a pan. It carries that reason rather than being dropped:
    a stimulus missing from the verdicts reads as one that was never asked, and a
    count of what moved is then taken over a denominator that quietly shrank."""
    root = saved(tmp_path, {"0": [(-20.0, -300.0)] * 4, "127": [(-20.0, -300.0)] * 4})

    found = balance.measure(root)

    assert len(found) == 1
    assert found[0].not_measured == balance.MONO_SOURCE
    assert found[0].settings == []
    assert found[0].channels is None
    assert found[0].to_json()["measured"] is False


def test_a_parameter_that_pans_hard_is_not_mistaken_for_a_mono_source(tmp_path) -> None:
    """Taken to its two ends a panpot leaves every take with one live channel and
    one empty one, so a pair of channels chosen from any single take answers mono
    -- and the one parameter this measurement exists for would be refused.

    Measured on the unit at note 36 of the first drum map, asked at 1 against 127:
    the balance sits at +58.3 dB and -58.5 dB, and the run was dropped before the
    channels were asked of every take rather than of one."""
    root = saved(tmp_path, {"1": [(-20.0, -300.0)] * 4, "127": [(-300.0, -20.0)] * 4})

    found = balance.measure(root)

    assert len(found) == 1
    assert found[0].not_measured is None
    assert set(found[0].channels) == {2, 3}
    assert found[0].moved_between_settings


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


def test_an_input_the_unit_is_not_on_does_not_refuse_the_run(tmp_path) -> None:
    """An input nothing is plugged into is not silent, and reading the floor over
    every input it has puts the bar thirty decibels above where the reading is.

    Measured: two unused inputs sat at -70 and -75 dBFS through every take of a
    control run while the two carrying the unit sat at -110 in their leads. Two of
    the run's three voices came back as mono sources -- the quieter channel
    reaching -59.7 against a floor of -79.4 -- which is not a bound being reported,
    it is a statement that the unit arrived on one channel, and it was false."""
    root = saved(
        tmp_path,
        {"0": [(-59.5, -300.0)] * 3, "127": [(-300.0, -59.7)] * 3},
        idle_db=-70.0,
    )

    found = balance.measure(root)

    assert len(found) == 1
    assert found[0].not_measured is None
    assert found[0].channels == (2, 3)


def test_the_sign_of_a_balance_does_not_depend_on_which_channel_was_louder(tmp_path) -> None:
    """A pan is a direction before it is a size, so the direction has to survive
    being read twice. Ordering the pair by level put the sign in the hands of
    whichever channel was louder by a hair: two readings of one saved run of the
    part panpot returned +10.18 dB and -10.18 dB, each self-consistent with the
    pair it also reported, and nothing in the figure said which it was under."""
    quieter_first = saved(tmp_path / "a", {"0": [(-20.0, -10.0)] * 3, "127": [(-20.0, -10.0)] * 3})
    louder_first = saved(tmp_path / "b", {"0": [(-10.0, -20.0)] * 3, "127": [(-10.0, -20.0)] * 3})

    (low,) = balance.measure(quieter_first)
    (high,) = balance.measure(louder_first)

    assert low.channels == (2, 3) and high.channels == (2, 3)
    assert low.settings[0].typical_db < 0 < high.settings[0].typical_db


def test_a_sweep_is_read_and_not_silently_given_a_pairs_verdict(tmp_path) -> None:
    """A byte printed as a pan is read at a dozen settings, and the screening
    verdicts were arithmetic defined on exactly two. Read that way a sweep
    returned false for every one of them -- not because nothing moved, but
    because the count was wrong, which is the shape of a negative that cannot be
    contradicted by the record carrying it."""
    root = saved(
        tmp_path,
        {
            str(value): [(-10.0 - value / 8.0, -30.0 + value / 8.0)] * 3
            for value in (0, 16, 32, 48, 64, 80, 96, 112, 127)
        },
    )

    (found,) = balance.measure(root)

    assert len(found.settings) == 9
    assert found.moved_between_settings
    assert not found.did_not_repeat


def test_a_sweep_that_turns_back_on_itself_is_not_read_off_its_ends(tmp_path) -> None:
    """The widest gap in the run rather than the gap between the ends. A table
    that returns to where it started would put its two ends in the same place,
    and a reading taken there says the byte did nothing at all."""
    root = saved(
        tmp_path,
        {
            "0": [(-10.0, -30.0)] * 3,
            "64": [(-30.0, -10.0)] * 3,
            "127": [(-10.0, -30.0)] * 3,
        },
    )

    (found,) = balance.measure(root)

    assert found.moved_between_settings


def test_a_sweep_of_a_byte_that_does_nothing_still_shows_nothing(tmp_path) -> None:
    """The guard has to have an outside at a dozen settings as much as at two,
    and the yardstick is now the steadiest of many rather than of a pair, so it
    runs lower and the verdict is easier to trip."""
    root = saved(
        tmp_path,
        {str(value): [(-20.0, -10.0)] * 3 for value in (0, 16, 32, 48, 64, 80, 96, 112, 127)},
    )

    (found,) = balance.measure(root)

    assert not found.moved_between_settings
    assert not found.did_not_repeat


def test_every_number_survives_the_json_round_trip(tmp_path) -> None:
    """The record is the archive, so a figure the JSON drops did not happen."""
    root = saved(tmp_path, {"0": [(-10.0, -20.0)] * 4, "127": [(-20.0, -10.0)] * 4})

    written = balance.measure(root)[0].to_json()

    assert written["moved_between_settings"]
    assert [len(s["balance_db"]) for s in written["by_setting"]] == [4, 4]
    assert written["channels"] == [3, 2] or written["channels"] == [2, 3]
