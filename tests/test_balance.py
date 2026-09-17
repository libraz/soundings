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
    count of what moved is then taken over a denominator that quietly shrank.

    The settings are still listed, and each take still says how far it stood over
    its own lead. The run measured levels and no separation, and publishing the
    refusal without them would turn a stated bound back into a silence -- which is
    the whole reason a null is worth reading here."""
    root = saved(tmp_path, {"0": [(-20.0, -300.0)] * 4, "127": [(-20.0, -300.0)] * 4})

    found = balance.measure(root)

    assert len(found) == 1
    assert found[0].not_measured == balance.MONO_SOURCE
    assert found[0].channels is None
    assert found[0].to_json()["measured"] is False

    assert [s.setting for s in found[0].settings] == ["0", "127"]
    assert all(s.balance_db == [] for s in found[0].settings)
    assert all(len(s.over_the_lead_db) == 4 for s in found[0].settings)
    # Every channel the take has, since the pair this could not find is the thing
    # it has no answer about.
    assert found[0].over_the_lead_on == tuple(range(4))


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


BAND_TONES = (125.0, 250.0, 500.0, 1000.0, 2000.0)
"""Third-octave centres a tone lands squarely inside, two octaves apart.

Two octaves rather than one third, so that a band carrying a tone and a band
carrying nothing but what the rest of the signal leaks into it are both in the
set. A separation that holds in the first and not in the second is the whole
question a band reading of one is asked.
"""


def _shaped(n: int, rate: int, levels: dict[float, float], seed: int) -> np.ndarray:
    """One signal whose bands stand at given levels, and valleys between them."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / rate
    out = rng.normal(0, 1.0, n) * 1e-6
    for frequency, level in levels.items():
        out += np.sin(2 * np.pi * frequency * t + rng.uniform(0, 6.28)) * 10 ** (level / 20)
    return out


def banded(
    path,
    *,
    levels: dict[float, float],
    attenuated_by_db: float,
    floor_db: float,
    seed: int,
    only_in_the_quieter: dict[float, float] | None = None,
) -> None:
    """A take whose second channel is the first scaled, with an optional floor under it.

    The floor is generated after the attenuation and does not go through it, which
    is what a term added to a channel downstream of a multiplier looks like. With
    the floor far below, the two channels are one signal and one number apart.

    `only_in_the_quieter` puts content in the second channel and nowhere else, so
    that one band of one take can be made to say something the other take does not.
    """
    rng = np.random.default_rng(seed)
    rate, seconds = 8000, 4.0
    n = int(rate * seconds)
    lead = int(rate * LEAD)
    near = np.zeros(n)
    near[lead:] = _shaped(n - lead, rate, levels, seed)
    far = near * 10 ** (-attenuated_by_db / 20)
    far[lead:] += rng.normal(0, 1.0, n - lead) * 10 ** (floor_db / 20)
    if only_in_the_quieter:
        far[lead:] += _shaped(n - lead, rate, only_in_the_quieter, seed + 1)
    frames = rng.normal(0, 1e-6, (n, 4))
    frames[:, 2] += near
    frames[:, 3] += far
    write(path, frames, rate)


def saved_bands(tmp_path, settings: dict[str, list[dict]], stimulus: str = "held") -> object:
    takes = []
    seed = 100
    for setting, per_take in settings.items():
        for index, how in enumerate(per_take):
            name = f"{stimulus}-{setting}-{index:02d}.wav"
            seed += 1
            banded(tmp_path / name, seed=seed, **how)
            takes.append({"stimulus": stimulus, "setting": setting, "take": index, "file": name})
    (tmp_path / "takes-manifest.json").write_text(
        json.dumps({"stimuli": [{"name": stimulus, "lead_s": LEAD}], "takes": takes})
    )
    return tmp_path


def test_a_channel_that_is_the_other_one_scaled_separates_alike_in_every_band(tmp_path) -> None:
    """What a pair of multipliers does, which is the reading's null.

    A gain has no frequency in it, so however the source is shaped the two
    channels stand the same distance apart in every band of it.
    """
    levels = {f: -20.0 for f in BAND_TONES}
    root = saved_bands(
        tmp_path,
        {
            "064": [{"levels": levels, "attenuated_by_db": 0.0, "floor_db": -120.0}] * 2,
            "112": [{"levels": levels, "attenuated_by_db": 12.0, "floor_db": -120.0}] * 2,
        },
    )

    (found,) = balance.by_band(root)

    assert not found.depends_on_frequency
    at = {s.setting: s for s in found.settings}
    held = [v for v in at["112"].separation_db if v is not None]
    assert held
    assert all(abs(v - 12.0) < 1.0 for v in held)


def test_a_floor_under_the_quieter_channel_shows_as_a_separation_that_has_a_shape(
    tmp_path,
) -> None:
    """The one thing a broadband balance cannot tell from a pair of multipliers.

    A term added after the multiplier is swamped where the signal is loud and
    takes over where it is not, so the separation stops being one number -- and
    broadband it still reads as a pan that stopped short of silence.
    """
    levels = {125.0: 0.0, 250.0: -40.0, 500.0: 0.0, 1000.0: -40.0, 2000.0: 0.0}
    root = saved_bands(
        tmp_path,
        {
            "064": [{"levels": levels, "attenuated_by_db": 0.0, "floor_db": -120.0}] * 2,
            "127": [{"levels": levels, "attenuated_by_db": 40.0, "floor_db": -30.0}] * 2,
        },
    )

    (found,) = balance.by_band(root)

    assert found.depends_on_frequency
    at = {s.setting: s for s in found.settings}
    by_centre = dict(zip(found.centres, at["127"].separation_db, strict=True))
    assert by_centre[125.0] is not None
    assert by_centre[250.0] is not None
    # The loud band still reports the multiplier; the quiet one reports the floor.
    assert by_centre[125.0] > 30.0
    assert by_centre[250.0] < 15.0


def test_a_setting_the_run_holds_one_take_of_is_left_out_and_named(tmp_path) -> None:
    """One take cannot say whether a band repeats, and zero scatter is not the answer.

    Left out rather than read, because a band whose scatter is unknown would pass
    the repeatability test it was never given.
    """
    levels = {f: -20.0 for f in BAND_TONES}
    root = saved_bands(
        tmp_path,
        {
            "000": [{"levels": levels, "attenuated_by_db": 0.0, "floor_db": -120.0}] * 2,
            "064": [{"levels": levels, "attenuated_by_db": 6.0, "floor_db": -120.0}],
            "127": [{"levels": levels, "attenuated_by_db": 12.0, "floor_db": -120.0}] * 2,
        },
    )

    (found,) = balance.by_band(root)

    assert found.settings_left_out == ["064"]
    assert [s.setting for s in found.settings] == ["000", "127"]


def test_a_band_the_two_takes_disagree_about_is_left_out_of_the_profile(tmp_path) -> None:
    """A band can stand well over the interface's floor and still not repeat."""
    steady = {f: -20.0 for f in BAND_TONES}
    root = saved_bands(
        tmp_path,
        {
            "064": [{"levels": steady, "attenuated_by_db": 0.0, "floor_db": -120.0}] * 2,
            "127": [
                {"levels": steady, "attenuated_by_db": 12.0, "floor_db": -120.0},
                {
                    "levels": steady,
                    "attenuated_by_db": 12.0,
                    "floor_db": -120.0,
                    "only_in_the_quieter": {500.0: -28.0},
                },
            ],
        },
    )

    (found,) = balance.by_band(root)

    at = {s.setting: s for s in found.settings}
    by_centre = dict(zip(found.centres, at["127"].separation_db, strict=True))
    assert by_centre[500.0] is None
    assert by_centre[250.0] is not None
    assert max(at["127"].scatter_db) > balance.BAND_REPEATS_WITHIN_DB


def test_the_rows_of_one_setting_are_on_one_scale_and_say_which(tmp_path) -> None:
    """A record whose normaliser is not in it cannot lay two settings side by side.

    The two channels' rows are against the two of them together, so their
    difference is the separation exactly and the reference puts either back where
    the take had it.
    """
    levels = {f: -20.0 for f in BAND_TONES}
    root = saved_bands(
        tmp_path,
        {
            "064": [{"levels": levels, "attenuated_by_db": 0.0, "floor_db": -120.0}] * 2,
            "127": [{"levels": levels, "attenuated_by_db": 12.0, "floor_db": -120.0}] * 2,
        },
    )

    written = balance.by_band(root)[0].to_json()

    for block in written["by_setting"]:
        assert isinstance(block["reference_db"], float)
        for first, second, separation in zip(
            block["first_db"], block["second_db"], block["separation_db"], strict=True
        ):
            if separation is not None:
                assert abs((first - second) - separation) < 0.02


def test_a_reading_says_how_far_its_quieter_channel_stood_over_its_own_lead(tmp_path) -> None:
    """A separation is a separation of the unit's signal only while both channels
    carry one, so the quieter side's distance from the silence the take begins with
    is what says whether the reading is a level at all.

    Read at every setting rather than at chosen ones. The setting that matters is
    whichever separated furthest, and naming it in advance is a table kept by hand
    beside figures that could pick it out themselves.
    """
    root = saved(
        tmp_path,
        {
            "064": [(-20.0, -20.0)] * 2,
            "127": [(-20.0, -95.0)] * 2,
        },
    )

    found = balance.measure(root)[0]

    centred, hard = found.settings
    assert centred.stood_over_the_lead_db > 60.0
    # The note at -95 dB against a lead at -100 is a channel barely off its own floor,
    # which is the reading a separation must not be published without.
    assert hard.stood_over_the_lead_db < 10.0
    assert found.stood_over_the_lead_db == hard.stood_over_the_lead_db
    assert found.to_json()["stood_over_the_lead_db"] == round(hard.stood_over_the_lead_db, 2)


def test_the_ceiling_a_separation_is_read_against_comes_out_of_a_record(tmp_path) -> None:
    """How far this pair of inputs can be driven apart is a fact about one rig on
    one day, so it is read out of a record of that rig rather than held beside the
    code, where it would reach the next unit as an assumption.

    Every measured run of the record is asked. The ceiling is the furthest the pair
    has been shown to reach, and taking it from a chosen run would make it the
    furthest that run happened to.
    """
    where = tmp_path / "data/units/some-unit-01/balance/a-record.json"
    where.parent.mkdir(parents=True)
    where.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "name": "one",
                        "measured": True,
                        "channels": [2, 3],
                        "by_setting": [{"balance_db": [4.0, -31.25]}],
                    },
                    {
                        "name": "two",
                        "measured": False,
                        "channels": None,
                        "by_setting": [{"balance_db": [-99.0]}],
                    },
                ]
            }
        )
    )

    found = balance.how_far_a_record_separated(where)

    assert found["db"] == 31.25
    # The run this could not read is not evidence about how far the pair reaches,
    # and the channels reported are the ones the figure itself came off.
    assert found["on_channels"] == [2, 3]
    # Named from the unit down, the way one record names another everywhere here.
    assert found["where"] == "balance/a-record.json"
