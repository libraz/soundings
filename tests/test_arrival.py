"""What arrived at once and what arrived late, which no reading of one instant can say.

A byte that mixes two signals onto the same channels is invisible to every other
reading here. A comparison made in one channel sees one level; a comparison
between the two channels sees a pan and not a mix; a band reading sees whatever
shape the sum has. What separates the two halves is time, and this file guards
that it is separated rather than assumed.

Three failures are guarded. A level change reaching both windows must not be
reported as signal moving between them. A run whose windows were never measured
must be left unread rather than read against a guess. And the figure the
candidate laws differ in -- the two windows together -- must come back flat when
the two halves hold a constant power and three decibels down in the middle when
they cross fade in amplitude, because a reading that cannot tell those apart
cannot decide anything about this class.

Takes are synthesised rather than recorded: what is under test is the reading.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from soundings import arrival
from soundings.takes import write

RATE = 8000
LEAD = 0.6
TAKE_S = 2.2

NOTE_AT, NOTE_FOR = 0.65, 0.30
RETURN_AT = 1.50
"""Where the note is gated and where a delayed copy of it lands, in seconds.

The figures are the ones measured on this unit: the note arrives a twentieth of a
second after it is asked for, and a delay with its feedback at nothing returns it
eighty-five hundredths of a second later.
"""

WINDOWS = {"length": 0.20, "lead": 0.10, "early": 0.68, "late": 1.53}
"""Three windows of one length, each inside the thing it is reading."""

SILENT_DB = -60.0
"""What an end of the sweep is written as when the law would put nothing there.

A multiplier taken to zero is minus infinity in decibels and there is no such
take: an end that empties a window empties it down to the floor of whatever is
carrying it, so the tests say so with a number rather than with an infinity the
arithmetic would have to be defended against.
"""


def take(path, *, early_db: float, late_db: float, seed: int, mono: bool = False) -> None:
    """A take of a gated note and a delayed copy of it, in two channels of four.

    Four channels because the interface has more inputs than the unit uses, and
    which of them carried it is something the reading has to work out rather than
    be told. The two the unit is not on hold noise well over the converter's own,
    since an input nothing is plugged into is not silent.
    """
    rng = np.random.default_rng(seed)
    n = int(RATE * TAKE_S)
    tone = np.sin(2 * np.pi * 220 * np.arange(n) / RATE)
    gate = np.zeros(n)
    gate[int(NOTE_AT * RATE) : int((NOTE_AT + NOTE_FOR) * RATE)] = 10 ** (early_db / 20)
    gate[int(RETURN_AT * RATE) : int((RETURN_AT + NOTE_FOR) * RATE)] = 10 ** (late_db / 20)
    frames = rng.normal(0, 1e-6, (n, 4))
    frames[:, 0] += rng.normal(0, 1.0, n) * 10 ** (-70.0 / 20)
    frames[:, 1] += rng.normal(0, 1.0, n) * 10 ** (-75.0 / 20)
    frames[:, 2] += tone * gate
    if not mono:
        frames[:, 3] += tone * gate
    write(path, frames, RATE)


def saved(tmp_path, settings, *, windows=WINDOWS, mono: bool = False, stimulus: str = "gated"):
    """A directory shaped like one a --save run leaves, with the windows it measured."""
    kept = []
    seed = 0
    for setting, levels in settings.items():
        for index, (early, late) in enumerate(levels):
            name = f"{stimulus}-{setting}-{index:02d}.wav"
            seed += 1
            take(tmp_path / name, early_db=early, late_db=late, seed=seed, mono=mono)
            kept.append({"stimulus": stimulus, "setting": setting, "take": index, "file": name})
    manifest = {"stimuli": [{"name": stimulus, "lead_s": LEAD}], "takes": kept}
    if windows is not None:
        manifest["windows_s"] = windows
    (tmp_path / "takes-manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_signal_moved_from_one_window_to_the_other_is_reported(tmp_path) -> None:
    """The reading this stage exists for: one window empties as the other fills,
    which is what a byte printed `D> 0E - D 0<E` is claimed to do and what no
    reading of the output at one instant can see."""
    root = saved(
        tmp_path,
        {
            "0": [(-6.0, SILENT_DB)] * 2,
            "64": [(-9.0, -9.0)] * 2,
            "127": [(SILENT_DB, -6.0)] * 2,
        },
    )

    (found,) = arrival.measure(root)

    assert found.moved_between_settings
    assert found.moved_in_opposite_directions


def test_a_level_change_reaching_both_windows_is_not_a_move_between_them(tmp_path) -> None:
    """The guard has to have an outside. A byte that turns the whole output down
    moves both windows together, and a reading that called that a mix would report
    the output level as a balance."""
    root = saved(
        tmp_path,
        {
            "0": [(-6.0, -12.0)] * 2,
            "64": [(-16.0, -22.0)] * 2,
            "127": [(-26.0, -32.0)] * 2,
        },
    )

    (found,) = arrival.measure(root)

    assert not found.moved_between_settings
    assert not found.moved_in_opposite_directions
    assert found.together_spread_db > 15.0


def test_the_two_windows_together_hold_still_for_a_constant_power_pair(tmp_path) -> None:
    """The figure the candidates differ in. A pair of multipliers whose squares sum
    to one puts the same power out at every setting, and a reading that could not
    show that could not separate this class at all."""
    turns = np.linspace(0.0, np.pi / 2, 9)
    root = saved(
        tmp_path,
        {
            f"{int(i * 127 / 8):03d}": [
                (
                    float(20 * np.log10(max(np.cos(turn), 1e-3))),
                    float(20 * np.log10(max(np.sin(turn), 1e-3))),
                )
            ]
            * 2
            for i, turn in enumerate(turns)
        },
    )

    (found,) = arrival.measure(root)

    assert found.moved_between_settings
    assert found.together_spread_db < 0.5


def test_the_two_windows_together_dip_three_decibels_for_a_crossfade(tmp_path) -> None:
    """The other law, and the one the total tells apart from the first. Two halves
    of an amplitude are a quarter of a power each, so the two together sit three
    decibels under either end in the middle."""
    parts = np.linspace(0.0, 1.0, 9)
    root = saved(
        tmp_path,
        {
            f"{int(i * 127 / 8):03d}": [
                (
                    float(20 * np.log10(max(1.0 - part, 1e-3))),
                    float(20 * np.log10(max(part, 1e-3))),
                )
            ]
            * 2
            for i, part in enumerate(parts)
        },
    )

    (found,) = arrival.measure(root)

    assert 2.8 < found.together_spread_db < 3.2


def test_a_run_that_does_not_say_where_its_windows_go_is_left_unread(tmp_path) -> None:
    """Where a return lands is a measurement. A directory that does not carry one
    is refused rather than read against the printed delay time, which is a
    statement about a page and not about this unit."""
    root = saved(tmp_path, {"0": [(-6.0, SILENT_DB)]}, windows=None)

    (found,) = arrival.measure(root)

    assert found.not_measured == arrival.NO_WINDOWS
    assert not found.settings


def test_windows_that_overlap_are_refused_rather_than_summed(tmp_path) -> None:
    """A return that begins before the direct sound has ended would be summed with
    it inside one window, and the figure would move with the setting exactly as a
    separated reading does while meaning something else."""
    root = saved(
        tmp_path,
        {"0": [(-6.0, SILENT_DB)]},
        windows={"length": 0.20, "lead": 0.10, "early": 0.68, "late": 0.80},
    )

    (found,) = arrival.measure(root)

    assert found.not_measured == arrival.WINDOWS_OVERLAP


AT_THE_FLOOR_DB = -130.0
"""A window holding less than the noise the take is carrying anyway.

Distinct from `SILENT_DB`, which is an end of a sweep that emptied a window down
to something still well over the floor -- which is what the unit actually does,
and what the reading has to be able to tell from this.
"""


def test_how_far_each_window_stood_over_the_lead_is_reported(tmp_path) -> None:
    """A window that has fallen to the lead is not a quiet reading of the unit, it
    is the interface, and the claim it supports is a bound in one direction rather
    than a level in either."""
    root = saved(
        tmp_path,
        {
            "0": [(-6.0, AT_THE_FLOOR_DB)] * 2,
            "64": [(-9.0, SILENT_DB)] * 2,
            "127": [(AT_THE_FLOOR_DB, -6.0)] * 2,
        },
    )

    (found,) = arrival.measure(root)

    by = {s.setting: s for s in found.settings}
    assert by["0"].early_over_the_lead_db > 40.0
    assert by["0"].late_over_the_lead_db < 3.0
    # The middle setting emptied its late window to a level, not to the floor, and
    # the two must not read alike: the whole use of this figure is to say which of
    # the two a sweep's end reached.
    assert by["64"].late_over_the_lead_db > 40.0
    assert found.quietest_late_over_the_lead_db < 3.0


def test_the_two_channels_the_unit_arrived_on_are_the_ones_read(tmp_path) -> None:
    """Two of the four inputs carry nothing but their own preamps, at seventy and
    seventy-five decibels down. A reading that took them in would be averaging the
    unit with an idle channel."""
    root = saved(tmp_path, {"0": [(-6.0, SILENT_DB)] * 2, "127": [(SILENT_DB, -6.0)] * 2})

    (found,) = arrival.measure(root)

    assert found.channels == [2, 3]


def test_a_unit_that_arrived_on_one_channel_is_still_read(tmp_path) -> None:
    """Unlike a balance between the channels, this reading has an answer on a mono
    source: what it separates is two parts of one take, not two legs of a cable.
    A stage that refused a mono source here would be carrying a limitation that
    belongs to a different question."""
    root = saved(
        tmp_path,
        {"0": [(-6.0, SILENT_DB)] * 2, "127": [(SILENT_DB, -6.0)] * 2},
        mono=True,
    )

    (found,) = arrival.measure(root)

    assert found.channels == [2]
    assert found.moved_in_opposite_directions


# Every ordering of the three windows against one length, rather than a sample of
# them. Four numbers decide whether a run can be read at all, and the space they
# span is twelve cells: enumerating it is narrower than choosing cases by eye and
# leaves nothing for a later reader to wonder about.
GEOMETRIES = [
    pytest.param({"length": 0.2, "lead": 0.1, "early": 0.7, "late": 1.5}, True, id="in-order"),
    pytest.param({"length": 0.2, "lead": 0.1, "early": 0.3, "late": 0.5}, True, id="touching"),
    pytest.param({"length": 0.2, "lead": 0.1, "early": 0.25, "late": 1.5}, False, id="lead-late"),
    pytest.param({"length": 0.2, "lead": 0.1, "early": 0.7, "late": 0.85}, False, id="early-late"),
    pytest.param({"length": 0.2, "lead": 0.7, "early": 0.1, "late": 1.5}, False, id="lead-after"),
    pytest.param({"length": 0.2, "lead": 0.1, "early": 1.5, "late": 0.7}, False, id="late-before"),
    pytest.param({"length": 0.0, "lead": 0.1, "early": 0.7, "late": 1.5}, False, id="no-length"),
    pytest.param({"length": -0.2, "lead": 0.1, "early": 0.7, "late": 1.5}, False, id="back-length"),
]


@pytest.mark.parametrize(("windows", "readable"), GEOMETRIES)
def test_the_windows_are_in_order_and_do_not_touch(windows, readable) -> None:
    assert arrival._apart(windows) is readable
