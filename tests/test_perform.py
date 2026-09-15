"""The gates a single take passes before it is kept, on the input the unit is on.

An interface has more inputs than the unit is plugged into, and the ones nobody
plugged anything into are not silent: on this rig an idle preamp sits around -70
dBFS while the unit's own two inputs floor at -110. Every reader in the archive
already picks a channel before it measures anything, for the reason
`test_archive_channel.py` sets out at length.

The gate that decides whether a take is *taken again* did not. It peaked across
the whole interface, so the idle preamp answered for the unit: every take on a
run was recorded three times and refused three times, and each record went out
saying the lead-in was never quiet -- about channels quieter than any the archive
holds. The cost was three times the machine time and a sentence that is not true.

No hardware: the recordings here are built in the test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soundings import perform

SR = 48000
LEAD = 0.6

ASKED_FOR_DBFS = -60.0
"""The bar a lead-in is held against, which is what makes these numbers a test.

Taken from the command's own default rather than invented, because the defect is
not that the wrong channel was read -- it is that the wrong channel was read
*across* this bar. An idle input under it and a unit's floor under it differ by
nothing a gate can act on, and a test written that way passes on the code it was
supposed to reject. The idle preamp measured on this rig peaks at -59.6 dBFS,
four tenths of a decibel the wrong side of the bar, and that is the whole of it.
"""


@dataclass
class FakeRecording:
    samples: np.ndarray
    sample_rate: int = SR
    device: str = "test input"
    overflows: int = 0
    open_seconds: float = 0.6


def _take(*channels: np.ndarray) -> FakeRecording:
    return FakeRecording(np.column_stack(channels))


READ_OVER = LEAD * 0.8
"""How much of the lead-in the gate reads, which is what a level here is set over.

Not the whole take. Scaling a channel so its peak over two seconds is the number
asked for puts that peak wherever the noise happened to put it, and the slice the
gate reads then falls a decibel short -- which on a margin of four tenths is the
difference between a test that rejects the old code and one that passes on it.
It did: the first version of the test below asserted a refusal the old code did
not make, and said so.
"""


def _quiet(peak_db: float, seed: int = 3) -> np.ndarray:
    """A channel holding nothing but a floor, peaking at the level asked for.

    Peak rather than RMS because the gate reads a peak, and the two differ by the
    crest factor of whatever noise the input is making -- twelve decibels on this
    rig, which is more than the margin the whole defect turned on.
    """
    rng = np.random.default_rng(seed)
    floor = rng.standard_normal(int(2.0 * SR))
    return floor / np.abs(floor[: int(READ_OVER * SR)]).max() * 10 ** (peak_db / 20.0)


def _across_every_channel(recording: FakeRecording, before: float) -> float:
    """What the gate used to answer: the peak over the whole interface.

    Written out here rather than described, so each test below can show the bar
    falling between the two answers instead of asserting only the new one.
    """
    span = recording.samples[: int(before * recording.sample_rate)]
    return float(20.0 * np.log10(np.abs(span).max()))


def _with_a_note(level_db: float, seed: int = 4) -> np.ndarray:
    """A channel whose lead-in is at `level_db` and which then carries a note."""
    channel = _quiet(level_db, seed=seed)
    index = np.arange(channel.size)
    sounding = index >= LEAD * SR
    channel[sounding] += 0.5 * np.sin(2 * np.pi * 440.0 * index[sounding] / SR)
    return channel


def test_the_lead_in_is_read_on_the_input_the_unit_arrived_on() -> None:
    """The defect, as the two channels that produced it and the bar between them.

    An idle preamp peaking four tenths of a decibel over the bar, beside a unit
    whose own lead-in is forty decibels under it. Read across the interface the
    take is refused and taken again; read on the channel the note is on it passes.
    Both answers are asserted, because only the pair shows the bar falling between
    them -- the new answer alone would pass on the code this replaced.
    """
    idle = _quiet(-59.6, seed=1)
    unit = _with_a_note(-100.0, seed=2)
    take = _take(idle, unit)

    assert _across_every_channel(take, LEAD * 0.8) > ASKED_FOR_DBFS
    assert perform.lead_in_dbfs(take, LEAD * 0.8) < ASKED_FOR_DBFS - 30.0


def test_a_tail_on_the_unit_s_own_input_is_still_caught() -> None:
    """The other half, without which the fix is only a louder bar. What the gate
    exists for is the take before sounding into this one's lead-in, and that
    arrives on the same channel the note does."""
    idle = _quiet(-100.0, seed=5)
    unit = _with_a_note(-40.0, seed=6)

    assert perform.lead_in_dbfs(_take(idle, unit), LEAD * 0.8) > ASKED_FOR_DBFS


def test_which_channel_is_chosen_is_the_one_the_note_is_on_not_the_loudest_floor() -> None:
    """The choice the fix rests on. An idle input forty dB over the unit's floor
    still loses to the channel the note is on, because the pick is made on the
    peak of the whole take rather than on its lead-in."""
    idle = _quiet(-59.6, seed=7)
    unit = _with_a_note(-100.0, seed=8)

    assert perform.loudest_channel(_take(idle, unit)) == 1
    assert perform.loudest_channel(_take(unit, idle)) == 0


def test_a_take_with_no_lead_in_to_read_says_so_rather_than_passing() -> None:
    """Zero samples is not a quiet lead-in. Reported as minus infinity, which is
    what it is, and every caller compares rather than tests for it."""
    assert perform.lead_in_dbfs(_take(_quiet(-59.6), _quiet(-59.6)), 0.0) == float("-inf")


def test_a_digitally_silent_channel_does_not_read_as_a_number() -> None:
    """A channel of exact zeros has no decibel value, and taking its logarithm is
    how a floor arrives in a record as a very large negative number that looks
    measured."""
    silent = np.zeros(int(2.0 * SR))

    assert perform.lead_in_dbfs(_take(silent, silent), LEAD) == float("-inf")
