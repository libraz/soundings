"""Controls for the audibility verdict, on differences whose size is already known.

The verdict's failure modes are both silent and both look like results about the
hardware: calling a parameter audible when the unit merely failed to repeat, and
calling it inaudible when the change was real but smaller than the yardstick was
set to see. Each is given a case it has to get right.
"""

from __future__ import annotations

import numpy as np

from soundings import audible

SR = 48000


def note(*, fundamental: float = 220.0, gain: float = 1.0, seed: int = 0, noise: float = 3e-4):
    """A struck note with independent noise, so takes differ the way real ones do."""
    rng = np.random.default_rng(seed)
    lead = np.zeros(int(0.3 * SR))
    t = np.arange(int(1.5 * SR)) / SR
    tone = sum(np.sin(2 * np.pi * fundamental * k * t) / k for k in (1, 2, 3, 5))
    body = gain * tone * np.exp(-2.5 * t)
    signal = np.concatenate([lead, body])
    return signal + noise * rng.standard_normal(signal.size)


def judge(first, second, **kw):
    return audible.judge(
        first, second, SR, label="test", stimulus="a struck note", silence_before=0.25, **kw
    )


def unrepeatable(seed: int) -> np.ndarray:
    """A take with a proper silent lead-in whose body never repeats.

    The shape a voice with free-running modulation actually has: the level is
    the same take to take, so the gain fit lands on 0 dB and the residual is the
    whole signal. Pure noise with no lead-in models it badly -- the level fit
    then has nothing to lock to and wanders by enough to trip on its own.
    """
    rng = np.random.default_rng(seed)
    lead = 3e-4 * rng.standard_normal(int(0.3 * SR))
    t = np.arange(int(1.5 * SR)) / SR
    body = rng.standard_normal(t.size) * np.exp(-2.5 * t) * 0.4
    return np.concatenate([lead, body])


def modulated(phase: float, *, rate: float = 6.0, depth: float = 0.5) -> np.ndarray:
    """A take whose amplitude is swept by an LFO starting at an arbitrary phase.

    The rate matters as much as the depth. A slow sweep barely moves before a
    struck note has decayed, so takes at different phases still resemble each
    other and the result models a weak wobble rather than a chorus. A few Hz is
    what a chorus actually runs at, and it is what the real one measured like.
    """
    base = note(seed=9)
    t = np.arange(base.size) / SR
    return base * (1.0 + depth * np.sin(2 * np.pi * rate * t + phase))


__all__ = ["SR", "judge", "modulated", "note", "unrepeatable"]


def test_the_same_setting_twice_is_not_audible() -> None:
    """The negative control. Takes differ by their noise, and that must not count."""
    a = [note(seed=s) for s in (0, 1, 2)]
    b = [note(seed=s) for s in (3, 4, 5)]
    verdict = judge(a, b)
    assert not verdict.audible
    assert not verdict.changed_the_shape
    assert not verdict.changed_the_level


def test_a_changed_timbre_is_audible() -> None:
    a = [note(fundamental=220.0, seed=s) for s in (0, 1, 2)]
    b = [note(fundamental=233.1, seed=s) for s in (3, 4, 5)]
    verdict = judge(a, b)
    assert verdict.audible
    assert verdict.changed_the_shape


def test_a_pure_level_change_is_audible_even_though_it_leaves_no_residual() -> None:
    """The case the alignment's own level fit would otherwise erase."""
    a = [note(gain=1.0, seed=s) for s in (0, 1, 2)]
    b = [note(gain=0.5, seed=s) for s in (3, 4, 5)]
    verdict = judge(a, b)
    assert verdict.changed_the_level
    assert verdict.across_level_db < -5.0
    assert verdict.audible
    # A quieter setting has less signal over the same noise, so its residual is
    # worse for a reason that is not the unit. Judged against each setting's own
    # floor, that cancels, and a level change must not read as a modulator.
    assert not verdict.changed_the_repeatability


def test_a_change_smaller_than_the_unit_repeats_to_is_not_claimed() -> None:
    """A real but tiny difference buried under a noisy unit reads as inaudible.

    That is the honest answer for this stimulus, and the verdict says so rather
    than reporting a difference it cannot separate from the noise.
    """
    a = [note(seed=s, noise=0.02) for s in (0, 1, 2)]
    b = [note(fundamental=220.02, seed=s, noise=0.02) for s in (3, 4, 5)]
    verdict = judge(a, b)
    assert not verdict.audible
    assert "not" in verdict.describe()


def test_a_noisy_unit_raises_the_yardstick_rather_than_the_verdict() -> None:
    quiet = judge([note(seed=s) for s in (0, 1)], [note(seed=s) for s in (2, 3)])
    noisy = judge(
        [note(seed=s, noise=0.02) for s in (0, 1)], [note(seed=s, noise=0.02) for s in (2, 3)]
    )
    assert noisy.within_db > quiet.within_db
    assert not quiet.audible and not noisy.audible


def test_the_verdict_carries_the_stimulus_it_was_reached_with() -> None:
    verdict = judge([note(seed=0), note(seed=1)], [note(seed=2), note(seed=3)])
    assert verdict.to_json()["stimulus"] == "a struck note"
    assert "stimulus" in verdict.to_json()["caveat"]


def test_a_modulator_switching_on_is_audible_though_no_residual_can_show_it() -> None:
    """The chorus case: takes stop repeating, so the yardstick swallows the change.

    The difference between the settings can never clear a yardstick the change
    itself created, so the collapse in repeatability has to be the evidence.
    """
    rng = np.random.default_rng(5)
    steady = [note(seed=s) for s in (0, 1, 2, 3)]
    # Each take modulated at its own phase, the way a free-running LFO leaves them.
    moving = [modulated(phase) for phase in rng.uniform(0, 2 * np.pi, 4)]
    verdict = judge(steady, moving)
    assert verdict.changed_the_repeatability
    assert verdict.audible
    assert "moves came on" in verdict.describe()


def test_two_equally_repeatable_settings_are_not_called_audible_by_repeatability() -> None:
    a = [note(seed=s) for s in (0, 1, 2)]
    b = [note(seed=s) for s in (3, 4, 5)]
    assert not judge(a, b).changed_the_repeatability


def test_one_disturbed_take_does_not_read_as_a_modulator() -> None:
    """Measured on hardware: the same reverb contrast flagged a modulator over
    three takes and not over four, on a unit that does repeat with reverb on.

    The worst pair in a group moves with a single bad take. A real modulator
    moves every pair, so the group is judged by its median instead.
    """
    rng = np.random.default_rng(21)
    clean = [note(seed=s) for s in (0, 1, 2, 3, 8)]
    disturbed = [note(seed=s) for s in (4, 5, 6)]
    spoiled = note(seed=7)
    spoiled[int(0.9 * SR) : int(0.95 * SR)] += 0.05 * rng.standard_normal(int(0.05 * SR))
    disturbed.append(spoiled)
    assert not judge(clean, disturbed).changed_the_repeatability


def test_too_few_takes_per_setting_cannot_claim_a_modulator() -> None:
    """Under four takes a group has at most two pairs, whose median is their mean."""
    a = [note(seed=0), note(seed=1), note(seed=2)]
    b = [note(seed=3), note(seed=4), note(seed=5)]
    verdict = judge(a, b)
    assert np.isnan(verdict.within_each_db[0])
    assert not verdict.changed_the_repeatability


def test_a_yardstick_nothing_could_clear_is_inconclusive_rather_than_a_null() -> None:
    """Seen on hardware: a disturbed take left a stimulus unable to measure anything,
    and it reported the volume control it was asked about as inaudible."""
    takes = [unrepeatable(s) for s in range(6)]
    verdict = judge(takes[:3], takes[3:])
    assert not verdict.audible
    assert verdict.inconclusive
    assert "inconclusive" in verdict.describe()
    assert "says nothing about the parameter" in verdict.describe()


def test_an_audible_verdict_is_never_called_inconclusive() -> None:
    """The chorus repeats terribly and is still measured, via the repeatability channel."""
    rng = np.random.default_rng(43)
    steady = [note(seed=s) for s in (0, 1, 2, 3)]
    moving = [modulated(phase) for phase in rng.uniform(0, 2 * np.pi, 4)]
    verdict = judge(steady, moving)
    assert verdict.audible
    assert not verdict.inconclusive


def test_a_pure_level_change_on_an_imperfect_voice_is_not_a_modulator() -> None:
    """Measured on CC7, whose 20 dB is nothing but level: the release tail and the
    organ both opened a repeatability gap, because whatever fails to repeat in a
    voice scales with the voice while the noise floor does not."""
    rng = np.random.default_rng(53)

    def wobbly(seed: int, gain: float) -> np.ndarray:
        base = note(seed=0, noise=0.0) * gain
        jitter = rng.standard_normal(base.size) * 0.01 * np.abs(base)
        return base + jitter + 3e-4 * rng.standard_normal(base.size)

    quiet = [wobbly(s, 0.1) for s in (0, 1, 2)]
    loud = [wobbly(s, 1.0) for s in (3, 4, 5)]
    verdict = judge(quiet, loud)
    assert verdict.changed_the_level
    assert not verdict.changed_the_repeatability
