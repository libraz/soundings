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


__all__ = ["SR", "judge", "note"]


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
    steady = [note(seed=s) for s in (0, 1, 2)]
    # Each take modulated at its own phase, the way a free-running LFO leaves them.
    moving = []
    for phase in rng.uniform(0, 2 * np.pi, 3):
        base = note(seed=9)
        t = np.arange(base.size) / SR
        moving.append(base * (1.0 + 0.5 * np.sin(2 * np.pi * 1.3 * t + phase)))
    verdict = judge(steady, moving)
    assert verdict.changed_the_repeatability
    assert verdict.audible
    assert "moves came on" in verdict.describe()


def test_two_equally_repeatable_settings_are_not_called_audible_by_repeatability() -> None:
    a = [note(seed=s) for s in (0, 1, 2)]
    b = [note(seed=s) for s in (3, 4, 5)]
    assert not judge(a, b).changed_the_repeatability
