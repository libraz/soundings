"""Controls for the motion tracker, on effects whose settings are already known.

Every failure mode here produces a number rather than an error, and each of those
numbers would be published as a fact about a machine. A tracker that finds a rate
in noise invents a modulator; one that reports no motion because the direct path
swamped the return misses one; one biased toward zero delay by its own window
reports every chorus as shallow. So each is given an effect it has to get right,
and each null is given something it has to stay silent about.
"""

from __future__ import annotations

import numpy as np
import pytest

from soundings import motion

SR = 48000


def source(seconds: float = 4.0, *, seed: int = 5, tonal: bool = False) -> np.ndarray:
    """Broadband decaying material, which is what a delay can be measured against.

    Noise by default: its autocorrelation has one peak, so a delay measured
    against it is unique. `tonal` gives the periodic case instead, which is what
    a held note actually is.
    """
    n = int(seconds * SR)
    t = np.arange(n) / SR
    if tonal:
        return sum(np.sin(2 * np.pi * 261.6 * k * t + k) / k for k in (1, 2, 3, 4, 5))
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    # Band-limit it, so linear interpolation in the delay line is not the thing
    # under test.
    spectrum = np.fft.rfft(noise)
    spectrum[np.fft.rfftfreq(n, 1 / SR) > 6000] = 0
    return np.fft.irfft(spectrum, n) * (0.3 + 0.7 * np.exp(-0.4 * t))


def chorused(
    dry: np.ndarray,
    *,
    rate: float,
    depth_ms: float,
    centre_ms: float = 20.0,
    mix: float = 0.7,
    shape: str = "sine",
) -> np.ndarray:
    """A send-return modulated delay: the dry signal plus a copy that moves."""
    n = dry.size
    t = np.arange(n) / SR
    phase = 2 * np.pi * rate * t
    swing = np.sin(phase) if shape == "sine" else 2 / np.pi * np.arcsin(np.sin(phase))
    delay = (centre_ms + depth_ms / 2.0 * swing) / 1000.0 * SR
    taps = np.arange(n) - delay
    return dry + mix * np.interp(taps, np.arange(n), dry, left=0.0, right=0.0)


def reverberated(dry: np.ndarray, *, seed: int = 9, seconds: float = 0.6) -> np.ndarray:
    """A static effect: one fixed impulse response, nothing moving."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    tail = rng.standard_normal(n) * np.exp(-6.0 * np.arange(n) / SR)
    return dry + 0.4 * np.convolve(dry, tail, mode="full")[: dry.size] / np.abs(tail).sum() * 8


def test_a_chorus_gives_up_its_rate_and_its_depth() -> None:
    dry = source()
    wet = chorused(dry, rate=1.3, depth_ms=4.0)
    found = motion.measure(dry, wet, SR)
    assert found.delay is not None
    assert found.delay.rate_hz == pytest.approx(1.3, abs=0.1)
    assert found.delay.depth == pytest.approx(4.0, rel=0.15)
    assert found.delay.sinusoidal


def test_the_centre_delay_is_recovered_not_only_the_swing() -> None:
    dry = source()
    wet = chorused(dry, rate=0.8, depth_ms=3.0, centre_ms=25.0)
    found = motion.measure(dry, wet, SR)
    assert found.delay is not None
    assert found.delay.centre == pytest.approx(25.0, abs=1.0)


def test_a_deep_slow_swing_is_not_reported_as_a_shallow_one() -> None:
    """The window-bias failure: a reference window that overlaps less and less as
    the lag grows pulls every peak toward no delay, and every chorus reads shallow."""
    dry = source(seconds=6.0)
    wet = chorused(dry, rate=0.5, depth_ms=12.0, centre_ms=30.0)
    found = motion.measure(dry, wet, SR)
    assert found.delay is not None
    assert found.delay.depth == pytest.approx(12.0, rel=0.15)


def test_a_triangle_is_told_apart_from_a_sine_by_what_it_leaves() -> None:
    dry = source(seconds=6.0)
    sine = motion.measure(dry, chorused(dry, rate=1.0, depth_ms=6.0), SR)
    triangle = motion.measure(dry, chorused(dry, rate=1.0, depth_ms=6.0, shape="triangle"), SR)
    assert sine.delay is not None and triangle.delay is not None
    assert sine.delay.sinusoidal
    assert not triangle.delay.sinusoidal
    assert triangle.delay.shape_error > sine.delay.shape_error * 3


def test_a_static_effect_reports_no_motion_rather_than_a_rate() -> None:
    """The negative control that matters most: a reverb is audible and moves nothing.

    A tracker that finds a line here would publish a modulation rate for every
    static effect on the machine.
    """
    dry = source()
    found = motion.measure(dry, reverberated(dry), SR)
    assert found.delay is None
    assert not found.moves


def test_two_takes_of_the_same_thing_have_no_motion_in_them() -> None:
    dry = source()
    rng = np.random.default_rng(21)
    wet = dry + 1e-4 * rng.standard_normal(dry.size)
    assert motion.measure(dry, wet, SR).delay is None


def test_a_tremolo_is_found_in_the_level_when_the_delay_holds_still() -> None:
    dry = source()
    t = np.arange(dry.size) / SR
    wet = dry * (1.0 + 0.5 * np.sin(2 * np.pi * 4.0 * t))
    found = motion.measure(dry, wet, SR)
    assert found.level is not None
    assert found.level.rate_hz == pytest.approx(4.0, abs=0.1)
    assert found.moves


def test_a_decaying_note_is_not_a_slow_tremolo() -> None:
    """The envelope's own decay is the largest slow thing in any take of a note."""
    assert motion.level_lfo(source(), SR) is None


def test_a_periodic_input_declares_the_period_a_delay_could_fold_into() -> None:
    tonal = source(tonal=True)
    track = motion.track_delay(tonal, chorused(tonal, rate=1.0, depth_ms=1.0), SR)
    assert track.periodicity > 0.5
    assert track.ambiguity_ms == pytest.approx(1000.0 / 261.6, rel=0.02)


def test_broadband_material_is_not_flagged_as_ambiguous() -> None:
    dry = source()
    track = motion.track_delay(dry, chorused(dry, rate=1.0, depth_ms=4.0), SR)
    assert track.periodicity < 0.5
    assert not track.wraps


def test_the_trigger_scatter_is_removed_before_the_delay_is_read() -> None:
    """MIDI scatter is milliseconds; a chorus depth is milliseconds. Confusing the
    two would put the trigger jitter into the effect's centre delay."""
    dry = source()
    wet = chorused(dry, rate=1.1, depth_ms=4.0, centre_ms=20.0)
    late = np.concatenate([np.zeros(400), wet])[: wet.size]
    found = motion.measure(dry, late, SR)
    assert found.delay is not None
    assert found.delay.centre == pytest.approx(20.0, abs=1.5)


def test_a_silent_lead_in_does_not_flatten_the_depth() -> None:
    """Every real take opens with silence to measure the floor in, and no frame in
    it finds anything. Carrying the track across that gap fills a fifth of it with
    a flat stretch the oscillator then has to account for, and the depth comes
    back short."""
    dry = source()
    lead = np.zeros(int(0.5 * SR))
    with_lead = np.concatenate([lead, dry])
    wet = np.concatenate([lead, chorused(dry, rate=0.9, depth_ms=5.0, centre_ms=18.0)])
    found = motion.measure(with_lead, wet, SR)
    assert found.delay is not None
    assert found.delay.depth == pytest.approx(5.0, rel=0.1)
    assert found.delay.centre == pytest.approx(18.0, abs=0.5)
    assert found.delay.sinusoidal


def test_a_static_effect_moves_no_level_either() -> None:
    """A reverb's level rattles, and the largest bin of a rattle clears the median
    of the band as readily as a tremolo does. Only whether a sinusoid explains the
    series tells the two apart."""
    note = np.concatenate([np.zeros(int(0.5 * SR)), source()])
    assert motion.measure(note, reverberated(note), SR).level is None


def test_a_bend_in_a_trend_is_not_a_very_slow_oscillation() -> None:
    """A note decaying and then flattening onto the noise is the commonest series
    there is here, and no straight line removes its bend."""
    n = int(4.0 * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(17)
    signal = rng.standard_normal(n) * (np.exp(-6.0 * t) + 1e-4)
    assert motion.level_lfo(signal, SR) is None


def test_a_track_of_pure_noise_produces_no_line() -> None:
    """The line test's own negative control: any spectrum has a largest bin."""
    rng = np.random.default_rng(4)
    track = motion.DelayTrack(
        times=np.arange(400) * 0.01,
        delay_samples=rng.standard_normal(400) * 20 + 900,
        confidence=np.ones(400),
        sample_rate=SR,
        hop=0.01,
        searched_ms=(0.0, 60.0),
        ambiguity_ms=float("nan"),
        periodicity=0.0,
        align_samples=0.0,
    )
    assert motion.fit_lfo(track) is None


def test_a_track_that_found_almost_nothing_is_refused() -> None:
    track = motion.DelayTrack(
        times=np.arange(200) * 0.01,
        delay_samples=np.where(np.arange(200) % 5 == 0, 900.0, np.nan),
        confidence=np.zeros(200),
        sample_rate=SR,
        hop=0.01,
        searched_ms=(0.0, 60.0),
        ambiguity_ms=float("nan"),
        periodicity=0.0,
        align_samples=0.0,
    )
    assert motion.fit_lfo(track) is None


# A null and an unanswerable question look identical coming out of the tracker:
# both are an absent LfoFit. The wrap flag is what separates them here, and it
# has to land in a field rather than in prose, since the field is what gets read.
# It is not the whole verdict: a track can be readable and the material still
# have been unable to give a modulation up, which only the control can say.


def test_a_wrapped_track_with_no_line_is_not_a_finding_of_no_motion() -> None:
    """A pitched note folds a wide modulation into its own period, and the folded
    track has no line left in it. Reporting that as an effect standing still
    inverts the answer -- and invites averaging the one signal that cannot be
    averaged."""
    tonal = source(tonal=True)
    found = motion.measure(tonal, chorused(tonal, rate=0.9, depth_ms=12.0, centre_ms=25.0), SR)
    assert found.track.wraps
    assert not found.delay_answered
    assert not found.moves
    assert not found.answered
    assert "no answer" in found.describe()
    assert "nothing moves" not in found.describe()
    assert found.to_json()["track_readable"] is False


def test_a_wrapped_track_can_fit_a_rate_that_is_simply_wrong() -> None:
    """Why the wrap withholds a fit and not only a null. The fold puts a line in
    the track at a rate the modulator never ran at, deep enough to clear the
    threshold and smooth enough to pass the shape gate."""
    tonal = source(tonal=True)
    found = motion.measure(tonal, chorused(tonal, rate=0.9, depth_ms=12.0, centre_ms=25.0), SR)
    assert found.delay is not None
    assert found.delay.rate_hz > 1.5
    assert found.delay.depth > 30.0
    assert "withheld" in found.describe()


def test_a_static_effect_on_broadband_material_still_answers() -> None:
    """The other side of it: the guard must not swallow the negative control, or
    no effect could ever be called static again."""
    dry = source()
    found = motion.measure(dry, reverberated(dry), SR)
    assert not found.track.wraps
    assert found.delay_answered
    assert found.answered
    assert "nothing moves" in found.describe()
    assert found.to_json()["track_readable"] is True


def test_a_readable_track_settles_nothing_while_the_control_has_failed() -> None:
    """The field a consumer filters on has to agree with the sentence beside it.

    A pair whose ladder recovered nothing was never shown able to report a
    modulation, so its silence is not the effect standing still. The track being
    followable says only that nothing structural stopped the search -- and a
    record that called that conclusive published the opposite of its own prose.
    """
    dry = source()
    found = motion.measure(dry, reverberated(dry), SR)
    assert found.answered and not found.moves

    assert motion.is_conclusive(found, {"detectable_ms": None}) is False
    assert motion.is_conclusive(found, {"detectable_ms": [3.0, 0.75]}) is True


def test_an_effect_that_moved_needs_no_control_to_have_settled_it() -> None:
    """Having moved is the finding. Demanding a ladder of a positive would throw
    away a real modulator because the search happened to bracket it narrowly."""
    dry = source()
    found = motion.measure(dry, chorused(dry, rate=1.1, depth_ms=3.0, centre_ms=8.0), SR)
    assert found.moves
    assert motion.is_conclusive(found, {"detectable_ms": None}) is True


def test_a_rate_read_off_a_wrapped_track_still_counts_as_an_answer() -> None:
    """Wrapping says a null would be empty, not that a line found in spite of it
    is worthless. A shallow modulation on a pitched note is still readable."""
    tonal = source(tonal=True)
    found = motion.measure(tonal, chorused(tonal, rate=1.0, depth_ms=1.0, centre_ms=20.0), SR)
    if found.track.wraps and found.delay is not None:
        assert found.delay_answered
        assert found.answered


def test_the_control_recovers_a_swing_it_injected_into_broadband_material() -> None:
    """The control's own control. It has to be able to pass, on material whose
    return has a delay to sweep, or a failure anywhere else says nothing. A fixed
    echo rather than a reverb: a diffuse tail correlates with the dry signal at no
    single lag, so sweeping it produces no track, and this method cannot vouch for
    itself on one."""
    dry = source()
    echoed = dry + 0.7 * np.roll(dry, int(0.02 * SR))

    vouched = motion.control(dry, echoed, SR, depths_ms=(4.0, 2.0))

    assert vouched["recovered"]
    assert vouched["detectable_ms"] == [4.0, 2.0]
    assert abs(vouched["attempts"][0]["recovered_rate_hz"] - motion.CONTROL_RATE_HZ) < 0.1


def test_a_swing_too_deep_to_follow_is_missed_like_one_too_shallow() -> None:
    """The reason sensitivity is reported as a band. A delay that moves far inside
    one tracking frame smears that frame's peak, so the deepest rungs of the ladder
    fail for a reason that has nothing to do with the takes -- and a null bounded
    only from below would claim to cover them."""
    dry = source()
    deep = motion.modulated_copy(dry, SR, rate_hz=1.3, depth_ms=24.0, centre_ms=16.0)
    shallow = motion.modulated_copy(dry, SR, rate_hz=1.3, depth_ms=4.0, centre_ms=16.0)

    assert motion.measure(dry, dry + 0.7 * deep, SR).delay is None
    assert motion.measure(dry, dry + 0.7 * shallow, SR).delay is not None


def test_the_control_fails_where_there_is_no_return_to_carry_it() -> None:
    """A wet take identical to the dry one has no return, so the injected swing is
    scaled to nothing and cannot be found. That is the case the control exists for:
    the measurement would report no motion, and does so for the wrong reason."""
    dry = source()

    vouched = motion.control(dry, dry.copy(), SR, depths_ms=(4.0,))

    assert not vouched["recovered"]
    assert vouched["detectable_ms"] is None


def test_a_ladder_that_recovered_nothing_does_not_say_it_recovered() -> None:
    """The caveat naming the band opens by stating that the ladder recovered at two
    depths. Emitted where there is no band, it stands beside rows saying every rung
    failed, and a reader who takes the prose has been told the opposite of the
    rows."""
    dry = source()
    echoed = dry + 0.7 * np.roll(dry, int(0.02 * SR))

    failed = motion.control(dry, dry.copy(), SR, depths_ms=(4.0,))
    passed = motion.control(dry, echoed, SR, depths_ms=(4.0, 2.0))

    assert failed["detectable_why"] == motion.WHY_NO_BAND
    assert passed["detectable_why"] == motion.WHY_DETECTABLE


def test_a_lone_rung_across_a_gap_does_not_widen_the_band() -> None:
    """A band drawn through a missed depth would claim every depth inside it. The
    longest unbroken run is taken instead, so the gap ends the band rather than
    being spanned by it."""
    attempts = [
        {"injected_depth_ms": 24.0, "recovered": False},
        {"injected_depth_ms": 12.0, "recovered": True},
        {"injected_depth_ms": 6.0, "recovered": True},
        {"injected_depth_ms": 3.0, "recovered": False},
        {"injected_depth_ms": 1.5, "recovered": True},
    ]
    assert motion._longest_recovered_run(attempts) == (12.0, 6.0)


def test_a_ladder_that_recovered_nothing_has_no_band() -> None:
    assert motion._longest_recovered_run([{"injected_depth_ms": 6.0, "recovered": False}]) is None


def test_the_ladder_is_tried_deepest_first_however_it_was_given() -> None:
    """The band is read by walking down the ladder, so an unsorted one would break
    its run at whichever depth happened to come first."""
    dry = source(seconds=1.0)
    vouched = motion.control(dry, reverberated(dry), SR, depths_ms=(2.0, 4.0))
    assert vouched["injected_depths_ms"] == [4.0, 2.0]


def test_an_injected_swing_is_scaled_to_the_return_the_take_really_has() -> None:
    """A control given a louder return than the real one vouches for a measurement
    nobody made. The level is reported so a reader can see which it was."""
    dry = source(seconds=1.0)
    quiet = dry + 0.001 * np.roll(dry, 400)
    loud = dry + 0.5 * np.roll(dry, 400)

    quiet_level = motion.control(dry, quiet, SR, depths_ms=(4.0,))["return_level_db"]
    loud_level = motion.control(dry, loud, SR, depths_ms=(4.0,))["return_level_db"]

    assert quiet_level < loud_level - 20


def test_a_modulated_copy_swings_by_the_depth_it_was_given() -> None:
    """Peak to peak, because that is the quantity a fit reports back. Halving one
    and not the other would make every control miss by a factor of two."""
    dry = source(seconds=3.0)
    made = motion.modulated_copy(dry, SR, rate_hz=1.0, depth_ms=4.0, centre_ms=20.0)
    track = motion.track_delay(dry, dry + made, SR, search_ms=(0.0, 40.0))
    fit = motion.fit_lfo(track)

    assert fit is not None
    assert abs(fit.depth - 4.0) < 1.0


def decaying_into_silence(
    *, lead: float = 0.6, seconds: float = 4.0, floor: float = 1e-5, seed: int = 11
) -> tuple[np.ndarray, np.ndarray]:
    """A pair of struck takes shaped like real ones: silence, a hit, a decay that
    reaches the recorder's own floor well before the take ends, and a floor that
    is each take's own rather than shared.

    Both properties are needed to show what the gate does. A source that never
    goes quiet -- which every other fixture here is -- has no silent frames to
    mistrack; and one built by adding to the dry take shares its noise, so its
    silent frames correlate cleanly at one lag instead of scattering. Two takes
    off a machine have neither. Measured on a real crash take from this unit:
    peak -31 dBFS, fallen to -102 by the last second.
    """
    n = int(seconds * SR)
    index = np.arange(n)
    rng = np.random.default_rng(seed)
    body = source(seconds=seconds) * np.where(
        index < lead * SR, 0.0, np.exp(-3.0 * (index / SR - lead))
    )
    wet = body + 0.5 * np.roll(body, int(0.02 * SR))
    return body + rng.standard_normal(n) * floor, wet + rng.standard_normal(n) * floor


def test_frames_holding_only_the_noise_floor_are_not_tracked() -> None:
    """A normalised correlation says how well two frames match, not how much sound
    was in them, and noise matches itself. Without the level gate those frames
    report a confident delay at whatever lag their noise peaked at, and the track
    ends up swinging across most of the range that was searched."""
    dry, wet = decaying_into_silence()

    ungated = motion.track_delay(dry, wet, SR)
    gated = motion.track_delay(dry, wet, SR, lead_s=0.6)

    assert ungated.frames_below_floor == 0 and np.isnan(ungated.floor_db)
    assert gated.frames_below_floor > gated.delay_samples.size // 4
    assert ungated.excursion_ms > 10.0
    assert gated.excursion_ms < 1.0


def test_the_floor_gate_is_what_lets_a_decaying_take_carry_a_control() -> None:
    """The measurement this was found by, and the reason it blocked anything else.
    On a real crash take the control recovered nothing at any depth until the
    silent frames stopped contributing; with them gone it recovered four rungs.
    Every take of a struck note has this shape, so without the gate no such pair
    could vouch for its own null."""
    dry, wet = decaying_into_silence()

    assert motion.control(dry, wet, SR)["detectable_ms"] is None
    assert motion.control(dry, wet, SR, lead_s=0.6)["detectable_ms"] is not None


def test_a_take_with_no_lead_in_declared_is_left_ungated() -> None:
    """A floor guessed from the take itself would be right for one that decays
    into silence and would throw away most of one that does not, so an absent
    lead-in leaves the track as it was rather than gated against a guess."""
    dry = source(seconds=1.0)
    track = motion.track_delay(dry, dry + 0.5 * np.roll(dry, 400), SR, lead_s=0.0)

    assert np.isnan(track.floor_db)
    assert track.to_json()["noise_floor_db"] is None
    assert "frames_gated_on_level" not in track.to_json()
