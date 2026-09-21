"""Controls for the partial reading, on modulators whose settings are known.

Three mechanisms produce a moving sound and this reading has to tell them apart:
a delay being swept, a filter section turning every partial by the same angle, and
a level going up and down with nothing delayed at all. Each of the three returns a
number under all three readings, so a route that cannot separate them publishes a
chorus's depth for a tremolo. Each is injected here and each has something it has
to be refused.

The takes are synthetic and short on purpose. The comb fit is the expensive part
of the stage and its cost is in the length of the series, so the controls that
exercise it run on two seconds rather than seven.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from soundings import efxpartials, partials

SR = 16000
F0 = 400.0


def tone(seconds: float = 3.0, *, orders: int = 8) -> np.ndarray:
    """A held tone of several orders, which is what this reading is taken on."""
    t = np.arange(int(seconds * SR)) / SR
    return sum(np.sin(2 * np.pi * F0 * k * t + k) / k for k in range(1, orders + 1))


def grid_of(lo: float = 0.2, hi: float = 6.0, step: float = 0.01) -> np.ndarray:
    return np.arange(lo, hi + step / 2, step)


def turned(signal: np.ndarray, *, hz: float, radians: float) -> np.ndarray:
    """Every partial's phase turned by the same angle, which a delay never does."""
    spectrum = np.fft.rfft(signal)
    turn = np.exp(1j * radians)
    n = signal.size
    t = np.arange(n) / SR
    # Modulated by turning the whole analytic signal, so the angle is common to
    # every partial rather than proportional to its frequency.
    analytic = np.zeros(n, dtype=complex)
    analytic[: spectrum.size] = spectrum * 2.0
    analytic[0] = spectrum[0]
    narrow = np.fft.ifft(analytic)
    swing = np.angle(turn) * np.sin(2 * np.pi * hz * t)
    return np.real(narrow * np.exp(1j * swing))


def read(signal: np.ndarray, keep=None) -> partials.Partials:
    held = partials.read(signal, SR, carrier_hz=F0, keep=keep)
    assert held is not None
    return held


def test_a_held_tone_gives_up_the_orders_it_was_built_from() -> None:
    held = read(tone(orders=6))
    assert held.orders >= 4
    assert held.freqs_hz[0] == pytest.approx(F0)
    # The resolution the record quotes is one over what was read, not one over
    # what was captured: the ends are cut before anything is projected.
    assert held.rates_are_separated_by == pytest.approx(1.0 / held.sounded_s)


def test_a_swept_delay_turns_every_partial_in_proportion_to_its_own_frequency() -> None:
    dry = tone()
    wet = partials.swept(dry, SR, 3.0, 2.0, 1.0, 1.0)
    held = read(wet, keep=read(dry).kept)
    said = partials.excursion(held, 1.0)
    assert said["excursion_ms"] == pytest.approx(2.0, rel=0.1)
    assert said["what_the_phase_looks_like"] == "a swept delay"


def test_the_same_angle_at_every_partial_is_not_read_as_a_delay() -> None:
    dry = tone()
    held = read(turned(dry, hz=1.0, radians=0.8), keep=read(dry).kept)
    said = partials.excursion(held, 1.0)
    assert said["as_a_flat_turn"] < said["as_a_delay"]
    assert said["what_the_phase_looks_like"] != "a swept delay"


def test_a_level_swing_and_a_comb_are_told_apart_by_whether_the_partials_agree() -> None:
    """The one separation the comb fit cannot make for itself.

    A plain level modulation is fitted as a comb explaining more of its own series
    than a real comb does, so the mechanism has to be settled before the fit is
    read at all.
    """
    dry = tone()
    keep = read(dry).kept
    t = np.arange(dry.size) / SR
    shaken = read(dry * (1.0 + 0.5 * np.sin(2 * np.pi * 1.0 * t)), keep=keep)
    combed = read(partials.swept(dry, SR, 3.0, 2.0, 1.0, 0.3), keep=keep)
    assert partials.together(shaken)["lowest_agreement"] > efxpartials.AGREES_AS_ONE_VOICE
    assert partials.together(combed)["lowest_agreement"] <= efxpartials.AGREES_AS_ONE_VOICE


def test_a_rate_is_placed_against_the_take_with_nothing_in_the_path() -> None:
    dry = tone()
    keep = read(dry).kept
    grid = grid_of()
    floor = partials.project(read(dry, keep=keep).phase, read(dry, keep=keep).at, grid)
    wet = read(partials.swept(dry, SR, 3.0, 1.0, 1.3, 1.0), keep=keep)
    found = partials.peak(wet.phase, wet.at, grid, floor)
    assert found["hz"] == pytest.approx(1.3, abs=0.05)
    assert found["stands_over_bypassed"] > efxpartials.STANDS_OVER_BYPASSED


def test_a_rate_at_the_bottom_of_the_grid_is_reported_as_the_bottom_of_the_grid() -> None:
    """A slow drift with no modulator in it, read on a grid that starts too low.

    This is the reading that got published: a projection walked from 0.2 Hz over a
    take six and three quarter seconds long reported 0.2 Hz on thirteen types, and
    nothing beside the figure said that 0.2 was the lowest rate looked at or that
    two cycles of it do not fit in the take. A drifting tone puts its energy at the
    bottom of any grid, so the peak lands on the boundary and stands enormously
    over the bypassed take while naming no modulator at all.
    """
    seconds = 3.0
    t = np.arange(int(seconds * SR)) / SR
    # A monotonic drift and nothing periodic: the phase walks once across the take.
    drifting = sum(
        np.sin(2 * np.pi * F0 * k * t + k + k * 0.6 * t / seconds) / k for k in range(1, 9)
    )
    grid = grid_of(lo=0.2)
    floor = efxpartials.Floor(tone(seconds), SR, carrier_hz=F0, grid=grid)
    found = efxpartials.measure_one(drifting, SR, carrier_hz=F0, floor=floor, fit_comb=False)
    assert found is not None
    assert found["slowest_measurable_hz"] == pytest.approx(2.0 / found["sounded_s"], rel=1e-3)
    # The drift reports a rate, it stands over the bypassed take, and there is no
    # modulator in the signal at all. What marks it is the figure beside it.
    assert found["phase"]["peak"]["hz"] < found["slowest_measurable_hz"]
    against = found["stands_on_the_edge_of_the_search"]
    assert against is not None
    assert any("two cycles" in line for line in against)


def test_an_idle_input_is_not_the_unit_s_other_output() -> None:
    """The rule this replaces would have read a rate out of a lead nobody plugged in.

    On the rig this was measured on, the interface's unused inputs sit around
    thirty decibels under the unit's own pair and are not silent, so "the loudest
    channel that is not the one read" names one of them. The unit's two outputs
    reached within 1.4 dB of each other.
    """
    reached = [-69.9, -75.1, -37.6, -39.0, -240.0, -240.0]
    assert efxpartials._the_other_output(2, reached) == 3
    assert efxpartials._the_other_output(3, reached) == 2
    # The same rig with one lead out: the question is not answered off an idle input.
    assert efxpartials._the_other_output(2, [-69.9, -75.1, -37.6, -240.0]) is None
    assert efxpartials._the_other_output(0, [-37.6, -39.0]) == 1


def test_the_lowest_rate_searched_is_named_as_the_lowest_rate_searched() -> None:
    """Largest at the boundary means the largest seen, not the largest there is.

    Eight of the thirteen readings arrived this way: exactly on the grid's first
    point, which is where a maximum outside the search lands.
    """
    grid = grid_of(lo=0.2, hi=8.0)
    assert efxpartials.on_the_edge(0.2, grid, slowest=0.1) == ["the lowest rate searched"]
    assert efxpartials.on_the_edge(8.0, grid, slowest=0.1) == ["the highest rate searched"]
    assert efxpartials.on_the_edge(1.3, grid, slowest=0.1) is None
    assert efxpartials.on_the_edge(None, grid, slowest=0.1) is None


def test_a_rate_the_take_can_carry_is_not_called_an_edge() -> None:
    """The guard above must not put a caveat on the readings that are fine."""
    dry = tone()
    keep = read(dry).kept
    grid = grid_of()
    floor = efxpartials.Floor(dry, SR, carrier_hz=F0, grid=grid)
    floor.keep = keep
    wet = partials.swept(dry, SR, 3.0, 1.0, 1.3, 1.0)
    found = efxpartials.measure_one(wet, SR, carrier_hz=F0, floor=floor, fit_comb=False)
    assert found is not None
    assert found["phase"]["peak"]["hz"] == pytest.approx(1.3, abs=0.05)
    assert found["stands_on_the_edge_of_the_search"] is None


def test_a_take_with_nothing_in_it_does_not_stand_over_itself() -> None:
    """The floor is what a peak has to beat, and against itself it is one."""
    dry = tone()
    held = read(dry)
    grid = grid_of()
    floor = partials.project(held.phase, held.at, grid)
    found = partials.peak(held.phase, held.at, grid, floor)
    assert found["stands_over_bypassed"] == pytest.approx(1.0, abs=1e-6)


def test_a_fully_wet_path_has_no_comb_and_the_fit_says_so() -> None:
    """The case the phase route answers, which this one has to refuse.

    A delayed path with no direct one beside it puts out no comb at all, so a
    number here would be a number for a structure that is not there.
    """
    wet = partials.swept(tone(seconds=2.0), SR, 3.0, 2.0, 1.0, 1.0)
    said = partials.comb(wet, SR, 1.0)
    assert said["excursion_ms"] is None
    assert "no level swing" in said["why"]


def test_a_mixed_comb_gives_up_its_excursion_and_its_mix() -> None:
    dry = tone(seconds=2.0)
    said = partials.comb(partials.swept(dry, SR, 3.0, 2.0, 1.0, 0.25), SR, 1.0)
    assert said["excursion_ms"] == pytest.approx(2.0, rel=0.1)
    assert said["mix"] == pytest.approx(0.25, abs=0.1)
    assert said["explains"] > efxpartials.COMB_EXPLAINS


def test_the_excursion_carries_the_range_that_explains_the_series_as_well() -> None:
    """One excursion is the deepest minimum, not the only one that fits.

    Where the notches sweep far enough to shape the series the range closes onto
    the answer. Where they barely move it opens, and it has to open there: the
    returned figure scatters over exactly that range on material whose excursion
    is known, so a reader given the number alone would be given a precision the
    take does not hold.
    """
    dry = tone(seconds=2.0)
    far = partials.comb(partials.swept(dry, SR, 12.0, 5.0, 1.0, 0.35), SR, 1.0)
    low, high = far["excursions_that_explain_it_about_as_well_ms"]
    assert low <= far["excursion_ms"] <= high
    assert high <= 1.1 * low, "a five millisecond sweep settles on one excursion"

    near = partials.comb(partials.swept(dry, SR, 12.0, 0.3, 1.0, 0.35), SR, 1.0)
    low, high = near["excursions_that_explain_it_about_as_well_ms"]
    assert high > 1.5 * low, "a sweep this shallow does not settle, and must say so"


def test_every_seed_is_refined_even_when_another_seed_fills_the_pool() -> None:
    """The coarse scan cannot rank seeds against each other, so it must not try.

    It holds the mix at three values, so a seed whose minimum needs a mix between
    two of them scores badly coarsely and well once refined. Pooling every seed's
    rows and refining the best of the pool left such a seed with no refinement at
    all, and the fit reported a rate the take is not running at.
    """
    dry = tone(seconds=2.0)
    body = partials.swept(dry, SR, 12.0, 5.0, 0.9, 0.5)
    crowded = partials.comb(body, SR, 4.5, 2.25, 1.5, 1.125, 0.9, 0.45)
    assert crowded["hz"] == pytest.approx(0.9, rel=0.05)
    assert crowded["excursion_ms"] == pytest.approx(5.0, rel=0.1)


def test_the_room_around_the_gate_is_read_off_the_run_and_not_typed_in() -> None:
    """How much the gate's exact figure decided is a fact about the run, not the gate.

    Typed into the prose as a constant it is right for the run it was read on and
    silently wrong for the next, which is the one sentence in a limits block a reader
    has no way to check. The quantity is the empty band around the line, not which
    side a reading fell: the line is what puts it there.
    """
    roomy = [
        {"comb": {"explains": 0.9, "excursion_ms": 2.0}},
        {"comb": {"explains": 0.4, "excursion_ms": None}},
    ]
    said = efxpartials.how_far_the_line_could_have_moved(roomy)
    assert "0.400" in said and "0.900" in said
    assert f"{efxpartials.COMB_EXPLAINS - 0.4:.3f}" in said

    crowded = [
        {"comb": {"explains": 0.61, "excursion_ms": 2.0}},
        {"comb": {"explains": 0.59, "excursion_ms": None}},
    ]
    assert "0.010" in efxpartials.how_far_the_line_could_have_moved(crowded)

    one_sided = [{"comb": {"explains": 0.9, "excursion_ms": 2.0}}]
    assert "one side" in efxpartials.how_far_the_line_could_have_moved(one_sided)


def test_the_seeds_offered_to_the_comb_are_the_submultiples_of_both_peaks() -> None:
    """A peak sits on a whole multiple of the rate, so the rate is enumerated.

    A comb dragging several notches past a partial in one cycle puts its largest
    peak at twice or three times the modulator. Searching for the rate across that
    does not work -- the surface is multimodal in it -- so the candidates are
    listed and the fit's own residual picks.
    """
    seeds = efxpartials.seeds_from({"hz": 1.35}, {"hz": 0.9})
    assert 0.45 in seeds
    assert 1.35 in seeds
    assert all(s > 0 for s in seeds)


def test_a_pattern_that_reads_the_stimulus_instead_of_the_type_is_not_silent(tmp_path) -> None:
    """The one way this survey can report a short list of verdicts as the whole set.

    A file name carries the stimulus before the setting, so an unanchored pattern
    matches the stimulus and the byte after it: every take of `held-16` whose type
    begins `01` reads as `16 01`, they land on one key, and nothing fails. The
    shipped default is anchored, and a pattern that collides anyway says which
    files collided.
    """
    from scipy.io import wavfile

    for name in ("held-16-bypassed-00.wav", "held-16-01-40-00.wav", "held-16-01-41-00.wav"):
        wavfile.write(tmp_path / name, SR, tone(seconds=0.5).astype(np.float32))

    loose = efxpartials.rows_of(
        tmp_path, re.compile(r"(?P<type>[0-9A-Fa-f]{2}-[0-9A-Fa-f]{2})"), "bypassed"
    )
    assert list(loose["types"]) == ["16 01"]
    assert loose["more_than_one_take_named"]["16 01"] == [
        "held-16-01-40-00.wav",
        "held-16-01-41-00.wav",
    ]

    from soundings.cli import build_parser

    sub = next(
        a
        for a in build_parser()._actions
        if hasattr(a, "choices") and "efx-partials" in (a.choices or {})
    )
    shipped = sub.choices["efx-partials"].get_default("setting")
    anchored = efxpartials.rows_of(tmp_path, re.compile(shipped), "bypassed")
    assert list(anchored["types"]) == ["01 40", "01 41"]
    assert anchored["more_than_one_take_named"] == {}


def test_takes_the_manifest_forgot_are_read_and_named(tmp_path) -> None:
    """A store rewrites its manifest when it closes, so the files are the subject.

    A record built from a shortened manifest reads exactly like a complete one,
    which is the one way this survey could report a short list of verdicts as the
    whole set.
    """
    from scipy.io import wavfile

    for name in ("held-bypassed-00.wav", "held-01-40-00.wav", "held-01-41-00.wav"):
        wavfile.write(tmp_path / name, SR, tone(seconds=0.5).astype(np.float32))
    (tmp_path / "takes-manifest.json").write_text(
        '{"takes": [{"file": "held-01-41-00.wav", "setting": "01-41"}]}\n'
    )
    where = efxpartials.rows_of(
        tmp_path, re.compile(r"(?P<type>[0-9A-F]{2}-[0-9A-F]{2})"), "bypassed"
    )
    assert where["types"] == {"01 40": "held-01-40-00.wav", "01 41": "held-01-41-00.wav"}
    assert where["controls"] == ["held-bypassed-00.wav"]
    assert set(where["manifest"]["not_listed"]) == {
        "held-bypassed-00.wav",
        "held-01-40-00.wav",
    }
