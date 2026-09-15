"""Controls for reading a parameter as a profile rather than as a number.

Every failure this guards against reads as a measurement. A floor computed from
one take is zero, so every deviation clears it and a record of pure noise looks
like a record of a filter. A reference taken from the wrong takes shifts every
profile by a constant and the sweep still looks monotonic. A sweep pattern that
also matches the reference folds the flat takes into the readings and the curve
grows a value that never moved. None of these raise; they publish.

The takes here are synthetic, so the answer is known: a band given a known
attenuation has to come back at that band and at that figure, and a band given
nothing has to come back inside the floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from soundings import efxbands, takes

SR = 48000
SECONDS = 3.0
BANDS = (250, 1000, 4000)


@dataclass
class FakeRecording:
    samples: np.ndarray
    sample_rate: int = SR
    device: str = "test input"
    overflows: int = 0
    open_seconds: float = 0.6

    @property
    def seconds(self) -> float:
        return self.samples.shape[0] / self.sample_rate


def noise(seed: int, *, cut_at: float | None = None, by_db: float = 0.0) -> np.ndarray:
    """Noise, optionally with one third octave scaled by a known amount.

    Shaped in the frequency domain so the attenuation is exactly the figure asked
    for over exactly the band asked for -- the reading has to return that figure,
    and a reading that returns most of it is a reading with a leak in it.
    """
    rng = np.random.default_rng(seed)
    body = rng.standard_normal(int(SECONDS * SR)) * 0.1
    if cut_at is not None:
        spectrum = np.fft.rfft(body)
        freq = np.fft.rfftfreq(body.size, 1.0 / SR)
        inside = (freq >= cut_at / 2 ** (1 / 6)) & (freq < cut_at * 2 ** (1 / 6))
        spectrum[inside] *= 10.0 ** (by_db / 20.0)
        body = np.fft.irfft(spectrum, n=body.size)
    return np.stack([body, body * 0.5], axis=1)


LONG = 12.0
"""How long a take the shapes below are put into.

Longer than the takes above because these are read a twelfth of an octave at a
time, and how well a noise stimulus repeats itself in a band is how many bins the
band holds. Over a take the length of the others, the lowest of the set below
holds about ninety of them and fails to repeat by half a decibel, which is the
size of the differences these tests are about.
"""


def shaped(seed: int, gain_db, seconds: float = LONG) -> np.ndarray:
    """Noise with a known gain applied at every frequency.

    The profile is put in exactly, so what the reading returns can be held against
    the shape rather than against another reading of it: a band set that averages
    a deviation away returns a figure, and only a known input says it is the wrong
    one.
    """
    rng = np.random.default_rng(seed)
    body = rng.standard_normal(int(seconds * SR)) * 0.1
    spectrum = np.fft.rfft(body)
    freq = np.maximum(np.fft.rfftfreq(body.size, 1.0 / SR), 1.0)
    spectrum *= 10.0 ** (np.asarray(gain_db(freq), dtype=float) / 20.0)
    body = np.fft.irfft(spectrum, n=body.size)
    return np.stack([body, body * 0.5], axis=1)


def between(low: float, high: float, by_db: float):
    """A gain of `by_db` from `low` to `high` and nothing outside it."""
    return lambda freq: np.where((freq >= low) & (freq < high), by_db, 0.0)


FINE = tuple(c for c in efxbands.TWELFTH_OCTAVES if 2000.0 <= c <= 16000.0)
SHAPED_AT = 8000.0
"""A twelfth-octave set around the band the shapes below are put in.

High in the set rather than low, because a band a twelfth of an octave wide holds
as many bins as its centre is high: what these tests are about is a difference of
a decibel between two bands, and at the bottom of the set a noise stimulus fails
to repeat itself by more than that.
"""


@pytest.fixture
def profiles(tmp_path):
    """A run whose sweep is four known shapes rather than one band.

    Every take carries the reference takes' own noise, so the difference between a
    take and the reference is the shape and nothing else.
    """
    store = takes.Store.open(tmp_path / "shapes")
    for index in range(4):
        store.keep(
            FakeRecording(shaped(index, lambda freq: np.zeros_like(freq))),
            stimulus="held",
            setting=f"flat-{index:02d}",
            take=0,
        )
    narrow = 2 ** (1 / 24)
    wide = 2 ** 0.5
    for value, gain in (
        # Nothing done to it, and a draw of its own rather than one of the repeats
        # above: a repeat read against the mean of the others differs from it by no
        # more than the spread the floor came from, so it could never clear that
        # floor and would say the reading was cleaner than it is.
        (0, lambda freq: np.zeros_like(freq)),
        # Narrower than a third octave and wider than a twelfth: one set can see
        # how deep it is and the other cannot.
        (1, between(SHAPED_AT / narrow, SHAPED_AT * narrow, -12.0)),
        # An octave wide, so the deviation comes back on both sides inside the set.
        (2, between(SHAPED_AT / wide, SHAPED_AT * wide, -12.0)),
        # Down to the bottom of the set and level there, which is what a profile
        # that never comes back looks like, and is not a narrow one.
        (3, lambda freq: np.where(freq < SHAPED_AT, -12.0, 0.0)),
        # Still deepening where the set ends, which is the other way the largest
        # figure can be the band set's rather than the effect's. Twelve decibels
        # to the octave, so a third of one is four.
        (4, lambda freq: np.clip(12.0 * np.log2(freq / FINE[-1]), -48.0, 0.0)),
    ):
        store.keep(
            FakeRecording(shaped(9 if value == 0 else 0, gain)),
            stimulus="held",
            setting=f"04-{value:03d}",
            take=0,
        )
    store.close(question="a run of known shapes")
    return tmp_path / "shapes"


def profile(where, value: int, **extra):
    found = efxbands.read_directory(
        where,
        type_id="01 00",
        address="40 03 04",
        setting=r"held-04-(?P<value>\d+)-00",
        reference=r"held-flat-\d+-00",
        **{"bands_hz": FINE, "band_width_octaves": 1 / 12, "hold_s": LONG - 1.2, **extra},
    )
    return next(r for r in found["readings"] if r["value"] == value)


def test_a_deviation_narrower_than_a_band_is_read_shallower_than_it_is(profiles) -> None:
    """The whole reason the same takes are read at a second resolution.

    Twelve decibels taken out of a twelfth of an octave is twelve decibels. Read
    through a band four times wider it is averaged with the untouched bins beside
    it and comes back as about one, which is an ordinary looking figure with
    nothing about it to say the band was the wrong size.
    """
    fine = profile(profiles, 1)
    coarse = profile(
        profiles, 1, bands_hz=(2000, 4000, 8000, 16000), band_width_octaves=1 / 3
    )
    assert fine["largest_db"] == pytest.approx(-12.0, abs=1.0)
    assert coarse["largest_db"] == pytest.approx(-1.2, abs=0.5)


def test_the_half_points_bracket_the_part_that_was_shaped(profiles) -> None:
    found = profile(profiles, 2)
    assert found["half_below_hz"] == pytest.approx(SHAPED_AT / 2**0.5, rel=0.06)
    assert found["half_above_hz"] == pytest.approx(SHAPED_AT * 2**0.5, rel=0.06)


def test_a_profile_that_never_comes_back_says_so_rather_than_naming_a_band(
    profiles,
) -> None:
    """A null half point and a settled end are one answer, not a missing one."""
    found = profile(profiles, 3)
    assert found["half_below_hz"] is None
    assert found["half_above_hz"] == pytest.approx(SHAPED_AT, rel=0.06)
    assert found["settled_below_db"] == pytest.approx(0.0, abs=0.5)


def test_a_profile_still_deepening_where_the_bands_end_says_how_much(profiles) -> None:
    """Otherwise the largest figure reads as the effect's own and is the set's edge."""
    found = profile(profiles, 4)
    assert found["half_below_hz"] is None
    # A third of an octave of a slope running twelve decibels to the octave.
    assert found["settled_below_db"] == pytest.approx(-4.0, abs=0.7)


def test_a_slope_of_a_known_rate_is_read_at_that_rate(profiles) -> None:
    """The figure the shelves are read with, checked against a shape that has one."""
    found = profile(profiles, 4)
    assert found["steepest_db_per_octave"] == pytest.approx(12.0, abs=1.0)
    # Not at either end: the outermost half window has no symmetric window to fit.
    assert FINE[0] < found["steepest_at_hz"] < FINE[-1]


def test_a_deviation_narrower_than_the_window_is_read_slower_than_it_ran(profiles) -> None:
    """The band width lesson again, one resolution up.

    The same twelve decibels: one of them spread over an octave, the other taken
    out in a twelfth of one and put back. The second ran far faster than the first
    and is reported as running far slower, because a window an octave wide averages
    the whole of it with the flat either side.
    """
    steep = profile(profiles, 4)["steepest_db_per_octave"]
    notch = profile(profiles, 1)["steepest_db_per_octave"]
    assert abs(notch) < 0.5 * abs(steep)


def test_the_window_a_slope_was_fitted_over_is_published_beside_it(profiles) -> None:
    """Two resolutions of the same takes are fitted over two windows, not one."""
    fine = profile(profiles, 4)
    coarse = profile(profiles, 4, bands_hz=FINE[::4], band_width_octaves=1 / 3)
    assert fine["steepest_over_octaves"] == pytest.approx(1.0, abs=0.01)
    assert coarse["steepest_over_octaves"] == pytest.approx(2 / 3, abs=0.01)
    # A rate that does not change with the window, because this shape holds the
    # same rate everywhere. One that changed with it would be the window's.
    assert coarse["steepest_db_per_octave"] == pytest.approx(
        fine["steepest_db_per_octave"], abs=1.5
    )


# ---- where a feature sits, off the whole feature


def lorentzian(centre: float, width_octaves: float, by_db: float):
    """A peak of `by_db` at `centre`, half of it `width_octaves` to each side.

    In log frequency rather than in hertz, which is where a filter's peak is
    symmetric and is the axis the fit works on. Near its top it is a parabola,
    which is the part of the shape the fit is entitled to, and further out it is
    not -- so a fit that reached past the half points would be fitting the wrong
    curve, and the window stops there.
    """
    def gain(freq):
        offset = np.log2(np.maximum(freq, 1.0) / centre) / width_octaves
        return by_db / (1.0 + offset**2)

    return gain


BETWEEN_BANDS = float(np.sqrt(FINE[len(FINE) // 2] * FINE[len(FINE) // 2 + 1]))
"""A peak put halfway between two band centres, which is the worst case for a band.

Nothing the takes do can bring `largest_at_hz` closer than half a band here, and
which of the two neighbours it names is decided by whichever way the noise fell.
"""


@pytest.fixture
def peaks(tmp_path):
    """A run whose sweep is peaks a fit can be held against."""
    store = takes.Store.open(tmp_path / "peaks")
    for index in range(4):
        store.keep(
            FakeRecording(shaped(index, lambda freq: np.zeros_like(freq))),
            stimulus="held",
            setting=f"flat-{index:02d}",
            take=0,
        )
    for value, gain in (
        # Halfway between two bands, and wide enough that several sit above half.
        (0, lorentzian(BETWEEN_BANDS, 0.25, 12.0)),
        # The same peak taken out instead of put in: a notch is the same question.
        (1, lorentzian(BETWEEN_BANDS, 0.25, -12.0)),
        # Narrow enough that the half points are two bands apart, so the window
        # holds fewer than the three a parabola needs.
        (2, lorentzian(BETWEEN_BANDS, 1 / 24, 12.0)),
    ):
        store.keep(
            FakeRecording(shaped(0, gain)), stimulus="held", setting=f"04-{value:03d}", take=0
        )
    store.close(question="a run of known peaks")
    return tmp_path / "peaks"


def peak(where, value: int, **extra):
    found = efxbands.read_directory(
        where,
        type_id="01 00",
        address="40 03 04",
        setting=r"held-04-(?P<value>\d+)-00",
        reference=r"held-flat-\d+-00",
        **{"bands_hz": FINE, "band_width_octaves": 1 / 12, "hold_s": LONG - 1.2, **extra},
    )
    return next(r for r in found["readings"] if r["value"] == value)


def test_a_peak_between_two_bands_is_fitted_nearer_than_either_of_them(peaks) -> None:
    """The reading the largest band cannot give however well the takes were made.

    A band names a band. Where the feature is between two of them the answer is
    half a band wrong at best, and the fit is entitled to land between them.
    """
    found = peak(peaks, 0)
    by_band = abs(np.log2(found["largest_at_hz"] / BETWEEN_BANDS))
    by_fit = abs(np.log2(found["fitted_at_hz"] / BETWEEN_BANDS))
    # Half a band is a twenty-fourth of an octave, and that is the best the band
    # can do rather than what it happened to do.
    assert by_band == pytest.approx(1 / 24, abs=0.005)
    assert by_fit < by_band / 3


def test_a_notch_is_fitted_the_same_way_a_peak_is(peaks) -> None:
    """Which way a byte moves the level is not the question this reading answers."""
    found = peak(peaks, 1)
    assert found["largest_db"] < 0
    assert abs(np.log2(found["fitted_at_hz"] / BETWEEN_BANDS)) < 1 / 72


def test_the_bands_a_position_was_fitted_over_are_published_beside_it(peaks) -> None:
    """A window half an octave wide, at a twelfth of an octave a band."""
    found = peak(peaks, 0)
    assert found["fitted_over_bands"] == pytest.approx(6, abs=1)


def test_a_feature_too_narrow_to_fit_says_how_narrow_rather_than_nothing(peaks) -> None:
    """Absent for want of width and absent for want of curvature are different."""
    found = peak(peaks, 2)
    assert found["fitted_at_hz"] is None
    assert found["fitted_over_bands"] is not None
    assert found["fitted_over_bands"] < 3


def test_a_feature_with_no_measured_width_is_not_fitted_at_all(profiles) -> None:
    """No half point on one side is no window, which is not a narrow window."""
    found = profile(profiles, 3)
    assert found["half_below_hz"] is None
    assert found["fitted_at_hz"] is None
    assert found["fitted_over_bands"] is None


def test_a_lopsided_feature_pulls_the_fit_towards_its_longer_flank() -> None:
    """The bias this reading has and the largest band does not.

    A symmetric curve fitted to an asymmetric feature does not top out where the
    feature does, and nothing in the figure says so. It is why the record keeps
    both readings: this one is finer and biased, that one is coarse and is not,
    and which is wanted depends on whether what it will be held against went
    through the same reading.
    """
    centres = [1000.0 * 2 ** (j / 12) for j in range(17)]
    top = 8
    spot = np.log2(np.asarray(centres) / centres[top])
    # Half as steep below the top as above it, so the half point below is twice as
    # far away and twice as many bands sit on that side.
    moved = [float(12.0 - (16.0 if s > 0 else 4.0) * s**2) for s in spot]
    span = {"half_below_hz": centres[0], "half_above_hz": centres[-1]}
    found = efxbands._fitted(moved, centres, 12.0, centres[top], span)
    assert found["fitted_at_hz"] < centres[top]


def test_a_window_that_holds_a_slope_and_not_a_top_is_refused() -> None:
    """A parabola through a slope has a top, and it is nowhere near the reading."""
    centres = [1000.0 * 2 ** (j / 12) for j in range(9)]
    # Rising the whole way, so the largest is the last band and there is no top
    # inside the window at all.
    moved = [float(j) for j in range(9)]
    span = {"half_below_hz": centres[0], "half_above_hz": None}
    assert efxbands._fitted(moved, centres, 8.0, centres[-1], span)["fitted_at_hz"] is None
    # And the same window with both ends named still refuses, on the bend rather
    # than on the width: a rising line curves neither way, and what little it does
    # curve is as likely to open upwards as down.
    bowl = {"half_below_hz": centres[0], "half_above_hz": centres[-1]}
    dipped = [4.0, 2.0, 1.0, 0.5, 0.0, 0.5, 1.0, 2.0, 4.0]
    assert efxbands._fitted(dipped, centres, 4.0, centres[0], bowl)["fitted_at_hz"] is None


def test_a_top_that_falls_outside_the_bands_it_was_fitted_over_is_refused() -> None:
    """An extrapolation is not a reading, however well the curve fitted."""
    centres = [1000.0 * 2 ** (j / 12) for j in range(7)]
    # A parabola whose top is well above the last band: inside the window it is
    # only ever rising, so the fit is good and the vertex is somewhere else.
    spot = np.log2(np.asarray(centres)) - np.log2(centres[-1]) - 1.0
    moved = [float(12.0 - 4.0 * s**2) for s in spot]
    span = {"half_below_hz": centres[0], "half_above_hz": centres[-1]}
    assert efxbands._fitted(moved, centres, 12.0, centres[-1], span)["fitted_at_hz"] is None


def test_a_reading_inside_the_floor_has_no_span_to_report(directory) -> None:
    found = read(directory)
    flat = next(r for r in found["readings"] if r["value"] == 64)
    assert flat["half_below_hz"] is None
    assert flat["settled_below_db"] is None
    assert flat["steepest_db_per_octave"] is None
    # This run's bands are an octave apart, so no window fits inside one centred on
    # a band. A set too coarse to hold the window says so rather than quietly
    # fitting a narrower one and reporting it as the same figure.
    assert flat["steepest_over_octaves"] is None


def test_a_setting_with_nothing_done_to_it_reads_far_below_one_with_a_shape(
    profiles,
) -> None:
    """What the floor does not bound, at the resolution that needs it bounded.

    `floor_db` is the spread of a handful of repeats, band by band, so a band where
    those repeats happened to agree has a small floor and a deviation just outside
    it is not thereby a reading. A take that holds nothing and was not one of the
    repeats is the figure that says how much the finer set returns anyway.
    """
    nothing = profile(profiles, 0)
    shape = profile(profiles, 2)
    assert abs(nothing["largest_db"] or 0.0) < 0.25 * abs(shape["largest_db"])


def test_every_band_set_the_command_line_offers_is_one_the_reader_has() -> None:
    """Spelled in two files so the parser need not load the reader, held together here."""
    from soundings.cli import blocks

    assert set(blocks.BAND_SET_NAMES) == set(efxbands.BAND_SETS)
    for centres, width in efxbands.BAND_SETS.values():
        assert width > 0 and len(centres) == len(set(centres))


@pytest.fixture
def directory(tmp_path):
    """A run's worth of takes: four repeats of flat, a control, and a sweep."""
    store = takes.Store.open(tmp_path / "run")
    for index in range(4):
        store.keep(FakeRecording(noise(index)), stimulus="held", setting=f"flat-{index:02d}", take=0)
    # The control and every swept take carry one of the reference takes' own
    # noise, so that the only difference between them is the band this shaped on
    # purpose. Independent noise would put the untouched bands on either side of a
    # floor drawn from four other draws, and the assertions below would pass or
    # fail on the seed rather than on the reading.
    store.keep(FakeRecording(noise(1)), stimulus="held", setting="bypassed-00", take=0)
    for value, by_db in ((0, -12.0), (64, 0.0), (127, 6.0)):
        store.keep(
            FakeRecording(noise(0, cut_at=1000, by_db=by_db)),
            stimulus="held",
            setting=f"04-{value:03d}",
            take=0,
        )
    # The same chain with nothing played, forty decibels down. A setting that
    # turns the output off lands here, and its bands are this shape.
    for index in range(2):
        store.keep(
            FakeRecording(noise(200 + index) * 0.01),
            stimulus="held",
            setting=f"silence-{index:02d}",
            take=0,
        )
    store.close(question="a synthetic run")
    return tmp_path / "run"


def read(where, **extra):
    return efxbands.read_directory(
        where,
        type_id="01 00",
        address="40 03 04",
        setting=r"held-04-(?P<value>\d+)-00",
        reference=r"held-flat-\d+-00",
        **{
            "control": r"held-bypassed-\d+-00",
            "silence": r"held-silence-\d+-00",
            "bands_hz": BANDS,
            "hold_s": SECONDS - 1.2,
            **extra,
        },
    )


def test_the_band_that_was_cut_is_the_band_that_comes_back(directory) -> None:
    found = read(directory)
    by_value = {r["value"]: r for r in found["readings"]}
    assert sorted(by_value) == [0, 64, 127]
    assert by_value[0]["largest_at_hz"] == 1000
    assert by_value[0]["largest_db"] == pytest.approx(-12.0, abs=0.5)
    assert by_value[127]["largest_at_hz"] == 1000
    assert by_value[127]["largest_db"] == pytest.approx(6.0, abs=0.5)


def test_a_setting_that_did_nothing_reports_nothing_rather_than_a_small_number(
    directory,
) -> None:
    """The floor is the point. Without it the same take reads as a tiny tilt."""
    found = read(directory)
    flat = next(r for r in found["readings"] if r["value"] == 64)
    assert flat["outside_the_floor_hz"] == []
    assert flat["largest_db"] is None
    assert all(edge > 0 for edge in found["reference"]["floor_db"])


def test_the_bands_that_were_not_touched_stay_inside_the_floor(directory) -> None:
    found = read(directory)
    cut = next(r for r in found["readings"] if r["value"] == 0)
    assert cut["outside_the_floor_hz"] == [1000]


def test_the_reference_takes_are_not_also_read_as_settings(directory) -> None:
    """A sweep pattern that reaches the flat takes grows a setting nobody swept."""
    found = read(directory)
    names = {r["take"] for r in found["readings"]}
    assert not names & set(found["reference"]["takes"])
    assert not names & set(found["control"]["takes"])


def test_the_control_is_read_against_the_same_reference(directory) -> None:
    found = read(directory)
    assert len(found["control"]["readings"]) == 1
    assert found["control"]["readings"][0]["outside_the_floor_hz"] == []


def test_a_setting_that_turned_the_output_off_says_how_far_above_silence_it_was(
    directory,
) -> None:
    """The trap this control exists for.

    A level slot at its bottom byte records the room rather than the effect, and
    read as a deviation from the flat setting that is a large, ragged, frequency
    dependent profile -- which looks exactly like a reading. The number that tells
    the two apart is how far the take was above a take with nothing played.
    """
    store = takes.Store.open(directory)
    store.keep(
        FakeRecording(noise(7) * 0.01), stimulus="held", setting="04-999", take=0
    )
    found = read(directory)
    off = next(r for r in found["readings"] if r["value"] == 999)
    loud = next(r for r in found["readings"] if r["value"] == 127)
    assert off["above_the_silence_db"] == pytest.approx(0.0, abs=1.0)
    assert loud["above_the_silence_db"] > 30.0
    assert found["silence"]["heard_db"] < found["reference"]["heard_db"]


def test_the_floors_own_profile_is_published_rather_than_only_its_level(
    directory,
) -> None:
    """So a reading suspected of being the floor can be held against the floor."""
    found = read(directory)
    assert len(found["silence"]["takes"]) == 2
    assert len(found["silence"]["band_db"]) == len(BANDS)
    assert found["reference"]["above_the_silence_db"] > 30.0


def test_without_silence_takes_the_record_says_so_rather_than_guessing(directory) -> None:
    found = efxbands.read_directory(
        directory,
        type_id="01 00",
        address="40 03 04",
        setting=r"held-04-(?P<value>\d+)-00",
        reference=r"held-flat-\d+-00",
        bands_hz=BANDS,
        hold_s=SECONDS - 1.2,
    )
    assert found["silence"]["takes"] == []
    assert found["silence"]["band_db"] is None
    assert all(r["above_the_silence_db"] is None for r in found["readings"])


def test_a_take_no_pattern_named_is_counted_rather_than_dropped(directory) -> None:
    store = takes.Store.open(directory)
    store.keep(FakeRecording(noise(9)), stimulus="held", setting="strays-000", take=0)
    found = read(directory)
    assert found["takes_not_matching"]["count"] == 1
    assert any("stray" in name for name in found["takes_not_matching"]["settings"])


def test_a_reference_that_matches_nothing_refuses_rather_than_returns(directory) -> None:
    """Left to compute, an empty reference is a mean of nothing and a floor of zero."""
    with pytest.raises(ValueError, match="reference"):
        efxbands.read_directory(
            directory,
            type_id="01 00",
            address="40 03 04",
            setting=r"held-04-(?P<value>\d+)-00",
            reference=r"held-nothing-\d+-00",
            bands_hz=BANDS,
            hold_s=SECONDS - 1.2,
        )


def test_a_setting_pattern_without_the_value_group_refuses(directory) -> None:
    with pytest.raises(ValueError, match="value"):
        efxbands.read_directory(
            directory,
            type_id="01 00",
            address="40 03 04",
            setting=r"held-04-\d+-00",
            reference=r"held-flat-\d+-00",
            bands_hz=BANDS,
            hold_s=SECONDS - 1.2,
        )


def test_the_record_says_where_it_was_read_from_and_what_was_held(directory) -> None:
    found = read(directory, held=[{"address": "40 03 06", "bytes": "40"}])
    assert found["type"] == "01 00"
    assert found["address"] == "40 03 04"
    assert found["bands_hz"] == list(BANDS)
    assert found["held"] == [{"address": "40 03 06", "bytes": "40"}]
    assert str(directory) == found["takes_from"]
    assert found["manifest"]["files_present"] >= 8


def test_the_reference_says_what_it_was_taken_under(directory) -> None:
    """The reference is a state, and on some runs it is a state of the swept byte.

    A printed range whose last position is the stage switched out gives the null as
    one value of the byte being read, so the block that says what was held while
    sweeping cannot carry it -- that value is not what the readings were taken at.
    Without somewhere of its own, every profile in such a record is reported
    against something the record cannot name.
    """
    found = read(
        directory,
        held=[{"address": "40 03 06", "bytes": "40"}],
        reference_held=[{"address": "40 03 04", "bytes": "7F"}],
    )
    assert found["reference"]["held"] == [{"address": "40 03 04", "bytes": "7F"}]
    assert found["held"] == [{"address": "40 03 06", "bytes": "40"}]


def test_a_reference_taken_under_the_sweeps_own_state_says_so_by_being_empty(
    directory,
) -> None:
    found = read(directory)
    assert found["reference"]["held"] == []
    assert found["reference"]["why_held"]


@pytest.fixture
def crossed(tmp_path):
    """A run on an interface whose second input carries something the unit is not.

    Channel 0 is the unit and channel 1 is another input, kept at a steady level
    the unit passes on its way down. One swept setting turns the unit's own output
    below it, which is where reading each take's loudest channel stops reading the
    unit and starts reading the other input.
    """
    other = np.random.default_rng(900).standard_normal(int(SECONDS * SR)) * 0.004

    def pair(unit: np.ndarray) -> np.ndarray:
        return np.stack([unit, other], axis=1)

    store = takes.Store.open(tmp_path / "run")
    for index in range(4):
        store.keep(FakeRecording(pair(noise(index)[:, 0])), stimulus="held",
                   setting=f"flat-{index:02d}", take=0)
    for value, scale in ((0, 0.0005), (64, 1.0), (127, 1.0)):
        store.keep(FakeRecording(pair(noise(0)[:, 0] * scale)), stimulus="held",
                   setting=f"16-{value:03d}", take=0)
    store.close(question="a synthetic run on a crowded interface")
    return tmp_path / "run"


def crossed_read(where, **extra):
    return efxbands.read_directory(
        where,
        type_id="01 00",
        address="40 03 16",
        setting=r"held-16-(?P<value>\d+)-00",
        reference=r"held-flat-\d+-00",
        bands_hz=BANDS,
        hold_s=SECONDS - 1.2,
        **extra,
    )


def test_the_record_names_the_channel_it_read_and_every_channel_it_saw(directory) -> None:
    found = read(directory)
    picked = found["channel"]
    assert picked["read"] == 0
    assert picked["chosen_by"] == "loudest in the reference takes"
    assert len(picked["reference_db"]) == 2
    # The fixture's second channel is half the first, which is 6 dB down.
    assert picked["reference_db"][0] - picked["reference_db"][1] == pytest.approx(6.0, abs=0.5)


def test_a_take_the_unit_went_quiet_in_is_still_read_from_the_units_channel(
    crossed,
) -> None:
    """The one that reads as a measurement. Reading each take's own loudest channel
    returns a full profile of the other input, which is not a number the byte
    produced and not a number anything in the record contradicts."""
    found = crossed_read(crossed)
    quiet = next(r for r in found["readings"] if r["value"] == 0)
    loud = next(r for r in found["readings"] if r["value"] == 127)
    # Read from the unit, the setting that turned it off is far below the others.
    assert quiet["heard_db"] < loud["heard_db"] - 40
    assert found["channel"]["read"] == 0


def test_a_take_whose_loudest_channel_is_elsewhere_is_named(crossed) -> None:
    found = crossed_read(crossed)
    astray = found["channel"]["loudest_elsewhere"]
    assert len(astray) == 1
    assert "16-000" in astray[0]


def test_a_channel_given_on_the_command_line_is_used_and_said_to_be(crossed) -> None:
    found = crossed_read(crossed, channel=1)
    assert found["channel"]["read"] == 1
    assert found["channel"]["chosen_by"] == "given"
    # Every reading is now of the other input, so the byte moves nothing.
    assert all(r["largest_db"] is None for r in found["readings"])


@pytest.fixture
def two_legs(tmp_path):
    """A run whose second channel is the unit's other output rather than an input.

    The fixture's takes carry the same shaping in both channels at half the level,
    so the second channel follows the parameter and disagrees with the first about
    nothing -- which is what a record has to be able to say, and what it cannot say
    from one channel.
    """
    store = takes.Store.open(tmp_path / "run")
    for index in range(4):
        store.keep(FakeRecording(noise(index)), stimulus="held", setting=f"flat-{index:02d}", take=0)
    for value, by_db in ((0, -12.0), (64, 0.0), (127, 6.0)):
        store.keep(
            FakeRecording(noise(0, cut_at=1000, by_db=by_db)),
            stimulus="held",
            setting=f"04-{value:03d}",
            take=0,
        )
    store.close(question="a synthetic run on two legs")
    return tmp_path / "run"


def test_the_second_channel_is_read_and_named(two_legs) -> None:
    found = read(two_legs, silence=None, control=None)
    beside = found["other_channel"]
    assert beside["read"] == 1
    assert len(beside["band_db"]) == len(BANDS)


def test_the_second_channel_follows_the_parameter_and_agrees_with_the_first(
    two_legs,
) -> None:
    """Both halves matter. That it moved says it carried the unit rather than an
    idle input; that it agrees says one channel was the whole answer here."""
    found = read(two_legs, silence=None, control=None)
    cut = next(r for r in found["readings"] if r["value"] == 0)
    assert cut["other_db"] == pytest.approx(-12.0, abs=0.5)
    assert abs(cut["apart_db"]) < 0.5


def test_a_second_channel_carrying_something_else_does_not_follow(crossed) -> None:
    """The control's own control. The other channel here is an unrelated input, so
    it stays put while the first swings, and the record shows exactly that."""
    found = crossed_read(crossed)
    assert found["other_channel"]["read"] == 1
    for reading in found["readings"]:
        assert abs(reading["other_db"]) < 1.0


def test_a_take_with_one_channel_reports_no_second_one(tmp_path) -> None:
    store = takes.Store.open(tmp_path / "run")
    mono = lambda seed: noise(seed)[:, :1]  # noqa: E731
    for index in range(4):
        store.keep(FakeRecording(mono(index)), stimulus="held", setting=f"flat-{index:02d}", take=0)
    for value in (0, 64):
        store.keep(FakeRecording(mono(0)), stimulus="held", setting=f"04-{value:03d}", take=0)
    store.close(question="a mono run")
    found = read(tmp_path / "run", silence=None, control=None)
    assert found["other_channel"]["read"] is None
    assert all("apart_db" not in r for r in found["readings"])


@pytest.fixture
def builds(tmp_path):
    """A run whose sweep does one thing early in each take and another late.

    The first half of every take is the reference's own noise, so a window over it
    reads nothing however the byte was set; the second half is cut at one band by
    an amount the byte chooses. A record read over the whole take averages the two
    and reports half of each, which is why the window exists.
    """
    half = int(SECONDS * SR / 2)

    def in_two(early: np.ndarray, late: np.ndarray) -> np.ndarray:
        return np.concatenate([early[:half], late[half:]])

    store = takes.Store.open(tmp_path / "run")
    for index in range(4):
        flat = noise(index)
        store.keep(
            FakeRecording(in_two(flat, flat)),
            stimulus="held",
            setting=f"flat-{index:02d}",
            take=0,
        )
    for value, by_db in ((0, -12.0), (64, -6.0)):
        store.keep(
            FakeRecording(in_two(noise(0), noise(0, cut_at=1000, by_db=by_db))),
            stimulus="held",
            setting=f"04-{value:03d}",
            take=0,
        )
    store.close(question="a synthetic run that changes half way through")
    return tmp_path / "run"


def windowed(where, window):
    return efxbands.read_directory(
        where,
        type_id="01 50",
        address="40 03 0A",
        setting=r"held-04-(?P<value>\d+)-00",
        reference=r"held-flat-\d+-00",
        bands_hz=BANDS,
        window=window,
    )


def at_1k(found, value: int) -> float:
    return found["readings"][[r["value"] for r in found["readings"]].index(value)]["band_db"][
        BANDS.index(1000)
    ]


def test_a_window_reads_the_part_of_the_take_it_names(builds) -> None:
    early = windowed(builds, (0.1, 1.2))
    late = windowed(builds, (1.6, 1.2))
    assert at_1k(early, 0) == pytest.approx(0.0, abs=0.5)
    assert at_1k(late, 0) == pytest.approx(-12.0, abs=0.5)
    assert at_1k(late, 64) == pytest.approx(-6.0, abs=0.5)


def test_reading_the_whole_take_averages_the_two_halves(builds) -> None:
    """Why a window is not a convenience. Whole, the twelve decibel cut reads as
    about three -- an ordinary looking figure that is neither of the two."""
    whole = at_1k(windowed(builds, None), 0)
    assert -6.0 < whole < -1.0


def test_the_record_says_which_window_and_still_says_the_hold(builds) -> None:
    found = windowed(builds, (1.6, 1.2))
    assert found["window_s"]["opens_at"] == 1.6
    assert found["window_s"]["wide"] == 1.2
    assert found["window_s"]["measured_from"] == "the start of the take"
    assert found["readings"][0]["hold_s"] == pytest.approx(SECONDS - 1.0)


def test_a_record_read_whole_carries_no_window(builds) -> None:
    assert "window_s" not in windowed(builds, None)
