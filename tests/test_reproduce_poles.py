"""The plain low and high pass section, and the noise floor a reading of one has."""

from __future__ import annotations

import numpy as np
import pytest

from soundings import reproduce

FS = 32000.0
CORNER = 1000.0


def _db(freq, **kwargs):
    rendered = reproduce._pole_cascade(np.asarray(freq, dtype=float), fs=FS, **kwargs)
    return 20.0 * np.log10(np.abs(rendered))


def _slope(low_hz, high_hz, **kwargs):
    """Decibels an octave between two frequencies, the way a profile is read."""
    a, b = _db([low_hz, high_hz], **kwargs)
    return (b - a) / np.log2(high_hz / low_hz)


@pytest.mark.parametrize("side", ["low", "high"])
def test_the_corner_is_where_the_section_is_three_decibels_down(side):
    assert _db([CORNER], side=side, corner_hz=CORNER, sections=1, q=None) == (
        pytest.approx(-3.0103, abs=0.01)
    )


@pytest.mark.parametrize("side", ["low", "high"])
def test_no_sections_is_not_a_wide_filter_but_no_filter(side):
    """The type byte's off state, and it has to be unity and not merely close.

    A section pushed out of the band would leave its zero at half the rate in the
    flat reference every profile is read against.
    """
    freq = np.geomspace(20.0, 15900.0, 40)
    assert _db(freq, side=side, corner_hz=CORNER, sections=0, q=None) == (
        pytest.approx(np.zeros(40), abs=1e-12)
    )


def test_one_pole_runs_at_six_decibels_an_octave_and_two_at_twelve():
    """Read well below half the rate, which is where a skirt is only a skirt.

    Four to eight kilohertz would not do it: at thirty-two the zero at Nyquist has
    already bent a single pole to seven and a half decibels an octave there, which
    is the reading the rate is taken from and not the order.
    """
    one = _slope(1000.0, 2000.0, side="low", corner_hz=100.0, sections=1, q=None)
    two = _slope(1000.0, 2000.0, side="low", corner_hz=100.0, sections=2, q=None)
    assert one == pytest.approx(-6.0, abs=0.3)
    assert two == pytest.approx(2 * one, rel=0.02)


def test_a_pair_and_two_cascaded_poles_share_a_skirt_and_differ_at_the_corner():
    """What separates the two second-order readings, and where it is not found."""
    far = dict(side="low", corner_hz=200.0)
    cascade = _slope(3000.0, 6000.0, sections=2, q=None, **far)
    pair = _slope(3000.0, 6000.0, sections=1, q=0.7071067811865476, **far)
    assert cascade == pytest.approx(pair, abs=0.2)

    at_corner = dict(side="low", corner_hz=CORNER)
    assert _db([CORNER], sections=2, q=None, **at_corner)[0] == pytest.approx(-6.02, abs=0.05)
    assert _db([CORNER], sections=1, q=0.7071067811865476, **at_corner)[0] == (
        pytest.approx(-3.01, abs=0.05)
    )


def test_a_low_pass_steepens_towards_half_the_rate_and_that_is_what_names_it():
    """The zero the bilinear transform leaves at Nyquist, which is the rate reading.

    Far below half the rate the section is its analogue self and says nothing
    about any clock. The same two frequencies read against a chain clocked higher
    are further from its Nyquist and so are shallower there.
    """
    middle = _slope(1000.0, 2000.0, side="low", corner_hz=200.0, sections=1, q=None)
    top = _slope(7500.0, 15000.0, side="low", corner_hz=200.0, sections=1, q=None)
    assert middle == pytest.approx(-6.0, abs=0.3)
    assert top < -9.0

    higher = reproduce._pole_cascade(
        np.array([7500.0, 15000.0]), side="low", corner_hz=200.0,
        sections=1, q=None, fs=48000.0,
    )
    ladder = 20.0 * np.log10(np.abs(higher))
    assert (ladder[1] - ladder[0]) > top + 2.0


def test_a_section_is_reachable_through_the_renderer_and_its_count_comes_off_a_byte():
    model = {
        "sample_rate_hz": FS,
        "chain": [
            {
                "kind": "pole",
                "side": "low",
                "sections": {
                    "byte": "40 03 03",
                    "map": {"kind": "states", "values": {"1": 1, "*": 0}},
                },
                "corner_hz": {"fixed": CORNER},
            }
        ],
    }
    freq = np.array([CORNER])
    on = reproduce.response(model, {"40 03 03": 1}, freq)
    off = reproduce.response(model, {"40 03 03": 0}, freq)
    assert 20.0 * np.log10(abs(on[0])) == pytest.approx(-3.0103, abs=0.01)
    assert off[0] == pytest.approx(1.0 + 0j)


def test_a_deep_cut_stops_where_the_chain_stops_and_a_shallow_one_is_untouched():
    """A model's band read through a chain whose own silence the record measured."""
    assert reproduce._over_the_chains_own_silence(-3.0, 40.0) == pytest.approx(-3.0, abs=0.01)
    assert reproduce._over_the_chains_own_silence(-60.0, 40.0) == pytest.approx(-40.0, abs=0.1)
    assert reproduce._over_the_chains_own_silence(+6.0, 40.0) == pytest.approx(6.0, abs=0.01)


def test_a_record_with_no_silence_takes_leaves_the_reading_alone():
    """Not a floor invented for it: such a record's deep cuts cannot be scored."""
    assert reproduce._over_the_chains_own_silence(-60.0, None) == -60.0
