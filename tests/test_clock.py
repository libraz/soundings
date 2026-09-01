"""Reading the unit's pitch, and what the report says when nothing held still."""

from __future__ import annotations

import pytest

from soundings import clock
from soundings.stability import ToneFit


def test_a440_is_a440():
    assert clock.equal_temperament(69) == pytest.approx(440.0)


def test_an_octave_up_doubles():
    assert clock.equal_temperament(81) == pytest.approx(880.0)


def test_master_tune_moves_the_target():
    """The unit's own tuning is part of what a note should sound at, not a departure from it."""
    assert clock.equal_temperament(69, 100.0) == pytest.approx(440.0 * 2 ** (1 / 12))
    assert clock.equal_temperament(69, 0.0) == pytest.approx(440.0)
    assert clock.equal_temperament(69, None) == pytest.approx(440.0)


def steady(hz: float, *, wobble: float = 0.001, seconds: float = 20.0) -> ToneFit:
    return ToneFit(frequency=hz, wobble_cycles=wobble, seconds=seconds)


def test_no_voice_read_is_reported_as_no_pitch_rather_than_as_zero():
    """An absent measurement must not be publishable as a measured value."""
    found = clock.Clock(note=69, expected_hz=440.0)
    assert found.to_json() is None
    assert found.describe() == ""


def test_departure_is_signed_parts_per_million():
    found = clock.Clock(note=69, expected_hz=440.0)
    assert found.departure_ppm(steady(440.0)) == pytest.approx(0.0)
    assert found.departure_ppm(steady(440.0044)) == pytest.approx(10.0, abs=0.1)
    assert found.departure_ppm(steady(439.9956)) == pytest.approx(-10.0, abs=0.1)


def test_voices_are_published_under_bank_and_program():
    found = clock.Clock(
        note=69,
        expected_hz=440.0,
        voices={(0, 16): steady(440.0), (8, 80): steady(440.022)},
    )
    published = found.to_json()
    assert set(published["voices"]) == {"0:16", "8:80"}
    assert published["voices"]["8:80"]["departure_ppm"] == pytest.approx(50.0, abs=0.5)
    assert published["caveat"] == clock.CAVEAT


def test_a_voice_that_wobbles_is_reported_as_not_steady_rather_than_dropped():
    """Its frequency is meaningless, but its absence has to be visible."""
    found = clock.Clock(note=69, expected_hz=440.0, voices={(0, 19): steady(441.0, wobble=0.4)})
    published = found.to_json()
    assert published["voices"]["0:19"]["steady"] is False
    assert "frequency_hz" in published["voices"]["0:19"]
    assert "0 of them steady" in found.describe()


def test_the_chain_is_bounded_by_its_calmest_voice():
    """Every take shared one converter, so it cannot wander more than the calmest does."""
    found = clock.Clock(
        note=69,
        expected_hz=440.0,
        voices={
            (0, 16): steady(440.0, wobble=0.002, seconds=20.0),
            (0, 73): steady(440.01, wobble=0.030, seconds=20.0),
        },
    )
    said = found.describe()
    assert "2 voices span" in said
    assert "2 of them steady" in said
    # 0.002 cycles over 20 s is 1e-4 Hz, which against 440 Hz is well under 1 ppm.
    assert "under 0 ppm" in said
