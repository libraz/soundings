"""The one section in the catalogue whose magnitude comes from a phase.

Every other section in the renderer shapes a magnitude directly. This one passes
all frequencies at unit gain and only turns the phase, and what a listener hears
is the cancellation when the turned copy is summed back with the untouched one.
That makes two things identifiable which a magnitude response hides: how many
sections there are, which the number of cancellations fixes on its own, and what
rate the chain runs at, which decides where the cancellations start to crowd
together on their way to Nyquist.

No hardware: every response here is computed.
"""

from __future__ import annotations

import math

import numpy as np

from soundings import reproduce

FS = 32000.0
CORNER = 1040.0
"""The corner the unit's own profile fits at the bottom of the phaser's byte."""


def _notches(sections: int, corner_hz: float) -> list[float]:
    freq = np.geomspace(20.0, FS / 2 - 100, 60000)
    mag = np.abs(
        reproduce._allpass_chain(freq, sections=sections, corner_hz=corner_hz, mix=1.0, fs=FS)
    )
    return [
        freq[i]
        for i in range(1, len(mag) - 1)
        if mag[i] < mag[i - 1] and mag[i] < mag[i + 1] and mag[i] < 0.05
    ]


def test_the_notch_count_is_the_section_count_and_nothing_else() -> None:
    """What makes this section identifiable at all. Four cancellations is eight
    sections, and no corner setting of six or ten produces four."""
    assert len(_notches(6, CORNER)) == 3
    assert len(_notches(8, CORNER)) == 4
    assert len(_notches(10, CORNER)) == 5


def test_the_notches_sit_where_the_tangent_puts_them_not_where_frequency_does() -> None:
    """The bilinear reading, which is the whole of why the ratios close up towards
    Nyquist. Asserted against the ratios an analogue ladder would have kept: with
    the corner up where the unit's own byte takes it, the top cancellation sits
    more than an octave below where a chain with no sample rate would put it."""
    for k, hz in enumerate(_notches(8, CORNER)):
        want = (
            FS
            / math.pi
            * math.atan(math.tan(math.pi * CORNER / FS) * math.tan(math.pi * (2 * k + 1) / 16))
        )
        assert abs(hz - want) / want < 0.01

    high = 4486.0  # what the byte's middle fits to
    analogue_top = high * math.tan(math.pi * 7 / 16)
    assert analogue_top / _notches(8, high)[-1] > 1.8


def test_a_section_that_passes_everything_passes_everything() -> None:
    """With nothing summed back the chain is transparent, whatever the corner. This
    is also the reference the unit's own profile is a deviation from."""
    freq = np.geomspace(20.0, 15000.0, 500)
    flat = reproduce._allpass_chain(freq, sections=8, corner_hz=CORNER, mix=0.0, fs=FS)
    assert np.allclose(np.abs(flat), 1.0)


def test_the_dry_and_the_wet_are_summed_rather_than_averaged() -> None:
    """Six decibels where the two arrive together, which is what the unit's profile
    stands over its own dry reference. Averaging instead would put the whole profile
    6 dB low and leave every notch where it is, so the error would be constant, lean
    nowhere, and pass both the gross gate and the breakdown gate."""
    together = reproduce._allpass_chain(
        np.array([1.0]), sections=8, corner_hz=CORNER, mix=1.0, fs=FS
    )
    assert 5.9 < 20.0 * math.log10(float(np.abs(together)[0])) < 6.1


def test_the_chain_is_reached_through_a_model_like_every_other_section() -> None:
    """Through `response`, so a model file can hold one without a path of its own."""
    model = {
        "sample_rate_hz": FS,
        "chain": [
            {
                "kind": "allpass-chain",
                "sections": 8,
                "corner_hz": {
                    "byte": "40 03 03",
                    "map": {"kind": "points", "log": True, "points": [[0, CORNER], [127, 13508.0]]},
                },
                "mix": {
                    "byte": "40 03 07",
                    "map": {"kind": "points", "log": False, "points": [[0, 0.0], [127, 1.0]]},
                },
            }
        ],
    }
    freq = np.array([212.0, 1000.0])

    deep = reproduce.response(model, {"40 03 03": 0, "40 03 07": 127}, freq)
    assert 20.0 * math.log10(abs(deep[0])) < -15.0

    bypassed = reproduce.response(model, {"40 03 03": 0, "40 03 07": 0}, freq)
    assert np.allclose(np.abs(bypassed), 1.0)
