"""A loop around an all-pass cascade, and the two things that are in it.

The cascade itself is covered where it was written. What is checked here is what a
returned signal does to it: where the resonances land, that the notches do not
move, what a sample of delay costs and where it costs it, and that a pole of a
high pass in the return path leaves the loop alone in the middle of the band and
takes it to nothing at the bottom.

The last test in the file is the one the claim leans on hardest and is not about
feedback at all -- that four second-order all-pass sections at a resonance of
exactly a half are eight first-order ones, exactly and not nearly. That is why a
section count cannot separate the two chains, and why the byte that raises the
resonance has to be asked instead.
"""

from __future__ import annotations

import numpy as np

from soundings import reproduce

FS = 32000.0
CORNER = 1000.0
SECTIONS = 8


def chain(freq, **kwargs):
    return reproduce._allpass_chain(
        freq, sections=SECTIONS, corner_hz=CORNER, mix=1.0, fs=FS, **kwargs
    )


def a_fine_sweep() -> np.ndarray:
    """Fine enough that a notch is found where it is and not where a band is."""
    return np.geomspace(20.0, 15900.0, 40000)


def test_no_feedback_is_the_cascade_it_always_was():
    """The knob's off state has to be the old code, not a close rendering of it."""
    freq = a_fine_sweep()
    t = np.tan(np.pi * CORNER / FS)
    c = (t - 1.0) / (t + 1.0)
    z1 = np.exp(-1j * 2 * np.pi * freq / FS)
    was = 1.0 + ((c + z1) / (1.0 + c * z1)) ** SECTIONS
    assert np.allclose(chain(freq), was, rtol=0, atol=0)


def test_a_loop_does_not_move_a_notch():
    """The whole of why the byte is a loop and not a corner.

    The dry signal is outside the loop, so it goes on cancelling wherever the
    cascade has turned through an odd half circle however hard the loop is driven.
    A byte that moved the sections' corners would move them all.
    """
    freq = a_fine_sweep()
    quiet = freq[np.argmin(np.abs(chain(freq)))]
    for gain in (0.3, 0.6, 0.9):
        driven = freq[np.argmin(np.abs(chain(freq, feedback=gain)))]
        assert abs(np.log2(driven / quiet)) < 0.01


def test_a_loop_puts_its_resonance_between_the_notches():
    """And at the height one over one minus the loop gain says it should."""
    freq = a_fine_sweep()
    quiet = np.abs(chain(freq))
    notches = freq[
        [i for i in range(1, len(freq) - 1) if quiet[i] < quiet[i - 1] and quiet[i] < quiet[i + 1]]
    ]
    assert len(notches) == 4

    driven = np.abs(chain(freq, feedback=0.8))
    peaks = freq[
        [
            i
            for i in range(1, len(freq) - 1)
            if driven[i] > driven[i - 1] and driven[i] > driven[i + 1] and driven[i] > 2.0
        ]
    ]
    # One resonance in each gap between consecutive notches, and none on a notch.
    for low, high in zip(notches, notches[1:], strict=False):
        assert sum(low < peak < high for peak in peaks) == 1
    for notch in notches:
        assert min(abs(np.log2(peak / notch)) for peak in peaks) > 0.1

    # Where the cascade comes back in phase the loop adds up, and the dry signal
    # is still there beside it: one plus the sum of the returns.
    assert np.isclose(driven.max(), 1.0 + 1.0 / (1.0 - 0.8), rtol=0.02)


def test_a_sample_of_delay_costs_nothing_at_the_bottom_and_a_sign_at_the_top():
    """The two ends of the reading, both exact rather than close.

    A sample of delay turns the returned signal by the frequency itself, so at
    nothing it turns it by nothing and the two loops are one loop; at half the
    rate it has turned it right round, which is a loop of the same size returned
    the other way. Everything the claim makes of the top band is that second
    identity: where an instantaneous loop resonates, a delayed one cancels.
    """
    nothing = np.array([0.0])
    assert np.allclose(
        chain(nothing, feedback=0.8, feedback_delay=1),
        chain(nothing, feedback=0.8),
        rtol=0,
        atol=0,
    )

    half_the_rate = np.array([FS / 2.0])
    assert np.allclose(
        chain(half_the_rate, feedback=0.8, feedback_delay=1),
        chain(half_the_rate, feedback=-0.8),
    )


def test_the_delay_is_read_at_the_top_of_the_band_and_nowhere_else():
    """Where the resonance nearest half the rate is, the two loops disagree most."""
    freq = a_fine_sweep()
    instant = np.abs(chain(freq, feedback=0.8))
    late = np.abs(chain(freq, feedback=0.8, feedback_delay=1))
    apart = np.abs(20 * np.log10(instant / late))
    top = int(np.argmax(instant * (freq > 8000.0)))
    assert apart[top] > 10.0
    assert apart[freq < 200.0].max() < apart[top] / 3.0


def test_a_blocked_return_carries_nothing_at_the_bottom_and_everything_above_it():
    """A pole of a high pass is zero at nothing, so the loop is open there exactly."""
    nothing = np.array([0.0])
    assert np.allclose(
        chain(nothing, feedback=0.9, feedback_highpass_hz=20.0),
        chain(nothing, feedback=0.0),
        rtol=0,
        atol=1e-12,
    )

    # Two decades above its corner the filter is unity to a part in a thousand,
    # so the loop is the loop and the byte's whole range is unaffected by it.
    well_above = np.array([2000.0])
    assert np.isclose(
        abs(chain(well_above, feedback=0.9, feedback_highpass_hz=20.0)[0]),
        abs(chain(well_above, feedback=0.9)[0]),
        rtol=0.02,
    )


def test_a_blocked_return_does_nothing_when_the_loop_is_open():
    """A filter inside an open loop is not in the signal path at all.

    This is the arithmetic behind the claim that the high pass is in the return
    and not in the chain, so it is worth having as a test rather than as a
    sentence: the setting where the byte is nothing has to render identically with
    the filter and without it.
    """
    freq = a_fine_sweep()
    assert np.allclose(
        chain(freq, feedback=0.0, feedback_highpass_hz=20.0),
        chain(freq, feedback=0.0),
        rtol=0,
        atol=0,
    )


def test_closing_the_loop_round_the_output_keeps_the_notch_a_true_zero():
    """How the two places a loop can be closed differ, which is not where anything sits.

    Around the cascade alone the dry path is outside the loop and the returned
    signal fills the cancellation in to the loop gain over one plus it. Around the
    summed output the dry is inside, and what comes back at the cancellation is
    nothing, so the notch stays a null.
    """
    freq = a_fine_sweep()
    around_chain = np.abs(chain(freq, feedback=0.5)).min()
    around_output = np.abs(
        chain(freq, feedback=0.5, feedback_around="the-output")
    ).min()
    assert np.isclose(around_chain, 0.5 / 1.5, rtol=0.05)
    assert around_output < 1e-3


def test_a_loop_round_half_the_cascade_resonates_twice_as_far_apart():
    """Which is what lands one of its resonances on a notch.

    The sub-chain turns through half the phase in the same band, so the places it
    comes back in phase are twice as far apart -- and every other one of them is
    where the whole cascade has turned through an odd half circle, which is a
    notch.
    """
    freq = a_fine_sweep()
    whole = np.abs(chain(freq, feedback=0.7))
    half = np.abs(chain(freq, feedback=0.7, feedback_sections=4))

    def resonances(profile):
        return freq[
            [
                i
                for i in range(1, len(freq) - 1)
                if profile[i] > profile[i - 1] and profile[i] > profile[i + 1]
                and profile[i] > 2.0
            ]
        ]

    assert len(resonances(half)) < len(resonances(whole))


def test_the_stage_is_reachable_through_response_with_the_loop_off_a_byte():
    model = {
        "sample_rate_hz": FS,
        "chain": [
            {
                "kind": "allpass-chain",
                "sections": SECTIONS,
                "corner_hz": {"fixed": CORNER},
                "mix": {"fixed": 1.0},
                "feedback": {
                    "byte": "40 03 06",
                    "map": {"kind": "points", "log": False, "points": [[0, 0.0], [127, 0.8]]},
                },
                "feedback_delay": 1,
                "feedback_highpass_hz": 20.0,
            }
        ],
    }
    freq = np.array([400.0, 4000.0])
    off = reproduce.response(model, {"40 03 06": 0}, freq)
    on = reproduce.response(model, {"40 03 06": 127}, freq)
    assert np.allclose(off, chain(freq, feedback=0.0))
    assert np.allclose(
        on, chain(freq, feedback=0.8, feedback_delay=1, feedback_highpass_hz=20.0)
    )


def test_four_second_order_sections_at_a_resonance_of_a_half_are_eight_first_order_ones():
    """The equivalence the closed reading carries, checked rather than asserted.

    A second-order all-pass at a resonance of exactly a half has two real poles at
    the same place, which is two first-order sections at that corner. So a chain of
    four of them and a chain of eight of these are one transfer function -- not
    close, the same -- and no reading of a magnitude can ever separate them. What a
    reading can separate is whether the byte beside them raises that resonance,
    which is a different question and the one the record answers.
    """
    freq = a_fine_sweep()
    eight = reproduce._allpass_chain(
        freq, sections=8, corner_hz=CORNER, mix=1.0, fs=FS
    )
    four = reproduce._allpass_chain(
        freq, sections=4, corner_hz=CORNER, mix=1.0, fs=FS, q=0.5
    )
    assert np.allclose(eight, four, rtol=1e-10, atol=1e-12)


def test_raising_that_resonance_crowds_the_notches_towards_the_corner():
    """Which is the reading the record contradicts, so it has to be a real prediction."""
    freq = a_fine_sweep()

    def notches(q):
        profile = np.abs(
            reproduce._allpass_chain(
                freq, sections=4, corner_hz=CORNER, mix=1.0, fs=FS, q=q
            )
        )
        return freq[
            [
                i
                for i in range(1, len(freq) - 1)
                if profile[i] < profile[i - 1] and profile[i] < profile[i + 1]
            ]
        ]

    flat, sharp = notches(0.5), notches(4.0)
    assert len(flat) == len(sharp) == 4
    # Every notch is nearer the sections' own corner at the higher resonance.
    for was, now in zip(flat, sharp, strict=True):
        assert abs(np.log2(now / CORNER)) < abs(np.log2(was / CORNER))
