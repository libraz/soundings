"""What a swept-sine run reports when its own sweeps disagree.

The failure this guards against was measured rather than imagined. One channel
of a working path was disturbed intermittently, and a single sweep reported its
distortion at -58.5 dB where ten further sweeps put it near -90. Nothing in that
sweep's own numbers said which kind of sweep it was: the impulse was there, the
latency was right, the response was flat. Only running it again said so.

So the guard is that a run of several sweeps says whether they agreed, and that
the response it reports comes from a sweep the path actually produced rather
than from an average of a clean one and a disturbed one.

No hardware. The responses here are built directly, not deconvolved.
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from soundings import probe
from soundings.cli import inject

SR = 48000


def response(noise_db: float, *, second_db: float = -90.0, gain: float = 1.0) -> probe.Response:
    """A response with a stated floor, and an impulse whose level is the gain."""
    impulse = np.zeros(SR)
    peak = SR // 2
    impulse[peak] = gain
    return probe.Response(
        impulse=impulse,
        peak=peak,
        origin=peak - 48,
        sample_rate=SR,
        harmonics=[probe.Harmonic(order=2, level_db=second_db, arrived_ms=100.0)],
        noise_db=noise_db,
    )


def read(found: list[probe.Response]) -> dict:
    args = argparse.Namespace(rate=SR, pad=1.0, reference=None)
    return inject._read_channel(3, found, None, args)


def test_sweeps_that_agree_are_reported_as_repeating() -> None:
    entry = read([response(-128.0), response(-129.0), response(-130.0)])
    assert entry["repeats"]
    assert entry["floor_spread_db"] == 2.0
    assert "caveat" not in entry


def test_one_disturbed_sweep_makes_the_whole_channel_unreadable() -> None:
    """The measured shape: two clean sweeps and one disturbed. Reporting the
    median as though it stood alone would publish a number whose own run said it
    could not be trusted."""
    entry = read([response(-92.4, second_db=-58.5), response(-128.0), response(-129.0)])
    assert not entry["repeats"]
    assert entry["floor_spread_db"] > inject.REPEATS_WITHIN_DB
    assert "describe the disturbance rather than the path" in entry["caveat"]


def test_every_sweep_is_kept_rather_than_collapsed_to_one_number() -> None:
    """A spread says something moved; the list says which sweep moved, which is
    what separates a settling run from one that comes and goes."""
    entry = read([response(-92.4, second_db=-58.5), response(-128.0), response(-129.0)])
    assert entry["noise_db_each"] == [-92.4, -128.0, -129.0]
    assert entry["distortion_db_each"] == [-58.5, -90.0, -90.0]
    assert entry["takes"] == 3


def test_the_reported_response_is_a_sweep_that_really_happened() -> None:
    """The median take, not an average. Averaging a clean impulse with a disturbed
    one gives a response neither sweep measured, and its level is not either of
    theirs."""
    entry = read(
        [response(-128.0, gain=1.0), response(-90.0, gain=0.5), response(-129.0, gain=1.0)]
    )
    assert entry["noise_db"] == -128.0


def test_a_single_sweep_still_answers_but_claims_no_repeatability() -> None:
    entry = read([response(-128.0)])
    assert entry["takes"] == 1
    assert entry["floor_spread_db"] == 0.0
    # One sweep cannot disagree with itself, so the flag says nothing either way
    # and the caveat is absent; the take count is what a reader has to look at.
    assert "caveat" not in entry


def test_a_loopback_run_refuses_a_preparation_it_cannot_apply() -> None:
    """--loopback puts the machine outside the path. A state written into it then
    changes nothing that comes back, so a run carrying both would record a
    preparation beside a response the preparation could not have reached."""
    with pytest.raises(SystemExit) as refused:
        inject.cmd_transfer(argparse.Namespace(prepare=[("40 03 00", (0x02, 0x01))], loopback=True))
    assert str(refused.value) == inject.NOT_WITH_LOOPBACK
