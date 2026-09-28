"""Controls for recovering the response a pair of note takes differ by.

The takes are synthetic so the answer is known: one broadband decaying voice,
struck several times with its own small take-to-take noise and an offset of a
few hundred samples, and a second set made the same way with a known stereo
response of the two channels' mean added to each channel. What comes back has to
be that response, below the ceiling the stage publishes to; and a second set
with nothing added has to come back at the noise the halves disagree by, not as
a plausible small response.
"""

from __future__ import annotations

import numpy as np
import pytest

from soundings import reproduce, sendresponse
from soundings.render import graph

SR = 48000
LEAD_S = 0.3
SECONDS = 1.5


def voice(seed: int) -> np.ndarray:
    """A broadband stereo strike, after a lead, that has decayed away before the take ends.

    Decayed away because a take that ends while the voice is still sounding has lost
    the part of the response to that tail, and the deconvolution divides that loss
    back in wherever the voice is thin. The unit's own takes reach the converter's
    floor before they end; `test_how_far_the_takes_had_decayed_is_reported` holds
    the stage to saying so of its own takes."""
    rng = np.random.default_rng(seed)
    n = int(SECONDS * SR)
    onset = int(LEAD_S * SR)
    body = np.zeros(n)
    t = np.arange(n - onset) / SR
    body[onset:] = rng.standard_normal(n - onset) * np.exp(-t / 0.1) * 0.1
    return np.stack([0.8 * body, 0.5 * body], axis=1)


def known_response(seed: int) -> np.ndarray:
    """A stereo response: a short gap, then two independent decaying noises."""
    rng = np.random.default_rng(seed)
    n = int(0.25 * SR)
    gap = int(0.01 * SR)
    t = np.arange(n - gap) / SR
    out = np.zeros((n, 2))
    for c in range(2):
        out[gap:, c] = rng.standard_normal(n - gap) * np.exp(-t / 0.03) * 0.05
    return out


def struck(base: np.ndarray, response: np.ndarray | None, seed: int) -> np.ndarray:
    """One take: the voice, its own noise, the response of its mean, and an offset."""
    rng = np.random.default_rng(seed)
    n = base.shape[0]
    take = base + rng.standard_normal(base.shape) * 1e-5
    if response is not None:
        mono = take.mean(axis=1)
        for c in range(2):
            take[:, c] = take[:, c] + np.convolve(mono, response[:, c])[:n]
    lag = float(rng.uniform(-200.0, 200.0))
    size = reproduce.correlation_size(take[:, 0], take[:, 0])
    return np.stack(
        [reproduce.delayed(take[:, c], lag, size)[:n] for c in range(2)], axis=1
    )


def takes_of(response: np.ndarray | None) -> tuple[list[np.ndarray], list[np.ndarray]]:
    base = voice(1)
    dry = [struck(base, None, 10 + k) for k in range(4)]
    wet = [struck(base, response, 20 + k) for k in range(4)]
    return dry, wet


def expected_at_the_published_rate(response: np.ndarray) -> np.ndarray:
    """The known response carried to the published rate.

    Carried as a signal and scaled by the ratio of the rates, since a response is a
    sum over samples and there are fewer of them per second."""
    scale = SR / sendresponse.RATE_HZ
    return np.stack(
        [graph.resampled(response[:, c], SR, sendresponse.RATE_HZ) * scale for c in range(2)],
        axis=1,
    )


def error_db(found: np.ndarray, start: int, expected: np.ndarray) -> float:
    """How far apart two responses are below the ceiling, relative to the expected one.

    `found` begins at lag `start`, which a response cut at the ceiling may put before
    zero. Compared as spectra rather than cut in time: a brick-wall cut of a short array
    wraps its own ringing round the array, which is an error of the comparison."""
    reach = abs(start) + max(found.shape[0], expected.shape[0])
    size = 1 << int(np.ceil(np.log2(2 * reach)))
    placed = np.zeros((size, 2))
    placed[np.arange(start, start + found.shape[0]) % size] = found
    below = np.fft.rfftfreq(size, 1.0 / sendresponse.RATE_HZ) < graph.CUT_HZ
    a = np.fft.rfft(placed, size, axis=0)[below]
    b = np.fft.rfft(expected, size, axis=0)[below]
    return 10.0 * np.log10(np.sum(np.abs(a - b) ** 2) / np.sum(np.abs(b) ** 2))


@pytest.fixture(scope="module")
def recovered() -> tuple[dict, np.ndarray]:
    response = known_response(3)
    dry, wet = takes_of(response)
    return sendresponse.recover(dry, wet, SR, lead_s=LEAD_S), response


def test_a_known_response_comes_back_below_the_ceiling(recovered) -> None:
    found, response = recovered
    got = np.stack(
        [np.asarray(found["response"]["left"]), np.asarray(found["response"]["right"])],
        axis=1,
    )
    start = found["response"]["starts_at_samples"]
    assert error_db(got, start, expected_at_the_published_rate(response)) < -40.0


def test_the_response_is_at_the_rate_models_are_drawn_at(recovered) -> None:
    found, _ = recovered
    assert found["response"]["rate_hz"] == sendresponse.RATE_HZ == 32000
    assert found["response"]["below_hz"] == graph.CUT_HZ


def test_the_half_split_floor_is_reported_overall_and_per_band(recovered) -> None:
    found, _ = recovered
    floor = found["half_split"]
    assert floor["overall_db"] < -30.0
    centres = [band["centre_hz"] for band in floor["bands"]]
    assert centres == list(sendresponse.BAND_CENTRES_HZ)
    assert all(band["difference_db"] is not None for band in floor["bands"])


def test_each_half_predicts_the_other(recovered) -> None:
    found, _ = recovered
    for direction in found["held_out"]:
        assert direction["residual_db"] < -30.0


def test_the_regularisation_is_stated(recovered) -> None:
    found, _ = recovered
    reg = found["regularisation"]
    assert reg["value"] > 0.0
    assert reg["under_the_input_db"] < 0.0


def test_how_far_the_takes_had_decayed_is_reported(recovered) -> None:
    found, _ = recovered
    ended = found["takes_end"]
    assert ended["dry_db"] < -60.0
    assert ended["wet_db"] < -60.0


def test_a_pair_with_nothing_added_returns_a_response_at_the_floor() -> None:
    dry, wet = takes_of(None)
    found = sendresponse.recover(dry, wet, SR, lead_s=LEAD_S)
    whole = found["whole_estimate"]
    assert abs(whole["energy_db"] - whole["noise_db"]) < 3.0
    assert found["response"]["length_s"] <= sendresponse.WINDOW_S
    for level in found["level"].values():
        assert level["return_over_dry_db"] is None or level["return_over_dry_db"] < -40.0


def test_too_few_takes_to_split_is_refused() -> None:
    dry, wet = takes_of(None)
    with pytest.raises(ValueError):
        sendresponse.recover(dry[:3], wet, SR, lead_s=LEAD_S)
