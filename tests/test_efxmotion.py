"""Sorting insertion effects into the ones that move and the ones that do not.

The failure this guards is a quiet one. A type whose takes were unusable produces
an empty delay track, an empty track produces no line, and no line reads as an
effect standing still -- so every broken pair sorts into `static`, which is the
answer that costs nothing and is wrong for exactly the effects a survey like this
exists to find. The three-way split is the whole point, and each test below is
about a type landing in the right one of the three piles.

No hardware. Takes are written to a temporary directory in the shape a --save run
writes them, so what is under test is the sort and not this machine's audio.
"""

from __future__ import annotations

import json

import numpy as np

from soundings import efxmotion, motion
from soundings.takes import write

SR = 48000


def material(seconds: float = 4.0, *, seed: int = 3) -> np.ndarray:
    """Broadband decaying noise, band-limited so interpolation is not under test."""
    n = int(seconds * SR)
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    spectrum[np.fft.rfftfreq(n, 1 / SR) > 6000] = 0
    tail = np.exp(-0.4 * np.arange(n) / SR)
    return np.fft.irfft(spectrum, n) * (0.3 + 0.7 * tail)


def a_type(root, name: str, dry: np.ndarray, wet: np.ndarray) -> None:
    """Write one type's pair of takes in the shape a --save run leaves them."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    write(directory / "unpitched-0-00", dry[:, None], SR)
    write(directory / "unpitched-1-00", wet[:, None], SR)
    (directory / "takes-manifest.json").write_text(
        json.dumps(
            {
                "takes": [
                    {"file": "unpitched-0-00.wav", "stimulus": "unpitched", "setting": "0"},
                    {"file": "unpitched-1-00.wav", "stimulus": "unpitched", "setting": "1"},
                ]
            }
        )
    )


def test_a_type_that_modulates_is_sorted_as_moving(tmp_path):
    dry = material()
    swung = motion.modulated_copy(dry, SR, rate_hz=1.3, depth_ms=4.0, centre_ms=20.0)
    a_type(tmp_path, "01-00", dry, dry + 0.7 * swung)

    found = efxmotion.survey(tmp_path)

    assert [f.verdict for f in found] == ["moves"]
    assert abs(found[0].rate_hz - 1.3) < 0.2
    assert found[0].where == "delay"


def test_a_type_with_a_fixed_delay_is_sorted_as_static(tmp_path):
    """A static effect has a return to track and no line in the track. It must
    come back as static rather than as unanswerable, or the survey has no
    positive pile to put anything in."""
    dry = material()
    a_type(tmp_path, "02-00", dry, dry + 0.7 * np.roll(dry, int(0.02 * SR)))

    found = efxmotion.survey(tmp_path)

    assert [f.verdict for f in found] == ["static"]
    assert found[0].detectable_ms is not None


def test_a_type_whose_takes_carry_no_return_is_sorted_as_neither(tmp_path):
    """The pile this module exists for. Two identical takes have nothing to
    track, and filing that as standing still would be a finding produced by a
    broken measurement."""
    dry = material()
    a_type(tmp_path, "03-00", dry, dry.copy())

    found = efxmotion.survey(tmp_path)

    assert [f.verdict for f in found] == ["could not say"]
    assert found[0].detectable_ms is None


def test_a_type_that_moved_is_not_put_through_a_control(tmp_path):
    """Having moved is itself proof the pair could show motion, and a control on
    a moving type is meaningless anyway: it sweeps the take's own return, so it
    adds a motion to a motion and recovers a line whatever the depth. Running it
    would buy a confounded number for seven more passes over the audio."""
    dry = material()
    swung = motion.modulated_copy(dry, SR, rate_hz=1.3, depth_ms=4.0, centre_ms=20.0)
    a_type(tmp_path, "04-00", dry, dry + 0.7 * swung)

    found = efxmotion.measure_type(
        "04 00",
        tmp_path / "04-00" / "unpitched-0-00.wav",
        tmp_path / "04-00" / "unpitched-1-00.wav",
        stimulus="unpitched",
    )

    assert found.moves and found.conclusive
    assert found.verdict == "moves"
    assert found.detectable_ms is None


def test_the_three_piles_account_for_every_type(tmp_path):
    """A type in none of them, or in two, would be a type the record does not
    state a verdict for."""
    dry = material()
    swung = motion.modulated_copy(dry, SR, rate_hz=1.3, depth_ms=4.0, centre_ms=20.0)
    a_type(tmp_path, "01-00", dry, dry + 0.7 * swung)
    a_type(tmp_path, "02-00", dry, dry + 0.7 * np.roll(dry, int(0.02 * SR)))
    a_type(tmp_path, "03-00", dry, dry.copy())

    found = efxmotion.survey(tmp_path)
    piles = efxmotion.partition(found)

    assert sum(len(v) for v in piles.values()) == len(found) == 3
    assert set(piles["moving"]) & set(piles["static"]) == set()


def test_a_directory_without_a_manifest_is_passed_over(tmp_path):
    """A save directory can hold anything; only what carries a manifest is a
    type's takes, and guessing from filenames would sort a stray directory."""
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "unpitched-0-00.wav").write_bytes(b"")
    dry = material(seconds=1.0)
    a_type(tmp_path, "01-00", dry, dry + 0.5 * np.roll(dry, 400))

    assert [f.type_id for f in efxmotion.survey(tmp_path)] == ["01 00"]


def test_a_wrapped_track_is_not_sorted_as_static_even_if_its_control_passes(tmp_path, monkeypatch):
    """A swing wider than half the input's own period folds, and a folded line can
    vanish -- which is the shape standing still has, so such a pair cannot carry a
    null whatever its control did.

    The control is forced rather than found. On every source tried, a track
    periodic enough to wrap defeated the injected control as well, so the two
    conditions were never seen apart and no recording could reach this branch.
    That makes the guard redundant on the evidence available rather than shown to
    be necessary, and the fake is what states the rule it encodes.
    """
    period_s = 0.004
    n = int(4.0 * SR)
    index = np.arange(n)
    dry = np.sin(2 * np.pi * index / (period_s * SR)) * np.exp(-0.4 * index / SR)
    swung = motion.modulated_copy(dry, SR, rate_hz=0.9, depth_ms=12.0, centre_ms=20.0)
    a_type(tmp_path, "06-00", dry, dry + 0.7 * swung)
    monkeypatch.setattr(
        efxmotion.motion,
        "control",
        lambda *a, **k: {"detectable_ms": [6.0, 0.375], "return_level_db": -3.2},
    )

    found = efxmotion.measure_type(
        "06 00",
        tmp_path / "06-00" / "unpitched-0-00.wav",
        tmp_path / "06-00" / "unpitched-1-00.wav",
        stimulus="unpitched",
    )

    assert found.track["may_have_wrapped"] and found.detectable_ms is not None
    assert not found.moves
    assert found.verdict == "could not say"


def test_a_type_the_unit_accepts_with_no_takes_is_reported_unsurveyed(tmp_path):
    """The capture stage's version of the same failure the three piles guard
    against. A type whose takes never got recorded leaves a shorter list of
    verdicts, and a short list reads exactly like a complete one."""
    dry = material(seconds=1.0)
    a_type(tmp_path, "01-00", dry, dry + 0.5 * np.roll(dry, 400))
    accepted = tmp_path / "efx-type-map.json"
    accepted.write_text(
        json.dumps({"effects": [{"type": "01 00"}, {"type": "02 00"}, {"type": "03 07"}]})
    )

    found = efxmotion.survey(tmp_path)

    assert efxmotion.accepted_types(accepted) == ["01 00", "02 00", "03 07"]
    assert efxmotion.missing(found, efxmotion.accepted_types(accepted)) == ["02 00", "03 07"]


def test_nothing_is_unsurveyed_when_every_accepted_type_has_takes(tmp_path):
    """The guard must be able to come back clean, or it says nothing when it fires."""
    dry = material(seconds=1.0)
    a_type(tmp_path, "01-00", dry, dry + 0.5 * np.roll(dry, 400))
    a_type(tmp_path, "02-00", dry, dry + 0.5 * np.roll(dry, 700))

    found = efxmotion.survey(tmp_path)

    assert efxmotion.missing(found, ["01 00", "02 00"]) == []


def test_a_type_missing_one_of_its_two_settings_is_left_out(tmp_path):
    """Half a pair cannot be compared, and pairing it with another type's take
    would produce a verdict about neither."""
    directory = tmp_path / "05-00"
    directory.mkdir()
    write(directory / "unpitched-0-00", material(seconds=1.0)[:, None], SR)
    (directory / "takes-manifest.json").write_text(
        json.dumps(
            {"takes": [{"file": "unpitched-0-00.wav", "stimulus": "unpitched", "setting": "0"}]}
        )
    )

    assert efxmotion.survey(tmp_path) == []
