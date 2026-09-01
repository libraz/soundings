"""Controls for the stimulus catalogue and the union verdict it feeds."""

from __future__ import annotations

import pytest

from soundings import audible, stimuli

from .test_audible import judge, note


def test_every_catalogue_entry_says_what_it_cannot_see() -> None:
    """A stimulus that only advertises its strengths invites a null to be read as a fact."""
    for s in stimuli.CATALOGUE.values():
        assert s.sees and s.blind_to
        assert s.hold <= s.seconds


def test_broad_and_all_expand_and_do_not_repeat_themselves() -> None:
    broad = stimuli.resolve(["broad"])
    assert [s.name for s in broad] == list(stimuli.BROAD)
    assert len(stimuli.resolve(["all"])) == len(stimuli.CATALOGUE)
    assert [s.name for s in stimuli.resolve(["struck", "broad", "struck"])] == list(stimuli.BROAD)


def test_an_unknown_stimulus_is_refused_rather_than_dropped() -> None:
    with pytest.raises(KeyError, match="no stimulus named"):
        stimuli.resolve(["struck", "nonesuch"])


def test_the_released_stimulus_actually_lets_go_early() -> None:
    """The whole point of it: most of the take is after note-off."""
    released = stimuli.CATALOGUE["released"]
    assert released.hold < released.seconds * 0.2


def _verdict(name: str, *, differ: bool) -> audible.Verdict:
    a = [note(seed=s) for s in (0, 1, 2)]
    b = [note(fundamental=277.2 if differ else 220.0, seed=s) for s in (3, 4, 5)]
    return judge(a, b, stimulus_name=name)


def test_one_stimulus_hearing_it_makes_the_parameter_audible() -> None:
    overall = audible.Overall(
        label="test",
        verdicts=[_verdict("struck", differ=False), _verdict("released", differ=True)],
    )
    assert overall.audible
    assert overall.heard_by == ["released"]
    assert overall.deaf_to == ["struck"]
    assert "AUDIBLE" in overall.describe()


def test_a_null_names_every_stimulus_it_was_asked_under() -> None:
    overall = audible.Overall(
        label="test",
        verdicts=[_verdict("struck", differ=False), _verdict("soft", differ=False)],
    )
    assert not overall.audible
    assert overall.deaf_to == ["struck", "soft"]
    assert "struck, soft" in overall.describe()
    assert "not about the parameter" in overall.describe()
    assert overall.to_json()["not_heard_by"] == ["struck", "soft"]


def test_nothing_asked_is_not_a_null() -> None:
    overall = audible.Overall(label="test")
    assert not overall.audible
    assert "nothing was asked" in overall.describe()
