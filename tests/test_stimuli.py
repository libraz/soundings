"""Controls for the stimulus catalogue and the union verdict it feeds."""

from __future__ import annotations

import pytest

from soundings import audible, stimuli

from .test_audible import judge, note, unrepeatable


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


def test_an_all_inconclusive_result_is_not_reported_as_a_null() -> None:
    takes = [unrepeatable(s) for s in range(6)]
    bad = judge(takes[:3], takes[3:], stimulus_name="struck")
    overall = audible.Overall(label="test", verdicts=[bad])
    assert not overall.audible
    assert overall.inconclusive_under == ["struck"]
    assert overall.deaf_to == []
    assert "INCONCLUSIVE" in overall.describe()
    assert not overall.to_json()["conclusive"]


def test_a_drum_stimulus_carries_its_own_channel() -> None:
    """An unpitched sound lives on channel 10 and nowhere else; the same note on
    any other channel is a pitched voice, which is what it exists not to be."""
    for name in ("unpitched", "wash"):
        assert stimuli.CATALOGUE[name].on(0) == 9, name
    assert stimuli.CATALOGUE["struck"].on(0) == 0
    assert stimuli.CATALOGUE["struck"].on(5) == 5


def test_a_stimulus_that_needs_a_part_set_up_says_which_address() -> None:
    """Written after the reset and before the note, and undone by the next reset."""
    kit = stimuli.CATALOGUE["struck_kit"]
    assert kit.writes == (("40 12 15", 1),)
    assert kit.on(0) == 1, "the part written to and the part played must be the same one"
    assert all(s.writes == () for s in stimuli.CATALOGUE.values() if s.name != "struck_kit")


def test_the_effect_set_holds_only_stimuli_a_delay_can_be_measured_against() -> None:
    """Every one of them is either unpitched or slow enough not to fold a delay
    into its own period. A pitched note at middle C repeats every 3.8 ms."""
    for name in stimuli.EFFECT:
        stimulus = stimuli.CATALOGUE[name]
        unpitched = stimulus.channel == 9 or stimulus.writes
        assert unpitched or stimulus.note <= 24, name


def test_effect_expands_like_broad_does() -> None:
    assert [s.name for s in stimuli.resolve(["effect"])] == list(stimuli.EFFECT)
    assert stimuli.resolve(["effect", "struck"])[-1].name == "struck"


def test_a_stimulus_describes_the_channel_only_when_it_overrides_one() -> None:
    assert "channel 10" in stimuli.CATALOGUE["wash"].describe()
    assert "channel" not in stimuli.CATALOGUE["struck"].describe()
