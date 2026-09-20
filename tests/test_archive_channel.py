"""A level published without the input it was measured on is not a level.

An interface has more inputs than the unit is plugged into, and the ones nobody
plugged anything into are not silent -- an idle preamp here sits around -70 dBFS
while the unit's own output floor is -110. A reader that takes the loudest
channel of each take separately therefore answers from the unit while the unit is
loud and from an idle input the moment a setting turns the output down, and
nothing in the figures marks the change: the level is plausible, the profile is
full and ragged, and the take behind it is real.

That happened. Twenty published records carried an idle input's noise as the
floor every reading was held against, and one reading of an output level slot was
published as a twenty five decibel frequency-dependent curve for a control that
only changes level.

So the rule is mechanical rather than remembered: **a record that publishes a
level says which channel the level came from.** A record naming the channel can
be checked against the interface it was recorded on; one that does not cannot be
checked at all, by anybody, ever again.

**A level is not the only quantity the input decides**, which is the wider rule
below. A decay time is a subtraction between two takes, and two takes answered
from different inputs subtract one signal from another: the whole of it is left
standing where a tail should be, and five published records carried decay times
up to two and a half times what their takes hold. A modulation rate is a track
through one take, and an idle input tracks as no modulation -- the same answer a
setting with no modulation in it returns. Neither of those publishes a figure in
dBFS, so neither was reached by the rule above and neither could be checked.

So the rule is keyed to the take rather than to a field name: **a record that
names the audio it read says which channel of the interface it read that audio
on.** A stage inventing a quantity this file has not heard of falls inside it by
having read a take at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

UNITS = Path(__file__).resolve().parents[1] / "data" / "units"


def _levels_somewhere(data: object) -> bool:
    """Whether any reading anywhere under `data` publishes a level in dBFS."""
    if isinstance(data, dict):
        if "heard_db" in data:
            return True
        return any(_levels_somewhere(value) for value in data.values())
    if isinstance(data, list):
        return any(_levels_somewhere(item) for item in data)
    return False


def _reads_a_take(data: object) -> bool:
    """Whether the record names a saved take anywhere in it.

    A record that names the audio it read is a record whose answer depends on
    which input that audio was read from, whatever quantity it went on to
    publish: a level, a decay time, a modulation rate, a pitch. Keying the rule
    to the take rather than to one field name is what stops a stage inventing a
    field this file has not heard of and falling outside the rule by doing so.
    """
    if isinstance(data, dict):
        return any(_reads_a_take(value) for value in data.values())
    if isinstance(data, list):
        return any(_reads_a_take(item) for item in data)
    return isinstance(data, str) and (data.endswith(".wav") or "takes" in data.split("/"))


def _names_a_channel(data: object) -> bool:
    """Whether anything anywhere under `data` says which channel was read."""
    if isinstance(data, dict):
        if {"channel", "channels", "reference_channel"} & set(data):
            return True
        return any(_names_a_channel(value) for value in data.values())
    if isinstance(data, list):
        return any(_names_a_channel(item) for item in data)
    return False


def _records() -> list[Path]:
    return sorted(
        path
        for path in UNITS.glob("*/*/*.json")
        if path.name not in {"index.json", "unit.json"}
    )


def _publishing_levels() -> list[Path]:
    out = []
    for path in _records():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:  # a malformed record is another gate's finding
            continue
        if isinstance(data, dict) and _levels_somewhere(data):
            out.append(path)
    return out


WITH_LEVELS = _publishing_levels()


def _reading_takes() -> list[Path]:
    out = []
    for path in _records():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:  # a malformed record is another gate's finding
            continue
        if isinstance(data, dict) and _reads_a_take(data):
            out.append(path)
    return out


FROM_TAKES = _reading_takes()


def test_the_archive_has_a_record_that_publishes_a_level() -> None:
    """Without one, every assertion below passes by having nothing to check."""
    assert WITH_LEVELS, f"no record under {UNITS} publishes a heard_db"


@pytest.mark.parametrize("path", WITH_LEVELS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_a_record_that_publishes_a_level_says_which_channel_it_read(path: Path) -> None:
    data = json.loads(path.read_text())
    picked = data.get("channel")
    assert isinstance(picked, dict), (
        f"{path.relative_to(UNITS)} publishes a level and does not say which channel "
        "it was read from. A level measured on an input the unit is not on is a "
        "level, and it is not the unit's."
    )
    assert isinstance(picked.get("read"), int), (
        f"{path.relative_to(UNITS)} has a channel block with no channel in it"
    )
    assert isinstance(picked.get("loudest_elsewhere"), list), (
        f"{path.relative_to(UNITS)} does not name the takes whose own loudest channel "
        "was not the one read, which is the only place a reading that changed channel "
        "can be seen"
    )


@pytest.mark.parametrize("path", WITH_LEVELS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_the_channel_read_is_one_the_record_measured(path: Path) -> None:
    """The index has to be inside the levels beside it, or it names nothing."""
    picked = json.loads(path.read_text())["channel"]
    seen = picked.get("reference_db") or picked.get("reached_db") or []
    if not seen:
        return  # a record built from no takes has no interface to describe
    assert 0 <= picked["read"] < len(seen), (
        f"{path.relative_to(UNITS)} says it read channel {picked['read']} of "
        f"{len(seen)} it measured"
    )


def test_the_archive_has_a_record_read_from_a_take() -> None:
    """Without one, the rule below passes by having nothing to apply to."""
    assert FROM_TAKES, f"no record under {UNITS} names a take it read"


@pytest.mark.parametrize("path", FROM_TAKES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_a_record_read_from_a_take_says_which_channel_it_was_read_on(path: Path) -> None:
    """The rule the level one is a special case of.

    A level is not the only quantity an input decides. A decay time is a
    subtraction between two takes, and two takes read from different inputs
    subtract one signal from another and leave the whole of it standing where a
    tail should be -- which published decay times two and a half times longer
    than the takes hold. A modulation rate is a track through one take, and an
    idle input tracks as no modulation, which is what a setting with none also
    returns. Neither publishes a dBFS figure, and neither could be checked.
    """
    assert _names_a_channel(json.loads(path.read_text())), (
        f"{path.relative_to(UNITS)} names the audio it read and does not say which "
        "channel of the interface that audio came from. Which input a take was read "
        "from is part of every quantity taken out of it, not only of a level."
    )


def _naming_a_second_output() -> list[Path]:
    out = []
    for path in WITH_LEVELS:
        data = json.loads(path.read_text())
        beside = data.get("other_channel")
        if isinstance(beside, dict) and isinstance(beside.get("read"), int):
            out.append(path)
    return out


NAMING_A_SECOND_OUTPUT = _naming_a_second_output()


def test_the_archive_has_a_record_naming_a_second_output() -> None:
    """Without one, the two rules below pass by having nothing to apply to."""
    assert NAMING_A_SECOND_OUTPUT, f"no record under {UNITS} names a second output"


@pytest.mark.parametrize(
    "path", NAMING_A_SECOND_OUTPUT, ids=lambda p: f"{p.parent.name}/{p.name}"
)
def test_a_second_output_is_the_other_half_of_the_pair(path: Path) -> None:
    """The unit's other output, and not the loudest channel that is not the first.

    An interface presents its inputs in pairs and the unit is on one of them, so
    the second output is the partner of the channel that was read. Choosing it by
    level instead is what three published records did, and on a type whose quieter
    side sits low an idle input of the interface stands over that side and wins:
    the record then reports the interface's own noise as the unit's second output,
    which is the reading that cannot be told from a stage doing nothing.
    """
    data = json.loads(path.read_text())
    read = data["channel"]["read"]
    beside = data["other_channel"]["read"]
    first = read - (read % 2)
    want = first + 1 if read == first else first
    assert beside == want, (
        f"{path.relative_to(UNITS)} read channel {read} and names channel {beside} as "
        f"the unit's other output, where the other half of that pair is {want}. A "
        "channel outside the pair is an input nothing is plugged into."
    )


@pytest.mark.parametrize(
    "path", NAMING_A_SECOND_OUTPUT, ids=lambda p: f"{p.parent.name}/{p.name}"
)
def test_a_second_output_says_how_far_apart_the_swept_takes_were(path: Path) -> None:
    """The one figure that says which side of the pair a record is of.

    `reached_db` beside it cannot: it is the highest each channel reached anywhere
    in the run, and a run's control take has the two sides equal by construction,
    so the run that was read 22 dB down on every swept take publishes a fifth of a
    decibel there. The comparison has to be over the swept takes and no others.
    """
    over = json.loads(path.read_text())["other_channel"].get("over_the_swept_takes")
    assert isinstance(over, dict) and isinstance(over.get("apart_db"), (int, float)), (
        f"{path.relative_to(UNITS)} names a second output and does not say how far "
        "the swept takes stood apart on the two, which is the only place a run read "
        "on the silent side of a pair can be seen."
    )
