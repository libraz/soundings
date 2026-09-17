"""Where a record sits, held against what the record says it is.

A unit's directory is one directory per stage, named after the command that wrote
the records in it. That convention is only worth having if it cannot quietly stop
being true, and it can be checked exactly, because a record's envelope carries the
same word its directory is named with.

So this is not a style rule. A record filed under the wrong stage is a record a
reader will not find when they look for that stage, and `soundings complete`
counts the stage as unrun -- which is the one failure a progress report must not
have. Two names for one thing is how the archive got here the first time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from soundings import index
from soundings.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]
UNITS = ROOT / "data" / "units"
PUBLISHED = sorted(UNITS.rglob("*.json"))


def _under_a_stage(path: Path) -> bool:
    """Whether this record sits in a stage's directory rather than at a unit's top."""
    return path.parent.parent.parent == UNITS


def _at_a_units_top(path: Path) -> bool:
    return path.parent.parent == UNITS


def _commands() -> set[str]:
    parser = build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return set(sub.choices)


def test_every_directory_under_a_unit_is_named_after_a_command() -> None:
    """The stage vocabulary is the CLI's, so a new directory is a new command."""
    commands = _commands()
    found = {path.parent.name for path in PUBLISHED if _under_a_stage(path)}
    unknown = sorted(found - commands)
    assert not unknown, (
        f"directories under a unit that name no command: {unknown}. A record is filed "
        "under the command that wrote it, so a directory with another name is either a "
        "record in the wrong place or a stage that was never a command."
    )


def test_a_record_sits_under_the_stage_that_wrote_it() -> None:
    """The envelope and the directory are two spellings of one fact."""
    wrong = []
    for path in PUBLISHED:
        # The listing of a unit sits above its stages, since its subject is the
        # directory rather than anything measured; it still carries the envelope
        # every writer stamps, and the two facts do not contradict each other.
        if not _under_a_stage(path):
            continue
        found = json.loads(path.read_text())
        if not isinstance(found, dict):
            continue
        stamp = found.get("record")
        if not isinstance(stamp, dict) or not stamp.get("stage"):
            continue
        if path.parent.name != stamp["stage"]:
            wrong.append(f"{path.relative_to(UNITS)} was written by {stamp['stage']}")
    assert not wrong, (
        "records filed under a stage other than the one that wrote them:\n  " + "\n  ".join(wrong)
    )


def test_only_the_hand_kept_records_sit_at_the_top_of_a_unit() -> None:
    """Everything else is a measurement, and a measurement has a stage."""
    loose = sorted(
        str(p.relative_to(UNITS))
        for p in PUBLISHED
        if _at_a_units_top(p) and p.name not in index.AT_THE_ROOT
    )
    assert not loose, (
        f"records at the top of a unit that are not hand-kept: {loose}. A measurement "
        "belongs under the stage that wrote it; only the unit's own identity and its "
        "hand-kept behaviours sit above them."
    )


@pytest.mark.parametrize(
    "unit", sorted(p for p in UNITS.iterdir() if p.is_dir()), ids=lambda p: p.name
)
def test_the_index_says_what_the_directory_holds(unit: Path) -> None:
    """A generated listing is only worth having while it is still generated."""
    written = unit / "index.json"
    assert written.exists(), (
        f"{unit.name} has no index.json. Write one with\n    soundings index {unit}"
    )
    published = json.loads(written.read_text())
    published.pop("record", None)
    assert published == index.survey(unit), (
        f"{unit.name}/index.json disagrees with the directory it lists. Regenerate it "
        f"with\n    soundings index {unit} --out {unit}/index.json"
    )


def test_a_subject_is_a_name_and_not_a_description() -> None:
    """`channel` means two things and only one of them is a subject.

    On a scan it is the MIDI channel a message went out on, which is what the
    record is about. On a stage that reads audio it is an object saying which
    interface input every figure came from and how loud the others were, which is
    a fact about the reading. Grouped by the second, every audio record would be
    its own subject and the coverage count would be the record count.
    """
    assert index.names_it("40 03 05")
    assert index.names_it(3)
    assert not index.names_it({"read": 2, "chosen_by": "highest across the takes read"})
    assert not index.names_it(["01 21", "11 08"])
    assert not index.names_it(None)


def test_two_records_about_one_thing_meet_under_one_key() -> None:
    """Which is the whole point: a rate from one stage and a band profile from
    another are two readings of one parameter, and a listing by stage cannot say
    so."""
    stages = {
        "efx-rate": [
            {"file": "efx-rate/a.json", "about": {"type": "01 22", "address": "40 03 03"}}
        ],
        "efx-bands": [
            {"file": "efx-bands/b.json", "about": {"address": "40 03 03", "type": "01 22"}}
        ],
    }
    found = index.subjects(stages)
    assert found["subjects"] == 1
    assert found["records_naming_one"] == 2
    key = next(iter(found["by_subject"]))
    assert found["by_subject"][key]["stages"] == ["efx-bands", "efx-rate"]


def test_a_record_naming_no_subject_is_listed_and_not_counted() -> None:
    """An absence, not a refusal. A count that absorbed these would be a count of
    what happened to be easy to read, and the stage that has to learn a flag would
    never surface."""
    stages = {
        "balance": [{"file": "balance/40-11.json"}],
        "efx-rate": [
            {"file": "efx-rate/a.json", "about": {"type": "01 22", "address": "40 03 03"}}
        ],
    }
    found = index.subjects(stages)
    assert found["naming_no_subject"] == ["balance/40-11.json"]
    assert found["subjects"] == 1
    assert found["records_naming_one"] == 1


def test_one_field_under_two_names_is_one_subject() -> None:
    """`decay` and `phase` spell the effect type `type_id` and five other stages
    spell it `type`, and both hold `01 00`. Left apart, a phase reading and a band
    profile of one parameter read as two parameters -- a coverage figure counting
    the same work twice and the same gap not at all."""
    stages = {
        "phase": [
            {"file": "phase/a.json", "about": {"type_id": "01 00", "address": "40 03 03"}}
        ],
        "efx-bands": [
            {"file": "efx-bands/b.json", "about": {"type": "01 00", "address": "40 03 03"}}
        ],
    }
    found = index.subjects(stages)
    assert found["subjects"] == 1
    assert list(found["by_subject"]) == ["address 40 03 03, type 01 00"]


def test_the_stages_that_read_one_parameter_can_all_say_which() -> None:
    """A flag on four siblings and not the fifth is a defect in the fifth: the
    fifth's records are the ones no coverage figure reaches."""
    parser = build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    for name in ("balance", "balance-bands", "arrival", "vibrato", "motion", "decay", "phase"):
        flags = {s for action in sub.choices[name]._actions for s in action.option_strings}
        assert {"--type", "--slot"} <= flags, f"{name} cannot say what its takes were of"


def test_a_record_about_a_list_is_read_one_run_at_a_time() -> None:
    """A record of three types at one address has no one subject to put at its
    top, and read only there it looks like a record about nothing. Its runs each
    say what they were of, which is still the record speaking about itself."""
    stages = {
        "balance": [
            {
                "file": "balance/three-types.json",
                "also_about": [
                    {"type": "01 01", "address": "40 03 15"},
                    {"type": "01 03", "address": "40 03 15"},
                ],
            }
        ]
    }
    found = index.subjects(stages)
    assert found["subjects"] == 2
    assert not found["naming_no_subject"]
    for row in found["by_subject"].values():
        assert row["records"] == ["balance/three-types.json"]


def test_a_top_level_subject_wins_over_the_runs() -> None:
    """Read one level down only where the top says nothing. A record that names
    its subject has named it, and going looking for more would file it under
    settings it merely held."""
    stages = {
        "efx-rate": [
            {
                "file": "efx-rate/a.json",
                "about": {"type": "01 22", "address": "40 03 03"},
                "also_about": [{"type": "99 99", "address": "40 03 99"}],
            }
        ]
    }
    found = index.subjects(stages)
    assert list(found["by_subject"]) == ["address 40 03 03, type 01 22"]
