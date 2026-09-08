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
