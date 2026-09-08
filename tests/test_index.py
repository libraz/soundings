"""Listing a unit's directory from the records in it.

The listing exists to answer "what is here", and it is only worth having while
every line of it is still read off the record it names. So what is checked is
that it derives: a record moved, renamed or added changes the listing without
anybody editing it, and a record that says nothing about itself gets an entry
that says nothing rather than a guess made from its file name.

No hardware and no unit: every directory here is built in the test.
"""

from __future__ import annotations

import json
from pathlib import Path

from soundings import index


def _write(unit: Path, name: str, payload: dict) -> None:
    (unit / name).parent.mkdir(parents=True, exist_ok=True)
    (unit / name).write_text(json.dumps(payload))


def test_records_are_grouped_by_the_directory_that_holds_them(tmp_path) -> None:
    _write(tmp_path, "meta.json", {"unit_id": "somebody-01"})
    _write(tmp_path, "contrast/40-11-00.json", {"address": "40 11 00"})
    _write(tmp_path, "contrast/40-11-01.json", {"address": "40 11 01"})
    _write(tmp_path, "block/40-11.json", {"block": "40 11"})

    found = index.survey(tmp_path)
    assert found["unit_id"] == "somebody-01"
    assert found["records"] == 3
    assert sorted(found["stages"]) == ["block", "contrast"]
    assert [e["file"] for e in found["stages"]["contrast"]] == [
        "contrast/40-11-00.json",
        "contrast/40-11-01.json",
    ]


def test_an_entry_says_what_the_record_says_about_itself(tmp_path) -> None:
    """Read from the record's fields, so an entry cannot contradict its record."""
    _write(tmp_path, "meta.json", {"unit_id": "somebody-01"})
    _write(
        tmp_path,
        "contrast/cc-91.json",
        {
            "controller": 91,
            "address": None,
            "record": {"stage": "contrast", "measured_at": "2026-01-01T00:00:00+00:00"},
        },
    )
    (entry,) = index.survey(tmp_path)["stages"]["contrast"]
    assert entry["about"] == {"controller": 91}, "a null field is not something it is about"
    assert entry["written_by"] == "contrast"
    assert entry["measured_at"] == "2026-01-01T00:00:00+00:00"


def test_a_record_that_says_nothing_about_itself_gets_no_guess(tmp_path) -> None:
    """The file name is not evidence, so a bare record is listed and left bare."""
    _write(tmp_path, "meta.json", {"unit_id": "somebody-01"})
    _write(tmp_path, "transfer/analogue-input.json", {"answer": "no"})
    (entry,) = index.survey(tmp_path)["stages"]["transfer"]
    assert entry == {"file": "transfer/analogue-input.json"}


def test_a_record_left_at_the_top_of_a_unit_is_named_rather_than_dropped(tmp_path) -> None:
    """Filed above every stage, it would otherwise vanish from the listing entirely."""
    _write(tmp_path, "meta.json", {"unit_id": "somebody-01"})
    _write(tmp_path, "measurements.json", {"note": "kept by hand"})
    _write(tmp_path, "strays.json", {"block": "40 11"})

    found = index.survey(tmp_path)
    assert found["filed_under_no_stage"] == ["strays.json"]
    assert found["records"] == 0, "a stray is not counted as one of the stages' records"


def test_the_hand_kept_records_are_not_listed_as_measurements(tmp_path) -> None:
    _write(tmp_path, "meta.json", {"unit_id": "somebody-01"})
    _write(tmp_path, "measurements.json", {"note": "kept by hand"})
    _write(tmp_path, "index.json", {"note": "this listing"})
    found = index.survey(tmp_path)
    assert found["records"] == 0
    assert "filed_under_no_stage" not in found


def test_the_rendering_names_every_record(tmp_path) -> None:
    _write(tmp_path, "meta.json", {"unit_id": "somebody-01"})
    _write(tmp_path, "block/40-11.json", {"block": "40 11"})
    _write(tmp_path, "contrast/40-11-00.json", {"address": "40 11 00"})
    shown = index.render(index.survey(tmp_path))
    assert "block/  (1)" in shown
    assert "40-11.json" in shown
    assert "address 40 11 00" in shown
