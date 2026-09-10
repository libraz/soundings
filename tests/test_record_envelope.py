"""The archive's shape, gated where prose cannot gate it.

Each of these is a ratchet rather than a state: green on the archive as it stands
and red on the next record that makes it worse.

Every record now carries an envelope, including the ones whose run predates it --
those carry null where the invocation and the moment would be, and name those
fields as ones the run did not record. That is the distinction the gates below
hold: a null that is accounted for is a stated absence, and a null that is not is
a field somebody will eventually fill in with a value nothing holds.

What must not happen is the archive growing a record with no identity, an
envelope with an unexplained hole in it, or another finding spelled as a key name.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soundings import record

ROOT = Path(__file__).resolve().parents[1]
UNITS = ROOT / "data" / "units"
SCHEMA = json.loads((ROOT / "data" / "schema" / "record.json").read_text())
PUBLISHED = sorted(UNITS.rglob("*.json"))

RECORDS_WITHOUT_AN_ENVELOPE = 0
"""How many records carry no envelope at all. It is zero and may not rise.

Every writer stamps one, and the records made before the envelope existed carry
one that says which of its fields their run did not record. So a file here with
no `record` key is a writer that bypassed report.write_json, not a record too old
to have been stamped.
"""

SENTENCE_KEYS = frozenset(
    {
        "answered_here_and_declined_there",
        "captured_is_asserted_not_measured",
        "does_not_see_a_mirrored_block",
        "input_is_asserted_not_measured",
        "landed_outside_its_own_block",
        "left_out_of_the_gesture",
        "marked_by_sysex_not_by_controller",
        "missed_by_the_first_pass",
        "moved_holding_no_value_this_run_sent",
        "records_the_plan_does_not_name",
        "regions_reaching_past_their_mapped_end",
        "shallower_than_the_control_recovered",
        "why_one_setting_carries_the_control",
        "why_the_asymmetry_is_the_evidence",
        "why_the_total_is_reported",
    }
)
"""Top-level keys shaped like a sentence, which is a signal and not a verdict.

The defect this guards against is a key that spells this unit's answer: a reader
has to know the answer in order to ask the question, and the next unit -- whose
window, if it has one, is at other blocks -- spells the same question
differently. A consumer would then match key names per unit rather than read a
value, which is the archive's own rule about one unit's findings turned on its
own records. Three such keys have come out into `findings` entries, whose kind is
the question and whose fields are this unit's answer to it.

The fifteen left are not that. The test cannot tell them apart, because what it
measures is how many underscores a key has, so it catches verbosity and the
defect alike. Each of these was read: some are a limitation of the stage rather
than a finding of it, some are a caveat about how the run was made, and the rest
name a question whose answer is in the value and whose spelling would not change
on the next unit. Moving them would put method statements under `findings`, where
they are not findings, and churn the archive for no reader's benefit.

`captured_is_asserted_not_measured` is the pair of `input_is_asserted_not_measured`
already here: both say that a condition the run was made under was stated by
whoever ran it and not shown by the run. A power-on capture is only a power-on
capture if the unit had just been switched on, and nothing in the reading can
tell that from a unit somebody had already sent to.

So a sixteenth entry is a key to go and look at, not a key that is wrong. Read the
value, decide which of the two it is, and either migrate it or add it here.
"""


def _envelopes():
    for path in PUBLISHED:
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "record" in data:
            yield path, data["record"]


def test_the_schema_describes_the_envelope_that_is_actually_written(tmp_path) -> None:
    """The published contract and the code that fills it, checked against each other.

    A schema nothing is validated against drifts from its writer, and it drifts
    silently because it is data rather than code. Building one envelope and
    comparing its fields is what keeps the file honest.
    """
    written = record.envelope({}, out_path=tmp_path / "any.json")["record"]
    stated = record.unrecorded(unit_id="somebody-01", stage="sweep", first_published="2026-01-01")
    described = SCHEMA["properties"]["record"]["properties"]
    required = set(SCHEMA["properties"]["record"]["required"])

    # Two shapes, and between them they are exactly what the schema describes: a
    # run that recorded itself, and one that says which of its fields it did not.
    assert set(written) | set(stated) == set(described), (
        "the envelope and its schema name different fields"
    )
    assert required <= set(written) and required <= set(stated)
    assert set(described) - set(written) == {"not_recorded"}
    assert written["schema_version"] == record.SCHEMA_VERSION == stated["schema_version"]


def test_a_field_the_run_did_not_record_is_the_only_one_left_null() -> None:
    """A stated absence and an unexplained hole read the same in a listing.

    They are opposite. The first says a value does not exist and cannot be
    recovered; the second is a field somebody will eventually fill in, and the
    only values available to fill it with would be invented. So every null among
    the required fields has to be one this record named.
    """
    unaccounted = []
    for path, stamp in _envelopes():
        named = set(stamp.get("not_recorded", {}).get("fields", []))
        for field in SCHEMA["properties"]["record"]["required"]:
            if stamp.get(field) is None and field not in named:
                unaccounted.append(f"{path.relative_to(UNITS)}: {field}")
    assert not unaccounted, (
        "envelope fields that are null without the record saying its run did not "
        "record them:\n  " + "\n  ".join(unaccounted)
    )


def test_a_record_naming_no_absent_field_carries_no_reason_for_one() -> None:
    """The reason travels with the absence, so a record with neither is complete."""
    stray = [
        str(path.relative_to(UNITS))
        for path, stamp in _envelopes()
        if "not_recorded" in stamp and not stamp["not_recorded"].get("fields")
    ]
    assert not stray, f"records explaining an absence they do not have: {stray}"


@pytest.mark.parametrize("path,stamp", list(_envelopes()), ids=lambda v: str(v))
def test_a_record_carrying_an_envelope_carries_a_complete_one(path, stamp) -> None:
    """Half an envelope is worse than none: it reads as migrated and is not."""
    described = SCHEMA["properties"]["record"]["properties"]
    for field in SCHEMA["properties"]["record"]["required"]:
        assert field in stamp, f"{path.name} has an envelope without {field}"
    assert not set(stamp) - set(described), f"{path.name} carries a field the schema does not name"


def test_a_published_record_under_a_unit_names_that_unit() -> None:
    """An enveloped record in the archive says which unit, never null.

    Null is the right answer while a record is still in a scratch directory --
    there is no unit above the path and nothing at write time can know one is
    coming. Inside `data/units` there always is one, so a null here is a record
    that was written elsewhere and filed afterwards, and the identity is settled
    at that end by `record.seal`.
    """
    nameless = [
        str(p.relative_to(UNITS)) for p, stamp in _envelopes() if stamp.get("unit_id") is None
    ]
    assert not nameless, (
        f"{len(nameless)} published records name no unit, having been written to a scratch "
        f"path and filed afterwards: {nameless[:5]}. Seal them with\n"
        f"    python -m soundings.record data/units/**/*.json"
    )


def test_a_record_that_already_names_a_unit_is_not_relabelled(tmp_path) -> None:
    """Sealing fills a blank; it never moves a record to a different machine.

    A record whose unit disagrees with the directory it sits in has been mis-filed,
    and overwriting the name it carries would turn that into a silent relabelling
    of somebody's measurement.
    """
    unit = tmp_path / "roland-sc8850-01"
    unit.mkdir()
    (unit / "meta.json").write_text(json.dumps({"unit_id": "roland-sc8850-01"}))
    filed = unit / "misfiled.json"
    filed.write_text(json.dumps({"record": {"unit_id": "roland-sc88pro-01"}, "value": 1}))

    assert record.seal(filed) is False
    assert json.loads(filed.read_text())["record"]["unit_id"] == "roland-sc88pro-01"


def test_the_records_predating_the_envelope_are_the_number_that_is_recorded() -> None:
    """Exact rather than an upper bound, in both directions and for two reasons.

    Upwards it means a record reached the archive without an envelope, which
    every writer now stamps -- so something bypassed report.write_json. Downwards
    it means records were migrated and the number was left saying otherwise,
    which is the staleness an allowlist entry has once its divergence is gone.
    """
    legacy = [p.name for p in PUBLISHED if "record" not in json.loads(p.read_text())]
    assert len(legacy) == RECORDS_WITHOUT_AN_ENVELOPE, (
        f"{len(legacy)} records predate the envelope, not {RECORDS_WITHOUT_AN_ENVELOPE}. "
        "Set RECORDS_WITHOUT_AN_ENVELOPE to the new number in the same change that "
        "migrated them, or find the writer that bypassed report.write_json."
    )


def test_no_new_finding_is_spelled_as_a_key_name() -> None:
    found = set()
    for path in PUBLISHED:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            found |= {k for k in data if k.count("_") >= 4}
    new = found - SENTENCE_KEYS
    assert not new, (
        f"new top-level keys stating a finding rather than holding one: {sorted(new)}. "
        "A finding belongs in a value that names its kind (record.finding), so a reader "
        "can ask the question without already knowing this unit's answer to it."
    )
    assert not SENTENCE_KEYS - found, (
        f"SENTENCE_KEYS names keys the archive no longer has: {sorted(SENTENCE_KEYS - found)}. "
        "Delete them in the change that migrated the record."
    )
