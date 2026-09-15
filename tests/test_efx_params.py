"""One insertion effect type's parameters, read into a verdict apiece.

The failure this guards against is a modulator wearing a parameter's name. A
setting that stops repeating is not a setting that sounds different, and the two
arrive at this code looking alike: both are addresses the comparison called
audible.

No hardware: every record here is built in the test.
"""

from __future__ import annotations

import json

from soundings import efxparams


def _record(
    address: str, *, audible: bool, shape=None, level=None, repeats=(-40.0, -40.0), conclusive=True
):
    """A contrast record shaped as the comparison writes one."""
    return {
        "address": address,
        "values": [0, 127],
        "audible": audible,
        "conclusive": conclusive,
        "by_stimulus": [
            {
                "stimulus": "a note",
                "takes_per_setting": 4,
                "each_setting_unrepeatable_db": list(repeats),
                "same_setting_residual_db": max(repeats),
                "across_setting_residual_db": -1.0 if audible else -40.0,
                "across_setting_level_db": 6.0 if level else 0.0,
                "noise_floor_db": -56.0,
                "changed_the_shape": shape,
                "changed_the_level": level,
                "margin_db": 6.0,
            }
        ],
    }


def _write(where, name, payload):
    where.mkdir(parents=True, exist_ok=True)
    (where / name).write_text(json.dumps(payload))


def test_a_setting_that_stops_repeating_is_not_a_setting_that_sounds_different(tmp_path) -> None:
    """Heard on neither shape nor level, so what separates the two is the asymmetry."""
    record = _record("40 03 03", audible=True, shape=False, level=False, repeats=(-40.0, -7.9))
    verdict = efxparams.row("04 02", 0, 64, record, None)
    assert verdict["verdict"] == efxparams.UNREADABLE
    assert verdict["audible"] is False
    assert "asymmetry" in verdict["why"]


def test_a_refusal_is_not_a_null(tmp_path) -> None:
    """A comparison that could not separate its settings reports the same `audible` false
    a real null does, and only `conclusive` tells them apart. Measured on type 01 20,
    seventeen of twenty parameters arrived this way -- every take of one setting already
    differing by most of the signal, because the type runs its modulator at the values it
    powers up holding. Read as nulls they would say the type has three live parameters."""
    record = _record(
        "40 03 09", audible=False, shape=False, level=False, repeats=(-2.0, -2.5), conclusive=False
    )
    verdict = efxparams.row("01 20", 6, 0, record, None)
    assert verdict["verdict"] == efxparams.NO_YARDSTICK
    assert verdict["verdict"] != efxparams.NULL
    assert verdict["audible"] is False
    assert verdict["conclusive"] is False
    assert "unasked, not inaudible" in verdict["why"]


def test_a_null_that_had_a_yardstick_stays_a_null(tmp_path) -> None:
    """The guard above must not swallow the negatives the archive exists to publish."""
    record = _record("40 03 0A", audible=False, shape=False, level=False, repeats=(-44.0, -45.0))
    verdict = efxparams.row("01 20", 7, 0, record, None)
    assert verdict["verdict"] == efxparams.NULL
    assert verdict["conclusive"] is True


def test_a_verdict_that_stands_still_names_the_modulator_it_also_started(tmp_path) -> None:
    """The settings differ by more than the takes do, and one of them still moves."""
    record = _record("40 03 04", audible=True, shape=True, level=True, repeats=(-39.0, -17.1))
    verdict = efxparams.row("04 02", 1, 60, record, None)
    assert verdict["verdict"] == efxparams.AUDIBLE
    assert "started something moving" in verdict["why"]


def test_a_parameter_heard_only_once_the_gain_was_divided_out_says_so(tmp_path) -> None:
    record = _record("40 03 05", audible=True, shape=False, level=True)
    verdict = efxparams.row("04 02", 2, 10, record, None)
    assert verdict["verdict"] == efxparams.AUDIBLE
    assert "how much of it there is" in verdict["why"]


def test_a_clean_verdict_carries_no_qualifier(tmp_path) -> None:
    record = _record("40 03 06", audible=True, shape=True, level=False)
    assert "why" not in efxparams.row("04 02", 3, 1, record, None)


def test_the_control_is_not_counted_as_one_of_the_parameters(tmp_path) -> None:
    """Counting it would shift every later slot onto another parameter's default."""
    _write(tmp_path, "a.json", _record("40 03 03", audible=False, shape=False, level=False))
    _write(tmp_path, "b.json", _record("40 03 04", audible=True, shape=True, level=False))
    control = tmp_path / "control.json"
    control.write_text(json.dumps(_record("40 42 22", audible=True, shape=True, level=True)))
    found = efxparams.read_directory(tmp_path, "04 02", [7, 9], control, [])
    assert [p["address"] for p in found["parameters"]] == ["40 03 03", "40 03 04"]
    assert [p["default"] for p in found["parameters"]] == [7, 9]


def test_an_address_asked_again_carries_the_pair_that_was_withdrawn(tmp_path) -> None:
    """The reason travels with the row, not with whoever remembers the directory."""
    _write(tmp_path, "first.json", _record("40 03 03", audible=True, shape=False, level=True))
    again = tmp_path / "again.json"
    again.write_text(
        json.dumps(
            _record("40 03 03", audible=True, shape=False, level=True, repeats=(-38.0, -38.0))
        )
    )
    control = tmp_path / "control.json"
    control.write_text(json.dumps(_record("40 42 22", audible=True, shape=True, level=True)))
    found = efxparams.read_directory(
        tmp_path, "04 02", [7], control, [], supersede={"40 03 03": str(again)}
    )
    row = found["parameters"][0]
    assert row["withdrawn_pair"]["values"] == [0, 127]
    assert "silent" in row["why_asked_again"]
    assert row["each_setting_unrepeatable_db"] == [-38.0, -38.0]


def test_the_record_carries_its_control_and_its_limits(tmp_path) -> None:
    """A run of nulls is readable only beside the thing that proves it could have found one."""
    _write(tmp_path, "a.json", _record("40 03 03", audible=False, shape=False, level=False))
    control = tmp_path / "control.json"
    control.write_text(json.dumps(_record("40 42 22", audible=True, shape=True, level=True)))
    found = efxparams.read_directory(tmp_path, "04 02", [7], control, [])
    assert found["positive_control"]["address"] == "40 42 22"
    assert found["positive_control"]["audible"] is True
    assert found["not_established"]
    assert found["results"]["asked"] == 1


def _refusal(address: str, why: str = "the lead-in was not quiet") -> dict:
    """A contrast record shaped as the comparison writes one when it will not answer."""
    return {
        "address": address,
        "values": [0, 127],
        "channel": 2,
        "refused": {
            "at": "a note",
            "why": why,
            "measured": {"asked_for_dbfs": -60, "loudest_lead_in_dbfs": -52.8},
        },
    }


def test_a_slot_whose_run_refused_is_not_counted_as_one_nobody_asked(tmp_path) -> None:
    """Only one of the two is work outstanding.

    Asking a refused slot again the same way gets the same refusal, so reading
    it as unasked puts it in a queue it can never leave.
    """
    _write(tmp_path, "a.json", _record("40 03 03", audible=False, shape=False, level=False))
    _write(tmp_path, "b.json", _refusal("40 03 04"))
    control = tmp_path / "control.json"
    control.write_text(json.dumps(_record("40 42 22", audible=True, shape=True, level=True)))
    found = efxparams.read_directory(
        tmp_path, "04 02", [7, 9], control, [], slots=["40 03 03", "40 03 04", "40 03 05"]
    )
    coverage = found["coverage"]
    assert coverage["never_asked"] == ["40 03 05"]
    assert [r["address"] for r in coverage["refused"]] == ["40 03 04"]
    assert coverage["refused"][0]["parameter"] == 1
    assert coverage["refused"][0]["measured"]["loudest_lead_in_dbfs"] == -52.8
    assert "why_refused" in coverage


def test_a_refused_slot_carries_no_verdict_either_way(tmp_path) -> None:
    """What was refused was the takes. Nothing says the parameter does nothing."""
    _write(tmp_path, "b.json", _refusal("40 03 04"))
    control = tmp_path / "control.json"
    control.write_text(json.dumps(_record("40 42 22", audible=True, shape=True, level=True)))
    found = efxparams.read_directory(
        tmp_path, "04 02", [7, 9], control, [], slots=["40 03 03", "40 03 04"]
    )
    assert found["parameters"] == []
    assert found["coverage"]["answered"] == 0


def test_a_directory_with_nothing_refused_says_so_without_the_note(tmp_path) -> None:
    """A key explaining a thing that did not happen is a key a reader has to rule out."""
    _write(tmp_path, "a.json", _record("40 03 03", audible=False, shape=False, level=False))
    control = tmp_path / "control.json"
    control.write_text(json.dumps(_record("40 42 22", audible=True, shape=True, level=True)))
    found = efxparams.read_directory(tmp_path, "04 02", [7], control, [], slots=["40 03 03"])
    assert found["coverage"]["refused"] == []
    assert "why_refused" not in found["coverage"]


def _held(address: str, value: str = "00") -> dict:
    return {"address": address, "bytes": value}


def test_a_slot_the_run_held_still_is_not_one_it_failed_to_reach(tmp_path) -> None:
    """The whole of a moving type is unaskable until its modulator is held, and
    the held byte then has no record -- which is what a slot the run could not
    reach also looks like. Read as unasked it goes into a queue of work
    outstanding, and asking it means writing it to the top of its range, which
    starts the modulator again and spoils the slots measured after it."""
    _write(tmp_path, "a.json", _record("40 03 03", audible=False, shape=False, level=False))
    control = tmp_path / "control.json"
    control.write_text(json.dumps(_record("40 42 22", audible=True, shape=True, level=True)))

    found = efxparams.read_directory(
        tmp_path,
        "01 20",
        [7, 9, 3],
        control,
        [_held("40 03 00", "01 20"), _held("40 03 05")],
        slots=["40 03 03", "40 03 04", "40 03 05"],
    )

    coverage = found["coverage"]
    assert coverage["held_still"] == ["40 03 05"]
    assert coverage["never_asked"] == ["40 03 04"]
    assert "why_held_still" in coverage


def test_the_value_a_held_slot_was_held_at_is_the_one_the_record_publishes(tmp_path) -> None:
    """The ground is read off `prepared` rather than from a list beside it, so the
    byte a reader is shown is the byte the run was given. Two lists would be two
    to keep in step, and the way that drifts is a record saying a slot was held
    with nothing saying what at."""
    control = tmp_path / "control.json"
    control.write_text(json.dumps(_record("40 42 22", audible=True, shape=True, level=True)))

    found = efxparams.read_directory(
        tmp_path, "01 20", [3], control, [_held("40 03 03")], slots=["40 03 03"]
    )

    assert found["coverage"]["held_still"] == ["40 03 03"]
    assert _held("40 03 03") in found["prepared"]


def test_a_null_measured_beside_a_held_byte_says_the_hold_narrows_it(tmp_path) -> None:
    """The hold is what makes the rest of the type askable, and it is also what
    makes one class of null unreadable: a parameter that does nothing but scale
    what the held byte moves has nothing to scale, and answers inaudible. Without
    the limit the record publishes that as a fact about the unit, which is the
    run's own doing reported as the unit's behaviour."""
    _write(tmp_path, "a.json", _record("40 03 04", audible=False, shape=False, level=False))
    control = tmp_path / "control.json"
    control.write_text(json.dumps(_record("40 42 22", audible=True, shape=True, level=True)))

    found = efxparams.read_directory(
        tmp_path,
        "01 20",
        [7, 9, 3],
        control,
        [_held("40 03 00", "01 20"), _held("40 03 05")],
        slots=["40 03 03", "40 03 04", "40 03 05"],
    )

    assert efxparams.WHY_NULL_WHILE_HELD in found["not_established"]
    assert found["parameters"][0]["verdict"] == efxparams.NULL


def test_an_unparked_run_does_not_carry_the_limit_about_holding(tmp_path) -> None:
    """Sixty-odd records held nothing, and a limit explaining a hold none of them
    made is one more sentence a reader has to rule out before trusting the nulls."""
    _write(tmp_path, "a.json", _record("40 03 03", audible=False, shape=False, level=False))
    control = tmp_path / "control.json"
    control.write_text(json.dumps(_record("40 42 22", audible=True, shape=True, level=True)))

    found = efxparams.read_directory(
        tmp_path, "04 02", [7], control, [_held("40 03 00", "04 02")], slots=["40 03 03"]
    )

    assert efxparams.WHY_NULL_WHILE_HELD not in found["not_established"]
    assert list(efxparams.LIMITS) == found["not_established"]


def test_a_run_that_held_nothing_carries_no_note_about_holding(tmp_path) -> None:
    """Every unparked run is one, so the note would stand over sixty-odd records
    explaining a thing none of them did."""
    _write(tmp_path, "a.json", _record("40 03 03", audible=False, shape=False, level=False))
    control = tmp_path / "control.json"
    control.write_text(json.dumps(_record("40 42 22", audible=True, shape=True, level=True)))

    found = efxparams.read_directory(
        tmp_path, "04 02", [7], control, [_held("40 03 00", "04 02")], slots=["40 03 03"]
    )

    assert found["coverage"]["held_still"] == []
    assert "why_held_still" not in found["coverage"]
