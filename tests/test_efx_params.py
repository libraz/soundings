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


def test_a_pair_withdrawn_for_naming_an_unprinted_value_says_that_and_not_silence(
    tmp_path,
) -> None:
    """Two things send an address round again, and the row must name the right one.

    A pair of nought against 127 on a parameter the page prints two states for was
    not withdrawn because a setting fell silent -- it was withdrawn because 127 is
    not one of that parameter's settings. Reported as the other reason the row
    would say the run had found something it did not look for.
    """
    _write(tmp_path, "first.json", _record("40 03 03", audible=False, shape=False, level=False))
    again = tmp_path / "again.json"
    again.write_text(json.dumps(_record("40 03 03", audible=True, shape=True, level=False)))
    control = tmp_path / "control.json"
    control.write_text(json.dumps(_record("40 42 22", audible=True, shape=True, level=True)))
    found = efxparams.read_directory(
        tmp_path,
        "04 02",
        [7],
        control,
        [],
        supersede={"40 03 03": str(again)},
        printed={"40 03 03": "00/01"},
    )
    row = found["parameters"][0]
    assert row["withdrawn_pair"]["values"] == [0, 127]
    assert "not printed as having" in row["why_asked_again"]
    assert "silent" not in row["why_asked_again"]


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


def test_a_null_taken_at_a_value_the_page_gives_the_parameter_none_of_is_not_a_null() -> None:
    """The store's accepted range is not the engine's reachable settings.

    This block accepts and reads back every seven-bit value at every one of its
    addresses, so a pair taken from the ends of what it accepts never clamps and
    nothing about such a null looks wrong. What is wrong is visible only over the
    class: not one parameter of it was ever heard.
    """
    record = _record("40 03 03", audible=False)
    verdict = efxparams.row("01 55", 0, 0, record, None, "00/01/02/03/04/05")
    assert verdict["verdict"] == efxparams.OUTSIDE_ITS_PRINTED_VALUES
    assert verdict["asked_inside_its_printed_values"] is False
    assert verdict["printed_values"] == "00/01/02/03/04/05", "the cell, not what it expands to"


def test_two_printed_values_that_are_nought_and_127_leave_the_pair_asked_correctly() -> None:
    """A rotor's speed switch is printed as those two bytes, so the ends of the
    accepted range are exactly its two settings -- and a rule counting the names
    between the slashes would withdraw a slot that was asked right."""
    record = _record("40 03 0D", audible=False)
    verdict = efxparams.row("01 22", 10, 0, record, None, "00/7F")
    assert verdict["verdict"] == efxparams.NULL
    assert verdict["asked_inside_its_printed_values"] is True


def test_a_parameter_pointed_at_a_conversion_table_narrows_nothing() -> None:
    """That column gives a setting at every one of the 128 values. A cell holding a
    slash -- `315-8k/Bypass` in the setting column beside it -- is not a list."""
    record = _record("40 03 13", audible=False)
    verdict = efxparams.row("04 03", 16, 0, record, None, "*8")
    assert verdict["verdict"] == efxparams.NULL
    assert verdict["asked_inside_its_printed_values"] is True


def test_a_parameter_heard_at_such_a_pair_was_still_heard() -> None:
    """Only a null taken there is uninterpretable. A tone gain's values run from 52
    to 76 and the pair asked was 0 and 127, and it was audible anyway."""
    record = _record("40 03 13", audible=True, shape=True, level=True)
    verdict = efxparams.row("01 00", 16, 64, record, None, "34–4C")
    assert verdict["verdict"] == efxparams.AUDIBLE
    assert verdict["asked_inside_its_printed_values"] is False


WRAPS = {0: "L180(=R180)", 64: "0", 127: "R180(=L180)"}
"""One column of the conversion grid comes back to where it began, at the two
values a pair is otherwise taken from."""


def test_a_null_at_two_values_the_page_calls_one_setting_is_not_a_null(tmp_path) -> None:
    """The takes are takes of one setting, so they differ by what a setting differs
    from itself by -- which is the number a null is, arriving the same way and
    meaning something else. One such row was published conclusive."""
    _write(tmp_path, "a.json", _record("40 03 03", audible=False, shape=False, level=False))
    control = tmp_path / "control.json"
    control.write_text(json.dumps(_record("40 42 22", audible=True, shape=True, level=True)))

    found = efxparams.read_directory(
        tmp_path,
        "01 71",
        [64],
        control,
        [],
        printed={"40 03 03": "*13"},
        settings={"40 03 03": WRAPS},
    )

    row = found["parameters"][0]
    assert row["verdict"] == efxparams.ASKED_AT_ONE_SETTING
    assert row["the_pair_names_one_setting"] is True
    # The referral does give every value a setting, so the other question's answer
    # stays yes and a row carrying only it reads as a pair asked properly.
    assert row["asked_inside_its_printed_values"] is True


def test_a_pair_the_page_calls_two_settings_is_still_read_as_a_null(tmp_path) -> None:
    """Thirteen of the grid's fourteen columns do not come back on themselves, and
    the rule has to leave every one of their nulls standing."""
    _write(tmp_path, "a.json", _record("40 03 03", audible=False, shape=False, level=False))
    control = tmp_path / "control.json"
    control.write_text(json.dumps(_record("40 42 22", audible=True, shape=True, level=True)))

    found = efxparams.read_directory(
        tmp_path,
        "01 20",
        [64],
        control,
        [],
        printed={"40 03 03": "*6"},
        settings={"40 03 03": {0: "0.05", 64: "2.60", 127: "10.00"}},
    )

    assert found["parameters"][0]["verdict"] == efxparams.NULL
    assert found["parameters"][0]["the_pair_names_one_setting"] is False


def test_a_pair_withdrawn_for_being_one_setting_twice_says_that_and_not_silence(
    tmp_path,
) -> None:
    """Three things send an address round again now, and a row that names the wrong
    one reports a finding the run never made."""
    _write(tmp_path, "first.json", _record("40 03 03", audible=False, shape=False, level=False))
    again = tmp_path / "again.json"
    again.write_text(json.dumps(_record("40 03 03", audible=True, shape=True, level=False)))
    control = tmp_path / "control.json"
    control.write_text(json.dumps(_record("40 42 22", audible=True, shape=True, level=True)))

    found = efxparams.read_directory(
        tmp_path,
        "01 71",
        [64],
        control,
        [],
        supersede={"40 03 03": str(again)},
        printed={"40 03 03": "*13"},
        settings={"40 03 03": WRAPS},
    )

    row = found["parameters"][0]
    assert row["why_asked_again"] == efxparams.WHY_ASKED_AT_ONE_SETTING
    assert "silent" not in row["why_asked_again"]


#: Two stages of one type as the page prints them, with the gate of each named the
#: way the effect list names it: a short tag, then the quantity.
TWO_STAGES = {
    "40 03 08": "W/P Sel",
    "40 03 09": "W/P LPF",
    "40 03 0A": "W/P Level",
    "40 03 0B": "Disc Type",
    "40 03 0D": "Disc Nz Lev",
}


def test_a_null_taken_with_its_own_stage_switched_off_is_not_a_null(tmp_path) -> None:
    """The filter of a noise generator whose level powers up at nought had nothing to
    pass, so the two settings are two takes of the same silence. Read as a null it
    would say the filter does nothing; what it says is that the run could not have
    heard it."""
    _write(tmp_path, "a.json", _record("40 03 09", audible=False, shape=False, level=False))
    control = tmp_path / "control.json"
    control.write_text(json.dumps(_record("40 42 22", audible=True, shape=True, level=True)))

    found = efxparams.read_directory(
        tmp_path,
        "01 73",
        [1, 127, 0, 0, 0],
        control,
        [],
        slots=list(TWO_STAGES),
        names=TWO_STAGES,
    )

    # One record in the directory, so one row: the rest of the type is `never_asked`.
    row = found["parameters"][0]
    assert row["address"] == "40 03 09"
    assert row["verdict"] == efxparams.ITS_STAGE_WAS_SHUT
    assert row["audible"] is False
    shut = row["at_nought_while_this_was_asked"]
    assert [g["address"] for g in shut] == ["40 03 0A"]
    assert shut[0]["at_nought_by"] == efxparams.BY_THE_UNIT


def test_a_gate_shut_by_another_stage_leaves_the_null_standing(tmp_path) -> None:
    """The page groups a type's parameters by printing a tag in front of each, and a
    gate reaches its own stage and no other. A disc generator at nought says nothing
    about the filter of the noise generator beside it."""
    _write(tmp_path, "a.json", _record("40 03 09", audible=False, shape=False, level=False))
    control = tmp_path / "control.json"
    control.write_text(json.dumps(_record("40 42 22", audible=True, shape=True, level=True)))

    found = efxparams.read_directory(
        tmp_path,
        "01 73",
        [127, 127, 0],
        control,
        [],
        slots=["40 03 09", "40 03 0A", "40 03 0D"],
        # The W/P generator is up; only the disc one is at nought.
        names={"40 03 09": "W/P LPF", "40 03 0A": "W/P Level", "40 03 0D": "Disc Nz Lev"},
    )

    row = found["parameters"][0]
    assert row["verdict"] == efxparams.NULL
    assert "at_nought_while_this_was_asked" not in row


def test_a_gate_asked_at_its_own_two_settings_is_asked(tmp_path) -> None:
    """A level byte resting at nought is what makes the parameters behind it unasked.
    It does not make the level byte itself unasked: what it was asked at is the pair,
    not what it rests at, and a rule that read otherwise would withdraw every gate on
    the unit."""
    _write(tmp_path, "a.json", _record("40 03 0A", audible=False, shape=False, level=False))
    control = tmp_path / "control.json"
    control.write_text(json.dumps(_record("40 42 22", audible=True, shape=True, level=True)))

    found = efxparams.read_directory(
        tmp_path,
        "01 73",
        [1, 127, 0, 0, 0],
        control,
        [],
        slots=list(TWO_STAGES),
        names=TWO_STAGES,
    )

    assert found["parameters"][0]["address"] == "40 03 0A"
    assert found["parameters"][0]["verdict"] == efxparams.NULL


def test_a_stage_the_run_itself_shut_says_which_of_the_two_it_was(tmp_path) -> None:
    """A parked run holds a modulator's depth at nought on purpose and the type's other
    record answers what sits behind it; a value nobody chose is a state no record
    mentions. Both are reported and they are not the same finding."""
    _write(tmp_path, "a.json", _record("40 03 09", audible=False, shape=False, level=False))
    control = tmp_path / "control.json"
    control.write_text(json.dumps(_record("40 42 22", audible=True, shape=True, level=True)))

    found = efxparams.read_directory(
        tmp_path,
        "01 73",
        [1, 127, 127, 0, 0],
        control,
        # The unit powers the generator up at full; this run wrote it down.
        [{"address": "40 03 0A", "bytes": "00"}],
        slots=list(TWO_STAGES),
        names=TWO_STAGES,
    )

    row = found["parameters"][0]
    assert row["address"] == "40 03 09"
    assert row["verdict"] == efxparams.ITS_STAGE_WAS_SHUT
    assert row["at_nought_while_this_was_asked"][0]["at_nought_by"] == efxparams.BY_THIS_RUN


#: A chorus stage as the page prints it: a mix deciding how much of it reaches the
#: output, a depth deciding how far its modulation travels, and two parameters.
ONE_CHORUS = {
    "40 03 0F": "CF Rate",
    "40 03 10": "CF Depth",
    "40 03 11": "CF Fb",
    "40 03 12": "CF Mix",
}


def test_a_modulation_at_nothing_is_not_a_stage_out_of_the_output(tmp_path) -> None:
    """The two silence different things and the verdict says which. A rate byte with
    its depth at nought scales nothing, which is what the parked runs were made to
    leave -- and it is not the same statement as a stage that is not there."""
    _write(tmp_path, "a.json", _record("40 03 0F", audible=False, shape=False, level=False))
    control = tmp_path / "control.json"
    control.write_text(json.dumps(_record("40 42 22", audible=True, shape=True, level=True)))

    found = efxparams.read_directory(
        tmp_path,
        "04 02",
        [8, 0, 89, 50],
        control,
        [],
        slots=list(ONE_CHORUS),
        names=ONE_CHORUS,
    )

    row = found["parameters"][0]
    assert row["verdict"] == efxparams.ITS_MODULATION_WAS_STILL
    assert row["at_nought_while_this_was_asked"][0]["leaves"] == efxparams.MODULATION_STILL


def test_a_mix_is_not_silenced_by_the_depth_beside_it(tmp_path) -> None:
    """A chorus at no depth is a fixed delay, so turning its mix up is still audible.
    A rule that read a depth as silencing everything in the stage would withdraw the
    one byte the run could most certainly hear."""
    _write(tmp_path, "a.json", _record("40 03 12", audible=False, shape=False, level=False))
    control = tmp_path / "control.json"
    control.write_text(json.dumps(_record("40 42 22", audible=True, shape=True, level=True)))

    found = efxparams.read_directory(
        tmp_path,
        "04 02",
        [8, 0, 89, 50],
        control,
        [],
        slots=list(ONE_CHORUS),
        names=ONE_CHORUS,
    )

    row = found["parameters"][0]
    assert row["address"] == "40 03 12"
    assert row["verdict"] == efxparams.NULL
    assert "at_nought_while_this_was_asked" not in row


def test_a_level_inside_a_stage_that_is_not_there_is_not_asked_either(tmp_path) -> None:
    """A rotary's own per-band level cannot be heard while the byte deciding how much
    of the rotary reaches the output is at nought. A gate is only exempt from its own
    reading, not from another gate's."""
    names = {"40 03 0B": "RT Lo Lev", "40 03 15": "RT Level", "40 03 14": "RT Pan"}
    _write(tmp_path, "a.json", _record("40 03 0B", audible=False, shape=False, level=False))
    control = tmp_path / "control.json"
    control.write_text(json.dumps(_record("40 42 22", audible=True, shape=True, level=True)))

    found = efxparams.read_directory(
        tmp_path,
        "11 04",
        [127, 127, 0],
        control,
        [{"address": "40 03 15", "bytes": "00"}],
        slots=["40 03 0B", "40 03 14", "40 03 15"],
        names=names,
    )

    row = found["parameters"][0]
    assert row["address"] == "40 03 0B"
    assert row["verdict"] == efxparams.ITS_STAGE_WAS_SHUT
    assert row["at_nought_while_this_was_asked"][0]["address"] == "40 03 15"
