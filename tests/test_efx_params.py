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


def _record(address: str, *, audible: bool, shape=None, level=None, repeats=(-40.0, -40.0)):
    """A contrast record shaped as the comparison writes one."""
    return {
        "address": address,
        "values": [0, 127],
        "audible": audible,
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
