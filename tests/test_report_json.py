"""A record built from numpy values has to survive being written.

Everything here is measured with numpy and a numpy scalar behaves like a number
until the last step of a run, where json refuses it: a numpy bool derived from
one dB comparison lost a three minute measurement with every take recorded.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from soundings.cli import report


def test_a_record_holding_numpy_values_is_written(tmp_path) -> None:
    where = tmp_path / "record.json"

    report.write_json(
        str(where),
        {
            "was_quiet": np.float64(-70.0) <= -60.0,
            "lead_in_dbfs": np.float64(-70.25),
            "takes": np.int64(4),
        },
    )

    written = json.loads(where.read_text())
    assert {k: v for k, v in written.items() if k != "record"} == {
        "was_quiet": True,
        "lead_in_dbfs": -70.25,
        "takes": 4,
    }


def test_a_record_written_outside_the_archive_names_no_unit(tmp_path) -> None:
    """Null is the answer, not an omission.

    A run writing to a scratch path is not publishing, and there is no unit for
    it to claim. The field is present and empty rather than absent, so a reader
    never has to tell "measured from nothing in particular" from "written before
    the envelope existed".
    """
    where = tmp_path / "scratch.json"

    report.write_json(str(where), {"whatever": 1})

    assert json.loads(where.read_text())["record"]["unit_id"] is None


def test_something_that_is_not_a_number_is_still_refused() -> None:
    """Converting numpy is not a licence to write anything at all."""
    with pytest.raises(TypeError):
        report._plain(object())
