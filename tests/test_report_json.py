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
    assert written == {"was_quiet": True, "lead_in_dbfs": -70.25, "takes": 4}


def test_something_that_is_not_a_number_is_still_refused() -> None:
    """Converting numpy is not a licence to write anything at all."""
    with pytest.raises(TypeError):
        report._plain(object())
