"""Every published file has to be JSON, which `json.dump` does not guarantee.

Python writes `-Infinity` and `NaN` and reads them back without complaint, so a
record holding one round-trips through this repo and passes every other test
here. `jq`, `JSON.parse`, Go and Rust all refuse the file outright, and those are
the readers the archive is published for. The gate is on the archive rather than
on any one writer, because the leak is one forgotten guard in one `to_json` and
there is no reason to expect the next one to be in a module anybody thought to
check.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from soundings import jsonio

UNITS = Path(__file__).resolve().parents[1] / "data" / "units"
PUBLISHED = sorted(UNITS.rglob("*.json"))


def _refuse(constant: str) -> float:
    raise ValueError(constant)


@pytest.mark.parametrize("path", PUBLISHED, ids=lambda p: str(p.relative_to(UNITS)))
def test_a_published_record_is_json_a_strict_reader_accepts(path: Path) -> None:
    json.loads(path.read_text(), parse_constant=_refuse)


def _walk(node, trail=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk(value, f"{trail}.{key}" if trail else key)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item, f"{trail}[]")
    else:
        yield trail, node


@pytest.mark.parametrize("path", PUBLISHED, ids=lambda p: str(p.relative_to(UNITS)))
def test_a_published_record_holds_no_non_finite_figure(path: Path) -> None:
    """The same thing said of the values rather than of the bytes.

    A reader that repairs the file by hand, or a writer that emits the string
    "-Infinity", would get past the parse and still leave a figure no arithmetic
    can use.
    """
    text = path.read_text()
    loose = json.loads(text)

    for trail, value in _walk(loose):
        assert not (isinstance(value, float) and not math.isfinite(value)), trail


def test_a_figure_that_ran_off_the_end_is_written_null_and_named() -> None:
    """Three of these occur and they do not mean the same thing, so the direction
    is written beside the null rather than left to be inferred from the token."""
    assert jsonio.db(float("-inf")) is None
    assert jsonio.db(float("nan")) is None
    assert jsonio.db(-12.345) == -12.35

    named = jsonio.beyond(level_db=float("-inf"), residual_db=float("nan"), floor_db=-40.0)

    assert named == {"level_db": jsonio.BELOW, "residual_db": jsonio.NOT_TAKEN}


def test_a_record_whose_figures_were_all_finite_says_nothing() -> None:
    """An empty explanation on every record is an explanation nobody reads."""
    assert jsonio.beyond(a=1.0, b=-40.0, c=None) == {}
