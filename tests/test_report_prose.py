"""The prose a report carries, held against the archive that already published it.

Every result file states its own method, and that statement is the only thing
standing between a number and a reader who has to decide what it means. It is
also the easiest thing in the repository to break without noticing, because
nothing executes it: a sentence moved, rewrapped or reworded still runs.

So the archive is used as the fixture. Each file under data/units was written
by this harness, and every method, note and caveat in one of them must still
exist verbatim as a string in the source. Rewording a published method statement
means the archive and the code no longer agree about how a measurement was made,
and that has to be a deliberate act with the data reissued, not a side effect of
moving the sentence to another module.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

UNIT = Path(__file__).parents[1] / "data" / "units" / "roland-sc8850-01"
SOURCE = Path(__file__).parents[1] / "src" / "soundings"

# Keys whose value is a statement about how something was measured, rather than
# something measured.
PROSE = frozenset(
    {
        "method",
        "note",
        "why",
        "why_preceded",
        "caveat",
        "verdict_rule",
        "sampling_caveat",
        "not_scanned",
        "rpn_parked",
    }
)

# Files no command writes, so there is no source literal for them to match.
# meta.json and measurements.json are kept by hand; power-on-state.json was read
# out before there was a command that reads a whole map.
BY_HAND = {"meta.json", "measurements.json", "power-on-state.json"}

# Written by an earlier form of alias-scan, whose prose the current code no
# longer contains. Left in the archive because it is a record of a run that
# happened; excluded here because it does not describe the code as it stands.
SUPERSEDED = {"cc-aliases-ch1-wholemap.json"}


def _literals() -> set[str]:
    """Every string constant in the package.

    Read with ast rather than by searching the text, so that a sentence split
    across several source lines by implicit concatenation is compared as the one
    string it becomes.
    """
    found: set[str] = set()
    for path in sorted(SOURCE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                found.add(node.value)
    return found


def _is_prose(key: str) -> bool:
    """Whether this key's value is a statement about how something was measured.

    By prefix as well as by name: every such key added since has been called
    why_something, and a set that has to be extended by hand silently stops
    covering the newest prose -- which is the prose most likely to be wrong.
    """
    return key in PROSE or key.startswith("why_")


def _prose(obj, path: str = "") -> list[tuple[str, str]]:
    if isinstance(obj, dict):
        out = []
        for key, value in obj.items():
            if isinstance(value, str) and _is_prose(key) and len(value) > 60:
                out.append((f"{path}/{key}", value))
            else:
                out.extend(_prose(value, f"{path}/{key}"))
        return out
    if isinstance(obj, list):
        return [x for i, v in enumerate(obj) for x in _prose(v, f"{path}[{i}]")]
    return []


# rglob, not glob: the per-address records live in audible/ and were not being
# checked at all, which is most of the archive by file count.
GENERATED = sorted(
    p for p in UNIT.rglob("*.json") if p.name not in BY_HAND and p.name not in SUPERSEDED
)


@pytest.mark.parametrize("path", GENERATED, ids=lambda p: p.name)
def test_published_prose_still_exists_in_the_source(path: Path):
    literals = _literals()
    missing = [
        (where, text)
        for where, text in _prose(json.loads(path.read_text()))
        if text not in literals
    ]
    assert not missing, "\n".join(f"{path.name}{where}: {text}" for where, text in missing)


def test_the_archive_is_actually_being_checked():
    """A filter that quietly matched nothing would make every case above vacuous."""
    counted = sum(len(_prose(json.loads(p.read_text()))) for p in GENERATED)
    assert counted >= 60, counted
