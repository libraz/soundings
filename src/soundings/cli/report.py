"""Writing a result out, when the run was asked for one."""

from __future__ import annotations

import json
from pathlib import Path

from .. import record


def write_json(where: str | None, payload: dict) -> None:
    """Write the payload as JSON and say where it went, or do nothing.

    Nothing is a valid outcome: a run without --out was asked for the summary on
    the terminal and not for a file. The directory is made rather than demanded,
    since the archive is laid out per unit and a new unit has no directory yet.

    The record's envelope is added here rather than at the twenty-nine places
    that call this, which is the whole reason those places do not have to
    remember it. Identity written per call site is identity that is optional, and
    it was already missing from most of the archive.
    """
    if not where:
        return
    path = Path(where)
    path.parent.mkdir(parents=True, exist_ok=True)
    whole = record.envelope(payload, out_path=path)
    path.write_text(json.dumps(whole, indent=2, default=_plain) + "\n")
    print(f"\nwrote {path}")


def _plain(value):
    """A numpy scalar as the Python number it stands for.

    Everything here is measured with numpy and a numpy scalar compares, rounds
    and prints exactly like a number right up to the point where it is written,
    which is the last step of a run: a numpy bool derived from one dB comparison
    lost a three minute measurement at `json.dump` with every take already
    recorded. Converting rather than raising, because there is nothing a reader
    gains from the distinction and nothing the run can do about it by then.
    """
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not something a record can hold")
