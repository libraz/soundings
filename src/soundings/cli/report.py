"""Writing a result out, when the run was asked for one."""

from __future__ import annotations

import json
from pathlib import Path


def write_json(where: str | None, payload: dict) -> None:
    """Write the payload as JSON and say where it went, or do nothing.

    Nothing is a valid outcome: a run without --out was asked for the summary on
    the terminal and not for a file. The directory is made rather than demanded,
    since the archive is laid out per unit and a new unit has no directory yet.
    """
    if not where:
        return
    path = Path(where)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {path}")
