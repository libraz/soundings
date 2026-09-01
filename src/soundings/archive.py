"""Reading the archive back, so one measurement can be aimed by an earlier one.

A sweep says which regions exist; a write probe says which of their bytes take
any value. Neither is worth repeating at the head of every later run, and both
are already published. What is read back here is this repository's own output,
so the shapes are the ones the `to_json` methods write.
"""

from __future__ import annotations

import json
from pathlib import Path

Address = tuple[int, int, int]

# The bytes a control change and an NRPN were both measured to reach. Writing
# them by SysEx asks whether the location has a third way in, and -- because the
# whole watched space is diffed, not just the byte written -- whether the value
# is also kept anywhere else.
ALIASED_BYTES = (
    "40 11 19",
    "40 11 1C",
    "40 11 21",
    "40 11 22",
    "40 11 30",
    "40 11 31",
    "40 11 32",
    "40 11 33",
    "40 11 34",
    "40 11 35",
    "40 11 36",
    "40 11 37",
    "40 21 04",
)


def regions(path: str | Path, prefix: str = "") -> list[tuple[Address, int]]:
    """Every region in an address map, as (start, size), optionally under a prefix."""
    data = json.loads(Path(path).read_text())
    out = []
    for region in data["regions"]:
        if not region["address"].startswith(prefix) or not region["size"]:
            continue
        start = tuple(int(b, 16) for b in region["address"].split())
        out.append((start, region["size"]))
    return out


def accepting_bytes(path: str | Path) -> list[str]:
    """Addresses the write probe found take any value and give it back.

    A reset probe needs somewhere it can put a mark. An address that clamps or
    refuses may keep what it had, and a byte that was never broken tells the
    reset nothing.
    """
    data = json.loads(Path(path).read_text())
    return [
        b["address"]
        for region in data["regions"]
        for b in region["bytes"]
        if b["classification"] == "accepts" and b["restored"]
    ]
