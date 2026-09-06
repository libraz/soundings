"""The envelope every published record carries, and where its identity comes from.

A record said which unit it was a measurement of only when its call site
remembered to say so, and most did not: of the sixty-eight records at the root of
the first unit's directory, three carried `unit_id`. The directory path was the
only thing holding the identity, so a record read outside its directory -- which
is what a consumer of the archive does -- could not say what it was a measurement
of. Identity written per call site is identity that is optional, and an optional
field is missing from whichever record turns out to need it most.

So it is written once, here, from what the invocation already knows. A stage
cannot omit it, cannot spell it differently, and cannot let it drift from the run
that produced the file.

**The envelope sits under one key rather than beside the stage's own fields.** A
payload is unchanged by this, so a reader that already knows a record's shape
keeps working; and whether a record has that key is what tells a written-since
file from one written before the envelope existed, which a migration needs and a
count cannot give.

**What the envelope holds is what identifies the run, not what it found.** The
unit, the stage, the invocation, the moment. A finding goes in the payload as a
value -- never as a key name, for the reason `findings` below gives.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 1
"""The envelope's own version, which is not the archive's and not a unit's.

It moves when the envelope gains or loses a field, so a reader can say which
shape it is holding. A stage's payload changing shape is not this version's
business: the payloads are many and this is one.
"""

MEASURED_AT = "measured_at"
"""Named rather than spelled inline, because `captured` was already taken.

`power-on-state.json` carries `captured` as prose -- "immediately after a power
cycle, before anything was sent to it" -- which is a description of the moment
rather than the moment. Reusing the key for a timestamp would put two meanings
and two types behind one name, which is the defect this module exists to remove
rather than one to add.
"""


@dataclass
class Invocation:
    """What the command line knew about the run, kept for the records it writes.

    Held for the process rather than passed down through twenty-nine call sites.
    A parameter threaded through every writer is a parameter some writer will be
    added without, and the field it fills would then be absent exactly where a
    new stage put it.
    """

    stage: str
    argv: list[str] = field(default_factory=list)
    midi_device_id: str | None = None


_current: Invocation | None = None


def invoked(invocation: Invocation | None) -> None:
    """Declare the run every record written from here belongs to.

    Called once, by the command line entry point. Passing None restores the
    unset state, which is what a test does when it wants to write a record
    without claiming an invocation produced it.
    """
    global _current
    _current = invocation


def current() -> Invocation | None:
    """The run in progress, or None outside one."""
    return _current


def unit_of(path: str | Path) -> str | None:
    """The unit a record written to this path is a measurement of, or None.

    Read from the archive's own layout by walking up for the `meta.json` that
    declares it, rather than taken from the directory name. The two must agree
    and the disagreement is worth raising: a unit directory copied to start a
    second one keeps the first one's `meta.json`, and every record written into
    it would otherwise be published under the wrong unit while looking right.

    None is a legitimate answer. A run writing outside `data/units` -- a scratch
    file, a verification pass over somebody's working directory -- is not
    publishing to the archive and has no unit to name.
    """
    here = Path(path).resolve().parent
    for directory in [here, *here.parents]:
        meta = directory / "meta.json"
        if not meta.is_file():
            continue
        declared = json.loads(meta.read_text()).get("unit_id")
        if declared and declared != directory.name:
            raise ValueError(
                f"{meta} declares unit_id {declared!r} inside a directory named "
                f"{directory.name!r}. A record written here would be published under a "
                f"unit it was not measured from."
            )
        return declared
    return None


def envelope(payload: dict, *, out_path: str | Path) -> dict:
    """The payload with its envelope in front of it.

    The envelope is first in insertion order so that it is first in the written
    file: what a record is should be readable without scrolling past what it
    found.
    """
    invocation = _current
    stamp = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": unit_of(out_path),
        "stage": invocation.stage if invocation else None,
        "invocation": _tidied(invocation.argv) if invocation else None,
        MEASURED_AT: datetime.now(UTC).replace(microsecond=0).isoformat(),
        "midi_device_id": invocation.midi_device_id if invocation else None,
    }
    return {"record": stamp, **payload}


def seal(path: str | Path) -> bool:
    """Fill in the unit of a record that was filed into the archive after it was written.

    A run whose `--out` points at a scratch directory is not publishing when it
    writes: there is no unit above the path, `unit_of` correctly answers None,
    and the record only becomes part of the archive when somebody copies it in.
    Nothing at write time can know that is going to happen, so the identity is
    settled at the other end instead.

    This is not a guess. A record's unit is unambiguous from where it now sits --
    the same `meta.json` walk every other record uses -- so sealing is reading the
    archive's own layout rather than attributing a measurement to a machine on a
    hunch. What it must never do is overwrite a unit the record already names,
    which would let a mis-filed record be silently relabelled to wherever it
    landed; a record that disagrees with its directory is a question for a person.

    Returns whether the file was changed.
    """
    path = Path(path)
    data = json.loads(path.read_text())
    stamp = data.get("record")
    if not isinstance(stamp, dict) or stamp.get("unit_id") is not None:
        return False
    declared = unit_of(path)
    if declared is None:
        return False
    stamp["unit_id"] = declared
    path.write_text(json.dumps(data, indent=2) + "\n")
    return True


def finding(kind: str, **fields) -> dict:
    """One thing a run established, as a value that names its own kind.

    Findings were reaching the archive as key names -- a record carrying
    `blocks_42_to_47_and_4a_to_4f_are_a_window` states the finding in the place a
    reader has to know in advance. The next unit's window is at other blocks, so
    its key is spelled differently, and a consumer asking "is any block here a
    window" has to match key names per unit instead of reading a value. That is
    the archive's own rule about defaults turned on its records: one unit's
    finding must not arrive as another unit's vocabulary.

    The kind is the question; the fields are this unit's answer to it.
    """
    return {"kind": kind, **fields}


def _tidied(argv: list[str]) -> list[str]:
    """The invocation with the operator's home directory out of it.

    The archive is published, and an absolute path says who ran the measurement
    and where their disk is laid out. Neither is a measurement of the unit.
    """
    home = str(Path.home())
    out = []
    for token in argv:
        if token.startswith(home):
            token = "~" + token[len(home) :]
        out.append(token.replace(os.sep, "/"))
    return out


def _main(argv: list[str] | None = None) -> int:
    """`python -m soundings.record <paths>` -- seal records filed by hand.

    A command rather than something the test does for you: filing a record into
    the archive is a decision, and a gate that quietly rewrote the files it checks
    would be making it on somebody's behalf.
    """
    import sys

    paths = argv if argv is not None else sys.argv[1:]
    if not paths:
        print("usage: python -m soundings.record <record.json>...")
        return 2
    sealed = [p for p in paths if seal(p)]
    for path in sealed:
        print(f"sealed {path}")
    print(f"{len(sealed)} of {len(paths)} needed it")
    return 0


__all__ = [
    "SCHEMA_VERSION",
    "Invocation",
    "current",
    "envelope",
    "finding",
    "invoked",
    "seal",
    "unit_of",
]

if __name__ == "__main__":
    raise SystemExit(_main())
