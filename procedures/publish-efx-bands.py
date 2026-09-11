#!/usr/bin/env python3
"""Read every saved band sweep into archive records, one per slot per run.

    publish-efx-bands.py --takes .cache/takes --out data/units/<unit-id>/efx-bands

**Nothing here reads a working file.** The profiles are taken from the takes
themselves, through the same `efx-bands` stage the command line exposes, so a
record can be rebuilt from the audio after the reader improves, and so there is
one implementation of the reading rather than one in a stage and another in a
publisher.

What this adds to the stage is the enumeration, and one thing more than the rate
publisher needs: **which takes in a directory are the reference and which are the
control.** A sweep of a level-like parameter is reported as a deviation, so the
setting it is a deviation from, and the takes made with the effect bypassed
entirely, are as much a part of the record as the sweep -- and they are takes in
the same directory rather than fields, because they were recorded in the same
session as the sweep and their whole value is that they were.

**A slot a run swept and this does not name produces no record and is listed at
the end**, because a publisher that quietly covers the runs it recognises leaves
the rest looking like runs that were never made.

No hardware.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundings import efxbands, record  # noqa: E402

STIMULUS = (
    "channel 2, program 126, note 60, velocity 100, held 8.0 s, captured 9.0 s, "
    "40 12 31 = 0"
)
"""Applause rather than the carrier the rest of the effect work uses.

A stage that only shapes what is already there can report nothing about a band the
stimulus does not reach, and the usual carrier has its strongest partial fifty
decibels below the lower of the two printed corners and nothing at all above the
higher one. What it costs is that applause is noise and does not repeat exactly,
which is why every run here measures its own floor first.
"""

ROUTED = [("40 42 22", "01"), ("40 12 31", "00")]
"""The part put through the effect, and its vibrato turned off. Both are writes
the run made and neither is the swept byte, so they belong with what was held."""

FLAT_GAINS = [
    ("40 03 04", "40"), ("40 03 06", "40"), ("40 03 09", "40"), ("40 03 0C", "40")
]
"""Every gain at the centre of its printed range, which is the setting the run
called flat and every profile in these records is a deviation from."""

PINNED_MIDS = [
    ("40 03 07", "48"), ("40 03 08", "00"), ("40 03 0A", "38"), ("40 03 0B", "00")
]
"""The two parametric mids' centre and width, held where this unit answers with
after a reset. A gain read through one of them is read through a stated filter
rather than through whatever the slot happened to hold."""

FULL, CENTRE = "7F", "40"
"""Where a gain is pinned while the corner it shapes is swept, and where it goes
for the null. A corner with its gain at the centre has nothing to shape, so the
same slot is asked twice and the two are separate records: what was held is the
whole difference between them, and one record holding both would have to carry
two answers to `held`."""


def corner(slot: str, label: str, gain: str, *pinned: tuple[str, str]) -> tuple[dict, dict]:
    """A corner's sweep and its null, as two entries of the table below."""
    swept = {
        "slot": slot,
        "setting": rf"held-126-{label}-(?P<value>\d+)-00",
        "with": [(gain, FULL), *pinned],
    }
    null = {
        "slot": slot,
        "setting": rf"held-126-{label}null-(?P<value>\d+)-00",
        "with": [(gain, CENTRE), *pinned],
        "as": "null",
    }
    return swept, null


RUNS = (
    {
        "dir": "how-much-is-twelve-db",
        "type": "01 00",
        "reference": r"held-126-flat-\d+-00",
        "control": r"held-126-bypassed-\d+-00",
        "silence": r"held-126-silence-\d+-00",
        "held": ROUTED + FLAT_GAINS,
        "swept": (
            {"slot": "40 03 04", "setting": r"held-126-04-(?P<value>\d+)-00"},
            {"slot": "40 03 06", "setting": r"held-126-06-(?P<value>\d+)-00"},
            # The null, and the only slot here whose takes are named for the
            # parameter rather than for its address: a width moved while the gain
            # it shapes is at its centre has nothing to shape, so a profile with a
            # shape in it would say the run was reading something other than gain.
            {"slot": "40 03 08", "setting": r"held-126-m1q-(?P<value>\d+)-00"},
        ),
    },
    {
        "dir": "how-much-is-twelve-db-fine",
        "type": "01 00",
        "reference": r"held-126-flat-\d+-00",
        "control": r"held-126-bypassed-\d+-00",
        "silence": r"held-126-silence-\d+-00",
        "held": ROUTED + FLAT_GAINS + PINNED_MIDS,
        "swept": (
            {"slot": "40 03 04", "setting": r"held-126-04-(?P<value>\d+)-00"},
            {"slot": "40 03 06", "setting": r"held-126-06-(?P<value>\d+)-00"},
            {"slot": "40 03 09", "setting": r"held-126-09-(?P<value>\d+)-00"},
            {"slot": "40 03 0C", "setting": r"held-126-0C-(?P<value>\d+)-00"},
        ),
    },
    {
        "dir": "where-the-eq-hinges",
        "type": "01 00",
        "reference": r"held-126-flat-\d+-00",
        "control": r"held-126-bypassed-\d+-00",
        "silence": r"held-126-silence-\d+-00",
        "held": ROUTED + FLAT_GAINS,
        "swept": (
            *corner("40 03 03", "lowfreq", "40 03 04"),
            *corner("40 03 05", "hifreq", "40 03 06"),
            *corner("40 03 07", "m1freq", "40 03 09", ("40 03 08", "02")),
            *corner("40 03 08", "m1q", "40 03 09", ("40 03 07", "40")),
            *corner("40 03 0A", "m2freq", "40 03 0C", ("40 03 0B", "02")),
            *corner("40 03 0B", "m2q", "40 03 0C", ("40 03 0A", "40")),
            # The one row of this type that should move every band by the same
            # amount. Published with the whole profile rather than one figure,
            # because a level control that is not only a level looks identical to
            # one that is until the bands are compared with each other.
            {"slot": "40 03 16", "setting": r"held-126-level-(?P<value>\d+)-00"},
        ),
    },
)


def main(argv: list[str]) -> int:
    root = Path(argv[argv.index("--takes") + 1]) if "--takes" in argv else Path(".cache/takes")
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else None
    if out is None:
        print("--out <dir> is required; it names the unit this is a record of")
        return 2
    out.mkdir(parents=True, exist_ok=True)

    written, missing = 0, []
    wrote: set[Path] = set()
    # Which takes any pattern claimed, against every take on disk. Counted per
    # slot, a take the sweep skips and the reference reads looks unpublished; the
    # difference of the two sets is the only honest count, and it names them.
    on_disk: set[tuple[str, str]] = set()
    claimed: set[tuple[str, str]] = set()
    for run in RUNS:
        where = root / run["dir"]
        if not where.exists():
            missing.append(run["dir"])
            continue
        for path in sorted(where.glob("*.wav")):
            on_disk.add((run["dir"], path.stem))
        for pattern in (run["reference"], run["control"], run.get("silence")):
            if not pattern:
                continue
            for path in sorted(where.glob("*.wav")):
                if re.fullmatch(pattern, path.stem):
                    claimed.add((run["dir"], path.stem))

        for entry in run["swept"]:
            address, setting = entry["slot"], entry["setting"]
            for path in sorted(where.glob("*.wav")):
                if re.fullmatch(setting, path.stem):
                    claimed.add((run["dir"], path.stem))
            # What was held is what the run wrote and is not the byte being swept.
            # Left in, a slot would appear as both the sweep and a constant, and a
            # reader would have no way to tell which of the two the record meant.
            # The sweep's own pins are written last, since a corner read through a
            # gain at the top of its range was written over the flat setting and
            # the record has to say the byte that was there when the take was made.
            by_address = dict(run["held"])
            by_address.update(entry.get("with") or ())
            held = [
                {"address": a, "bytes": b} for a, b in by_address.items() if a != address
            ]
            argv_of = [
                "efx-bands", str(where), "--type", run["type"], "--slot", address,
                "--setting", setting, "--reference", run["reference"],
                "--control", run["control"], "--stimulus", STIMULUS,
                *(["--silence", run["silence"]] if run.get("silence") else []),
            ]
            record.invoked(
                record.Invocation(
                    stage="efx-bands",
                    argv=argv_of,
                    started=time.monotonic(),
                    midi_device_id=None,
                )
            )
            payload = efxbands.read_directory(
                where,
                type_id=run["type"],
                address=address,
                setting=setting,
                reference=run["reference"],
                control=run["control"],
                silence=run.get("silence"),
                stimulus=STIMULUS,
                held=held,
            )
            if not payload["readings"]:
                print(f"  {run['dir']} {address}: no take matched {setting!r}")
                continue
            kind = run["type"].replace(" ", "-")
            stem = f"{kind}-{address.split()[-1]}-{run['dir']}"
            if word := entry.get("as"):
                stem += f"-{word}"
            path = out / f"{stem}.json"
            # Two runs can reach the same name, and the second would overwrite the
            # first without anything failing -- a record lost to a naming collision
            # looks exactly like a record never made.
            if path in wrote:
                print(f"  {path.name}: a second run reached this name")
                continue
            path.write_text(
                json.dumps(record.envelope(payload, out_path=path), indent=2, default=float)
                + "\n"
            )
            wrote.add(path)
            written += 1
            # The widest deviation any setting reached, beside the count. A slot
            # whose whole sweep clears the floor by a tenth of a decibel and one
            # that swings twelve print the same count, and only one of them is a
            # curve worth reading.
            widest = max(
                (abs(r["largest_db"]) for r in payload["readings"] if r["largest_db"]),
                default=0.0,
            )
            moved = sum(1 for r in payload["readings"] if r["largest_db"] is not None)
            print(
                f"  {path.name}  {len(payload['readings'])} settings, "
                f"{moved} outside the floor, widest {widest:.2f} dB"
            )

    record.invoked(None)
    print(f"\n{written} records -> {out}")
    if left := sorted(on_disk - claimed):
        print(f"{len(left)} takes matched no pattern and are in no record:")
        for where_name, stem in left:
            print(f"  {where_name}/{stem}")
    # A record under `out` that this did not write is one whose run has been
    # dropped or renamed, and it stays on disk being read as current.
    if stale := sorted(p.name for p in out.glob("*.json") if p not in wrote):
        print(f"\n{len(stale)} records under {out} this run did not write:")
        for name in stale:
            print(f"  {name}")
    if missing:
        print(f"no takes on disk for: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
