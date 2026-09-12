#!/usr/bin/env python3
"""Read the saved equaliser takes for what they say about phase, one pair per record.

    publish-efx-phase.py --takes .cache/takes --out data/units/<unit-id>/phase

The same takes `publish-efx-bands.py` reads for band energy, asked the other
question about them. A band energy says how much of each band came out; this says
when, which is the half a rendering has to get right for a recording and a
rendered buffer to be held against each other at all.

**One pair per record, as the other pair stages publish.** A phase is measured
between two takes and a record here names both of them, in the same shape `decay`
and `motion` use, rather than being folded into a sweep: the pair is the
measurement, and a file holding ten of them would have one control standing for
ten comparisons that were each made separately.

**Which pairs, and why those.** Each row of the type at the ends of what it was
swept over, against the flat setting of the same run -- the ends because that is
where a row does the most and a phase too small to clear the control at the ends
is too small anywhere. Beside them, one pair of two flat takes per run: that is
the null, and it is published as a record rather than kept as a number so that
what this method returns for a byte that did nothing is in the archive next to
what it returns for bytes that did.

**Every record carries a control of its own** -- every pair of the run's flat
takes, read on the channel the record's own pair was read on. A phase between two
takes of one note carries the alignment and the stimulus as well as the effect,
and without those pairs the record says what a phase was and not whether it was
one. Every pair rather than one because one pair is one draw: measured here, the
pairs of one run's four repeats disagree with each other by more than some of the
rows disagree with the flat setting, so a bound taken from whichever pair came
first would pass the run's own noise as a reading, and three of these records
lost a verdict when the bound was widened to all of them.

**The null's own pair is left inside its bound rather than taken out of it.**
Excluding the pair being measured is right for a row and wrong for a null: the
null IS one of the repeats, so holding it against the largest of the others makes
the largest of the draws stand above the rest by construction, about one time in
as many pairs as there are.

No hardware.
"""

from __future__ import annotations

import json
import re
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundings import efxbands, phase, record, takes  # noqa: E402

STIMULUS = (
    "channel 2, program 126, note 60, velocity 100, held 8.0 s, captured 9.0 s, "
    "40 12 31 = 0"
)
"""The same stimulus the band records were read from, restated here.

A record that names the takes it read and not what was sounded into them leaves a
reader holding a filename. It is applause rather than a tone because a stage that
only shapes what is already there says nothing about a band the stimulus does not
reach -- and applause is also why the coherence in these records falls where it
does, since noise repeats as a spectrum more readily than as a waveform.
"""

LEAD_S = 0.6
"""Seconds dropped from the head of every take before anything is measured."""

FLAT = r"held-126-flat-(?P<take>\d+)-00"
"""The setting every pair below is measured against, and the pairs of it that are
the null and the control."""

RUNS = (
    {
        "dir": "how-much-is-twelve-db",
        "type": "01 00",
        "rows": (
            ("40 03 04", r"held-126-04-(?P<value>\d+)-00"),
            ("40 03 06", r"held-126-06-(?P<value>\d+)-00"),
            ("40 03 08", r"held-126-m1q-(?P<value>\d+)-00"),
        ),
    },
    {
        "dir": "how-much-is-twelve-db-fine",
        "type": "01 00",
        "rows": (
            ("40 03 04", r"held-126-04-(?P<value>\d+)-00"),
            ("40 03 06", r"held-126-06-(?P<value>\d+)-00"),
            ("40 03 09", r"held-126-09-(?P<value>\d+)-00"),
            ("40 03 0C", r"held-126-0C-(?P<value>\d+)-00"),
        ),
    },
    {
        "dir": "where-the-eq-hinges",
        "type": "01 00",
        "rows": (
            ("40 03 03", r"held-126-lowfreq-(?P<value>\d+)-00"),
            ("40 03 05", r"held-126-hifreq-(?P<value>\d+)-00"),
            ("40 03 07", r"held-126-m1freq-(?P<value>\d+)-00"),
            ("40 03 08", r"held-126-m1q-(?P<value>\d+)-00"),
            ("40 03 0A", r"held-126-m2freq-(?P<value>\d+)-00"),
            ("40 03 0B", r"held-126-m2q-(?P<value>\d+)-00"),
            ("40 03 16", r"held-126-level-(?P<value>\d+)-00"),
        ),
    },
    {
        # The output level row again, from the run that asked it at every value
        # and in a different session. The band records of the two runs are two
        # resolutions of one question; here they are two draws of one answer, and
        # a row that turns no angle is worth having twice from separate takes.
        "dir": "where-the-level-steps",
        "type": "01 00",
        "rows": (("40 03 16", r"held-126-level-(?P<value>\d+)-00"),),
    },
)


def flats(where: Path) -> list[Path]:
    return sorted(p for p in where.glob("*.wav") if re.fullmatch(FLAT, p.stem))


def ends(where: Path, setting: str) -> list[tuple[int, Path]]:
    """The lowest and the highest value the row was swept over, with its take."""
    found = []
    for path in sorted(where.glob("*.wav")):
        if (match := re.fullmatch(setting, path.stem)) is not None:
            found.append((int(match.group("value")), path))
    if not found:
        return []
    found.sort()
    return [found[0]] if len(found) == 1 else [found[0], found[-1]]


def one(dry: Path, wet: Path, repeats: list[Path], *, type_id: str, address: str,
        value, where: Path) -> dict:
    """One pair read, with every pair of the run's repeats read on the same channel."""
    dry_frames, wet_frames, rate, picked = takes.read_pair(str(dry), str(wet))
    head = int(LEAD_S * rate)
    how = {"bands": efxbands.THIRD_OCTAVES, "width_octaves": 1 / 3}
    found = phase.measure(dry_frames[head:], wet_frames[head:], rate, **how)
    # Every pair of the repeats, including the pair being measured where that pair
    # is itself two repeats. A bound drawn from one pair is one draw; a bound drawn
    # from every pair BUT the one measured makes the largest of the draws stand
    # above the rest of them by construction, which is what a null record is.
    control = list(combinations(repeats, 2))
    loaded = []
    for a, b in control:
        first, second, control_rate, _ = takes.read_pair(str(a), str(b), on=picked["read"])
        if control_rate != rate:
            raise SystemExit(f"a control pair is {control_rate} Hz and the pair is {rate} Hz")
        loaded.append((first[head:], second[head:]))
    vouched = phase.control(loaded, rate, **how)
    return {
        "type_id": type_id,
        "address": address,
        "value": value,
        "stimulus": STIMULUS,
        "takes_from": str(where),
        "dry": str(dry),
        "wet": str(wet),
        "control_from": [[str(a), str(b)] for a, b in control],
        "sample_rate": rate,
        "channel": picked,
        "lead_s": LEAD_S,
        "limits": phase.LIMITS,
        **found,
        "control": vouched,
        **(
            {"control_failed": phase.CONTROL_FAILED}
            if vouched["largest_deg"] is None
            else {}
        ),
        "conclusive": phase.is_conclusive(found, vouched),
    }


def write(payload: dict, path: Path, argv_of: list[str]) -> None:
    record.invoked(
        record.Invocation(
            stage="phase", argv=argv_of, started=time.monotonic(), midi_device_id=None
        )
    )
    path.write_text(
        json.dumps(record.envelope(payload, out_path=path), indent=2, default=float) + "\n"
    )


def said(payload: dict) -> str:
    readable = sum(
        1
        for r in payload["readings"]
        if (r["coherence"] or 0.0) >= phase.READABLE
    )
    largest = max(
        (
            (abs(r["phase_deg_less_delay"]), r["hz"])
            for r in payload["readings"]
            if r["phase_deg_less_delay"] is not None
            and (r["coherence"] or 0.0) >= phase.READABLE
        ),
        default=(None, None),
    )
    bound = payload["control"]["largest_deg"]
    return (
        f"{readable:2d} readable bands, "
        + (
            f"largest {largest[0]:.1f} deg at {largest[1]:.0f} Hz"
            if largest[0] is not None
            else "no readable band"
        )
        + (f", control {bound:.1f} deg" if bound is not None else ", control found nothing")
        + ("  => stands" if payload["conclusive"] else "  => inside the control")
    )


def main(argv: list[str]) -> int:
    root = Path(argv[argv.index("--takes") + 1]) if "--takes" in argv else Path(".cache/takes")
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else None
    if out is None:
        print("--out <dir> is required; it names the unit this is a record of")
        return 2
    out.mkdir(parents=True, exist_ok=True)

    written, missing = 0, []
    for run in RUNS:
        where = root / run["dir"]
        if not where.exists():
            missing.append(run["dir"])
            continue
        repeats = flats(where)
        if len(repeats) < 4:
            print(f"  {run['dir']}: {len(repeats)} flat takes, and a pair and a "
                  "control of its own need four")
            continue
        kind = run["type"].replace(" ", "-")

        # The null first, so that what the method returns for a byte that did
        # nothing is read before anything it returns for a byte that did.
        payload = one(repeats[0], repeats[-1], repeats, type_id=run["type"],
                      address=None, value=None, where=where)
        path = out / f"{kind}-flat-{run['dir']}.json"
        write(payload, path, ["phase", str(repeats[0]), str(repeats[-1]),
                              "--control", *[str(p) for p in repeats[1:-1]]])
        print(f"  {path.name}  {said(payload)}")
        written += 1

        for address, setting in run["rows"]:
            for value, wet in ends(where, setting):
                payload = one(repeats[0], wet, repeats, type_id=run["type"],
                              address=address, value=value, where=where)
                path = out / f"{kind}-{address.split()[-1]}-{value:03d}-{run['dir']}.json"
                write(payload, path, ["phase", str(repeats[0]), str(wet),
                                      "--control", *[str(p) for p in repeats[1:]]])
                print(f"  {path.name}  {said(payload)}")
                written += 1

    if missing:
        print(f"\nno takes on disk for: {', '.join(missing)}")
    print(f"\n{written} records -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
