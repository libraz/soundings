#!/usr/bin/env python3
"""Read every saved rate sweep into archive records, one per slot per run.

    publish-efx-rate.py --takes .cache/takes --out data/units/<unit-id>/efx-rate

**Nothing here reads a working file.** The readings are taken from the takes
themselves, through the same `efx-rate` stage the command line exposes, so a
record can be rebuilt from the audio after the reader improves -- which has
happened twice -- and so there is one implementation of the reading rather than
one in a stage and another in a publisher.

What this adds to the stage is the enumeration: a sweep names its takes for the
question it was asking that day, so each run has its own pattern here, and the
pattern says which part of a name is the type, which the address, which the byte,
and which the state the run was holding. **A run whose takes are on disk and
whose pattern is not here produces no record and is listed at the end**, because
a publisher that quietly covers the runs it recognises leaves the rest looking
like runs that were never made.

The state each run held is the other half. Three of these sweeps exist only
because a type with two modulators returns whichever dominates, so the reading is
meaningless without what was turned down beside it; that is carried per record
rather than described here.

No hardware.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundings import efxrate, record  # noqa: E402

# Each run: the directory under --takes, a pattern over a take's file name, the
# state the run held while it read, and how long it waited after writing a
# setting. The pattern must name `value`; `kind` and `slot` are taken from it
# where the run's names carry them and from the entry otherwise.
RUNS = (
    {
        "dir": "which-rate-table-005to100",
        "name": r"held-16-(?P<kind>[0-9A-F]{4})-(?P<slot>[0-9A-F]{2})-(?P<value>\d+)-00",
        "held": [],
        "settled": None,
    },
    {
        "dir": "which-rate-table-005to640",
        "name": r"held-16-(?P<kind>[0-9A-F]{4})-(?P<slot>[0-9A-F]{2})-(?P<value>\d+)-00",
        "held": [],
        "settled": None,
    },
    {
        "dir": "the-same-rotor-elsewhere",
        "name": r"held-16-(?P<kind>[0-9A-F]{4})-(?P<slot>[0-9A-F]{2})-(?P<value>\d+)-00",
        "held": [],
        "settled": 10.0,
    },
    {
        "dir": "one-rate-slot-0404-400306",
        "name": r"held-16-(?P<kind>[0-9A-F]{4})-(?P<slot>[0-9A-F]{6})-(?P<value>\d+)-00",
        "held": [],
        "settled": None,
    },
    {
        "dir": "one-value-long-0121-400307",
        "name": r"held-16-(?P<kind>[0-9A-F]{4})-(?P<slot>[0-9A-F]{6})-(?P<value>\d+)-long-00",
        "held": [],
        "settled": 10.0,
    },
    {
        "dir": "rate-slot-followup",
        "name": r"held-16-(?P<kind>[0-9A-F]{4})-(?P<slot>40-03-[0-9A-F]{2})-(?P<value>\d+)-00",
        "held": [],
        "settled": None,
    },
    # The runs that had to quiet something. `state` is part of the take's name,
    # so it separates readings of the same slot taken under different settings --
    # which is the whole reason those runs exist.
    {
        "dir": "silence-the-other-stage",
        "name": r"held-16-(?P<kind>[0-9A-F]{4})-(?P<slot>[0-9A-F]{2})"
        r"-(?P<state>[^-]+)-(?P<value>\d+)-00",
        "held": [],
        "settled": 10.0,
    },
    {
        "dir": "which-end-is-dry",
        "name": r"held-16-(?P<kind>[0-9A-F]{4})-(?P<slot>[0-9A-F]{2})"
        r"-(?P<state>c\d+-f\d+)-(?P<value>\d+)-00",
        "held": [],
        "settled": 2.0,
    },
    {
        "dir": "two-lines-one-take",
        "name": r"held-16-(?P<kind>[0-9A-F]{4})-(?P<slot>[0-9A-F]{2})"
        r"-(?P<state>h[\d-]+)-(?P<value>\d+)-00",
        "held": [],
        "settled": 2.0,
    },
    {
        "dir": "the-bottom-of-the-constant",
        "name": r"held-16-(?P<kind>[0-9A-F]{4})-(?P<slot>[0-9A-F]{2})"
        r"-(?P<state>low|high)-(?P<value>\d+)-long-00",
        "held": [],
        "settled": 10.0,
    },
)

STATE_MEANS = {
    # what a run's own name for a state means as addresses, so the record carries
    # the setting rather than the run's shorthand for it
    "LoSlowalone": [("40 03 0A", "00"), ("40 03 06", "7F")],
    "HiSlowalone": [("40 03 06", "00"), ("40 03 0A", "7F")],
    "HiFast_switchfast": [("40 03 06", "00"), ("40 03 0A", "7F"), ("40 03 0D", "01")],
    "HiSlow_switchfast": [("40 03 06", "00"), ("40 03 0A", "7F"), ("40 03 0D", "01")],
    "HiFast_switchslow": [("40 03 06", "00"), ("40 03 0A", "7F"), ("40 03 0D", "00")],
    "RTHalone": [("40 03 13", "00"), ("40 03 15", "7F"),
                 ("40 03 0B", "00"), ("40 03 0F", "7F")],
    "low": [("40 03 0A", "00"), ("40 03 06", "7F")],
    "high": [("40 03 06", "00"), ("40 03 0A", "7F")],
}
"""A state this cannot spell out is carried as the run's own word for it, and the
record says so. Inventing addresses for a shorthand would be worse than keeping
the shorthand: one is unreadable, the other is wrong."""


def held_for(state: str | None) -> tuple[list[dict], str | None]:
    if not state:
        return [], None
    if state in STATE_MEANS:
        return [{"address": a, "bytes": b} for a, b in STATE_MEANS[state]], None
    if (found := re.fullmatch(r"c(\d+)-f(\d+)", state)):
        return [
            {"address": "40 03 07", "bytes": f"{int(found.group(1)):02X}"},
            {"address": "40 03 0C", "bytes": f"{int(found.group(2)):02X}"},
        ], None
    if (found := re.fullmatch(r"h(\d+)", state)):
        return [], f"the run's other rate slot held at byte {int(found.group(1))}"
    return [], f"the run's own name for the state it held: {state!r}"


def main(argv: list[str]) -> int:
    root = Path(argv[argv.index("--takes") + 1]) if "--takes" in argv else Path(".cache/takes")
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else None
    if out is None:
        print("--out <dir> is required; it names the unit this is a record of")
        return 2
    out.mkdir(parents=True, exist_ok=True)

    written, missing, unmatched = 0, [], 0
    for run in RUNS:
        where = root / run["dir"]
        if not where.exists():
            missing.append(run["dir"])
            continue
        pattern = re.compile(run["name"])
        groups: dict[tuple[str, str, str | None], None] = {}
        for path in sorted(where.glob("*.wav")):
            found = pattern.fullmatch(path.stem)
            if not found:
                unmatched += 1
                continue
            got = found.groupdict()
            groups[(got["kind"], got["slot"], got.get("state"))] = None
        for kind, slot, state in groups:
            address = "40 03 " + slot[-2:] if len(slot) == 2 else slot.replace("-", " ")
            type_id = f"{kind[:2]} {kind[2:]}"
            held, note = held_for(state)
            here = dict(run)
            setting = run["name"]
            if state is not None:
                setting = setting.replace(
                    "(?P<state>[^-]+)", re.escape(state)
                ).replace("(?P<state>c\\d+-f\\d+)", re.escape(state)).replace(
                    "(?P<state>h[\\d-]+)", re.escape(state)
                ).replace("(?P<state>low|high)", re.escape(state))
            setting = setting.replace("(?P<kind>[0-9A-F]{4})", kind)
            setting = setting.replace("(?P<slot>[0-9A-F]{2})", slot)
            setting = setting.replace("(?P<slot>[0-9A-F]{6})", slot)
            setting = setting.replace("(?P<slot>40-03-[0-9A-F]{2})", slot)
            record.invoked(
                record.Invocation(
                    stage="efx-rate",
                    argv=[
                        "efx-rate", str(where), "--type", type_id, "--slot", address,
                        "--setting", setting, "--settled", str(here["settled"]),
                    ],
                    started=time.monotonic(),
                    midi_device_id=None,
                )
            )
            payload = efxrate.read_directory(
                where,
                type_id=type_id,
                address=address,
                setting=setting,
                held=held,
                settled_s=here["settled"],
            )
            if note:
                payload["held_not_spelled_out"] = note
            if not payload["readings"]:
                continue
            stem = f"{kind[:2]}-{kind[2:]}-{address.split()[-1]}-{run['dir']}"
            if state:
                stem += f"-{re.sub(r'[^0-9A-Za-z]+', '', state)}"
            path = out / f"{stem}.json"
            path.write_text(
                json.dumps(record.envelope(payload, out_path=path), indent=2, default=float)
                + "\n"
            )
            written += 1
            print(f"  {path.name}  {len(payload['readings'])} readings")
    record.invoked(None)
    print(f"\n{written} records -> {out}")
    if unmatched:
        print(f"{unmatched} takes matched no pattern and are in no record")
    if missing:
        print(f"no takes on disk for: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
