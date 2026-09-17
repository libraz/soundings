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
        "settled": None,
    },
    {
        "dir": "which-rate-table-005to640",
        "name": r"held-16-(?P<kind>[0-9A-F]{4})-(?P<slot>[0-9A-F]{2})-(?P<value>\d+)-00",
        "settled": None,
    },
    {
        "dir": "the-same-rotor-elsewhere",
        "name": r"held-16-(?P<kind>[0-9A-F]{4})-(?P<slot>[0-9A-F]{2})-(?P<value>\d+)-00",
        "settled": 10.0,
    },
    {
        "dir": "one-rate-slot-0404-400306",
        "name": r"held-16-(?P<kind>[0-9A-F]{4})-(?P<slot>[0-9A-F]{6})-(?P<value>\d+)-00",
        "settled": None,
    },
    {
        "dir": "one-value-long-0121-400307",
        "name": r"held-16-(?P<kind>[0-9A-F]{4})-(?P<slot>[0-9A-F]{6})-(?P<value>\d+)-long-00",
        "settled": 10.0,
    },
    {
        "dir": "rate-slot-followup",
        "name": r"held-16-(?P<kind>[0-9A-F]{4})-(?P<slot>40-03-[0-9A-F]{2})-(?P<value>\d+)-00",
        "settled": None,
    },
    # The runs that had to quiet something. `state` is part of the take's name,
    # so it separates readings of the same slot taken under different settings --
    # which is the whole reason those runs exist.
    {
        "dir": "silence-the-other-stage",
        "name": r"held-16-(?P<kind>[0-9A-F]{4})-(?P<slot>[0-9A-F]{2})"
        r"-(?P<state>[^-]+)-(?P<value>\d+)-00",
        "settled": 10.0,
    },
    # The same directory's earlier half, whose names carry no state because the run
    # had not yet needed one: it began asking each slot once and grew a second
    # reading of three of them later. A take named without a state is not a take
    # made without one -- what those readings had turned down is in the run's own
    # table and is carried here by slot, in `HELD_BY_SLOT`.
    {
        "dir": "silence-the-other-stage",
        "name": r"held-16-(?P<kind>[0-9A-F]{4})-(?P<slot>[0-9A-F]{2})-(?P<value>\d+)-00",
        "settled": 10.0,
        "state_name": "the-other-stage-down",
    },
    {
        "dir": "which-end-is-dry",
        "name": r"held-16-(?P<kind>[0-9A-F]{4})-(?P<slot>[0-9A-F]{2})"
        r"-(?P<state>c\d+-f\d+)-(?P<value>\d+)-00",
        "settled": 2.0,
    },
    {
        "dir": "two-lines-one-take",
        "name": r"held-16-(?P<kind>[0-9A-F]{4})-(?P<slot>[0-9A-F]{2})"
        r"-(?P<state>h[\d-]+)-(?P<value>\d+)-00",
        "settled": 2.0,
    },
    # The sweeps of one slot of one type, whose takes name the slot and not the
    # type because the type did not change all evening. Several of these cover the
    # same slot at different value ranges and take lengths; each is its own record,
    # and what its take length could carry is in every reading.
    {"dir": "rate-curve", "kind": "0121", "slot_named": True, "settled": None},
    {"dir": "rate-curve-low", "kind": "0121", "slot_named": True, "settled": None},
    {"dir": "rate-curve-every", "kind": "0121", "slot_named": True, "settled": None},
    {"dir": "rate-curve-slow", "kind": "0121", "slot_named": True, "settled": None},
    {"dir": "rate-curve-slow2", "kind": "0121", "slot_named": True, "settled": None},
    {"dir": "rate-curve-top", "kind": "0121", "slot_named": True, "settled": None},
    {"dir": "rate-curve-0201", "kind": "0201", "slot_named": True, "settled": None},
    {"dir": "rate-curve-1108", "kind": "1108", "slot_named": True, "settled": None},
    {"dir": "rate-curve-1108c", "kind": "1108", "slot_named": True, "settled": None},
    # The rotor sweeps, where the state in the name is which side the speed switch
    # was on. Twice over: the second time with ten seconds of settling in the name,
    # the first time with a wait the run did not record, which is what its records
    # say rather than a figure reconstructed from the script as it stands now.
    {
        "dir": "which-way-the-rotor-turns",
        "kind": "0122",
        "name": r"held-16-(?P<slot>[0-9A-F]{6})-(?P<value>\d+)-(?P<state>slow|fast)-w10-00",
        "settled": 10.0,
        "as": "settled10",
    },
    {
        "dir": "which-way-the-rotor-turns",
        "kind": "0122",
        "name": r"held-16-(?P<slot>[0-9A-F]{6})-(?P<value>\d+)-(?P<state>slow|fast)-00",
        "settled": None,
        "as": "settle-not-recorded",
    },
    {
        "dir": "the-bottom-of-the-constant",
        "name": r"held-16-(?P<kind>[0-9A-F]{4})-(?P<slot>[0-9A-F]{2})"
        r"-(?P<state>low|high)-(?P<value>\d+)-long-00",
        "settled": 10.0,
    },
)

STATE_MEANS = {
    # What a run's own name for a state means as addresses, so the record carries
    # the setting rather than the run's shorthand for it. **Keyed by the run as
    # well as by the word**: `low` and `slow` are one run's names for its own
    # arrangement and mean nothing outside it, and a table keyed by the word alone
    # would publish one run's addresses under another's takes the first time two
    # evenings reached for the same adjective.
    ("silence-the-other-stage", "LoSlowalone"): [("40 03 0A", "00"), ("40 03 06", "7F")],
    ("silence-the-other-stage", "HiSlowalone"): [("40 03 06", "00"), ("40 03 0A", "7F")],
    ("silence-the-other-stage", "HiFast_switchfast"): [
        ("40 03 06", "00"), ("40 03 0A", "7F"), ("40 03 0D", "01")],
    ("silence-the-other-stage", "HiSlow_switchfast"): [
        ("40 03 06", "00"), ("40 03 0A", "7F"), ("40 03 0D", "01")],
    ("silence-the-other-stage", "HiFast_switchslow"): [
        ("40 03 06", "00"), ("40 03 0A", "7F"), ("40 03 0D", "00")],
    ("silence-the-other-stage", "RTHalone"): [
        ("40 03 13", "00"), ("40 03 15", "7F"), ("40 03 0B", "00"), ("40 03 0F", "7F")],
    ("the-bottom-of-the-constant", "low"): [("40 03 0A", "00"), ("40 03 06", "7F")],
    ("the-bottom-of-the-constant", "high"): [("40 03 06", "00"), ("40 03 0A", "7F")],
    # The speed switch, which is the whole of what this run's state names.
    ("which-way-the-rotor-turns", "slow"): [("40 03 0D", "00")],
    ("which-way-the-rotor-turns", "fast"): [("40 03 0D", "01")],
}
"""A state this cannot spell out is carried as the run's own word for it, and the
record says so. Inventing addresses for a shorthand would be worse than keeping
the shorthand: one is unreadable, the other is wrong."""

SLOT_NAMED = r"held-16-(?P<slot>40-03-[0-9A-F]{2})-(?P<value>\d+)-00"
"""How a sweep of one type names its takes when the type is not in them."""

HELD_BY_SLOT = {
    # Where a run held something and did not put it in the take's name, what it
    # held is keyed by the slot it was reading. Each of these is one row of that
    # run's own table, and each slot appears once among the rows that named no
    # state, so there is nothing to choose between.
    ("11 04", "40 03 08"): [("40 03 13", "00"), ("40 03 15", "7F")],
    ("11 04", "40 03 0C"): [("40 03 13", "00"), ("40 03 15", "7F")],
    ("11 07", "40 03 0C"): [("40 03 13", "00"), ("40 03 15", "7F")],
    ("11 00", "40 03 04"): [("40 03 15", "00"), ("40 03 13", "7F")],
    ("11 02", "40 03 04"): [("40 03 15", "00"), ("40 03 13", "7F")],
    ("11 08", "40 03 04"): [("40 03 15", "00"), ("40 03 13", "7F")],
}

UNTOUCHED = {
    "dir": "rate-slot-followup",
    "name": r"held-16-(?P<type>[0-9A-F]{4})-default-long-00",
}
"""The takes in a sweep's directory that are not sweep points.

Five types were loaded and nothing was written to any of them, held long enough
that the reading's own floor sat an order of magnitude below what had been read
at eight seconds. Those takes have no setting, so they are one record of what
each type carried untouched rather than rows of a sweep -- and the circumstance
is in the record as `hold_s` and `slowest_measurable_hz` rather than as a
sentence, because a sentence written here would be a second place where the
archive's prose lives.

A run that asks one thing usually saves takes for another beside them, and a
publisher keyed only on sweeps drops those without noticing. This one is named so
it is published rather than counted as a name nobody recognised."""


def address_of(slot: str) -> str:
    """A run's own spelling of the slot, as the bytes a record spells an address with.

    One run names the whole address, another only the offset under `40 03`, and a
    third runs the bytes together. All three become the spaced form every other
    record in the archive uses, because an address that is spelled differently in
    two records is an address a reader looking for one of them will not find.
    """
    bare = slot.replace("-", "").replace(" ", "").upper()
    if len(bare) == 2:
        return f"40 03 {bare}"
    return " ".join(bare[at : at + 2] for at in range(0, len(bare), 2))


GROUP = re.compile(r"\(\?P<(\w+)>[^)]*\)")
"""A named group in a run's pattern. The bodies above hold no parentheses of their
own, which is what lets this be a match rather than a parser."""


def specialised(name: str, **pinned: str | None) -> str:
    """The run's pattern with every group but `value` pinned to one take's answer.

    One directory holds several sweeps, so the pattern that enumerates them is not
    the pattern that reads one of them: what goes into a record has to match that
    record's takes and no others. Pinning by group name rather than by the text of
    the group means a pattern can be rewritten without this quietly ceasing to pin
    anything -- which is the failure that leaves a record covering the neighbours'
    takes as well as its own.
    """

    def one(found: re.Match) -> str:
        group = found.group(1)
        if pinned.get(group) is None:
            return found.group(0)
        return re.escape(pinned[group])

    return GROUP.sub(one, name)


def held_for(where: str, state: str | None) -> tuple[list[dict], str | None]:
    if not state:
        return [], None
    if (spelled := STATE_MEANS.get((where, state))) is not None:
        return [{"address": a, "bytes": b} for a, b in spelled], None
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

    written, missing = 0, []
    wrote: set[Path] = set()
    # Which takes any run claimed, against every take on disk. Counted per run, a
    # take that one pattern skips and another reads looks unpublished; the
    # difference of the two sets is the only honest count, and it names them.
    on_disk: set[tuple[str, str]] = set()
    claimed: set[tuple[str, str]] = set()
    for run in RUNS:
        where = root / run["dir"]
        if not where.exists():
            missing.append(run["dir"])
            continue
        pattern = re.compile(run.get("name") or SLOT_NAMED)
        groups: dict[tuple[str, str, str | None], None] = {}
        for path in sorted(where.glob("*.wav")):
            on_disk.add((run["dir"], path.stem))
            found = pattern.fullmatch(path.stem)
            if not found:
                continue
            claimed.add((run["dir"], path.stem))
            got = found.groupdict()
            # A run that swept one type names the slot and not the type, because
            # the type was the same all evening. It is taken from the entry then,
            # which is the only place it survives.
            kind = got.get("kind") or run["kind"]
            groups[(kind, got["slot"], got.get("state"))] = None
        for kind, slot, state in groups:
            address = address_of(slot)
            type_id = f"{kind[:2]} {kind[2:]}"
            if state is None and "state_name" in run:
                # A run that held something and did not name it in the take. What
                # it held is looked up, never guessed: a record that says nothing
                # was held is wrong in a way a reader cannot see.
                spelled = HELD_BY_SLOT.get((type_id, address))
                if spelled is None:
                    print(f"  {type_id} {address}: no state on the take and none on"
                          f" record for it; skipped rather than published as unheld")
                    continue
                held = [{"address": a, "bytes": b} for a, b in spelled]
                note = None
            else:
                held, note = held_for(run["dir"], state)
            here = dict(run)
            setting = specialised(
                run.get("name") or SLOT_NAMED, kind=kind, slot=slot, state=state
            )
            record.invoked(
                record.Invocation(
                    stage="efx-rate",
                    # A flag whose value is nothing is left off rather than
                    # spelled `None`. The two read the same in a file and are not
                    # the same on a command line: the parser takes `--settled` as
                    # a number and stops at the word, so a line written that way
                    # is one no reader can run.
                    argv=[
                        "efx-rate", str(where), "--type", type_id, "--slot", address,
                        "--setting", setting,
                        *(["--settled", str(here["settled"])]
                          if here["settled"] is not None else []),
                        *[a for h in held for a in ("--held", f"{h['address']}={h['bytes']}")],
                        *(["--held-not-spelled-out", note] if note else []),
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
                held_not_spelled_out=note,
                settled_s=here["settled"],
            )
            if not payload["readings"]:
                continue
            stem = f"{kind[:2]}-{kind[2:]}-{address.split()[-1]}-{run['dir']}"
            for word in (state, run.get("state_name"), run.get("as")):
                if word:
                    stem += f"-{re.sub(r'[^0-9A-Za-z]+', '', word)}"
            path = out / f"{stem}.json"
            # Two runs over one directory can reach the same name, and the second
            # would overwrite the first without anything failing -- a record lost
            # to a naming collision looks exactly like a record never made.
            if path in wrote:
                print(f"  {path.name}: a second run reached this name; both are"
                      " the same type, slot and state, so one of them needs an `as`")
                continue
            path.write_text(
                json.dumps(record.envelope(payload, out_path=path), indent=2, default=float)
                + "\n"
            )
            wrote.add(path)
            written += 1
            print(f"  {path.name}  {len(payload['readings'])} readings")

    written += untouched(root, out, wrote, on_disk, claimed)

    record.invoked(None)
    print(f"\n{written} records -> {out}")
    if left := sorted(on_disk - claimed):
        print(f"{len(left)} takes matched no pattern and are in no record:")
        for where, stem in left:
            print(f"  {where}/{stem}")
    # A record under `out` that this did not write is one whose run has been
    # dropped or renamed, and it stays on disk being read as current.
    if stale := sorted(p.name for p in out.glob("*.json") if p not in wrote):
        print(f"\n{len(stale)} records under {out} this run did not write:")
        for name in stale:
            print(f"  {name}")
    if missing:
        print(f"no takes on disk for: {', '.join(missing)}")
    return 0


def untouched(
    root: Path,
    out: Path,
    wrote: set[Path],
    on_disk: set[tuple[str, str]],
    claimed: set[tuple[str, str]],
) -> int:
    """The takes made with a type loaded and nothing written to it."""
    where = root / UNTOUCHED["dir"]
    if not where.exists():
        return 0
    pattern = re.compile(UNTOUCHED["name"])
    for path in sorted(where.glob("*.wav")):
        on_disk.add((UNTOUCHED["dir"], path.stem))
        if pattern.fullmatch(path.stem):
            claimed.add((UNTOUCHED["dir"], path.stem))
    record.invoked(
        record.Invocation(
            stage="efx-rate",
            argv=["efx-rate", str(where), "--untouched", "--setting", UNTOUCHED["name"]],
            started=time.monotonic(),
            midi_device_id=None,
        )
    )
    payload = efxrate.read_untouched(where, setting=UNTOUCHED["name"])
    if not payload["readings"]:
        return 0
    path = out / f"untouched-{UNTOUCHED['dir']}.json"
    path.write_text(
        json.dumps(record.envelope(payload, out_path=path), indent=2, default=float) + "\n"
    )
    wrote.add(path)
    print(f"  {path.name}  {len(payload['readings'])} readings")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
