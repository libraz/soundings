"""At what rate one insertion effect makes a held tone's phase jump, setting by setting.

Read from saved takes, with no machine attached. A modulator that moves a delay
smoothly turns a partial's phase smoothly; one whose value is re-taken at instants and
held between them turns it in a train of discontinuities, and the rate of that train is
a different quantity from the rate of the waveform underneath it. `efx-rate` answers the
second. This answers the first, and on a type carrying both they are not the same number.

**Rectified, which is why this reads where the signed series does not.** The train's
jumps alternate in sign as the waveform under them rises and falls, so a projection of
the signed series is pulled onto the waveform's own line -- measured here, at nine of
twelve settings. Taking the size of each step and not its direction leaves the train
and drops the waveform, and the train then stands over the take with nothing in its
path by two orders of magnitude at every setting.

**What rectifying costs, and it is stated rather than corrected.** A series whose
fundamental is one rate has energy at twice it once the sign is folded away, so the line
this returns is the train's rate or twice it and this reading does not say which. What
it does say is how the line moves, and that is what a byte is read by: `line_hz` is
published as measured, the record carries the signed series' own peak beside it where
that could be read at all, and no factor is applied to either.

**The floor is the take with the part routed past the effect, at the same rate.** A held
tone is not steady and its own drift projects onto a slow grid, so a peak is reported as
how far it stands over the same projection of the routed-past take rather than over the
middle of its own grid. That is the comparison `efx-partials` already makes, made here on
a different series.

**No structure and no count.** Whether the jumps are a sample and hold, where such a
hold would sit, and how many of them fall in one cycle of anything else are not read
here: this returns a rate and how far it stands over nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import efxpartials, partials, takes

VALUE = "value"
"""The named group a `--setting` pattern has to capture, as the stages beside it name it.

A run names its takes however its own question needed, so the pattern that pulls the
byte back out belongs to the invocation rather than to this module -- and because it is
in the invocation it is in the record, where a reader can see how the settings were read
rather than trusting that they were.
"""

QUESTION = (
    "At what rate one insertion effect type makes the phase of a held tone's orders "
    "jump, at each setting of one byte."
)

METHOD = (
    "One take per setting of a held tone. Each order of the tone that cleared the "
    "threshold on the take with the part routed past the effect was demodulated by its "
    "own frequency, leaving that order's phase over the take; the orders come from that "
    "take and never from the take being read. The size of each step of that phase, with "
    "its direction dropped, is the series projected across a grid of rates, and the rate "
    "reported is where the take stands furthest over the same projection of the "
    "routed-past take. The signed phase is projected across the same grid beside it. "
    "Every take is read on one channel named for the whole run, and a setting taken more "
    "than once gives the run its floor."
)

LIMITS = (
    "`line_hz` is where the rectified series stood furthest over the routed-past take. "
    "Rectifying a series puts energy at twice its fundamental, so the line is the jump "
    "train's own rate or twice it, and nothing here decides which: no factor is applied "
    "and none is implied. `signed_hz` is the same projection of the phase itself, which "
    "is pulled onto the waveform under the train wherever that waveform is larger. "
    "`stands_over_bypassed` is the comparison every figure rests on, and a reading near "
    "one is the routed-past take's own behaviour whatever the rate beside it says. "
    "`slowest_measurable_hz` is the rate two cycles of which fill the part of the take "
    "that sounded, and a line at or under it is the take's length rather than the unit. "
    "A line at an end of the grid is the search's own limit and says so per reading. "
    "The projection of a train is largest at whichever of its harmonics the material "
    "favours, and which one that is moves between takes of one state, so a peak that "
    "lands a whole factor from another take's is the same train and not a different "
    "rate. `also_at` is the height at rates the run named, which is how a reading is "
    "asked what it holds at a rate rather than where it is largest. "
    "`heard_db` is the level over the take of the one channel named in `channel`: a "
    "reading taken from a take near the noise floor is a reading of the floor, and it is "
    "stable, which is what makes it dangerous."
)

NOT_HERE = (
    "No structure, no count and no verdict. Whether the jumps are a sample and hold, "
    "where such a hold sits in a chain, how many fall in one cycle of anything else, and "
    "which curve these rates lie on as the byte rises are fits across settings and types "
    "and are not made here. The printed range for this address is in documents/, under "
    "the same address, and is evidence about a page rather than about this unit; the two "
    "are not joined."
)

WHY_RECTIFIED = (
    "A train of jumps whose sizes follow a waveform is two things at once, and a "
    "projection of the phase returns whichever is larger. Dropping the direction of each "
    "step leaves the train's own periodicity and takes the waveform's out, which is what "
    "makes this a reading of the clock rather than a second reading of the modulator. "
    "The cost is that rectifying is not linear: a series with a fundamental has energy "
    "at twice it afterwards, so what this returns is the train's rate or twice it. Both "
    "projections are published and neither is corrected by the other."
)

WHY_FLOOR = (
    "How far apart this reading put one state of the unit twice, on the quantity the "
    "record publishes. It is the only figure a difference between two settings has to "
    "clear. Where no setting was taken twice the record says so and offers nothing in "
    "its place."
)

WHY_HELD = (
    "Every address written before the takes were made, beside the one that was swept. "
    "A type with two modulators returns whichever dominates a given reading, so what the "
    "other one was doing decides what this is a reading of -- and a modulator parked at "
    "the bottom of its own table is running slowly rather than switched off."
)

WHY_ONE_CHANNEL = efxpartials.WHY_ONE_CHANNEL


def _jumps(held: partials.Partials) -> np.ndarray:
    """Each order's phase step by step, with its direction dropped and its mean out."""
    step = np.abs(np.diff(held.phase, axis=1))
    return step - step.mean(axis=1, keepdims=True)


def _reading(
    held: partials.Partials, rectified_floor, signed_floor, grid, also_at: tuple[float, ...]
) -> dict:
    """Where this take stands furthest over the routed-past one, both ways round."""
    at = held.at[:-1]
    jumps = _jumps(held)
    line = partials.peak(jumps, at, grid, rectified_floor)
    signed = partials.peak(held.phase, held.at, grid, signed_floor)
    slowest = 2.0 / held.sounded_s if held.sounded_s > 0 else float("inf")
    return {
        "line_hz": line["hz"],
        "stands_over_bypassed": line["stands_over_bypassed"],
        "also_at": [
            partials.height_at(jumps, at, grid, rectified_floor, hz) for hz in also_at
        ],
        "signed_hz": signed["hz"],
        "the_signed_series_stands": signed["stands_over_bypassed"],
        "orders": held.orders,
        "slowest_measurable_hz": round(slowest, 4),
        "rates_are_separated_by": round(held.rates_are_separated_by, 4),
        "stands_on_the_edge_of_the_search": efxpartials.on_the_edge(
            line["hz"], grid, slowest
        ),
    }


def _floor_of(readings: list[dict]) -> dict:
    """What the run's repeated settings say about the line it published."""
    seen: dict[int, list[float]] = {}
    for row in readings:
        got = row.get("line_hz")
        if got is not None:
            seen.setdefault(row[VALUE], []).append(float(got))
    repeated = {value: got for value, got in seen.items() if len(got) > 1}
    if not repeated:
        return {"hz": None, "settings_taken_twice": []}
    spreads = {value: max(got) - min(got) for value, got in repeated.items()}
    return {
        "hz": round(max(spreads.values()), 4),
        "widest_at": max(spreads, key=lambda value: spreads[value]),
        "settings_taken_twice": sorted(repeated),
        "takes_per_setting": {str(value): len(got) for value, got in sorted(repeated.items())},
    }


def read_directory(
    where: str | Path,
    *,
    type_id: str,
    address: str,
    setting: str,
    carrier_hz: float,
    bypassed: str = "bypassed",
    held: list[dict] | None = None,
    held_not_spelled_out: str | None = None,
    settled_s: float | None = None,
    lead_s: float = 0.6,
    hold_s: float = 8.0,
    grid_hz: tuple[float, float] = (0.30, 30.00),
    step_hz: float = 0.01,
    also_at: tuple[float, ...] = (),
    channel: int | None = None,
) -> dict:
    """Every take under `where` whose setting matches, read into one record.

    The take with the part routed past the effect decides which orders are read and is
    the floor both projections are read against, so it is found first and by its own
    name.

    A take neither pattern names is counted rather than dropped: a pattern that matches
    nothing and a directory that holds nothing produce the same empty record otherwise,
    and they are different mistakes.
    """
    where = Path(where)
    listed, files = takes.listing(where)
    swept = takes.capturing(setting, VALUE)
    routed_past = re.compile(bypassed)

    controls = [
        name
        for name in files
        if routed_past.search(name)
        or routed_past.search(str(listed.get(name, {}).get("setting") or ""))
    ]
    if not controls:
        raise ValueError(f"no take under {where} matched the control {bypassed!r}")

    matched = []
    for name in files:
        if name in controls:
            continue
        source, found = takes.named_by(swept, listed.get(name, {}), name)
        if found is not None:
            matched.append((name, source, found))
    skipped = [
        str(listed.get(name, {}).get("setting") or name)
        for name in files
        if name not in controls and all(name != got[0] for got in matched)
    ]
    every = sorted({*controls, *(name for name, _, _ in matched)})
    reached, levels = takes.channel_reaching(where, every)
    used = reached if channel is None else int(channel)
    beside = takes.other_of_the_pair(used, len(levels))
    elsewhere: list[str] = []

    grid = np.arange(grid_hz[0], grid_hz[1] + step_hz / 2, step_hz)

    def body_of(name: str, on: int):
        got, rate, own = efxpartials.body_of(where / name, lead_s, hold_s, on=on)
        if on == used and own != used and name not in elsewhere:
            elsewhere.append(name)
        return got, rate

    control, rate = body_of(controls[0], used)
    standing = partials.read(control, rate, carrier_hz=carrier_hz)
    if standing is None:
        raise ValueError(
            f"no order of {carrier_hz} Hz cleared the threshold in the bypassed take"
        )
    keep = standing.kept
    rectified_floor = partials.project(_jumps(standing), standing.at[:-1], grid)
    signed_floor = partials.project(standing.phase, standing.at, grid)

    # One control read as though it were a setting, against *another* control. Never
    # against its own floor, which is one by construction and would publish a guaranteed
    # refusal as a measured one. Null where the run made only one such take.
    nothing_in_its_path = None
    if len(controls) > 1:
        other, at = body_of(controls[1], used)
        second = partials.read(other, at, carrier_hz=carrier_hz, keep=keep)
        if second is not None:
            nothing_in_its_path = {
                "take": controls[0],
                "against": controls[1],
                **_reading(
                    standing,
                    partials.project(_jumps(second), second.at[:-1], grid),
                    partials.project(second.phase, second.at, grid),
                    grid,
                    also_at,
                ),
            }

    readings: list[dict] = []
    for name, source, found in sorted(matched):
        body, at = body_of(name, used)
        here = partials.read(body, at, carrier_hz=carrier_hz, keep=keep)
        row = {
            VALUE: int(found.group(VALUE)),
            "take": name,
            "named_by": source,
            "heard_db": round(takes.level_db(takes.rms(body)), 1),
            "also_heard_db": (
                None if beside is None
                else round(takes.level_db(takes.rms(body_of(name, beside)[0])), 1)
            ),
            "settled_s": settled_s,
        }
        if here is None:
            row["why"] = "no order of the carrier cleared the threshold in this take"
            readings.append(row)
            continue
        readings.append(
            {**row, **_reading(here, rectified_floor, signed_floor, grid, also_at)}
        )

    readings.sort(key=lambda r: (r[VALUE], r["take"]))

    return {
        "question": QUESTION,
        "type": type_id,
        "address": address,
        "method": METHOD,
        "limits": LIMITS,
        "not_in_this_record": NOT_HERE,
        "carrier_hz": carrier_hz,
        "grid_hz": list(grid_hz),
        "grid_step_hz": step_hz,
        "also_at_hz": list(also_at),
        "channel": {
            "read": used,
            "chosen_by": "given" if channel is not None else "highest across the takes read",
            "reached_db": levels,
            "loudest_elsewhere": sorted(elsewhere),
            "why": WHY_ONE_CHANNEL,
        },
        "other_channel": takes.other_channel_block(used, levels, readings),
        "routed_past_the_effect": {
            "takes": sorted(controls),
            "read_against": controls[0],
            "why": efxpartials.WHY_THE_ORDERS_COME_FROM_THE_BYPASSED_TAKE,
            "with_nothing_in_its_path": nothing_in_its_path,
        },
        "why_rectified": WHY_RECTIFIED,
        "floor": _floor_of(readings),
        "why_floor": WHY_FLOOR,
        "held": held or [],
        "why_held": WHY_HELD,
        **({"held_not_spelled_out": held_not_spelled_out} if held_not_spelled_out else {}),
        "takes_from": str(where),
        "manifest": takes.manifest_note(listed, files),
        "settings_asked": sorted({r[VALUE] for r in readings}),
        "readings": readings,
        "takes_not_matching": takes.not_matching(skipped),
    }


__all__ = [
    "LIMITS",
    "METHOD",
    "NOT_HERE",
    "QUESTION",
    "VALUE",
    "read_directory",
]
