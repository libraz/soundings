"""How deep the notching one insertion effect puts on a held tone, setting by setting.

Read from saved takes, with no machine attached. The quantity is decibels, and it is
how far **one order of the tone moves in level** while the effect runs -- not how loud
the take is. `efx-sway` reads the take's own level over time, which is the sum over
the orders, and that is the one reading this quantity is invisible to: notching moves
energy between the orders rather than removing it, so a series that swings fifteen
decibels on a single order comes back under one decibel summed. Neither stage is a
second way of asking the other's question.

**Why a stage rather than a column on the excursion reading.** Beside this one,
`efx-excursion` answers how far the modulator moves the delay. That is a length and
this is a depth, and the two come apart exactly where the mixture does: a delay can
sweep just as far through a path nothing is combed against, where there is barely a
notch to read. A byte that changes the mixture and not the delay therefore moves this
and leaves that alone, which is the reading a balance byte needs and the excursion
stage cannot give.

**The reading has a null, and each order carries its own.** A take with the part
routed past the effect goes through the same demodulation, and the orders the tone is
strongest in return three tenths of a decibel there while its weakest wander two. So a
depth is read against the same order's null rather than against the largest of them,
and it is admitted against that and against the run's own repeats -- both measured by
the run being read, and neither a threshold chosen here.

**Read while the notching moves, so it is not a profile of one standing still.** A
band profile taken with the modulator parked returns how far down a notch goes at a
frequency. This returns how far one order moves as notches are dragged past it. The
two answer to the same structure and are not interchangeable: a notch narrower than
the order's own demodulator reads shallower here, and one that never reaches the order
reads as nothing here whatever a profile says.

**One reading, not a second demodulator.** The orders come off `partials.read`, the
same call `efx-partials` and `efx-excursion` make, with the same window and the same
set of orders carried in from the bypassed take. What this adds is one statistic over
the level series it already returns.
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

BETWEEN = (10.0, 90.0)
"""Which percentiles of a level series the swing is taken between.

End to end is one frame's opinion, and the bottom of a notch that reaches the take's own
floor is a reading of the floor rather than of the effect. The pair is symmetric so that
the figure does not depend on which way the notch passes, and it is wide enough that a
series spending most of its time at one level still returns most of its excursion.
"""

PRESENT_WITHIN_DB = 40.0
"""How far under its own working level an order may sit and still be counted present.

Read off a smoothed copy, because a notch reaching the take's floor is not the note
ending -- and a window with holes in it is not one window, so the longest single run of
frames is taken rather than every frame over the line.
"""

SMOOTHED_OVER_S = 0.25
"""Over how long an order's level is smoothed before deciding where it is present.

Long enough that a notch passing is inside the window rather than ending it, short
enough that the note's own decay is still followed.
"""

QUESTION = (
    "How deep the notching one insertion effect puts on a held tone, at each setting of "
    "one byte, in decibels of level swing on a single order."
)

METHOD = (
    "One take per setting of a held tone. Each order of the tone that cleared the "
    "threshold on the take with the part routed past the effect was demodulated by its "
    "own frequency, leaving that order's level over the take; the orders come from that "
    "take and never from the take being read, so an effect that raises its own quiet "
    "orders cannot bring a different set of them into its own answer. Over the frames "
    "the order was present in, its swing is the distance between the tenth and the "
    "ninetieth percentile of that level. The order standing furthest over its own swing "
    "on the routed-past take is the figure for the setting, and the same reading was made "
    "on the unit's other output beside it. Every "
    "take is read on one channel named for the whole run, and a setting taken more than "
    "once gives the run its floor."
)

LIMITS = (
    "A swing is decibels on one order and not a mixture: how much of the output each "
    "path of a mixture is, and where the two are equal, are not read here and do not "
    "follow from a depth. "
    "It is read while the notching moves, so it is not a profile of one standing still: "
    "a notch narrower than an order's own demodulator is read shallower here than a band "
    "profile of the same effect parked would return it, and a notch that never reaches "
    "an order is read as nothing here whatever a profile says. "
    "The level series is thinned to the rate `read_at_hz` reports, so a notch passing "
    "faster than that is read through a smoothed copy of itself and comes back "
    "shallower. What bounds that is the run's own repeats rather than an assumption: a "
    "modulator fast enough for the thinning to matter does not repeat to the figure in "
    "`floor`. "
    "`swing_db` is the swing of the order standing furthest over its own null and "
    "`at_order_hz` says which order that was; `largest_db` is the largest swing over the "
    "orders whatever its null did, and `per_order_db` is all of them, because an effect "
    "that notches one order and not another has done something a single figure cannot "
    "show. "
    "`heard_db` is the level over the take of the one channel named in `channel`: a "
    "reading taken from a take near the noise floor is a reading of the floor, and it is "
    "stable, which is what makes it dangerous. "
    "`swing_elsewhere_db` is the same reading on the unit's other output. It is here "
    "because a pan byte decides which output a notched path arrives on, and a run that "
    "read one channel can lose the whole of an effect to a pan without losing any level "
    "-- which has happened in this archive and was found only when the other output was "
    "read."
)

NOT_HERE = (
    "No curve, no table and no verdict. Which shape these depths lie on as the byte "
    "rises, what structure puts a notch where, and what fraction of the output each path "
    "of a mixture is are fits across settings and types and are not made here. The "
    "printed range for this address is in documents/, under the same address, and is "
    "evidence about a page rather than about this unit; the two are not joined."
)

WHY_ADMITTED = (
    "A swing is admitted where it stands over what the same reading returns with nothing "
    "in the effect's path by more than the run's own repeats of one setting disagree. "
    "Both numbers are the run's: the null is the take routed past the effect, read "
    "through this reading rather than asserted, and the floor is two takes of one "
    "setting. Neither is a threshold chosen here, and a run that made neither take is "
    "reported as having no bar rather than given one. "
    "The bar is therefore the run's own, and two runs of the same state can land either "
    "side of it: a run that repeated its setting to a hundredth of a decibel admits a "
    "tenth of a decibel over the null, and one that repeated to a decibel does not. Each "
    "row carries the margin it was judged on, so a reader can apply a bar of their own."
)

WHY_FLOOR = (
    "How far apart this reading put one state of the unit twice, on the quantity the "
    "record publishes. It is the only figure a difference between two settings has to "
    "clear, and only admitted rows can give one -- two settings that both sat at the "
    "null are two nulls, and how far apart they landed says nothing. Where no setting "
    "was taken twice the record says so and offers nothing in its place."
)

WHY_THE_DEEPEST_SETTING = (
    "Which setting of the byte notched deepest, and whether it stands clear of the "
    "setting next to it by more than the run repeats to. A deepest that does not stand "
    "clear is a reading of where the sweep happened to land rather than of the byte, and "
    "it is published saying so instead of being withheld: the rows are beside it and a "
    "reader who wants the curve rather than its peak has it."
)

WHY_NOTHING_IN_ITS_PATH = (
    "The take routed past the effect, read as though it were a setting. This is the "
    "reading's null, and a number is a depth only against it: the demodulation returns a "
    "level series for a steady tone too, and what that series does on its own is the "
    "size of every figure here that is not the effect. It is read on the same channel "
    "over the same stretch as the settings are."
)

WHY_HELD = (
    "Every address written before the takes were made, beside the one that was swept. "
    "A depth is read through the whole chain, so what else was in the chain decides what "
    "the reading is of: a modulator parked at no depth leaves nothing sweeping to notch "
    "with, a feedback path deepens a notch without moving it, and a second stage left in "
    "the path smears a notch into a shelf."
)

WHY_ONE_CHANNEL = efxpartials.WHY_ONE_CHANNEL


def _present(series: np.ndarray, at_hz: float) -> np.ndarray:
    """The one run of frames an order is present in, read off a smoothed copy."""
    window = max(1, int(SMOOTHED_OVER_S * at_hz))
    padded = np.pad(series, (window // 2, window - window // 2 - 1), mode="edge")
    loudest = np.array([padded[i : i + window].max() for i in range(series.size)])
    over = loudest >= loudest.max() - PRESENT_WITHIN_DB
    if not over.any():
        return series[:0]
    edges = np.diff(np.concatenate(([0], over.astype(int), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    longest = int(np.argmax(ends - starts))
    return series[starts[longest] : ends[longest]]


def _swing_of(held: partials.Partials) -> dict | None:
    """Each order's level swing in decibels, and the deepest of them."""
    at_hz = 1.0 / float(held.at[1] - held.at[0]) if held.at.size > 1 else 0.0
    per_order, orders_hz = [], []
    for index, hz in enumerate(held.freqs_hz):
        inside = _present(held.level_db[index], at_hz)
        if inside.size < 16:
            continue
        low, high = np.percentile(inside, BETWEEN)
        per_order.append(round(float(high - low), 2))
        orders_hz.append(round(float(hz), 1))
    if not per_order:
        return None
    return {
        "largest_db": max(per_order),
        "per_order_db": per_order,
        "orders_hz": orders_hz,
        "read_at_hz": round(at_hz, 1),
    }


def _against(swing: dict, null: dict | None) -> dict:
    """The swings read against the same orders' swings with nothing in the path.

    Order by order rather than largest against largest, and the order reported is the
    one standing furthest over its own null rather than the one that swung most. A weak
    order's level series wanders on a take with nothing in its path -- measured here,
    one order of a held tone swings two decibels through a bypass while the strong ones
    swing three tenths -- so a largest taken over the orders picks whichever wandered
    most wherever the effect did little, and a null taken the same way then hides it.
    `largest_db` beside this is that unqualified largest, so the two can be compared.
    """
    if null is None or null["orders_hz"] != swing["orders_hz"]:
        return {"swing_db": None, "at_order_hz": None, "over_the_null_db": None}
    over = [
        round(mine - theirs, 2)
        for mine, theirs in zip(swing["per_order_db"], null["per_order_db"], strict=True)
    ]
    at = int(np.argmax(over))
    return {
        "swing_db": swing["per_order_db"][at],
        "at_order_hz": swing["orders_hz"][at],
        "over_the_null_db": over[at],
        "over_the_null_per_order_db": over,
    }


def _floor_of(admitted: list[dict]) -> dict:
    """What the run's repeated settings say about the depth it published."""
    seen: dict[int, list[float]] = {}
    for row in admitted:
        got = row.get("swing_db")
        if got is not None:
            seen.setdefault(row[VALUE], []).append(float(got))
    repeated = {value: got for value, got in seen.items() if len(got) > 1}
    if not repeated:
        return {"db": None, "settings_taken_twice": []}
    spreads = {value: max(got) - min(got) for value, got in repeated.items()}
    return {
        "db": round(max(spreads.values()), 2),
        "widest_at": max(spreads, key=lambda value: spreads[value]),
        "settings_taken_twice": sorted(repeated),
        "takes_per_setting": {str(value): len(got) for value, got in sorted(repeated.items())},
    }


def _deepest(readings: list[dict], floor: dict) -> dict:
    """Which setting notched deepest, and whether it stands clear of the next one.

    Settings are ranked on the mean of their own takes and never on the deepest take of
    each: a setting taken twice gets two draws at being the largest, so a ranking over
    takes hands the answer to whichever setting the run happened to repeat -- which here
    is always the one the run chose for its floor.
    """
    got = [r for r in readings if r.get("swing_db") is not None and r.get("admitted")]
    if not got:
        return {"value": None, "db": None, "stands_clear": None, "why": WHY_THE_DEEPEST_SETTING}
    per_setting: dict[int, list[float]] = {}
    for row in got:
        per_setting.setdefault(row[VALUE], []).append(float(row["swing_db"]))
    means = {value: sum(seen) / len(seen) for value, seen in per_setting.items()}
    ranked = sorted(means, key=lambda value: means[value], reverse=True)
    best = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else None
    over = None if runner is None else round(means[best] - means[runner], 2)
    bar = floor.get("db")
    return {
        "value": best,
        "db": round(means[best], 2),
        "over_takes": len(per_setting[best]),
        "next_is": runner,
        "over_the_next_db": over,
        "stands_clear": None if over is None or bar is None else bool(over > bar),
        "why": WHY_THE_DEEPEST_SETTING,
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
    channel: int | None = None,
) -> dict:
    """Every take under `where` whose setting matches, read into one record.

    The take with the part routed past the effect decides which orders are read and is
    the null every depth is admitted against, so it is found first and by its own name.

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
    null = _swing_of(standing)

    null_elsewhere = None
    if beside is not None:
        aside, aside_at = body_of(controls[0], beside)
        read = partials.read(aside, aside_at, carrier_hz=carrier_hz, keep=keep)
        null_elsewhere = None if read is None else _swing_of(read)

    readings: list[dict] = []
    for name, source, found in sorted(matched):
        body, at = body_of(name, used)
        here = partials.read(body, at, carrier_hz=carrier_hz, keep=keep)
        swing = None if here is None else _swing_of(here)
        other, other_swing = (None, None)
        if beside is not None:
            other, other_at = body_of(name, beside)
            read = partials.read(other, other_at, carrier_hz=carrier_hz, keep=keep)
            other_swing = None if read is None else _swing_of(read)
        row = {
            VALUE: int(found.group(VALUE)),
            "take": name,
            "named_by": source,
            "heard_db": round(takes.level_db(takes.rms(body)), 1),
            "also_heard_db": (
                None if other is None else round(takes.level_db(takes.rms(other)), 1)
            ),
            "settled_s": settled_s,
        }
        if swing is None:
            row["why"] = "no order of the carrier cleared the threshold in this take"
            readings.append(row)
            continue
        aside = None if other_swing is None else _against(other_swing, null_elsewhere)
        readings.append({
            **row,
            **swing,
            **_against(swing, null),
            "swing_elsewhere_db": None if aside is None else aside["swing_db"],
            "over_the_null_elsewhere_db": None if aside is None else aside["over_the_null_db"],
            "admitted": None,
        })

    readings.sort(key=lambda r: (r[VALUE], r["take"]))

    # Admission needs the floor and the floor needs admitted rows, so the bar is read
    # once off every row that returned a depth and then applied. A first pass gated on
    # the null alone would let the repeats be measured on rows the bar then refused.
    over_the_null = [r for r in readings if r.get("over_the_null_db") is not None]
    floor = _floor_of(over_the_null)
    bar = floor.get("db")
    for row in over_the_null:
        row["admitted"] = bool(row["over_the_null_db"] > bar) if bar is not None else None
    admitted = [r for r in readings if r.get("admitted")]

    return {
        "question": QUESTION,
        "type": type_id,
        "address": address,
        "method": METHOD,
        "limits": LIMITS,
        "not_in_this_record": NOT_HERE,
        "carrier_hz": carrier_hz,
        "between_percentiles": list(BETWEEN),
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
            "with_nothing_in_its_path": null,
            "on_the_other_output": null_elsewhere,
            "why_nothing_in_its_path": WHY_NOTHING_IN_ITS_PATH,
        },
        "why_admitted": WHY_ADMITTED,
        "floor": floor,
        "why_floor": WHY_FLOOR,
        "deepest": _deepest(readings, floor),
        "held": held or [],
        "why_held": WHY_HELD,
        **({"held_not_spelled_out": held_not_spelled_out} if held_not_spelled_out else {}),
        "takes_from": str(where),
        "manifest": takes.manifest_note(listed, files),
        "settings_asked": sorted({r[VALUE] for r in readings}),
        "settings_admitted": sorted({r[VALUE] for r in admitted}),
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
