"""How far one insertion effect's delay swings, setting by setting.

Read from saved takes, with no machine attached. The quantity is a length in
milliseconds and it is the length a modulator *moves* a delay through, which is a
different reading from where the delay sits: `efx-time` places a copy that stands
still, and a copy that stands still is the one thing this stage cannot see.

**Why a stage rather than a column on an existing one.** A modulated quantity read
once per type answers what a type does and cannot answer what a byte buys, and a
byte's own curve is what separates two structures that both modulate. The reading
itself is the one `efx-partials` makes -- the same demodulation, the same gates,
the same two quantities -- indexed by the setting it was taken at instead of by the
type it was taken on, and with the floor that a per-setting reading needs and a
per-type one has nowhere to get.

**Two excursions, and the smaller of them is not the worse one.** A swept delay
turns each partial's phase in proportion to that partial's own frequency, so an
excursion converts straight out of the phase series -- but only for the delayed
path, and the output holds that path mixed against a direct one. Mixed, the
composite turns by far less and not in proportion, so the figure off the phase is a
lower bound whose distance from the answer is not stated by the number. The comb
fit is the reading that answers the mixed case, and both are published: a setting
where they part company is a setting where the mixture moved, which is a finding
about the balance rather than about the byte.

**The floor is the run's own repeats, on the quantity being published.** A setting
taken more than once says how far apart this reading put one state twice, and that
is the only figure a difference between two settings has to clear. Only admitted
rows can give one -- two settings that both refused to name an excursion are two
refusals, and how far apart they landed says nothing. Where no setting was taken
twice the record says so and offers nothing in its place.

**No curve and no table.** Which shape these excursions lie on as the byte rises,
whether two types share one of them, and what structure would produce it are fits
across settings and types and are not made here.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import efxpartials, takes

VALUE = "value"
"""The named group a `--setting` pattern has to capture.

A run names its takes however its own question needed, so the pattern that pulls
the byte back out belongs to the invocation rather than to this module -- and
because it is in the invocation it is in the record, where a reader can see how
the settings were read rather than trusting that they were.
"""

QUESTION = (
    "How far one insertion effect's modulator swings the delay under it, at each "
    "setting of one byte, in milliseconds."
)

METHOD = (
    "One take per setting of a held tone, with the effect's balance at the point "
    "that carries both paths. Each order of the tone that cleared the threshold on "
    "the take with the part routed past the effect was demodulated by its own "
    "frequency, leaving that partial's phase and its level over the take; a "
    "projection was walked across a grid of rates and the largest taken, separately "
    "for the phase and for the level. Where the partials disagreed in shape -- which "
    "is what a sweeping notch does and a level envelope cannot -- a two path comb was "
    "fitted forward to the level series with its rate free. The excursion is that "
    "fit's, and the one the phase implies is beside it. Every take is read on one "
    "channel named for the whole run, and a setting taken more than once gives the "
    "run its floor."
)

LIMITS = (
    "The reading returns how far a delay moved and not how long it is: a comb fit "
    "returns a centre as well, but the centre is the one parameter an even mixture "
    "leaves nearly free, so it is published inside the fit and is not what this "
    "stage answers. "
    "An excursion is withheld where the fit does not account for the series it was "
    "taken on, and a withheld row is not a setting that does not modulate -- it is a "
    "setting this reading could not tell a right answer from a wrong rate at. The "
    "rows are published with what they returned so the difference can be seen. "
    "The model the fit is forward in is two paths and no return, so a run made with "
    "a loop closed around the delay is outside it: the series is deeper and shaped "
    "differently than any excursion of that model produces, and the fit withholds "
    "across the whole sweep rather than at one setting. A record whose rows all "
    "withhold is reporting that, and the figure off the phase is what it has left. "
    "`slowest_measurable_hz` is the slowest rate the take was long enough to carry "
    "two cycles of, and a rate at or under it is the take's length rather than the "
    "unit's modulator; `stands_on_the_edge_of_the_search` says so per reading. "
    "`heard_db` is the level over the take of the one channel named in `channel`: a "
    "reading taken from a take near the noise floor is a reading of the floor, and it "
    "is stable, which is what makes it dangerous."
)

NOT_HERE = (
    "No curve, no table and no verdict. Which shape these excursions lie on as the "
    "byte rises, whether two types share one, and what structure would produce it "
    "are fits across settings and types and are not made here. The printed range for "
    "this address is in documents/, under the same address, and is evidence about a "
    "page rather than about this unit; the two are not joined."
)

WHY_TWO_EXCURSIONS = (
    "Both figures are published and neither is a correction of the other. The one "
    "off the phase is a lower bound wherever the effect is not fully wet -- measured "
    "by injection on this archive's own carrier, a fifth wet turns a two millisecond "
    "sweep into five thousandths of one while detection survives it -- and what it "
    "bounds is not stated by the number. The one off the comb fit answers the mixed "
    "case and is the figure this stage is for. A setting where the two part company "
    "further than its neighbours did is a setting where the mixture moved, which is a "
    "reading about the balance and not about the byte that was swept."
)

WHY_ADMITTED = (
    "An excursion is read only where the level series was a comb being swept and the "
    "forward fit explained enough of that series to have been told apart from a wrong "
    "rate. Both bars come from `efx-partials`, were set against injected combs of "
    "known excursion there, and are carried into this record beside the numbers they "
    "decide. A setting that fails either is listed with everything it did return, "
    "because a record that omitted them would read as a shorter sweep -- and the "
    "bottom of a depth byte failing is a reading about the bottom of the range."
)

WHY_FLOOR = (
    "How far apart this reading put one setting's own repeats, which is what a "
    "difference between two settings has to clear to be a difference. Reported as "
    "the widest spread any repeated setting showed and as a proportion of that "
    "setting's own mean, because an excursion an order of magnitude smaller repeats "
    "to a smaller number of milliseconds without repeating any better. Null where no "
    "setting was taken twice, in which case this run measured no floor and nothing "
    "here substitutes for one. "
    "There are two floors because there are two quantities, and each is taken over "
    "the rows its own gate admitted: two refusals are not two readings of a quantity, "
    "and the gate that admits a fitted excursion is not the gate that admits a phase "
    "one. A run where the fit withheld throughout still has a floor for the figure it "
    "did return, and a record carrying one floor would have had to drop it."
)

WHY_NOTHING_IN_ITS_PATH = (
    "One take with the part routed past the effect, read as though it were a setting, "
    "against another take with the part routed past the effect as its floor. That is "
    "the whole reading with the swing taken out of it, so what it returns is what a "
    "row has to stand over to be a modulated delay rather than the voice -- measured "
    "on this run's own stimulus instead of assumed from the bar. It is the sensitivity "
    "every refusal needs: a bottom setting that named no excursion means nothing "
    "unless a take that certainly holds no swing also names none here. "
    "Never a control read against its own floor. That comparison is one by "
    "construction and it refuses by construction, so a record built from it would "
    "publish a guaranteed refusal as a measured one -- which is the shape of every "
    "negative that turns out to be about the reading. Null where the run made only one "
    "such take, in which case this run measured no null and nothing here stands in for "
    "one."
)

WHY_HELD = (
    "Every address written before the takes were made, beside the one that was swept. "
    "An excursion is read through the whole chain, so what else was in the chain "
    "decides what the reading is of: a feedback path around the delay deepens the "
    "notch without moving it, a balance away from the point that carries both paths "
    "shallows the series the fit is taken on, and a second modulator on the type "
    "returns whichever of the two dominates."
)

WHY_ONE_CHANNEL = efxpartials.WHY_ONE_CHANNEL


def _floor_of(admitted: list[dict], key: str) -> dict:
    """What the run's repeated settings say about the quantity under `key`."""
    seen: dict[int, list[float]] = {}
    for row in admitted:
        got = row.get(key)
        if got is not None:
            seen.setdefault(row[VALUE], []).append(float(got))
    repeated = {value: got for value, got in seen.items() if len(got) > 1}
    if not repeated:
        return {"ms": None, "as_a_fraction": None, "settings_taken_twice": []}
    spreads = {
        value: (max(got) - min(got), sum(got) / len(got)) for value, got in repeated.items()
    }
    widest = max(spreads.values(), key=lambda pair: pair[0])
    loosest = max(spreads.values(), key=lambda pair: pair[0] / max(pair[1], 1e-12))
    return {
        "ms": round(widest[0], 4),
        "as_a_fraction": round(loosest[0] / max(loosest[1], 1e-12), 4),
        "settings_taken_twice": sorted(repeated),
        "takes_per_setting": {str(value): len(got) for value, got in sorted(repeated.items())},
    }


def _reading(row: dict) -> dict:
    """The two excursions and the gates that decided them, pulled flat out of a take."""
    comb = row.get("comb") or {}
    return {
        "excursion_ms": comb.get("excursion_ms"),
        "excursions_that_explain_it_about_as_well_ms": comb.get(
            "excursions_that_explain_it_about_as_well_ms"
        ),
        "off_the_phase_ms": row["phase"].get("excursion_ms"),
        "admitted": comb.get("excursion_ms") is not None,
        "the_phase_stands": bool(
            (row["phase"]["peak"].get("stands_over_bypassed") or 0.0)
            >= efxpartials.STANDS_OVER_BYPASSED
        ),
        "the_level_swing_is": row.get("the_level_swing_is"),
        "lowest_agreement": (row["level"]["partials"] or {}).get("lowest_agreement"),
        "explains": comb.get("explains"),
        "fitted_at_hz": comb.get("hz"),
        "mix": comb.get("mix"),
        "read_at_hz": row["read_at_hz"],
        "stands_over_bypassed": row["phase"]["peak"].get("stands_over_bypassed"),
        "slowest_measurable_hz": row["slowest_measurable_hz"],
        "stands_on_the_edge_of_the_search": row["stands_on_the_edge_of_the_search"],
        "orders": row["orders"],
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
    grid_hz: tuple[float, float] = (0.20, 8.00),
    step_hz: float = 0.01,
    channel: int | None = None,
    progress=None,
) -> dict:
    """Every take under `where` whose setting matches, read into one record.

    The take with the part routed past the effect is the floor every projection is
    read against and it decides which orders are read, so it is found first and by
    its own name: which partials a reading is taken on must not be decided by the
    take being read, or an effect that adds sidebands brings a different set of
    them into its own answer and two settings stop being comparable.

    A take neither pattern names is counted rather than dropped: a pattern that
    matches nothing and a directory that holds nothing produce the same empty
    record otherwise, and they are different mistakes.
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
    elsewhere: list[str] = []

    grid = np.arange(grid_hz[0], grid_hz[1] + step_hz / 2, step_hz)

    beside = takes.other_of_the_pair(used, len(levels))

    def body_of(name: str):
        got, rate, own = efxpartials.body_of(where / name, lead_s, hold_s, on=used)
        if own != used and name not in elsewhere:
            elsewhere.append(name)
        return got, rate

    def also_heard_db(name: str) -> float | None:
        """The same stretch of the same take, on the unit's other output.

        The body and not the whole take, because that is what `heard_db` beside it
        is, and two levels taken over different stretches are not a comparison.
        """
        if beside is None:
            return None
        other, _, _ = efxpartials.body_of(where / name, lead_s, hold_s, on=beside)
        return round(takes.level_db(takes.rms(other)), 1)

    first, rate = body_of(controls[0])
    floor = efxpartials.Floor(first, rate, carrier_hz=carrier_hz, grid=grid)

    # One control read as though it were a setting, against *another* control. Never
    # against its own floor, which is one by construction and would publish a
    # guaranteed refusal as a measured one. Null where the run made only one such
    # take, because then there is no second one to read it against.
    nothing_in_its_path = None
    if len(controls) > 1:
        other, at = body_of(controls[1])
        nothing_in_its_path = {
            "take": controls[0],
            "against": controls[1],
            **_reading(
                efxpartials.measure_one(
                    first,
                    rate,
                    carrier_hz=carrier_hz,
                    floor=efxpartials.Floor(other, at, carrier_hz=carrier_hz, grid=grid),
                )
            ),
        }

    seen_as_rows: list[dict] = []
    readings: list[dict] = []
    for name, source, found in sorted(matched):
        got, at = body_of(name)
        row = efxpartials.measure_one(got, at, carrier_hz=carrier_hz, floor=floor)
        if row is None:
            readings.append(
                {
                    VALUE: int(found.group(VALUE)),
                    "take": name,
                    "named_by": source,
                    "why": "no order of the carrier cleared the threshold in this take",
                }
            )
            continue
        seen_as_rows.append(row)
        reading = {
            VALUE: int(found.group(VALUE)),
            "take": name,
            **_reading(row),
            "heard_db": round(takes.level_db(takes.rms(got)), 1),
            "also_heard_db": also_heard_db(name),
            "settled_s": settled_s,
            "named_by": source,
        }
        readings.append(reading)
        if progress:
            progress(reading)

    readings.sort(key=lambda r: (r[VALUE], r["take"]))
    admitted = [r for r in readings if r.get("admitted")]
    stood = [r for r in readings if r.get("the_phase_stands")]

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
            "why_nothing_in_its_path": WHY_NOTHING_IN_ITS_PATH,
        },
        "why_two_excursions": WHY_TWO_EXCURSIONS,
        "why_admitted": WHY_ADMITTED,
        "floor": _floor_of(admitted, "excursion_ms"),
        "floor_off_the_phase": _floor_of(stood, "off_the_phase_ms"),
        "why_floor": WHY_FLOOR,
        "gates": efxpartials.limits(floor, seen_as_rows),
        "held": held or [],
        "why_held": WHY_HELD,
        **({"held_not_spelled_out": held_not_spelled_out} if held_not_spelled_out else {}),
        "takes_from": str(where),
        "manifest": takes.manifest_note(listed, files),
        "settings_asked": sorted({r[VALUE] for r in readings}),
        "settings_admitted": sorted({r[VALUE] for r in admitted}),
        "settings_the_phase_admitted": sorted({r[VALUE] for r in stood}),
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
