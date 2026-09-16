"""What one insertion effect's modulator does to level and to balance, setting by setting.

Read from saved takes, with no machine attached, the same way `efx-rate` is and
for the same reason: a reading improved later can be taken again off takes that
are still there.

**One take, not two.** A free-running modulator is at a different phase on every
note struck, so a comparison of two takes disagrees about nearly everything and
the yardstick swallows the change. Its rate, its depth and the shape of its cycle
are the same on every take even though its phase is not, so all three are read
inside one -- which is the argument `vibrato` already makes about pitch, applied
to the two quantities `sway` reads.

**The cycle is reported and not named.** Every row carries one averaged cycle of
whatever was modulating, the size of each harmonic in it, and how much of it was
spent climbing. Which printed shape that answers to is a comparison between this
unit and a page, and it is not made here.

**What the run had set while it read.** A type with more than one modulator
returns whichever dominates, so a sweep of one has to hold the other still, and
two readings taken under different holds are readings of different things. What
the run applied is carried into the record beside every row it covered.
"""

from __future__ import annotations

import re
from pathlib import Path

from . import sway, takes

QUESTION = (
    "What one insertion effect type's modulator does to the level of its output and to "
    "the balance between its two channels, at each setting of the byte."
)

METHOD = sway.METHOD

LIMITS = (
    "`noise_floor_db` is the level of the take's own lead-in, and a frame that does not "
    "stand twelve decibels over it is dropped rather than tracked: a level read out of "
    "silence is the noise's own level and it does not decay, so a tail tracked past the "
    "floor reads as a modulation flattening out. `searched_hz` bounds every null here -- "
    "a row saying nothing was found says nothing about a rate outside it. `cycles_folded` "
    "is how many cycles went into the curve reported, and a curve folded from few of them "
    "carries whatever else moved the track at that rate. Every row is bounded by the "
    "control, which is beside them."
)

NOT_HERE = (
    "No shape is named. A cycle is reported as a curve, as harmonics and as a fraction "
    "spent climbing, and which printed name it answers to is a comparison between this "
    "unit and a page -- the printed names are in documents/, under the same address, and "
    "the two are not joined here. Nor is there a law: whether the depth byte beside this "
    "one scales the cycle or reshapes it is a fit across settings and is not made here."
)

VALUE = "value"
"""The named group a `--setting` pattern has to capture, as `efx-rate` names it.

A run names its takes however its own question needed, so the pattern that pulls
the byte back out belongs to the invocation rather than to this module -- and
because it is in the invocation it is in the record.
"""


def sweep(
    where: str | Path,
    *,
    type_id: str,
    address: str,
    setting: str,
    still: str | None = None,
    held: list[dict] | None = None,
    lead_s: float = 0.6,
    search_hz: tuple[float, float] = sway.SEARCH_HZ,
    least_db: float = 0.5,
    channels: tuple[int, int] | None = None,
    progress=None,
) -> dict:
    """Every take under `where` whose setting matches, read into one record.

    A take the pattern does not match is skipped and counted. The count is
    reported rather than dropped: a pattern that matches nothing and a directory
    that holds nothing produce the same empty record otherwise, and they are
    different mistakes.

    One pair of channels is chosen for the whole run before any of it is read.
    A pair rather than a channel because half of what is read is the difference
    between two, and one pair for the run because a difference taken across a
    pair chosen per take is a difference between two different things -- which is
    what a setting that empties one side would otherwise produce.

    `still` names the take with no modulation in it -- the modulator switched off,
    or the part routed past the effect -- and the control goes into that one. It has to be named rather than guessed: a take already carrying a
    modulation ends up with two, the search finds the unit's own, and the control
    reads as having failed on material it can read perfectly well. Where a run
    made no such take the quietest row carries the control and the record says so,
    which bounds every row by whatever that take could still be read through.
    """
    where = Path(where)
    listed, files = takes.listing(where)
    pattern = takes.capturing(setting, VALUE)

    matched = [
        (name, listed.get(name, {}), *takes.named_by(pattern, listed.get(name, {}), name))
        for name in files
    ]
    wanted = [(name, entry, by, found) for name, entry, by, found in matched if found]
    skipped = [
        str(entry.get("setting") or name) for name, entry, _, found in matched if not found
    ]
    if not wanted:
        picked, levels = (0, 1), []
    else:
        picked, levels = takes.channel_pair_reaching(where, [n for n, _, _, _ in wanted])
    used = tuple(picked) if channels is None else tuple(channels)

    readings: list[dict] = []
    by_setting: dict[str, list[sway.Swing]] = {}
    _found: dict[str, sway.Swing] = {}
    for name, _entry, named_by, found in wanted:
        frames, rate = takes.read(where / name)
        swing = sway.measure(
            frames[:, list(used)],
            rate,
            lead_s=lead_s,
            search_hz=search_hz,
            least_db=least_db,
        )
        value = int(found.group(VALUE))
        by_setting.setdefault(str(value), []).append(swing)
        _found[name] = swing
        reading = {VALUE: value, **swing.to_json(), "take": name, "named_by": named_by}
        readings.append(reading)
        if progress:
            progress(reading)

    # The control goes into the take the modulator was off in where the run made
    # one, and into the quietest row where it did not.
    vouched, vouched_from, chosen_by = None, None, None
    parked = [n for n in files if re.search(still, n)] if still else []
    if parked:
        vouched_from, chosen_by = parked[0], "the take named as the modulator switched off"
    elif wanted:
        vouched_from = min(
            (n for n, _, _, _ in wanted),
            key=lambda n: sum(
                1 for t in (
                    _found[n].level, _found[n].level_in_db, _found[n].balance
                ) if t.found
            ),
        )
        chosen_by = "the row that showed least, no still take having been named"
    if vouched_from is not None:
        frames, rate = takes.read(where / vouched_from)
        vouched = sway.control(
            frames[:, list(used)], rate, lead_s=lead_s, search_hz=search_hz
        )

    readings.sort(key=lambda r: r[VALUE])
    return {
        "question": QUESTION,
        "type": type_id,
        "address": address,
        "method": METHOD,
        "limits": LIMITS,
        "not_in_this_record": NOT_HERE,
        "why_the_tracks": sway.WHY_THE_TRACKS,
        "why_the_cycle": sway.WHY_THE_CYCLE,
        "why_the_going_up_fraction": sway.WHY_GOING_UP,
        "cycle_points": sway.CYCLE_POINTS,
        "searched_hz": list(search_hz),
        "channel": {
            "read": list(used),
            "chosen_by": "given" if channels is not None else "the pair that reached highest",
            "reached_db": levels,
            "why": sway.WHY_CHANNEL,
        },
        "control": vouched,
        "control_taken_from": vouched_from,
        "control_chosen_by": chosen_by,
        "why_one_take_carries_the_control": (
            "The control was injected into one take rather than into each, and the take is "
            "named beside it. It cannot be otherwise: a take that already carries a "
            "modulation ends up with two, the search finds the unit's own, and the control "
            "reads as having failed on material it can read perfectly well. The cost is that "
            "the depth reached bounds the other rows only as far as their takes resemble this "
            "one, and each row's tracked frames are beside it so a reader can see where they "
            "do not."
        ),
        "held": held or [],
        "why_held": (
            "What else the run had written when it took these readings. A type with more than "
            "one modulator returns whichever dominates, so a reading taken with the other "
            "stage turned down and one taken with it running are readings of different things, "
            "and nothing in the numbers says which is which."
        ),
        "takes_from": str(where),
        "manifest": takes.manifest_note(listed, files),
        "settings_asked": sorted({r[VALUE] for r in readings}),
        "readings": readings,
        "takes_not_matching": takes.not_matching(skipped),
    }


__all__ = ["LIMITS", "METHOD", "NOT_HERE", "QUESTION", "VALUE", "sweep"]
