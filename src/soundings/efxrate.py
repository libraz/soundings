"""What frequency one insertion effect's rate slot modulates at, setting by setting.

Read from saved takes, with no machine attached. A run holds a note through the
effect at one setting of one byte, saves the take, writes the next setting, and
repeats; this reads what came back. Keeping the reading out of the run is what
lets a reading be improved without the unit sounding again -- twice already a
figure changed because the reader changed, and both times the takes were still
there to be asked.

**A reading is reported with what bounds it, or it is not reported.** Each of the
take's partials is followed separately and each returns its own period; the figure
here is what they agreed on, and how many of them agreed, out of how many, is
beside it -- with every partial's own figure, so that one which found half or
twice the rate is visible rather than averaged in. Beside those: how loud the take
was, because a reading taken from a take at the noise floor is a reading of the
floor and is otherwise indistinguishable from a good one; and the slowest rate
that take length could have carried, because a figure near that is a floor rather
than a rate.

**What the run had set while it read.** A type with two modulators returns
whichever dominates, so a sweep of one of them has to turn the other down or hold
it still, and the two kinds of reading are not comparable afterwards. The settings
the run applied are carried into the record with every reading they covered.

**A byte that names no rate can still make the output repeat, and that is a third
question rather than the first one told loosely.** The reading is the same: what
periodicity the take carries, by both routes, with what bounds it. What differs is
what may be concluded from it -- a rate slot returning its own setting is the slot
answering, while a pitch byte returning a periodicity at all is a fact about the
structure behind it. The record says which of the two it is, because nothing in
the numbers does.

**No law, no table, no verdict.** Whether a slot's readings follow one curve or
another is a fit across several types, and the fit is not made here -- see the
archive's own note on what this repository does not derive. The printed range for
the address lives in `documents/`, is evidence about a page rather than about a
unit, and is not joined to this.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import rates, takes

QUESTION = (
    "What frequency one insertion effect type's rate slot modulates at, at each "
    "setting of the byte."
)

METHOD = (
    "A note was held through the effect and the level and frequency of each of the "
    "take's strongest partials were followed and searched for a period. The figure "
    "reported is the one the partials agreed on, and each partial's own is beside it."
)

LIMITS = (
    "`slowest_measurable_hz` is the slowest rate this take was long enough to carry "
    "two cycles of, and a reading within a few per cent of it is a floor rather than "
    "a rate. `fastest_measurable_hz` is the top of the band the period was looked for "
    "in, and it is not a property of the take: a modulation above it cannot be "
    "returned whatever the take holds, and what comes back instead is a divisor of it "
    "that does fit. A reading near half the ceiling is the one to check against "
    "`lines`, which are found in the same band but are not made to choose. "
    "`heard_db` is the level over the take of the one channel named in "
    "`channel`: a "
    "reading taken from a take near the noise floor is a reading of the floor, and it "
    "is stable, which is what makes it dangerous. `settled_s` is how long the run "
    "waited after writing the setting, which matters for any type that accelerates -- "
    "a take begun before it settles returns the ramp's average."
)

WHY_CHANNEL = (
    "The channel every figure in this record was read from, and the highest each "
    "channel of the interface reached across the takes this record was built from. "
    "Chosen once for the run rather than per take: the interface carries inputs the "
    "unit is not on, those inputs are not silent, and a take whose output falls below "
    "one of them is read from that input instead -- which returns a level, and a rate, "
    "belonging to something else. `loudest_elsewhere` names every take whose own "
    "loudest channel is not the one used, because a reading that changed channel is "
    "exactly what the figures cannot say on their own."
)

NOT_HERE = (
    "No law, no table and no verdict: which curve these readings follow is a fit "
    "across types and is not made here. The printed range for this address is in "
    "documents/, under the same address, and is evidence about a page rather than "
    "about this unit; the two are not joined."
)

VALUE = "value"
"""The named group a `--setting` pattern has to capture.

A run names its takes however its own question needed, so the pattern that pulls
the byte back out belongs to the invocation rather than to this module -- and
because it is in the invocation it is in the record, where a reader can see how
the settings were read rather than trusting that they were.
"""

TYPE = "type"
"""The named group an `--untouched` pattern captures instead of `value`.

Nothing was written, so there is no byte to name; what separates one take from
the next is which type was loaded.
"""

CAME_FROM = "from"
"""A named group a `--setting` pattern may capture beside `value`.

What the same address held, and the modulator had reached, when the byte in
`value` was written -- with no reset in between, so the reading is of a parameter
approached from there rather than from wherever a reset leaves it.

Optional, because on most types it is not a question: a byte that names a rate
outright returns that rate whatever it was set to before. It stops being not a
question wherever the parameter is one a modulator has to travel to, and there a
record shaped as one byte to one rate cannot hold the reading at all -- two takes
of the same byte answer differently and nothing in the row says why. `rest` is
what a run writes here for a take that followed a reset and nothing else.
"""

NO_RATE_QUESTION = (
    "What frequency one insertion effect type's output was measured to repeat at, at "
    "each setting of a byte that names no rate."
)

NO_RATE_WHY = (
    "The byte this record sweeps is not a rate slot, and the periodicity below is "
    "what the output was measured to do rather than what the byte asked for. Stated "
    "in the question rather than left to be inferred from the address, because the "
    "two cases are read differently and nothing in the numbers separates them: a "
    "rate slot returning its own setting is the slot answering, while a byte that "
    "names no rate returning one at all is a fact about the structure behind it -- "
    "and what that structure is, is not decided here. Which of the two a record is "
    "belongs to the run, which knows what it wrote; it is not read off the printed "
    "page, because a page is evidence about a page."
)

UNTOUCHED_QUESTION = (
    "What a take carried when one insertion effect type was loaded and no parameter "
    "of it was written."
)

UNTOUCHED_WHY = (
    "The baseline a swept reading of the same type is read against. A sweep says what "
    "changed with the byte and cannot say what was there before it, so a line already "
    "present with nothing written is not attributable to the slot -- and whether it "
    "belongs to the type's own defaults, to the note that carried it, or to the "
    "reading is not decided here."
)


def _body(samples, rate: int, *, index: int, lead_s: float, hold_s: float, trim_s: float):
    first = int((lead_s + trim_s) * rate)
    last = int((lead_s + hold_s - trim_s) * rate)
    return takes.channel(samples, index)[first:last]


def _loudness_db(samples, index: int) -> float:
    body = takes.channel(samples, index)
    return float(20.0 * np.log10(max(float(np.sqrt((body**2).mean())), 1e-12)))


def _read_take(
    where: Path,
    name: str,
    entry: dict,
    *,
    index: int,
    lead_s: float,
    trim_s: float,
    hold_s: float | None,
    shared_lines: bool,
) -> dict:
    """One take read into the fields every reading here carries."""
    samples, rate = takes.read(where / name)
    seconds = float(entry.get("seconds") or samples.shape[0] / rate)
    hold = hold_s if hold_s is not None else seconds - 1.0
    body = _body(samples, rate, index=index, lead_s=lead_s, hold_s=hold, trim_s=trim_s)
    per_partial = rates.read_partials(body, rate)
    agreed = rates.agreed_rate(per_partial)
    swings = [r["level_swing"] for r in per_partial if r["level_swing"]]
    reading = {
        "rate_hz": agreed["rate_hz"],
        # Present only where the partials split evenly between two rates, which is
        # a take saying both and counting for neither. Carried through rather than
        # dropped: a row with no rate and no reason reads as a take nothing was
        # found in, and this one found two things.
        **({"split_between_hz": agreed["split_between_hz"]}
           if "split_between_hz" in agreed else {}),
        "agreeing": agreed["agreeing"],
        "of": agreed["of"],
        "rates": agreed["rates"],
        "heard_db": round(_loudness_db(samples, index), 1),
        # Dropped from the reading before it is published and gathered into the
        # record's own channel block, so both stages say this the same way.
        "_own": int(np.argmax(takes.channel_levels(samples))),
        "slowest_measurable_hz": swings[0].get("slowest_measurable_hz") if swings else None,
        "fastest_measurable_hz": rates.FASTEST_HZ,
        "hold_s": round(hold, 3),
        "take": name,
    }
    if shared_lines:
        # What the partials' spectra have in common, for the takes that may hold
        # more than one modulation. A vote returns one answer and can land between
        # two lines; this returns both, so a type with a modulator per stage is
        # readable without silencing either.
        #
        # Both series: the same rate's harmonics come out in different proportions
        # in each, so a line plain in one can be buried in the other. Neither is a
        # check on the other for whether a rate is the rate -- see `rates.LEVEL`.
        reading["lines"] = rates.common(body, rate, which=rates.LEVEL)
        reading["lines_in_the_frequency_series"] = rates.common(
            body, rate, which=rates.FREQUENCY)
    return reading


def read_directory(
    where: str | Path,
    *,
    type_id: str,
    address: str,
    setting: str,
    held: list[dict] | None = None,
    held_not_spelled_out: str | None = None,
    settled_s: float | None = None,
    names_a_rate: bool = True,
    lead_s: float = 0.6,
    trim_s: float = 0.5,
    hold_s: float | None = None,
    shared_lines: bool = False,
    channel: int | None = None,
    progress=None,
) -> dict:
    """Every take under `where` whose setting matches, read into one record.

    A take the pattern does not match is skipped and counted. The count is
    reported rather than dropped: a pattern that matches nothing and a directory
    that holds nothing produce the same empty record otherwise, and they are
    different mistakes.

    One channel is chosen for the whole run before any of it is read, from the
    highest each channel reached across the matched takes. There are no reference
    takes here to choose from -- a sweep of a rate is all there is -- so the
    channel is the one that ever carried the unit rather than the one that carried
    it on average, which a setting that silences the output would drag away.
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
        reached, levels = 0, []
    else:
        reached, levels = takes.channel_reaching(where, [name for name, _, _, _ in wanted])
    used = reached if channel is None else int(channel)

    readings: list[dict] = []
    elsewhere: list[str] = []
    for name, entry, named_by, found in wanted:
        came_from = (
            found.groupdict().get(CAME_FROM)
            if CAME_FROM in (pattern.groupindex or {})
            else None
        )
        reading = {
            VALUE: int(found.group(VALUE)),
            **({"came_from": came_from} if came_from is not None else {}),
            **_read_take(
                where,
                name,
                entry,
                index=used,
                lead_s=lead_s,
                trim_s=trim_s,
                hold_s=hold_s,
                shared_lines=shared_lines,
            ),
            "settled_s": settled_s,
            "named_by": named_by,
        }
        if reading.pop("_own") != used:
            elsewhere.append(name)
        readings.append(reading)
        if progress:
            progress(reading)

    readings.sort(key=lambda r: (r[VALUE], str(r.get("came_from") or "")))
    approached = any("came_from" in r for r in readings)
    return {
        "question": QUESTION if names_a_rate else NO_RATE_QUESTION,
        **({} if names_a_rate else {"why_the_byte_names_no_rate": NO_RATE_WHY}),
        "type": type_id,
        "address": address,
        "method": METHOD,
        "limits": LIMITS,
        "not_in_this_record": NOT_HERE,
        "channel": {
            "read": used,
            "chosen_by": "given" if channel is not None else "highest across the takes read",
            "reached_db": levels,
            "loudest_elsewhere": sorted(elsewhere),
            "why": WHY_CHANNEL,
        },
        "held": held or [],
        "why_held": "What else the run had written when it took these readings. A "
        "type with more than one modulator returns whichever dominates, so a reading "
        "taken with the other stage turned down and one taken with it running are "
        "different readings of different things, and nothing in the numbers says "
        "which is which.",
        # Beside the block it qualifies, and emitted here rather than added to the
        # payload by whoever called: a caveat assembled somewhere else is one the
        # next run publishes without.
        **(
            {"held_not_spelled_out": held_not_spelled_out}
            if held_not_spelled_out
            else {}
        ),
        "takes_from": str(where),
        "manifest": takes.manifest_note(listed, files),
        "settings_asked": sorted({r[VALUE] for r in readings}),
        **(
            {
                "why_came_from": "What this address held, and the modulation had "
                "reached, when each row's own value was written -- with no reset in "
                "between. `rest` is a take that followed a reset and nothing else, "
                "which is what every other record here is made of. Present because "
                "this run found two takes of one byte answering differently, and a "
                "row that carries only the byte cannot say which of them it is."
            }
            if approached
            else {}
        ),
        "readings": readings,
        "takes_not_matching": takes.not_matching(skipped),
    }


def _type_read(text: str) -> str:
    """A captured type as the two bytes a record spells it with, where it can be.

    A run names its takes with the bytes run together; where they are not four hex
    digits the capture is carried through as it was written rather than reshaped
    into something that looks like an address.
    """
    bare = text.replace(" ", "").replace("-", "").upper()
    if len(bare) == 4 and all(c in "0123456789ABCDEF" for c in bare):
        return f"{bare[:2]} {bare[2:]}"
    return text


def read_untouched(
    where: str | Path,
    *,
    setting: str,
    settled_s: float | None = None,
    lead_s: float = 0.6,
    trim_s: float = 0.5,
    hold_s: float | None = None,
    shared_lines: bool = False,
    channel: int | None = None,
    progress=None,
) -> dict:
    """Takes made with a type loaded and no parameter written, read into one record.

    The same takes, the same reading, and a different question: there is no byte,
    so what separates one take from the next is the type. Kept here rather than
    forced into a sweep, because a reading with nothing written has no setting and
    a column of settings with a word in it is worse than a second shape.
    """
    where = Path(where)
    listed, files = takes.listing(where)
    pattern = takes.capturing(setting, TYPE)

    matched = [
        (name, listed.get(name, {}), *takes.named_by(pattern, listed.get(name, {}), name))
        for name in files
    ]
    wanted = [(name, entry, by, found) for name, entry, by, found in matched if found]
    skipped = [
        str(entry.get("setting") or name) for name, entry, _, found in matched if not found
    ]
    if not wanted:
        reached, levels = 0, []
    else:
        reached, levels = takes.channel_reaching(where, [name for name, _, _, _ in wanted])
    used = reached if channel is None else int(channel)

    readings: list[dict] = []
    elsewhere: list[str] = []
    for name, entry, named_by, found in wanted:
        reading = {
            TYPE: _type_read(found.group(TYPE)),
            **_read_take(
                where,
                name,
                entry,
                index=used,
                lead_s=lead_s,
                trim_s=trim_s,
                hold_s=hold_s,
                shared_lines=shared_lines,
            ),
            "settled_s": settled_s,
            "named_by": named_by,
        }
        if reading.pop("_own") != used:
            elsewhere.append(name)
        readings.append(reading)
        if progress:
            progress(reading)

    readings.sort(key=lambda r: r[TYPE])
    return {
        "question": UNTOUCHED_QUESTION,
        "wrote": None,
        "method": METHOD,
        "limits": LIMITS,
        "not_in_this_record": NOT_HERE,
        "channel": {
            "read": used,
            "chosen_by": "given" if channel is not None else "highest across the takes read",
            "reached_db": levels,
            "loudest_elsewhere": sorted(elsewhere),
            "why": WHY_CHANNEL,
        },
        "why_untouched": UNTOUCHED_WHY,
        "takes_from": str(where),
        "manifest": takes.manifest_note(listed, files),
        "types_asked": sorted({r[TYPE] for r in readings}),
        "readings": readings,
        "takes_not_matching": takes.not_matching(skipped),
    }
