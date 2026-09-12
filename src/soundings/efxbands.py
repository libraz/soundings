"""What one insertion effect parameter does to the level of each third octave.

Read from saved takes, with no machine attached, the same way the rate stage is:
a run holds a stimulus through the effect at one setting of one byte, saves the
take, writes the next setting, and repeats; this reads what came back. What is
different here is the quantity. A rate is a number per take; a filter or a trim is
a *profile*, and one figure per setting would hide where it hinges, which is the
part the printed range names.

**The band energy, not the band's loudest bin.** Where the stimulus is noise --
and it has to be, for a stage that only shapes what is already there -- a peak bin
is one sample of a random variable and moves by decibels between takes of the same
setting. The energy summed over a band averages thousands of bins and moves by
hundredths.

**Nothing is a reading until it clears the floor this run measured.** The floor is
the spread, band by band, across several takes of one setting, taken in the same
session as everything read against it: a run's repeatability belongs to its
evening, its converter and its stimulus, and a profile read against another run's
floor is read against a reference that was never in the room.

**The control says whether the reference is unity.** A sweep is reported as a
deviation from the setting the run called flat, and that only means what it looks
like if the flat setting is the effect doing nothing. So the same profile is taken
with the part routed past the effect entirely. If bypass and flat differ, every
deviation below is still a measurement of the byte -- but of the byte against a
stage that was doing something, and the record says so instead of implying unity.

**One channel for the whole run, chosen once.** An interface has inputs the unit
is not plugged into, and they are not silent. Picking the loudest channel of each
take separately means a setting that turns the output down far enough is read from
whichever input happened to be noisiest instead -- a complete, plausible, entirely
wrong profile, with nothing in the figures to say the reading changed channel. The
channel is chosen from the reference takes, where the unit is certainly sounding,
and every take that disagrees with that choice is named.

**The band set is a resolution, and the same takes read at two of them are two
records.** A band wider than the deviation inside it averages that deviation with
what is beside it and reports it shallower than it was, and a deviation still
growing where the bands run out is reported as though it had stopped there. Both
look like ordinary figures. So the width of a band and the ends of the set are
stated per reading -- how far the profile had levelled off where the set ended,
and how wide it was around its largest -- rather than left for a reader to infer
from a centre list, and a set that answers one row's question badly is rerun over
the same takes instead of being argued with.

**No filter, no corner, no shape.** Which curve these bands lie on, where a shelf
hinges, what order it is -- that is a fit, and the fit is not made here. The
printed range for the address lives in `documents/`, is evidence about a page
rather than about a unit, and is not joined to this.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import takes

VALUE = "value"
"""The named group a `--setting` pattern has to capture.

A run names its takes however its own question needed, so the pattern that pulls
the byte back out belongs to the invocation rather than to this module -- and
because it is in the invocation it is in the record, where a reader can see how
the settings were read rather than trusting that they were.
"""

THIRD_OCTAVES = (
    100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250,
    1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500,
)
"""Third octaves rather than octaves.

The default, not a constant of the unit. Octave bands cannot separate the two
corners a printed range offers when they are one octave apart, which is the case
for both of this type's shelves -- an octave band centred between them would
report one number for either.
"""

TWELFTH_OCTAVES = tuple(round(1000.0 * 2 ** (k / 12), 1) for k in range(-64, 49))
"""Twelfth octaves from about 25 Hz to 16 kHz, for what a third octave averages away.

Four times finer than the set above and reaching two octaves below it, which is
two different limits of that set rather than one. A deviation that is still
growing at the lowest band is reported as though it had stopped there, and a
deviation narrower than a band is reported shallower than it is; a third octave
does both to this type, and the same takes answer at this resolution with no
machine attached.

**Where it starts is a measurement, not a preference.** The stimulus is noise and
its bands are read against the run's own repeats of one setting; those repeats
agree to within about a tenth of a decibel above 80 Hz, a few tenths down to
25 Hz, and more than a decibel by 20 Hz, which is where a band stops being able to
report anything smaller than what it fails to repeat. The stimulus itself is still
fifteen decibels above the same chain's silence at 20 Hz, so what ends the set
below is the repeatability and not the reach.

**Where it ends is a default and not a bound.** What a band above 16 kHz can
report depends on the unit's own output band and on the stimulus, and both are
measured rather than assumed: on the machine this was written against, the takes
stand thirty decibels above the same chain's silence at 16 kHz and are level with
it by 18, so a band above the set would report the converter. A run whose unit
reaches further is given its own centres.

Anchored on 1000 Hz so that every fourth centre is a third octave of the same
series and a reader can lay the two records of one sweep against each other.
"""

BAND_SETS = {
    "third-octave": (THIRD_OCTAVES, 1 / 3),
    "twelfth-octave": (TWELFTH_OCTAVES, 1 / 12),
}
"""The sets a run can ask for by name, each with the width its centres are read at.

The width belongs to the set rather than being a second choice beside it, because
the two only mean anything together: twelfth-octave centres read as third-octave
bands are the coarse measurement sampled four times as densely, which looks like a
finer reading of a narrow deviation and is not one.
"""

QUESTION = (
    "What one insertion effect type's parameter does to the level of each third "
    "octave band, at each setting of the byte."
)

METHOD = (
    "A stimulus was held through the effect at one setting of one byte and the energy "
    "of each third octave band of the take was measured. The figure reported per band "
    "is that energy against the same band of a setting the run held flat, and the "
    "spread of several takes of that flat setting is the floor each of them has to "
    "clear."
)

LIMITS = (
    "`floor_db` is the observed spread across the run's own repeats of one setting, "
    "band by band, and a deviation inside it is not a reading. It is a range over a "
    "handful of takes and not a bound: a band where those takes happened to agree "
    "closely has a small floor because it was sampled a few times, so a deviation just "
    "outside the floor is not thereby a reading either, and how many takes drew it is "
    "in `reference`. `heard_db` is the level over the whole take of the one channel "
    "named in `channel`, and `above_the_silence_db` is how far "
    "that is above what the same chain recorded with nothing played: a setting that "
    "turns the output down far enough returns the room and the converter, and the "
    "bands of that are the floor's own shape rather than anything the byte did. The "
    "floor's profile is in `silence` so the comparison can be made rather than taken "
    "on trust. Band energy is measured over the held part of the take only, so a "
    "setting whose effect is in the attack or the release is not in these figures at "
    "all. A band the stimulus does not reach cannot report what the effect did there, "
    "which is a limit of the stimulus and not a bound on the unit. "
    "Every figure is bounded by the width of a band and by where the set ends as well. "
    "A deviation narrower than one band is averaged with what is beside it and read "
    "shallower than it is, so where `half_below_hz` and `half_above_hz` are a band or "
    "two apart the `largest_db` beside them is a lower bound rather than a height; and "
    "where `settled_below_db` or `settled_above_db` is not near zero the profile was "
    "still changing when the bands ran out, so the largest figure is where the set "
    "ended rather than where the effect did. Both are answered by reading the same "
    "takes again at another resolution, not by reading further into these figures."
)

NOT_HERE = (
    "No filter, no corner and no order: which curve these bands lie on and where a "
    "shelf hinges is a fit across settings and types, and the fit is not made here. "
    "The printed range for this address is in documents/, under the same address, and "
    "is evidence about a page rather than about this unit; the two are not joined."
)

WHY_REFERENCE = (
    "The setting every profile below is reported against, and the repeats that give "
    "it its floor. Taken in the same session rather than carried from an earlier run: "
    "what a noise stimulus fails to repeat belongs to the evening it was recorded in, "
    "and a deviation read against another run's floor is read against a reference that "
    "was never in the room. "
    "The takes here cannot be read as a setting to show what the reading returns for "
    "one that holds nothing: a repeat read against the mean of the others differs from "
    "it by no more than the spread those same repeats drew the floor from, so it "
    "clears that floor in no band and would publish a control that cannot fire. What "
    "does show it is a run's own null -- the same byte swept with the stage it shapes "
    "turned off -- which is a separate record made from separate takes, and at this "
    "band set those nulls return a few tenths of a decibel."
)

WHY_CONTROL = (
    "The same profile with the part routed past the effect instead of through it. A "
    "deviation from the flat setting only reads as what the byte did if the flat "
    "setting is the effect doing nothing, and that is a measurement rather than an "
    "assumption. Where this differs from the reference, the readings below are still "
    "what the byte did -- measured against a stage that was doing something, which is "
    "stated here rather than implied away."
)

WHY_SILENCE = (
    "The same chain with nothing played, as a level and as a profile. A setting that "
    "turns the output off does not stop the take being recorded, so what comes back is "
    "the room, the converter and whatever the machine puts out idle -- and read as a "
    "deviation from the flat setting that is a large, ragged, frequency-dependent "
    "figure, which is what a reading of the floor looks like and is not what a reader "
    "would guess it was. Published as a profile rather than as a single number so that "
    "a reading suspected of being the floor can be held against the floor's own shape."
)

WHY_CHANNEL = (
    "The channel every figure in this record was read from, and the highest each "
    "channel of the interface reached while the reference takes were sounding. Chosen "
    "once for the run rather than per take: the interface carries inputs the unit is "
    "not on, those "
    "inputs are not silent, and a take whose output falls below one of them is read "
    "from that input instead -- which returns a full, ragged, plausible profile of "
    "something else entirely. `loudest_elsewhere` names every take whose own loudest "
    "channel is not the one used, because a reading that changed channel is exactly "
    "what the figures cannot say on their own."
)

WHY_OTHER = (
    "The same readings taken from a second channel of the interface, each against "
    "that channel's own repeats of the flat setting. A profile measured in one "
    "channel is the whole answer only if the other answers the same way, and for a "
    "type printed as a stereo one that is a measurement rather than an assumption. "
    "Per reading, `other_db` is how far this channel itself moved -- which is also "
    "what says whether it carried the unit at all, because an input nobody plugged "
    "anything into does not follow a parameter -- and `apart_db` is the largest "
    "disagreement between the two, band by band. Each is taken against its own "
    "reference first, so a standing difference in level between the channels is not "
    "counted as a disagreement about frequency; what is left is the shape, which is "
    "the thing one channel cannot report on its own. Which channel this is was not "
    "chosen by knowing the wiring: it is the second highest of the reference takes, "
    "and `channel.reference_db` is what a reader checks that against."
)

WHY_SPAN = (
    "Four figures per reading that say what the band set could and could not see of "
    "the profile, so that `largest_db` is read as a height where it is one. "
    "`half_below_hz` and `half_above_hz` are the nearest band on each side of the "
    "largest where the deviation had fallen to less than half of it, and either is "
    "null where it had not fallen that far before the bands ran out -- which is what a "
    "profile that levels off rather than returning looks like, and is not the same as "
    "a narrow one. `settled_below_db` and `settled_above_db` are how much the "
    "deviation was still changing over the outermost third octave at each end of the "
    "set: near zero says the profile had levelled off inside the bands, and anything "
    "else says the largest figure is where the set ended. None of the four is a "
    "corner, a width or an order -- a filter would give each of them a name, and "
    "naming them is the fit this record does not make."
)

WHY_HELD = (
    "What else the run had written when it took these readings. A band profile is the "
    "whole chain's, so a parameter read with another of the type's stages moved and "
    "one read with it at its centre are readings of different things, and nothing in "
    "the numbers says which is which."
)


def _body(samples, rate: int, *, index: int, lead_s: float, hold_s: float, trim_s: float):
    first = int((lead_s + trim_s) * rate)
    last = int((lead_s + hold_s - trim_s) * rate)
    return takes.channel(samples, index)[first:last]


def _loudness_db(samples, index: int) -> float:
    body = takes.channel(samples, index)
    return float(20.0 * np.log10(max(float(np.sqrt((body**2).mean())), 1e-12)))


def energies(
    body: np.ndarray, rate: int, centres=THIRD_OCTAVES, width_octaves: float = 1 / 3
) -> list[float]:
    """Energy per band, in dB, summed over the bins the band covers.

    Zero padded well past the take's own resolution so that the lowest band still
    has bins in it to sum: at a hundred hertz a third octave is twenty three hertz
    wide, which a transform of the take's own length resolves into a handful.

    **How wide a band is does not follow from where its centre is.** A set of
    centres a twelfth of an octave apart can be read as bands a twelfth wide, which
    is a finer measurement, or as third-octave bands sampled four times as densely,
    which is the same measurement read at more points -- and the two differ most on
    exactly the narrow deviation the finer set was asked for. So the width is given
    rather than taken from the spacing, and it is in the record beside the centres.
    """
    win = np.hanning(body.size)
    power = np.abs(np.fft.rfft(body * win, n=1 << 19)) ** 2
    freq = np.fft.rfftfreq(1 << 19, 1.0 / rate)
    # Sliced rather than masked. The bins are already in order, so a band is a
    # range of them and not a test over all of them -- which is what lets a set of
    # a hundred bands cost what a set of twenty does over a directory this size.
    edge = 2 ** (width_octaves / 2)
    out = []
    for centre in centres:
        first = int(np.searchsorted(freq, centre / edge, side="left"))
        last = int(np.searchsorted(freq, centre * edge, side="left"))
        total = float(power[first:last].sum())
        out.append(round(10.0 * np.log10(max(total, 1e-30)), 3))
    return out


def _profile(
    where: Path,
    name: str,
    entry: dict,
    *,
    channels: tuple[int, ...],
    lead_s: float,
    trim_s: float,
    hold_s: float | None,
    centres,
    width_octaves: float,
) -> tuple[dict[int, list[float]], dict[int, float], float, int]:
    """Every channel asked for, out of one read of the take.

    Read once and measured twice rather than read twice: the second channel is a
    control over the first, and a control that costs another pass over a hundred
    and fifty gigabytes is a control that gets left out of the next run.
    """
    samples, rate = takes.read(where / name)
    seconds = float(entry.get("seconds") or samples.shape[0] / rate)
    hold = hold_s if hold_s is not None else seconds - 1.0
    bands, heard = {}, {}
    for index in channels:
        body = _body(samples, rate, index=index, lead_s=lead_s, hold_s=hold, trim_s=trim_s)
        bands[index] = energies(body, rate, centres, width_octaves)
        heard[index] = round(_loudness_db(samples, index), 1)
    own = int(np.argmax(takes.channel_levels(samples)))
    return bands, heard, round(hold, 3), own


def _settled(moved: list[float], centres, *, low: bool) -> float | None:
    """How much the deviation was still changing where the bands ran out.

    Read over the outermost third octave rather than the outermost band, so that
    the figure means the same thing whatever the set's spacing is and so that one
    band's own scatter does not decide it. Near zero says the profile had levelled
    off inside the set; anything else says the largest deviation is where the bands
    ended, and nothing read at the edge can tell those two apart on its own.
    """
    step = 2 ** (1 / 3)
    if low:
        end, inner = 0, next(
            (j for j, c in enumerate(centres) if c >= centres[0] * step), None
        )
    else:
        end, inner = -1, next(
            (
                j
                for j in range(len(centres) - 1, -1, -1)
                if centres[j] <= centres[-1] / step
            ),
            None,
        )
    return None if inner is None else round(moved[end] - moved[inner], 2)


def _span(moved: list[float], centres, largest_db, largest_at) -> dict:
    """How wide the deviation was around its largest, and whether it levelled off."""
    if largest_db is None:
        return dict.fromkeys(
            ("half_below_hz", "half_above_hz", "settled_below_db", "settled_above_db")
        )
    peak = list(centres).index(largest_at)
    half = abs(largest_db) / 2.0
    return {
        "half_below_hz": next(
            (centres[j] for j in range(peak, -1, -1) if abs(moved[j]) < half), None
        ),
        "half_above_hz": next(
            (centres[j] for j in range(peak, len(moved)) if abs(moved[j]) < half), None
        ),
        "settled_below_db": _settled(moved, centres, low=True),
        "settled_above_db": _settled(moved, centres, low=False),
    }


def _against(profile, reference, floor, centres) -> dict:
    """One profile as a deviation from the reference, with what cleared the floor."""
    moved = [round(a - b, 2) for a, b in zip(profile, reference, strict=True)]
    outside = [
        centre for centre, value, edge in zip(centres, moved, floor, strict=True)
        if abs(value) > edge
    ]
    largest = max(
        (
            (value, centre)
            for centre, value, edge in zip(centres, moved, floor, strict=True)
            if abs(value) > edge
        ),
        key=lambda pair: abs(pair[0]),
        default=(None, None),
    )
    return {
        "band_db": moved,
        "outside_the_floor_hz": outside,
        "largest_db": largest[0],
        "largest_at_hz": largest[1],
        **_span(moved, centres, largest[0], largest[1]),
    }


def _matched(pattern, listed: dict, files: list[str]) -> list[tuple[str, dict, object]]:
    out = []
    for name in files:
        entry = listed.get(name, {})
        source, found = takes.named_by(pattern, entry, name)
        if found is not None:
            out.append((name, entry, (source, found)))
    return out


def read_directory(
    where: str | Path,
    *,
    type_id: str,
    address: str,
    setting: str,
    reference: str,
    control: str | None = None,
    silence: str | None = None,
    stimulus: str | None = None,
    held: list[dict] | None = None,
    bands_hz=THIRD_OCTAVES,
    band_width_octaves: float = 1 / 3,
    channel: int | None = None,
    lead_s: float = 0.6,
    trim_s: float = 0.5,
    hold_s: float | None = None,
    progress=None,
) -> dict:
    """Every take under `where` whose setting matches, read into one record.

    Four patterns rather than one: the sweep, the repeats of the flat setting every
    profile is reported against, the takes made with the effect bypassed, and the
    takes made with nothing played at all. All four live in the same directory
    because they are the same session, and a reference fetched from another
    directory would be a reference from another evening.

    A take none of the four patterns names is counted rather than dropped: a
    pattern that matches nothing and a directory that holds nothing produce the
    same empty record otherwise, and they are different mistakes.
    """
    where = Path(where)
    listed, files = takes.listing(where)
    centres = list(bands_hz)

    swept = takes.capturing(setting, VALUE)
    # The reference and the control select takes rather than name a setting, so
    # neither has to capture anything -- there is no byte to pull out of a take
    # that holds the flat setting or none of the effect at all.
    flat = re.compile(reference)
    bypassed = re.compile(control) if control else None
    quiet = re.compile(silence) if silence else None

    flats = _matched(flat, listed, files)
    if not flats:
        raise ValueError(f"no take under {where} matched the reference {reference!r}")

    # The channel before anything else, and from the reference takes, which are the
    # ones the unit is certainly sounding in. Every other take is then read from the
    # same input rather than from whichever one was loudest in it.
    reached, reference_levels = takes.channel_reaching(
        where, [name for name, _, _ in flats]
    )
    used = reached if channel is None else int(channel)
    # The second highest of the reference takes, read alongside as a control. It is
    # named rather than assumed to be the unit's other output: what says whether it
    # carried the unit is that it follows the parameter, which is reported per
    # reading, and `reference_db` is beside it for a reader to judge.
    ranked = sorted(range(len(reference_levels)), key=lambda i: -reference_levels[i])
    beside = next((i for i in ranked if i != used), None)
    wanted = (used,) if beside is None else (used, beside)
    elsewhere: list[str] = []

    def profile(name: str, entry: dict):
        bands, loud, hold, own = _profile(
            where, name, entry, channels=wanted,
            lead_s=lead_s, trim_s=trim_s, hold_s=hold_s, centres=centres,
            width_octaves=band_width_octaves,
        )
        if own != used and name not in elsewhere:
            elsewhere.append(name)
        return bands, loud, hold

    def averaged(rows: list[list[float]]) -> list[float]:
        return [round(float(np.mean([row[i] for row in rows])), 3) for i in range(len(centres))]

    # The floor next, because the reference itself is a take and a reader has to
    # be able to see how far above the floor even that was.
    quiets: list[str] = []
    floor_bands: list[float] | None = None
    floor_heard: float | None = None
    if quiet is not None:
        heard = []
        found = []
        for name, entry, _ in _matched(quiet, listed, files):
            quiets.append(name)
            bands, loud, _hold = profile(name, entry)
            found.append(bands[used])
            heard.append(loud[used])
        if found:
            floor_bands = averaged(found)
            floor_heard = round(float(np.mean(heard)), 1)

    def above(heard: float) -> float | None:
        return None if floor_heard is None else round(heard - floor_heard, 1)

    flat_read = [profile(name, entry) for name, entry, _ in flats]
    rows = [bands[used] for bands, _, _ in flat_read]
    floor = [
        round(max(row[i] for row in rows) - min(row[i] for row in rows), 2)
        for i in range(len(centres))
    ]
    middle = averaged(rows)
    beside_middle = (
        averaged([bands[beside] for bands, _, _ in flat_read]) if beside is not None else None
    )
    flat_heard = round(float(np.mean([loud[used] for _, loud, _ in flat_read])), 1)


    def apart(bands: dict[int, list[float]]) -> dict:
        """How far the second channel's own deviation is from the first channel's.

        Each channel against its own repeats of the flat setting, so a standing
        difference in level between the two is not counted as a disagreement about
        frequency -- what is left is the shape, which is the thing one channel
        cannot report on its own.
        """
        if beside is None or beside_middle is None:
            return {}
        mine = [a - b for a, b in zip(bands[used], middle, strict=True)]
        theirs = [a - b for a, b in zip(bands[beside], beside_middle, strict=True)]
        gap = max(
            ((round(a - b, 2), centre) for a, b, centre in zip(mine, theirs, centres, strict=True)),
            key=lambda pair: abs(pair[0]),
        )
        return {
            "other_db": round(max(theirs, key=abs), 2),
            "apart_db": gap[0],
            "apart_at_hz": gap[1],
        }

    claimed = {name for name, _, _ in flats} | set(quiets)
    controls = []
    if bypassed is not None:
        for name, entry, _ in _matched(bypassed, listed, files):
            claimed.add(name)
            found, heard, _hold = profile(name, entry)
            controls.append(
                {
                    **_against(found[used], middle, floor, centres),
                    **apart(found),
                    "heard_db": heard[used],
                    "above_the_silence_db": above(heard[used]),
                    "take": name,
                }
            )

    readings: list[dict] = []
    for name, entry, (source, found) in _matched(swept, listed, files):
        if name in claimed:
            continue
        claimed.add(name)
        measured, heard, hold = profile(name, entry)
        reading = {
            VALUE: int(found.group(VALUE)),
            **_against(measured[used], middle, floor, centres),
            **apart(measured),
            "heard_db": heard[used],
            "above_the_silence_db": above(heard[used]),
            "hold_s": hold,
            "take": name,
            "named_by": source,
        }
        readings.append(reading)
        if progress:
            progress(reading)

    readings.sort(key=lambda r: r[VALUE])
    record = {
        "question": QUESTION,
        "type": type_id,
        "address": address,
        "method": METHOD,
        "limits": LIMITS,
        "not_in_this_record": NOT_HERE,
        "bands_hz": centres,
        "band_width_octaves": round(band_width_octaves, 6),
        "why_span": WHY_SPAN,
        "channel": {
            "read": used,
            "chosen_by": "given" if channel is not None else "loudest in the reference takes",
            "reference_db": reference_levels,
            "loudest_elsewhere": sorted(elsewhere),
            "why": WHY_CHANNEL,
        },
        "other_channel": {
            "read": beside,
            "band_db": beside_middle,
            "why": WHY_OTHER,
        },
        "silence": {
            "takes": sorted(quiets),
            "band_db": floor_bands,
            "heard_db": floor_heard,
            "why": WHY_SILENCE,
        },
        "reference": {
            "takes": sorted(name for name, _, _ in flats),
            "band_db": middle,
            "floor_db": floor,
            "heard_db": flat_heard,
            "above_the_silence_db": above(flat_heard),
            "why": WHY_REFERENCE,
        },
        "control": {
            "takes": [row["take"] for row in controls],
            "readings": controls,
            "why": WHY_CONTROL,
        },
        "held": held or [],
        "why_held": WHY_HELD,
        "takes_from": str(where),
        "manifest": takes.manifest_note(listed, files),
        "settings_asked": sorted({r[VALUE] for r in readings}),
        "readings": readings,
        "takes_not_matching": takes.not_matching(
            [str(listed.get(name, {}).get("setting") or name)
             for name in files if name not in claimed]
        ),
    }
    if stimulus:
        record["stimulus"] = stimulus
    return record
