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
    "in `reference`. `heard_db` is the loudest "
    "channel's level over the body of the take, and `above_the_silence_db` is how far "
    "that is above what the same chain recorded with nothing played: a setting that "
    "turns the output down far enough returns the room and the converter, and the "
    "bands of that are the floor's own shape rather than anything the byte did. The "
    "floor's profile is in `silence` so the comparison can be made rather than taken "
    "on trust. Band energy is measured over the held part of the take only, so a "
    "setting whose effect is in the attack or the release is not in these figures at "
    "all. A band the stimulus does not reach cannot report what the effect did there, "
    "which is a limit of the stimulus and not a bound on the unit."
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
    "was never in the room."
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

WHY_HELD = (
    "What else the run had written when it took these readings. A band profile is the "
    "whole chain's, so a parameter read with another of the type's stages moved and "
    "one read with it at its centre are readings of different things, and nothing in "
    "the numbers says which is which."
)


def _body(samples, rate: int, *, lead_s: float, hold_s: float, trim_s: float):
    first = int((lead_s + trim_s) * rate)
    last = int((lead_s + hold_s - trim_s) * rate)
    return np.asarray(takes.loudest(samples)[first:last], dtype=np.float64)


def _loudness_db(samples) -> float:
    body = np.asarray(takes.loudest(samples), dtype=np.float64)
    return float(20.0 * np.log10(max(float(np.sqrt((body**2).mean())), 1e-9)))


def energies(body: np.ndarray, rate: int, centres=THIRD_OCTAVES) -> list[float]:
    """Energy per band, in dB, summed over the bins the band covers.

    Zero padded well past the take's own resolution so that the lowest band still
    has bins in it to sum: at a hundred hertz a third octave is twenty three hertz
    wide, which a transform of the take's own length resolves into a handful.
    """
    win = np.hanning(body.size)
    power = np.abs(np.fft.rfft(body * win, n=1 << 19)) ** 2
    freq = np.fft.rfftfreq(1 << 19, 1.0 / rate)
    out = []
    for centre in centres:
        lo, hi = centre / 2 ** (1 / 6), centre * 2 ** (1 / 6)
        inside = (freq >= lo) & (freq < hi)
        total = float(power[inside].sum()) if inside.any() else 0.0
        out.append(round(10.0 * np.log10(max(total, 1e-30)), 3))
    return out


def _profile(
    where: Path,
    name: str,
    entry: dict,
    *,
    lead_s: float,
    trim_s: float,
    hold_s: float | None,
    centres,
) -> tuple[list[float], float, float]:
    samples, rate = takes.read(where / name)
    seconds = float(entry.get("seconds") or samples.shape[0] / rate)
    hold = hold_s if hold_s is not None else seconds - 1.0
    body = _body(samples, rate, lead_s=lead_s, hold_s=hold, trim_s=trim_s)
    return energies(body, rate, centres), round(_loudness_db(samples), 1), round(hold, 3)


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

    def profile(name: str, entry: dict):
        return _profile(
            where, name, entry,
            lead_s=lead_s, trim_s=trim_s, hold_s=hold_s, centres=centres,
        )

    # The floor first, because the reference itself is a take and a reader has to
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
            found.append(bands)
            heard.append(loud)
        if found:
            floor_bands = [
                round(float(np.mean([row[i] for row in found])), 3)
                for i in range(len(centres))
            ]
            floor_heard = round(float(np.mean(heard)), 1)

    def above(heard: float) -> float | None:
        return None if floor_heard is None else round(heard - floor_heard, 1)

    flats = _matched(flat, listed, files)
    if not flats:
        raise ValueError(f"no take under {where} matched the reference {reference!r}")
    rows = [profile(name, entry)[0] for name, entry, _ in flats]
    floor = [
        round(max(row[i] for row in rows) - min(row[i] for row in rows), 2)
        for i in range(len(centres))
    ]
    middle = [round(float(np.mean([row[i] for row in rows])), 3) for i in range(len(centres))]

    flat_heard = round(
        float(np.mean([profile(name, entry)[1] for name, entry, _ in flats])), 1
    )

    claimed = {name for name, _, _ in flats} | set(quiets)
    controls = []
    if bypassed is not None:
        for name, entry, _ in _matched(bypassed, listed, files):
            claimed.add(name)
            found, heard, _hold = profile(name, entry)
            controls.append(
                {
                    **_against(found, middle, floor, centres),
                    "heard_db": heard,
                    "above_the_silence_db": above(heard),
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
            **_against(measured, middle, floor, centres),
            "heard_db": heard,
            "above_the_silence_db": above(heard),
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
