"""How much of a stereo take is not one L/R direction, and how loud the take was.

Read from saved takes, with no machine attached, the same way the other efx
readers are: a run holds a stimulus through the chain at one setting of one byte,
saves the take, writes the next setting, and repeats; this reads what came back.

**A single L/R direction is anything a fixed pair of gains can make.** A mono
source panned any way at all, including inverted or hard to one side, puts all of
its power on one line through the origin of the L/R plane -- the covariance of
the two channels is rank one, and its smaller eigenvalue is the power left once
that line is subtracted out. A reverb tail, a chorus voice or a delay's early
copies that do not sit on that one line are what is left, and the smaller
eigenvalue is a straight reading of how much of the take that is.

**The line is found separately in every 50 ms frame, not once over the whole
take.** A modulator or a moving pan changes the best direction from frame to
frame, and reading one direction over the whole take would count that motion
itself as decorrelation. Read frame by frame, a pan that moves is still one
direction in every frame it is measured in, and only what a frame's own best
direction cannot explain is counted.

**Two figures, not one.** The incoherent share is a power and says nothing about
how loud the take was -- a byte that turns the whole output down by twenty
decibels turns its incoherent share down by the same twenty without becoming more
or less diffuse. The overall level is published beside it for that reason, and
neither is a deviation from a reference: both are read directly off the take, the
way a delay's time is.

**Nothing here says what the decorrelated power is.** A reverb, a chorus, a
comb of early reflections and the room the unit's own converter sits in all put
power off the fixed line, and none of that is told apart from any other here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import takes

VALUE = "value"
"""The named group a `--setting` pattern has to capture."""

FRAME_S = 0.05
"""The width of one frame a direction is fitted over.

Fifty milliseconds, which is short enough that an ordinary panning motion moves
little inside one of them and long enough to hold thousands of samples at any
rate this archive records at -- a covariance estimated over fewer than that is
mostly its own estimation noise, and the smaller eigenvalue is the one of the
two that noise inflates.
"""

QUESTION = (
    "How much of a stereo take is not one fixed L/R direction, and how loud the take "
    "was, at each setting of the byte."
)

METHOD = (
    "The held part of each take was cut into non-overlapping 50 ms frames on the two "
    "channels named as the unit's stereo pair. Each frame's 2x2 covariance of the two "
    "channels was decomposed into its two eigenvalues; the smaller of them -- the power "
    "left once the single strongest linear combination of the two channels for that "
    "frame is subtracted out -- was averaged across the frames as the incoherent share, "
    "and the mean of the two eigenvalues -- half the pair's combined power, which is its "
    "power per channel -- was averaged across the frames beside it as the take's overall "
    "level. The direction is found separately in every frame, so a pan that moves between "
    "frames is not itself counted as decorrelation."
)

LIMITS = (
    "`incoherent_db` and `level_db` are both powers averaged over the frames and read "
    "back as decibels, so each moves by the frame-to-frame scatter of the take and not "
    "by any one sample. `floor_incoherent_db` and `floor_level_db` are the spread of the "
    "run's own repeats of one setting, and are what a difference between settings has to "
    "clear; both are null where no setting was taken more than once. "
    "The reading has no notion of what put power off the fixed line: an inverted, "
    "filtered or hard-panned copy of the same source is still one direction in every "
    "frame and reads as fully coherent, whatever it sounds like, and a genuinely "
    "decorrelated return is not told apart from room noise or the converter's own "
    "channel separation. A return that moves slower than the frame width has part of "
    "its own motion absorbed into the frame's best direction instead of left as the "
    "remainder, so a slow return reads less diffuse than it is at the frame width used."
)

NOT_HERE = (
    "No claim about what the decorrelated power is -- a reverb tail, a chorus voice, a "
    "comb of early reflections or the room the unit's own converter sits in. This reads "
    "a share and a level, not a structure behind either of them."
)

WHY_CHANNEL = (
    "Which pair of channels every figure in this record was read from, and the highest "
    "each channel of the interface reached across the swept takes. An interface carries "
    "inputs the unit is not on, and those are not silent, so the pair is chosen once for "
    "the run and named rather than picked per take -- a take the byte turned down far "
    "enough would otherwise be read from a pair of idle inputs and its incoherent share "
    "would be the interface's own crosstalk."
)

WHY_FLOOR = (
    "How far apart the reading put one setting's own repeats, on each figure, which is "
    "what a difference between two settings has to clear to be a difference. Null where "
    "no setting was taken more than once, in which case this run measured no floor."
)

WHY_CONTROL = (
    "The same reading taken at two settings of a directory known to carry a real "
    "decorrelated return -- a positive control, so a reader can see this reading move on "
    "material it is known to have something to find, rather than trusting the method on "
    "the strength of the send bytes alone. `shows` states the two figures and the gap "
    "between them; without a control the record cannot say whether this reading would "
    "see a return at all."
)

WHY_SILENCE = (
    "The same reading taken from a setting known to carry nothing -- the chain's own "
    "floor, read the same way as every other setting rather than assumed. What a setting "
    "at the floor returns is the room and the converter's own channel separation, and "
    "without this a reading near it cannot be told from silence."
)

WHY_HELD = (
    "What else the run had written when it took these readings. The return this stage "
    "reads is downstream of every stage the type has, so a byte read with another of "
    "them moved is a reading of a different chain, and nothing in the numbers says so."
)

WHY_HELD_FROM = (
    "Where `held` above was read from, rather than carried in by hand from another "
    "record's held block."
)


def _eigen_pair(cov_ll: np.ndarray, cov_rr: np.ndarray, cov_lr: np.ndarray):
    """The two eigenvalues of a batch of symmetric 2x2 matrices, closed form.

    Vectorised over frames rather than decomposed one at a time: `[[a, b], [b, d]]`
    has eigenvalues `(a+d)/2 +/- sqrt(((a-d)/2)^2 + b^2)`, which is exact and does
    not touch `numpy.linalg` per frame -- the only thing that matters at the size
    of this archive's directories.
    """
    mean = (cov_ll + cov_rr) / 2.0
    half_diff = (cov_ll - cov_rr) / 2.0
    spread = np.sqrt(np.maximum(half_diff * half_diff + cov_lr * cov_lr, 0.0))
    return mean - spread, mean + spread


def measure(pair: np.ndarray, rate: int, *, frame_s: float = FRAME_S) -> dict:
    """The incoherent share and the overall level of one stereo take.

    `pair` is the two channels to read, already selected and already trimmed to
    the stretch the figures are read over. Returns nulls and a frame count of
    zero where the stretch is shorter than one frame, rather than a covariance
    estimated over too little to mean anything.
    """
    n = int(round(frame_s * rate))
    if n <= 0 or pair.shape[0] < n:
        return {"incoherent_db": None, "level_db": None, "frames": 0}
    usable = (pair.shape[0] // n) * n
    frames = np.asarray(pair[:usable], dtype=np.float64).reshape(-1, n, 2)
    demeaned = frames - frames.mean(axis=1, keepdims=True)
    left, right = demeaned[..., 0], demeaned[..., 1]
    cov_ll = np.mean(left * left, axis=1)
    cov_rr = np.mean(right * right, axis=1)
    cov_lr = np.mean(left * right, axis=1)
    small, large = _eigen_pair(cov_ll, cov_rr, cov_lr)
    incoherent = float(np.mean(small))
    overall = float(np.mean((small + large) / 2.0))
    return {
        "incoherent_db": round(10.0 * np.log10(max(incoherent, 1e-24)), 2),
        "level_db": round(10.0 * np.log10(max(overall, 1e-24)), 2),
        "frames": int(frames.shape[0]),
    }


def _body(
    samples: np.ndarray, rate: int, *, indices: tuple[int, int],
    lead_s: float, trim_s: float, hold_s: float | None,
) -> np.ndarray:
    pair = np.asarray(samples[:, list(indices)], dtype=np.float64)
    seconds = pair.shape[0] / rate
    hold = hold_s if hold_s is not None else seconds - lead_s - trim_s
    first = int((lead_s + trim_s) * rate)
    last = int((lead_s + hold - trim_s) * rate)
    return pair[first:last]


def _matched(pattern, listed: dict, files: list[str]) -> list[tuple[str, dict, object]]:
    out = []
    for name in files:
        entry = listed.get(name, {})
        source, found = takes.named_by(pattern, entry, name)
        if found is not None:
            out.append((name, entry, (source, found)))
    return out


def _readings_by_value(
    directory: Path, pattern, *, indices, lead_s, trim_s, hold_s, frame_s
) -> list[dict]:
    """Every take under `directory` matching `pattern`, averaged per captured value.

    For a control or a silence read from a directory of its own rather than from
    `where`: the takes there are somebody else's repeats of somebody else's
    settings, and the one thing worth doing with several of them is exactly what
    the main sweep already does with its own -- average within a setting, not
    across them.
    """
    listed, files = takes.listing(directory)
    grouped: dict[int, list[tuple[float, float]]] = {}
    names: dict[int, list[str]] = {}
    for name, _entry, (_source, found) in _matched(pattern, listed, files):
        samples, rate = takes.read(directory / name)
        body = _body(samples, rate, indices=indices, lead_s=lead_s, trim_s=trim_s, hold_s=hold_s)
        result = measure(body, rate, frame_s=frame_s)
        if result["incoherent_db"] is None:
            continue
        value = int(found.group(VALUE))
        grouped.setdefault(value, []).append((result["incoherent_db"], result["level_db"]))
        names.setdefault(value, []).append(name)
    return [
        {
            "value": value,
            "incoherent_db": round(float(np.mean([row[0] for row in rows])), 2),
            "level_db": round(float(np.mean([row[1] for row in rows])), 2),
            "takes": sorted(names[value]),
        }
        for value, rows in sorted(grouped.items())
    ]


def _control_shows(readings: list[dict]) -> str | None:
    """What a positive control's two readings say, in one sentence -- or nothing.

    Needs two settings to show a gap at all; a control read at one setting has
    nothing to be a control against.
    """
    if len(readings) < 2:
        return None
    lo = min(readings, key=lambda r: r["value"])
    hi = max(readings, key=lambda r: r["value"])
    delta = round(hi["incoherent_db"] - lo["incoherent_db"], 2)
    return (
        f"At {hi['value']} the incoherent share reads {hi['incoherent_db']:.2f} dB, "
        f"{delta:+.2f} dB over {lo['value']}'s {lo['incoherent_db']:.2f} dB, on a "
        "directory known to carry a real decorrelated return -- which is what this "
        "reading returns where one is knowingly present."
    )


def _silence_shows(readings: list[dict]) -> str | None:
    """What a silence reading says, in one sentence -- or nothing where there is none."""
    if not readings:
        return None
    row = readings[0]
    return (
        f"At {row['value']}, known to carry nothing, the take reads "
        f"{row['incoherent_db']:.2f} dB incoherent and {row['level_db']:.2f} dB overall, "
        "which is the floor a reading near either figure cannot be told from."
    )


def read_directory(
    where: str | Path,
    *,
    type_id: str,
    address: str | None = None,
    controller: int | None = None,
    setting: str,
    control_from: str | Path | None = None,
    control_setting: str | None = None,
    silence_from: str | Path | None = None,
    silence_setting: str | None = None,
    stimulus: str | None = None,
    held: list[dict] | None = None,
    held_from: str | None = None,
    held_from_shows: str | None = None,
    held_not_spelled_out: str | None = None,
    channels: tuple[int, int] | None = None,
    lead_s: float = 0.6,
    trim_s: float = 0.5,
    hold_s: float | None = None,
    frame_s: float = FRAME_S,
    progress=None,
) -> dict:
    """Every take under `where` whose setting matches, read into one record.

    Exactly one of `address` and `controller`: a run swept one thing, and a
    record naming both would not say which of them the figures belong to.

    A take that does not match `setting` is counted rather than dropped: a
    pattern that matches nothing and a directory that holds nothing produce the
    same empty record otherwise, and they are different mistakes.

    `control_from`/`control_setting` and `silence_from`/`silence_setting` each
    name a directory of their own and a `--setting`-shaped pattern over it,
    because the control and the silence this stage wants are not takes this
    directory holds -- they are another directory's own settings, read the same
    way `where`'s own sweep is. Either is optional and either is empty here
    where the other one is not given.
    """
    if (address is None) == (controller is None):
        raise ValueError("a record is about one address or one controller, not both or neither")
    where = Path(where)
    listed, files = takes.listing(where)
    swept = takes.capturing(setting, VALUE)

    matched = [
        (name, listed.get(name, {}), *takes.named_by(swept, listed.get(name, {}), name))
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
    by_setting: dict[int, list[tuple[float, float]]] = {}
    for name, _entry, named_by, found in wanted:
        samples, rate = takes.read(where / name)
        body = _body(samples, rate, indices=used, lead_s=lead_s, trim_s=trim_s, hold_s=hold_s)
        result = measure(body, rate, frame_s=frame_s)
        value = int(found.group(VALUE))
        if result["incoherent_db"] is not None:
            by_setting.setdefault(value, []).append(
                (result["incoherent_db"], result["level_db"])
            )
        reading = {VALUE: value, **result, "take": name, "named_by": named_by}
        readings.append(reading)
        if progress:
            progress(reading)

    readings.sort(key=lambda r: (r[VALUE], r["take"]))

    repeated = [got for got in by_setting.values() if len(got) > 1]
    spreads_i = [max(v[0] for v in got) - min(v[0] for v in got) for got in repeated]
    spreads_l = [max(v[1] for v in got) - min(v[1] for v in got) for got in repeated]
    floor_incoherent_db = round(float(max(spreads_i)), 2) if spreads_i else None
    floor_level_db = round(float(max(spreads_l)), 2) if spreads_l else None

    control_readings: list[dict] = []
    if control_from is not None:
        control_pattern = takes.capturing(control_setting, VALUE)
        control_readings = _readings_by_value(
            Path(control_from), control_pattern, indices=used,
            lead_s=lead_s, trim_s=trim_s, hold_s=hold_s, frame_s=frame_s,
        )
    control_block = {
        "from": str(control_from) if control_from is not None else None,
        "readings": control_readings,
        "shows": _control_shows(control_readings),
        "why": WHY_CONTROL,
    }

    silence_readings: list[dict] = []
    if silence_from is not None:
        silence_pattern = takes.capturing(silence_setting, VALUE)
        silence_readings = _readings_by_value(
            Path(silence_from), silence_pattern, indices=used,
            lead_s=lead_s, trim_s=trim_s, hold_s=hold_s, frame_s=frame_s,
        )
    silence_block = {
        "from": str(silence_from) if silence_from is not None else None,
        "readings": silence_readings,
        "shows": _silence_shows(silence_readings),
        "why": WHY_SILENCE,
    }

    record = {
        "question": QUESTION,
        "type": type_id,
        **({"address": address} if address is not None else {"controller": controller}),
        "method": METHOD,
        "limits": LIMITS,
        "not_in_this_record": NOT_HERE,
        "frame_s": frame_s,
        "channel": {
            "read": list(used),
            "chosen_by": "given" if channels is not None else "the pair that reached highest",
            "reached_db": levels,
            "why": WHY_CHANNEL,
        },
        "control": control_block,
        "silence": silence_block,
        "held": held or [],
        "why_held": WHY_HELD,
        **(
            {
                "held_from": held_from,
                "why_held_from": (
                    f"{WHY_HELD_FROM} {held_from_shows}" if held_from_shows else WHY_HELD_FROM
                ),
            }
            if held_from
            else {}
        ),
        **(
            {"held_not_spelled_out": held_not_spelled_out}
            if held_not_spelled_out
            else {}
        ),
        "takes_from": str(where),
        "manifest": takes.manifest_note(listed, files),
        "settings_asked": sorted({r[VALUE] for r in readings}),
        "readings": readings,
        "floor_incoherent_db": floor_incoherent_db,
        "floor_level_db": floor_level_db,
        "settings_taken_twice": sorted(v for v, got in by_setting.items() if len(got) > 1),
        "why_floor": WHY_FLOOR,
        "takes_not_matching": takes.not_matching(skipped),
    }
    if stimulus:
        record["stimulus"] = stimulus
    return record


__all__ = [
    "FRAME_S",
    "LIMITS",
    "METHOD",
    "NOT_HERE",
    "QUESTION",
    "VALUE",
    "measure",
    "read_directory",
]
