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

import re
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
    "channels was decomposed and its smaller eigenvalue -- the power left once the "
    "single strongest linear combination of the two channels for that frame is "
    "subtracted out -- was averaged across the frames, with the larger eigenvalue beside "
    "it as the take's overall level. The direction is found separately in every frame, "
    "so a pan that moves between frames is not itself counted as decorrelation."
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
    "The same reading taken from the takes made with the part routed past the effect, "
    "where the run made any. Without it the record cannot say whether the byte's own "
    "lowest setting already carries decorrelated return from somewhere else in the "
    "chain, or none at all."
)

WHY_SILENCE = (
    "The same reading taken from takes made with the chain and nothing played, where the "
    "run made any. What a setting at the floor returns is the room and the converter's "
    "own channel separation read as an incoherent share, and without this a reading near "
    "that floor cannot be told from silence."
)

WHY_HELD = (
    "What else the run had written when it took these readings. The return this stage "
    "reads is downstream of every stage the type has, so a byte read with another of "
    "them moved is a reading of a different chain, and nothing in the numbers says so."
)

WHY_HELD_FROM = (
    "Where `held` above was read from, rather than carried in by hand from another "
    "record's held block. The capture this stage reads was itself made by a `contrast` "
    "run, and a `contrast` run applies a GS Reset before it writes its own `--prepare` "
    "list -- once, at the start of the run, not between the settings it then swept -- so "
    "`held` is exactly that list, applied to a freshly reset unit and left standing for "
    "every setting read here."
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


def _averaged(where: Path, names: list[str], *, indices, lead_s, trim_s, hold_s, frame_s) -> dict:
    incoherent, level = [], []
    for name in names:
        samples, rate = takes.read(where / name)
        body = _body(samples, rate, indices=indices, lead_s=lead_s, trim_s=trim_s, hold_s=hold_s)
        found = measure(body, rate, frame_s=frame_s)
        if found["incoherent_db"] is not None:
            incoherent.append(found["incoherent_db"])
            level.append(found["level_db"])
    return {
        "takes": sorted(names),
        "incoherent_db": round(float(np.mean(incoherent)), 2) if incoherent else None,
        "level_db": round(float(np.mean(level)), 2) if level else None,
    }


def read_directory(
    where: str | Path,
    *,
    type_id: str,
    address: str | None = None,
    controller: int | None = None,
    setting: str,
    control: str | None = None,
    silence: str | None = None,
    stimulus: str | None = None,
    held: list[dict] | None = None,
    held_from: str | None = None,
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

    A take none of the three patterns names is counted rather than dropped: a
    pattern that matches nothing and a directory that holds nothing produce the
    same empty record otherwise, and they are different mistakes.
    """
    if (address is None) == (controller is None):
        raise ValueError("a record is about one address or one controller, not both or neither")
    where = Path(where)
    listed, files = takes.listing(where)
    swept = takes.capturing(setting, VALUE)
    bypassed = re.compile(control) if control else None
    quiet = re.compile(silence) if silence else None

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

    claimed = {name for name, _, _, found in matched if found}
    control_names = [name for name, _, _ in _matched(bypassed, listed, files)] if bypassed else []
    silence_names = [name for name, _, _ in _matched(quiet, listed, files)] if quiet else []
    claimed |= set(control_names) | set(silence_names)
    control_block = {
        **_averaged(
            where, control_names, indices=used, lead_s=lead_s, trim_s=trim_s,
            hold_s=hold_s, frame_s=frame_s,
        ),
        "why": WHY_CONTROL,
    }
    silence_block = {
        **_averaged(
            where, silence_names, indices=used, lead_s=lead_s, trim_s=trim_s,
            hold_s=hold_s, frame_s=frame_s,
        ),
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
        **({"held_from": held_from, "why_held_from": WHY_HELD_FROM} if held_from else {}),
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
