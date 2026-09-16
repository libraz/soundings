"""Keeping the audio a measurement was made from, rather than only its verdict.

A run costs device time and produces two things: a number, and the takes the
number came out of. Only the first was ever kept, and that makes every later
question about the same sound another hour with the machine switched on -- which
is the scarce thing here. Worse, it makes a method change unaffordable: the
chorus and reverb verdicts had to be re-recorded rather than recomputed when the
statistic behind them changed, for want of the audio they were computed from.

So a run can put its takes on disk, and every analysis in this package can be run
again from them with the machine unplugged. What is stored is the whole
recording, all channels: the loudest one is what the verdict used, and a stereo
effect is invisible in it.

**The audio is not the archive.** `data/units/` is text, published, and small;
these are hundreds of megabytes of WAV and stay out of it. What travels with them
is a manifest holding the same method fields the JSON result carries, so a
directory of takes says what it is without the run that made it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _wav(path: str | Path) -> Path:
    path = Path(path)
    return path if path.suffix == ".wav" else path.with_suffix(".wav")


def write(path: str | Path, samples: np.ndarray, sample_rate: int) -> Path:
    """Write float32 samples, keeping every channel and the full resolution.

    Float rather than 16 or 24 bit integer: the takes are compared against a
    noise floor a few dB above the converter's own, and requantising them adds a
    floor of its own right where the measurement lives.
    """
    from scipy.io import wavfile

    path = _wav(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sample_rate, np.asarray(samples, dtype=np.float32))
    return path


def read(path: str | Path) -> tuple[np.ndarray, int]:
    """Return (frames, channels) as float64, whatever the file was stored as."""
    from scipy.io import wavfile

    rate, samples = wavfile.read(str(path))
    samples = np.asarray(samples)
    if samples.ndim == 1:
        samples = samples[:, None]
    if np.issubdtype(samples.dtype, np.integer):
        samples = samples.astype(np.float64) / float(np.iinfo(samples.dtype).max)
    return samples.astype(np.float64), int(rate)


def loudest(samples: np.ndarray) -> np.ndarray:
    """The channel carrying the most, which is what a mono measurement uses.

    **Per take, which is not what a run wants.** An interface has inputs the unit
    is not plugged into and they are not silent, so a take whose output falls
    below one of them is answered from that input instead, with nothing in the
    figures to say the reading changed channel. A run that sweeps a parameter
    which can turn the output down picks one channel for all its takes instead --
    `channel_reaching` is what picks it.
    """
    if samples.ndim == 1:
        return samples
    return samples[:, int(np.argmax(np.sqrt(np.mean(np.square(samples), axis=0))))]


def channel(samples: np.ndarray, index: int) -> np.ndarray:
    """One named channel, whatever shape the take was stored in."""
    frames = np.asarray(samples, dtype=np.float64)
    if frames.ndim == 1:
        return frames
    return frames[:, min(index, frames.shape[1] - 1)]


def channel_levels(samples: np.ndarray) -> list[float]:
    """Every channel's level over the take, in dBFS."""
    frames = np.asarray(samples, dtype=np.float64)
    if frames.ndim == 1:
        frames = frames[:, None]
    rms = np.sqrt(np.mean(np.square(frames), axis=0))
    return [round(float(20.0 * np.log10(max(float(v), 1e-12))), 1) for v in rms]


def channel_across(*samples: np.ndarray) -> tuple[int, list[float]]:
    """One channel for takes that are going to be compared, and what each reached.

    **Two takes compared must be read from one input.** A measurement that
    subtracts one take from another, or tracks one against the other, is a
    measurement of the difference between them -- and an interface carries inputs
    the unit is not on which are not silent, so a take whose output falls below one
    of them is answered from that input instead. Each take choosing its own loudest
    channel then subtracts one input from a different one, and what comes back is
    not a residual of anything.

    The highest each channel reached across all of them, so a take the effect made
    quiet does not move the choice: that is the direction that loses the unit to an
    idle input. `channel_reaching` is the same choice made from files on disk,
    where the takes are too many to hold at once.
    """
    highest: list[float] = []
    for frames in samples:
        levels = channel_levels(frames)
        highest = (
            levels if not highest else [max(a, b) for a, b in zip(highest, levels, strict=True)]
        )
    if not highest:
        raise ValueError("no take to choose a channel from")
    return int(np.argmax(highest)), highest


def channel_reaching(root: str | Path, names, *, seconds: float = 2.0) -> tuple[int, list[float]]:
    """Which channel the unit is on, and the highest each channel reached.

    **The highest a channel ever reached, not its average.** A run sweeps a
    parameter that can turn the output off, and a channel is not the wrong one
    for having been quiet in the takes where the effect was doing its job -- an
    average over the whole sweep is dragged toward the settings that silenced it,
    which is the direction that loses the unit to an idle input.

    Read from a slice in the middle of each take, with the file memory mapped:
    the question is which input carried the unit, and that does not need the whole
    take or the whole directory in memory.
    """
    from scipy.io import wavfile

    root = Path(root)
    highest: list[float] = []
    for name in names:
        rate, data = wavfile.read(str(root / name), mmap=True)
        if data.ndim == 1:
            data = data[:, None]
        middle = data.shape[0] // 2
        half = int(seconds * rate / 2)
        body = np.asarray(data[max(0, middle - half) : middle + half], dtype=np.float64)
        if np.issubdtype(data.dtype, np.integer):
            body = body / float(np.iinfo(data.dtype).max)
        levels = channel_levels(body)
        highest = levels if not highest else [max(a, b) for a, b in zip(highest, levels)]
    if not highest:
        raise ValueError(f"no take under {root} to choose a channel from")
    return int(np.argmax(highest)), highest


def channel_pair_reaching(
    root: str | Path, names, *, seconds: float = 2.0
) -> tuple[tuple[int, int], list[float]]:
    """Which pair of channels the unit is on, and the highest each channel reached.

    The loudest channel and the one it is paired with, rather than the two loudest.
    An interface presents its inputs in stereo pairs, and a measurement that reads
    a difference between channels has to read it across a pair the unit's two
    outputs are on -- not across whichever two happened to be loudest, which on a
    parameter that empties one side is a different pair at each setting.
    """
    picked, highest = channel_reaching(root, names, seconds=seconds)
    first = picked - (picked % 2)
    return (first, min(first + 1, len(highest) - 1)), highest


WHY_CHANNEL = (
    "Which channel of the interface both takes were read from, and the highest each channel "
    "reached across the two. One channel for the pair rather than the loudest of each take: this "
    "measurement subtracts one take from the other, and an interface carries inputs the unit is "
    "not on which are not silent, so a take whose output fell below one of them would be answered "
    "from that input and subtracted from a different one -- leaving a residual of nothing that "
    "reads as an effect doing nothing."
)


SECOND_CHANNEL_ABOVE_DB = 20.0
"""How far over the noise the second channel has to sit to be a channel at all.

Below this the pair is one channel and a floor, and anything computed from it is
the floor's own wander. The inputs this ran on sit near -94 dB unloaded against a
unit arriving at -48, so the separation is not a fine judgement.
"""


WHY_ACROSS_THE_SETTINGS = (
    "A channel the unit arrived on is one that carried signal at some point in the run, so "
    "each channel is taken at the loudest it ever reached. Asking a single take instead "
    "cannot see a parameter that moves signal from one channel to the other: taken to its "
    "two ends a panpot leaves every take with one live channel and one empty one, so "
    "whichever take is asked answers mono -- and the one parameter this measurement exists "
    "for is the one it would refuse."
)


WHY_THE_FLOOR_IS_THE_PAIRS_OWN = (
    "The floor the quieter channel has to clear is read from the lead of the channels the unit "
    "arrived on, and not from the lead of every input the interface has. An input the unit is "
    "not on is not silent: measured here, two unused inputs sat at -70 and -75 dBFS through "
    "every take while the two carrying the unit sat at -110, so a floor taken over all of them "
    "was thirty decibels above the one the reading is against. A run whose quieter channel "
    "reached -59.7 was refused as a mono source by three tenths of a decibel on that account -- "
    "which is not a bound being reported, it is a statement that the unit arrived on one "
    "channel, and it was false."
)


def level_db(x: float) -> float:
    """A linear amplitude as decibels, with a floor that keeps silence finite."""
    return float(20.0 * np.log10(x + 1e-15))


def rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(frame * frame)))


def spread(values: list[float]) -> float:
    """How far a set of readings of one thing moved, which is nothing when there is one."""
    return float(max(values) - min(values)) if len(values) > 1 else 0.0


def pair_of_channels(
    frames_of_every_take: list[np.ndarray], lead: np.ndarray | None
) -> tuple[tuple[int, int] | None, float]:
    """The two channels the unit arrived on, in the interface's own order, and their floor.

    Each channel is taken at the loudest it reached across every take of every
    setting, per `WHY_ACROSS_THE_SETTINGS`. The pair is picked on level alone,
    which needs no floor, and the floor is then read from those two channels' own
    lead, per `WHY_THE_FLOOR_IS_THE_PAIRS_OWN`. None when the quieter of the two
    does not clear it.

    The pair comes back sorted by input number rather than by level. Which of the
    two is subtracted from the other decides the sign of every figure read out of
    them, and ordering by level puts that sign in the hands of whichever channel
    was louder by a hair.
    """
    if not frames_of_every_take:
        return None, -120.0
    width = min(frames.shape[1] for frames in frames_of_every_take)
    levels = [
        max(level_db(rms(frames[:, c])) for frames in frames_of_every_take) for c in range(width)
    ]
    order = sorted(range(len(levels)), key=lambda c: levels[c], reverse=True)
    if len(order) < 2:
        return None, -120.0
    pair = sorted(order[:2])
    floor = level_db(rms(lead[:, pair])) if lead is not None and lead.shape[0] else -120.0
    if levels[order[1]] < floor + SECOND_CHANNEL_ABOVE_DB:
        return None, floor
    return (pair[0], pair[1]), floor


def grouped(
    root: str | Path,
) -> tuple[dict[str, float], dict[str, dict[str, list[Path]]], dict[str, list[str]]]:
    """A run's takes, by stimulus and then by setting, in the order it asked them.

    The order matters and a dictionary's insertion order is the only thing holding
    it: a sweep's settings are a table's index, and a record that lists them
    sorted as strings puts 8 after 120.
    """
    root = Path(root)
    manifest = json.loads((root / "takes-manifest.json").read_text())
    leads = {s.get("name"): s.get("lead_s") or 0.6 for s in manifest.get("stimuli", [])}
    by_stimulus: dict[str, dict[str, list[Path]]] = {}
    order: dict[str, list[str]] = {}
    for entry in manifest.get("takes", []):
        setting = str(entry.get("setting"))
        stimulus = str(entry.get("stimulus"))
        by_stimulus.setdefault(stimulus, {}).setdefault(setting, []).append(root / entry["file"])
        seen = order.setdefault(stimulus, [])
        if setting not in seen:
            seen.append(setting)
    return leads, by_stimulus, order


def read_pair(dry_path, wet_path, *, on: int | None = None):
    """Load two takes, read both from one channel, and say which and what each reached.

    `on` names the channel instead of choosing one, which is what a second pair
    read as a control over the first needs: a floor measured on the other leg of
    the unit bounds a comparison that was never made.

    Here rather than beside the commands that read pairs, because a publisher
    building the same records without the command line has to make the same choice
    and a second implementation of it is a second thing to get wrong.
    """
    dry, dry_rate = read(dry_path)
    wet, wet_rate = read(wet_path)
    if dry_rate != wet_rate:
        raise SystemExit(f"the two takes were captured at {dry_rate} and {wet_rate} Hz")
    chosen, reached = channel_across(dry, wet)
    picked = chosen if on is None else on
    said = {"read": picked, "reached_db": reached, "why": WHY_CHANNEL}
    return channel(dry, picked), channel(wet, picked), dry_rate, said


@dataclass
class Store:
    """A directory of takes and the manifest that says what they are."""

    root: Path
    entries: list[dict]

    @classmethod
    def open(cls, root: str | Path) -> Store:
        return cls(root=Path(root), entries=[])

    @classmethod
    def reopen(cls, root: str | Path) -> Store:
        """A store that will write its manifest back with what is already in it.

        `close` replaces the manifest, so a second run that adds a few takes to a
        finished directory leaves a manifest naming those alone. The readers here
        survive that -- the files are the subject and what the manifest forgot is
        reported -- but what it forgets is every earlier take's sample rate,
        channel count and overflow count, and that is not recoverable from a name.
        A run that means to append says so, and keeps them.
        """
        manifest = Path(root) / "takes-manifest.json"
        kept = json.loads(manifest.read_text()) if manifest.exists() else {"takes": []}
        return cls(root=Path(root), entries=list(kept.get("takes", ())))

    def keep(self, recording, *, stimulus: str, setting: str, take: int, **extra) -> Path:
        name = f"{_safe(stimulus)}-{_safe(setting)}-{take:02d}"
        path = write(self.root / name, recording.samples, recording.sample_rate)
        self.entries.append(
            {
                "file": path.name,
                "stimulus": stimulus,
                "setting": setting,
                "take": take,
                "sample_rate": recording.sample_rate,
                "channels": int(recording.samples.shape[1]),
                "seconds": round(recording.seconds, 4),
                "device": recording.device,
                "overflows": recording.overflows,
                "opened_ms": round(recording.open_seconds * 1000.0, 1),
                **extra,
            }
        )
        return path

    def close(self, **method) -> Path:
        path = self.root / "takes-manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**method, "takes": self.entries}, indent=2) + "\n")
        return path


def _safe(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(text))


def listing(where: str | Path) -> tuple[dict, list[str]]:
    """The files under `where`, and the manifest beside them as a lookup.

    **The files are the subject.** A store rewrites its manifest when it closes,
    so a run repeated for one setting leaves a manifest naming that setting alone
    while every earlier take is still on disk. Read from the manifest, such a
    directory reports one reading and looks complete; read from the files, it
    reports all of them and says how many the manifest had forgotten.
    """
    where = Path(where)
    manifest = where / "takes-manifest.json"
    kept = json.loads(manifest.read_text()) if manifest.exists() else {"takes": []}
    listed = {entry["file"]: entry for entry in kept.get("takes", ())}
    files = sorted(path.name for path in where.glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"no takes under {where}")
    return listed, files


def named_by(pattern, entry: dict, name: str):
    """Which of the two names a take has the pattern answered to, and the match.

    The setting the manifest recorded, and the file's own name, which is what the
    store wrote that setting into. Both are offered and which one answered is
    reported per take: a manifest is rewritten when its store closes, so the name
    is sometimes the only copy, and a pattern written against one should not
    silently find nothing under the other.
    """
    for source, against in (("manifest", entry.get("setting")), ("file name", name)):
        if not against:
            continue
        if (found := pattern.search(str(against))) is not None:
            return source, found
    return None, None


def manifest_note(listed: dict, files: list[str]) -> dict:
    return {
        "lists": len(listed),
        "files_present": len(files),
        "not_listed": sorted(set(files) - set(listed)),
        "why": "A take store rewrites its manifest when it closes, so a run repeated "
        "for one setting leaves a manifest naming that setting alone. The takes are "
        "still on disk and are read here; which of them the manifest had forgotten is "
        "named, because a record silently built from a shortened manifest reads exactly "
        "like a complete one.",
    }


def not_matching(skipped: list[str]) -> dict:
    return {
        "count": len(skipped),
        "settings": sorted(set(skipped))[:40],
        "why": "Takes under the same directory whose setting the pattern did not name. "
        "Counted rather than dropped: a pattern that matches nothing and a directory "
        "that holds nothing leave the same empty record otherwise.",
    }


def capturing(setting: str, group: str) -> re.Pattern:
    pattern = re.compile(setting)
    if group not in (pattern.groupindex or {}):
        raise ValueError(
            f"the --setting pattern must capture a group named {group!r}; "
            f"{setting!r} captures {sorted(pattern.groupindex)}"
        )
    return pattern


__all__ = [
    "Store",
    "WHY_CHANNEL",
    "capturing",
    "channel",
    "channel_across",
    "channel_levels",
    "channel_reaching",
    "listing",
    "loudest",
    "manifest_note",
    "named_by",
    "not_matching",
    "read",
    "read_pair",
    "write",
]
