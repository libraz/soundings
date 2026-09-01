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
    """The channel carrying the most, which is what a mono measurement uses."""
    if samples.ndim == 1:
        return samples
    return samples[:, int(np.argmax(np.sqrt(np.mean(np.square(samples), axis=0))))]


@dataclass
class Store:
    """A directory of takes and the manifest that says what they are."""

    root: Path
    entries: list[dict]

    @classmethod
    def open(cls, root: str | Path) -> Store:
        return cls(root=Path(root), entries=[])

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


__all__ = ["Store", "loudest", "read", "write"]
