"""Audio capture, with the dropout it can suffer treated as a first-class result.

A capture path can lose samples without failing: the file is written, the levels
are right, and only the timeline is wrong. Amplitude summaries -- peak, RMS,
noise floor, channel separation -- are all blind to it, so nothing downstream
notices. On the machine this harness was written on, one widely used capture
route lost 11 percent of samples in exactly that way.

So every recording carries the evidence of its own health, and
`soundings.selftest` verifies the timeline against a known stimulus before any
measurement is trusted.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd


class CaptureError(RuntimeError):
    pass


@dataclass
class Recording:
    samples: np.ndarray  # (frames, channels) float32
    sample_rate: int
    device: str
    requested_seconds: float
    overflows: int
    """Times PortAudio reported it had dropped input. Non-zero invalidates the take."""

    started_at: float = field(default=0.0)

    open_seconds: float = field(default=0.0)
    """Stream open to first block. Whatever is triggered before this is never captured."""

    @property
    def frames(self) -> int:
        return self.samples.shape[0]

    @property
    def seconds(self) -> float:
        return self.frames / self.sample_rate

    @property
    def length_ratio(self) -> float:
        """Captured duration over requested duration. Well below 1.0 means loss."""
        return self.seconds / self.requested_seconds if self.requested_seconds else 0.0

    @property
    def healthy(self) -> bool:
        return self.overflows == 0 and 0.97 <= self.length_ratio <= 1.05

    def channel(self, index: int) -> np.ndarray:
        return self.samples[:, index]

    def peak_dbfs(self, index: int) -> float:
        peak = float(np.abs(self.samples[:, index]).max())
        return 20.0 * np.log10(peak) if peak > 0 else -np.inf

    def health_report(self) -> str:
        return (
            f"{self.seconds:.3f}s / {self.requested_seconds:.3f}s requested "
            f"(ratio {self.length_ratio:.4f}), overflows {self.overflows}, "
            f"opened in {self.open_seconds * 1000:.0f} ms, "
            f"{'OK' if self.healthy else 'UNHEALTHY'}"
        )


def list_devices() -> list[dict]:
    return [d for d in sd.query_devices() if d["max_input_channels"] > 0]


def resolve_device(needle: str | None) -> int:
    devices = sd.query_devices()
    candidates = [(i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    if not candidates:
        raise CaptureError("no audio input devices")
    if needle is None:
        return sd.default.device[0]
    hits = [i for i, d in candidates if needle.lower() in d["name"].lower()]
    if not hits:
        names = ", ".join(d["name"] for _, d in candidates)
        raise CaptureError(f"no input device matching {needle!r}; have: {names}")
    if len(hits) > 1:
        names = ", ".join(devices[i]["name"] for i in hits)
        raise CaptureError(f"{needle!r} matches several input devices: {names}")
    return hits[0]


def record(
    seconds: float,
    *,
    device: str | None = None,
    sample_rate: int | None = None,
    channels: int | None = None,
    blocksize: int = 2048,
    ready: threading.Event | None = None,
) -> Recording:
    """Record from one input device and return the samples with their health.

    This never raises on a lossy capture. A lossy capture is a measurement about
    the setup, and the caller has to be able to see it.

    **Opening the device takes time, and on this chain it takes most of a
    second.** Anything triggered on a timer started alongside the call to this
    function therefore happens before a single sample has been captured, and what
    comes back is the tail of a sound whose beginning was never recorded -- at
    the right length, with no overflow, and healthy by every check here. So the
    duration is counted from the first block rather than from the call, and
    `ready` is set at that same moment for a caller that has a stimulus to
    trigger. A stimulus anchored to anything else is anchored to the wrong clock.
    """
    index = resolve_device(device)
    info = sd.query_devices(index)
    sr = int(sample_rate or info["default_samplerate"])
    ch = int(channels or info["max_input_channels"])

    blocks: queue.Queue = queue.Queue()
    overflows = 0
    flowing = threading.Event()

    def callback(indata, frames, time_info, status):
        nonlocal overflows
        if status.input_overflow:
            overflows += 1
        blocks.put(indata.copy())
        if not flowing.is_set():
            flowing.set()
            if ready is not None:
                ready.set()

    started = time.monotonic()
    with sd.InputStream(
        device=index,
        samplerate=sr,
        channels=ch,
        dtype="float32",
        blocksize=blocksize,
        callback=callback,
    ):
        flowing.wait(timeout=max(5.0, seconds))
        opened = time.monotonic()
        time.sleep(seconds)

    if ready is not None:
        ready.set()  # so a waiting stimulus thread cannot hang on a stream that never ran
    chunks = []
    while not blocks.empty():
        chunks.append(blocks.get())
    samples = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, ch), dtype=np.float32)
    return Recording(
        samples=samples,
        sample_rate=sr,
        device=info["name"],
        requested_seconds=seconds,
        overflows=overflows,
        started_at=started,
        open_seconds=opened - started,
    )


def onsets(
    signal: np.ndarray,
    sample_rate: int,
    *,
    threshold: float = 0.15,
    min_gap: float = 0.3,
    smooth: float = 0.005,
) -> np.ndarray:
    """Onset times in seconds, from an amplitude envelope crossing a relative threshold.

    Deliberately crude: it exists to check a timeline against a stimulus whose
    spacing is already known, not to analyse music.
    """
    window = max(1, int(sample_rate * smooth))
    envelope = np.convolve(np.abs(signal), np.ones(window) / window, mode="same")
    peak = envelope.max()
    if peak <= 0:
        return np.array([])
    above = envelope > peak * threshold
    crossings = np.flatnonzero(np.diff(above.astype(np.int8)) == 1) / sample_rate
    if crossings.size == 0:
        return crossings
    kept = [crossings[0]]
    for t in crossings[1:]:
        if t - kept[-1] > min_gap:
            kept.append(t)
    return np.array(kept)


def dropout_runs(signal: np.ndarray, *, min_length: int = 8) -> int:
    """Count runs of exactly-zero samples, the signature of a dropped buffer."""
    zero = (signal == 0.0).view(np.int8)
    edges = np.diff(np.concatenate(([0], zero, [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return int(((ends - starts) >= min_length).sum())
