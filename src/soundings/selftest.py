"""The gate that runs before any measurement is trusted.

Both failure modes this guards against are silent. A MIDI path that drops bytes
produces a missing row rather than an error, so an address space comes out with
holes in it that read as absences. A capture path that drops samples produces a
recording whose levels are all correct and whose timeline is compressed, so
every decay time comes out short by the same factor and nothing internal to the
data disagrees.

Neither is detectable after the fact from the measurements themselves. They have
to be excluded beforehand, against a stimulus whose answer is already known.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

import numpy as np

from . import capture as cap
from . import perform, roland
from .midi import MidiLink


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append(Check(name, passed, detail))

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def __str__(self) -> str:
        lines = [f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}" for c in self.checks]
        lines.append(f"  => {'ALL PASS' if self.passed else 'FAILED'}")
        return "\n".join(lines)


def midi_selftest(
    link: MidiLink,
    *,
    probe_address: str = "40 01 30",
    probe_size: int = 16,
    repeats: int = 100,
    device_id: int = roland.DEFAULT_DEVICE_ID,
    model_id: int = roland.GS_MODEL_ID,
) -> Report:
    """Prove the MIDI round trip before believing anything it returns."""
    report = Report()

    identity = link.exchange(roland.IDENTITY_REQUEST, timeout=1.0)
    report.add(
        "identity reply",
        len(identity) > 0,
        " ".join(f"{b:02X}" for b in identity) if identity else "no reply",
    )
    if not identity:
        return report

    request = roland.rq1(probe_address, probe_size, device_id=device_id, model_id=model_id)
    reference = roland.parse_dt1(link.exchange(request, timeout=1.0))
    report.add(
        "probe address readable",
        reference is not None,
        f"{probe_address} returned {reference.size} bytes" if reference else "no DT1 reply",
    )
    if reference is None:
        return report

    report.add(
        "reply checksum",
        reference.checksum_ok,
        "verified" if reference.checksum_ok else "MISMATCH -- the path corrupted the reply",
    )

    mismatched = 0
    empty = 0
    latencies: list[float] = []
    for _ in range(repeats):
        t0 = time.monotonic()
        reply = roland.parse_dt1(link.exchange(request, timeout=1.0))
        latencies.append(time.monotonic() - t0)
        if reply is None:
            empty += 1
        elif reply.data != reference.data or not reply.checksum_ok:
            mismatched += 1

    report.add(
        f"{repeats} identical reads",
        mismatched == 0 and empty == 0,
        f"{mismatched} mismatched, {empty} unanswered",
    )
    median_ms = statistics.median(latencies) * 1000
    lo_ms, hi_ms = min(latencies) * 1000, max(latencies) * 1000
    report.add(
        "round trip",
        True,
        f"median {median_ms:.1f} ms, range {lo_ms:.1f}-{hi_ms:.1f} ms",
    )
    return report


def timeline_selftest(
    link: MidiLink,
    *,
    device: str | None = None,
    ticks: int = 10,
    interval: float = 1.0,
    note: int = 84,
    program: int = 56,
    channel_index: int | None = None,
    tolerance: float = 0.005,
) -> Report:
    """Play a note at a known interval and check the recording agrees.

    File duration alone cannot separate sample loss from start-up latency -- one
    is proportional to the take, the other is constant. Onset spacing separates
    them, which is why the stimulus is a metronome rather than a single tone.
    """
    report = Report()
    seconds = interval * (ticks + 1) + 3.0

    import threading

    def play() -> None:
        time.sleep(1.5)
        link.send([0xC0, program])
        for cc, value in ((7, 127), (11, 127), (91, 0), (93, 0)):
            link.send([0xB0, cc, value])
        time.sleep(0.5)
        start = time.monotonic()
        for i in range(ticks):
            target = start + i * interval
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            link.send([0x90, note, 127])
            time.sleep(0.08)
            link.send([0x80, note, 0])

    thread = threading.Thread(target=play, daemon=True)
    thread.start()
    rec = cap.record(seconds, device=device)
    thread.join(timeout=1.0)

    # Named before anything is judged, because a run with no --audio records
    # from whatever the system calls its default input, and a laptop microphone
    # hears none of this. Then every check below fails for a reason that is not
    # about the unit, and the report as it stood said only "0 of 10 onsets" --
    # which reads as a silent machine and was believed as one.
    report.add("recorded from", True, rec.device)
    report.add("capture reported no overflow", rec.overflows == 0, f"{rec.overflows} overflows")
    report.add(
        "captured length",
        0.97 <= rec.length_ratio <= 1.05,
        f"{rec.seconds:.3f}s / {rec.requested_seconds:.3f}s (ratio {rec.length_ratio:.4f})",
    )

    if rec.frames == 0:
        report.add("onsets", False, "nothing captured")
        return report

    if channel_index is None:
        channel_index = perform.loudest_channel(rec)

    signal = rec.channel(channel_index)
    found = cap.onsets(signal, rec.sample_rate, min_gap=interval * 0.3)
    report.add(
        "onsets found",
        len(found) == ticks,
        f"{len(found)} of {ticks} on channel {channel_index}",
    )

    if len(found) < 2:
        return report

    gaps = np.diff(found)
    mean = float(gaps.mean())
    report.add(
        "onset spacing",
        abs(mean - interval) < tolerance,
        f"mean {mean:.5f}s vs {interval:.5f}s sent, "
        f"sd {gaps.std() * 1000:.2f} ms, worst {abs(gaps - interval).max() * 1000:.2f} ms",
    )
    report.add(
        "no zero-run dropouts",
        cap.dropout_runs(signal) == 0,
        f"{cap.dropout_runs(signal)} runs of >=8 zero samples",
    )
    report.add("peak level", True, f"{rec.peak_dbfs(channel_index):.1f} dBFS")
    return report
