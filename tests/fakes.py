"""Stand-ins for the hardware, shared across tests that exercise a command end to end.

Kept in one place rather than redefined per test file: a link or a recorder that
drifts from what the real one does is a fixture that passes tests against a
contract nothing real has.
"""

from __future__ import annotations

import threading

import numpy as np

from soundings.capture import Recording


class FakePorts:
    output_name = "Fake MIDI Out"


class FakeLink:
    """A link that opens and closes, and says nothing about what it holds.

    Enough for the selftest gate itself; a run that goes on to read or write
    through it needs `Answering` as well.
    """

    ports = FakePorts()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False


class Answering:
    """A link whose reads answer with whatever the unit is standing in for holds."""

    def __init__(self, holds: dict[str, list[int]]):
        self.holds = holds
        self.sent: list[list[int]] = []

    def send(self, message) -> None:
        self.sent.append(list(message))

    def exchange(self, message):
        from soundings import roland

        # An RQ1's address is the three bytes after the command, which is where
        # the harness puts it; there is no parser for the request side because
        # nothing but a test ever reads one back.
        asked = " ".join(f"{b:02X}" for b in message[5:8])
        got = self.holds.get(asked)
        # An empty list, as the real link returns when nothing arrived before the
        # timeout; it never returns None, and a silence is a result rather than an
        # error there.
        return roland.dt1(asked, got, device_id=0x10) if got is not None else []


def fake_capture_record(
    seconds: float,
    *,
    device: str | None = None,
    sample_rate: int | None = None,
    channels: int | None = None,
    blocksize: int = 2048,
    ready: threading.Event | None = None,
) -> Recording:
    """Stand in for `capture.record`, with a canned take of the shape asked for.

    Quiet for the first third and a tone for the rest, on every channel, so a
    stimulus's lead-in reads quiet and its note reads well above it whatever the
    caller asked for. The quiet third carries a low tone rather than exact
    silence, since a lead-in of exact zeros divides by zero in the comparison
    against it and reads as no rise at all rather than a large one.

    `ready` is set immediately: `perform.record_notes`'s playing thread waits on
    it before sending anything, and a fake that never sets it hangs for 30 s on
    every take.
    """
    if ready is not None:
        ready.set()
    rate = int(sample_rate or 48000)
    ch = int(channels or 2)
    frames = max(1, int(round(seconds * rate)))
    quiet_frames = frames // 3
    t = np.arange(frames) / rate
    tone = 1e-4 * np.sin(2 * np.pi * 440.0 * t)
    tone[quiet_frames:] += 0.5 * np.sin(2 * np.pi * 440.0 * t[quiet_frames:])
    samples = np.tile(tone.astype(np.float32)[:, None], (1, ch))
    return Recording(
        samples=samples,
        sample_rate=rate,
        device="fake",
        requested_seconds=seconds,
        overflows=0,
        open_seconds=0.01,
    )
