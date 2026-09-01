"""Playing a note into the recorder, and finding which channel it landed on.

Both are the mechanics around a measurement rather than the measurement itself,
and both have been wrong in a way that produced numbers rather than an error: a
note played before the audio device was open reads as an absent note, and a
level read off the wrong input channel reads as a unit that answers nothing.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from . import capture as cap
from .capture import Recording
from .midi import MidiLink


def record_note(
    link: MidiLink,
    *,
    device: str | None,
    channel: int,
    note: int,
    velocity: int,
    hold: float,
    seconds: float,
    lead: float,
) -> Recording:
    """Record while one note is played, with silence before it to measure the floor."""
    ready = threading.Event()

    def play() -> None:
        # Wait for audio to be flowing, not merely for the recorder to have been
        # called. Opening the device outlasts any lead-in worth having.
        ready.wait(timeout=30.0)
        time.sleep(lead)
        link.send([0x90 | (channel & 0x0F), note & 0x7F, velocity & 0x7F])
        time.sleep(hold)
        link.send([0x80 | (channel & 0x0F), note & 0x7F, 0])

    thread = threading.Thread(target=play, daemon=True)
    thread.start()
    recording = cap.record(seconds, device=device, ready=ready)
    thread.join(timeout=hold + lead + 1.0)
    return recording


def loudest_channel(recording: Recording) -> int:
    """Which input channel the unit actually arrived on.

    By peak rather than by RMS: the question is which input the instrument is
    plugged into, and a peak answers it from the first transient. `takes.loudest`
    is the RMS form, which is the one to use when choosing between channels that
    both carry signal.
    """
    peaks = [
        float(np.abs(recording.samples[:, c]).max()) for c in range(recording.samples.shape[1])
    ]
    return int(np.argmax(peaks))
