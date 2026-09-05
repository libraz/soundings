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


def record_notes(
    link: MidiLink,
    *,
    device: str | None,
    channel: int,
    notes,
    seconds: float,
    lead: float,
) -> Recording:
    """Record while a set of notes is played, with silence before them for the floor.

    Each note is `(note, velocity, at, hold)`, with `at` measured from the first
    note-on rather than from the start of the take, so a stimulus reads as what
    the player does and not as what the recorder does.

    **More than one note is the only way to ask a parameter about polyphony.**
    Whether a part is monophonic, and what it does when a voice is asked for
    twice, are both invisible to a single note: it sounds the same either way, so
    the address answers inaudible and the null is a fact about the stimulus.

    The note-offs are sorted in with the note-ons rather than sent per note,
    since two notes that overlap have their events interleaved and sending each
    note's pair in turn would hold the first until the second was over.
    """
    ready = threading.Event()
    status = channel & 0x0F
    events: list[tuple[float, list[int]]] = []
    for note, velocity, at, hold in notes:
        events.append((at, [0x90 | status, note & 0x7F, velocity & 0x7F]))
        events.append((at + hold, [0x80 | status, note & 0x7F, 0]))
    events.sort(key=lambda e: e[0])
    last = events[-1][0] if events else 0.0

    def play() -> None:
        # Wait for audio to be flowing, not merely for the recorder to have been
        # called. Opening the device outlasts any lead-in worth having.
        ready.wait(timeout=30.0)
        time.sleep(lead)
        started = time.monotonic()
        for at, message in events:
            # Against the start rather than the previous event: sleeping the gap
            # each time accumulates every send's own latency into the last note's
            # position, which on a two-note stimulus is the thing being measured.
            late = at - (time.monotonic() - started)
            if late > 0:
                time.sleep(late)
            link.send(message)

    thread = threading.Thread(target=play, daemon=True)
    thread.start()
    recording = cap.record(seconds, device=device, ready=ready)
    thread.join(timeout=last + lead + 1.0)
    return recording


OPEN_ATTEMPTS = 3
"""Times opening the audio device is retried before a take is given up on.

Not a hidden failure: the device on this chain refuses to start a stream every so
often with an unspecified hardware error, and it has cost three whole runs, each
after the machine time the run had already spent. A retry is a fresh stream
recording the same note, so nothing about the measurement changes -- and how many
were needed is kept, because a rate that climbs is a fact about the chain.
"""

OPEN_REST_S = 1.5


def record_notes_retrying(link: MidiLink, **asked) -> tuple[Recording, int]:
    """Record, retrying a device that refused to open, and say how many it took."""
    last_error: Exception | None = None
    for attempt in range(OPEN_ATTEMPTS):
        try:
            return record_notes(link, **asked), attempt
        except Exception as exc:  # sounddevice raises its own error type
            if "PortAudio" not in type(exc).__name__ and "PortAudio" not in str(exc):
                raise
            last_error = exc
            time.sleep(OPEN_REST_S)
    raise RuntimeError(
        f"the audio device refused to open {OPEN_ATTEMPTS} times in a row: {last_error}"
    )


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
    return record_notes(
        link,
        device=device,
        channel=channel,
        notes=((note, velocity, 0.0, hold),),
        seconds=seconds,
        lead=lead,
    )


RETAKES = 3
"""Attempts a take gets at finding a quiet lead-in before the run measures anyway.

Bounded rather than open ended, because a setting that never goes quiet is a
finding about the setting and the lead-in check on the takes still refuses to
measure through it.
"""

REST_S = 1.2
"""Waited between attempts, which is what the tail is being given time to die in."""

WHY_RETAKEN = (
    "A take whose own lead-in was not quiet was recorded again after a rest rather than kept. The "
    "lead-in is where the noise floor is measured and the floor is the yardstick every figure is "
    "judged against, so a tail left by the take before does not merely add noise -- it raises the "
    "bar the parameter is then asked to clear, and the run reads as a parameter that does "
    "nothing. Judged from the take's own lead-in rather than by listening separately first, since "
    "opening the audio device is the one part of this chain that fails outright and a listen "
    "before every take doubles how often it is asked to."
)


def lead_in_dbfs(recording: Recording, before: float) -> float:
    """How loud the take's own lead-in was, in dBFS.

    A Python float rather than whatever numpy returned. A numpy scalar compares
    and rounds like a number and then refuses to be written: the record is built
    from these and `json.dump` raised on a numpy bool derived from one, after the
    machine time the run had already spent measuring.
    """
    span = recording.samples[: int(before * recording.sample_rate)]
    if not span.size:
        return float("-inf")
    peak_level = float(np.abs(span).max())
    return float(20.0 * np.log10(peak_level)) if peak_level > 0 else float("-inf")


def peak(recording: Recording) -> float:
    """The largest sample in any channel, for choosing between whole takes."""
    return float(np.abs(recording.samples).max())


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
