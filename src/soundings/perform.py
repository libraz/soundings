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


QUIET_LISTEN_S = 0.3
"""How much is listened to at a time while waiting for a tail to die."""

QUIET_TIMEOUT_S = 8.0
"""How long the wait gives up after, so a parameter that never goes quiet is a
finding rather than a hang. Whatever it leaves behind is still caught by the
lead-in check the takes are judged against."""

WHY_WAIT_FOR_QUIET = (
    "Each take waited for the room to go quiet before it was recorded, rather than starting on a "
    "timer. The lead-in is where the noise floor is measured and the floor is the yardstick every "
    "figure is judged against, so a tail from the take before it does not merely add noise -- it "
    "raises the bar the parameter then fails to clear, and the run reads as a parameter that does "
    "nothing. How long each wait took is kept, since a setting that takes longer to go quiet than "
    "its pair is itself a difference between the two."
)


def wait_until_quiet(
    *,
    device: str | None,
    below_dbfs: float,
    listen_s: float = QUIET_LISTEN_S,
    timeout_s: float = QUIET_TIMEOUT_S,
) -> tuple[float, float, bool]:
    """Listen until nothing is sounding, and say how long that took.

    Returns the seconds waited, the level it ended at, and whether it got under
    the line before giving up. A timeout is not an error here: an address that
    leaves something sounding for longer than this is a fact about the address,
    and it is the lead-in check on the takes themselves that refuses to measure
    through it.
    """
    waited = 0.0
    level = float("inf")
    while waited < timeout_s:
        heard = cap.record(listen_s, device=device)
        level = (
            20.0 * np.log10(float(np.abs(heard.samples).max()))
            if heard.samples.size and np.abs(heard.samples).max() > 0
            else float("-inf")
        )
        if level <= below_dbfs:
            return waited, level, True
        waited += listen_s
    return waited, level, False


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
