"""Read the unit's pitch against equal temperament, and bound the chain's own wander.

Several voices, because most are not a frequency reference: one that layers two
detuned oscillators, or carries its own vibrato, wobbles regardless of what is
sent to it, and its phase slope is a confident number about nothing. A voice that
is not steady has no frequency to report and is recorded as having none, so that
the absence is visible rather than averaged in.

Every take passes through the same converter, so the chain cannot be wobbling by
more than the calmest voice does. That bounds it without a second instrument to
check against, and it is what makes a wobble common to all of them a fact about
the voices rather than about the measurement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import perform, roland, stability
from .midi import MidiLink
from .tonemap import Asker

CAVEAT = (
    "A departure here is the unit's tuning and the ratio of its sample clock to the "
    "converter's, together. Nothing measured here separates them. It is the correction a "
    "comparison against software rendered at exactly 48000 Hz needs; a comparison of two "
    "takes from this unit needs none of it, since both carry it equally. A program whose "
    "tone is not steady has no frequency to report and its number is recorded only so that "
    "the absence is visible."
)


def equal_temperament(note: int, cents: float | None = None) -> float:
    """The frequency a note should sound at, moved by the unit's own master tune."""
    hz = 440.0 * 2 ** ((note - 69) / 12.0)
    if cents is not None:
        hz *= 2 ** (cents / 1200.0)
    return hz


@dataclass
class Clock:
    note: int
    expected_hz: float
    voices: dict[tuple[int, int], stability.ToneFit] = field(default_factory=dict)

    def departure_ppm(self, fit: stability.ToneFit) -> float:
        return (fit.frequency / self.expected_hz - 1.0) * 1e6

    def describe(self) -> str:
        if not self.voices:
            return ""
        spread = [self.departure_ppm(f) for f in self.voices.values()]
        calmest_at = min(self.voices, key=lambda k: self.voices[k].wobble_cycles)
        calmest = self.voices[calmest_at].wobble_cycles
        steady = [k for k, f in self.voices.items() if f.steady]
        # The calmest voice bounds the chain: it carried the same converter as
        # every other take, so the chain cannot wander more than it does.
        bound = calmest / self.voices[calmest_at].seconds
        return (
            f"{len(self.voices)} voices span {min(spread):+.0f} to {max(spread):+.0f} ppm "
            f"off equal temperament, {len(steady)} of them steady\n"
            f"the calmest wobbles {calmest:.4f} cycles, so the chain's own wander is "
            f"under {bound / self.expected_hz * 1e6:.0f} ppm and the rest is the voices"
        )

    def to_json(self) -> dict | None:
        """None when no voice held still enough to read, which is itself the finding."""
        if not self.voices:
            return None
        return {
            "note_asked": self.note,
            "expected_hz": round(self.expected_hz, 6),
            "caveat": CAVEAT,
            "voices": {
                f"{bank}:{program}": {
                    **fit.to_json(),
                    "departure_ppm": round(self.departure_ppm(fit), 1),
                }
                for (bank, program), fit in sorted(self.voices.items())
            },
        }


def measure(
    link: MidiLink,
    *,
    programs: list[tuple[int, int]],
    expected_hz: float,
    note: int = 69,
    seconds: float = 20.0,
    channel: int = 0,
    device_id: int = roland.DEFAULT_DEVICE_ID,
    audio: str | None = None,
    progress=None,
) -> Clock:
    """Hold one long note per program and read the frequency of each.

    Each program is asked for through the tone map's Asker rather than sent
    blind. A bank and program the unit does not have is discarded whole and
    leaves the previous voice playing, which would be measured and filed under
    the voice that was asked for.
    """
    found = Clock(note=note, expected_hz=expected_hz)
    asker = Asker(link, channel=channel, device_id=device_id)
    asker.settle_on(0, 0)

    for bank, program in programs:
        where = f"bank {bank:3d} program {program:3d}"
        if not asker.ask(bank, program):
            if progress:
                progress(f"{where}: the unit does not have it")
            continue
        for controller, value in ((7, 127), (11, 127), (91, 0), (93, 0)):
            link.send([0xB0 | channel, controller, value])
        time.sleep(0.3)
        held = perform.record_note(
            link,
            device=audio,
            channel=channel,
            note=note,
            velocity=100,
            hold=seconds,
            seconds=seconds + 1.5,
            lead=0.5,
        )
        fit = (
            stability.tone_frequency(
                held.channel(perform.loudest_channel(held)),
                held.sample_rate,
                expected=expected_hz,
                skip=1.0,
                trim=0.5,
            )
            if held.healthy
            else None
        )
        if fit is None:
            if progress:
                progress(f"{where}: no tone to read")
            continue
        found.voices[(bank, program)] = fit
        if progress:
            progress(
                f"{where}: {fit.frequency:9.4f} Hz  "
                f"{found.departure_ppm(fit):+9.1f} ppm  "
                f"wobble {fit.wobble_cycles:7.4f} cycles  "
                f"{'steady' if fit.steady else 'not steady'}"
            )

    if progress:
        for line in found.describe().splitlines():
            progress(line)
    return found
