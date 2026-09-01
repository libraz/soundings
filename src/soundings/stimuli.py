"""The notes a parameter is asked under, since a null is a fact about the question.

Whether a parameter is audible is not a property of the parameter alone. It is a
property of the parameter and the note it was asked with, and the two are easy to
confuse because the answer looks the same either way: no residual above the
noise. A release time is inaudible while the key is still down. A filter cutoff
on a struck piano note is over before it has done much. A velocity curve compared
at one velocity is compared at a point where it has no slope.

So the stimulus is named, chosen, and carried with the verdict, and a parameter
that comes back inaudible is inaudible *under the stimuli it was tried with* --
which is a claim that can be extended by trying another, rather than a fact that
has to be overturned.

**A stimulus is only as good as its voice repeats.** The yardstick is what two
takes of one setting differ by, so a voice that does not repeat cannot detect
anything at all, however large the change. Every sustaining voice on this unit
is 20 to 58 dB worse than the struck piano note, which is why `struck` is the
default and why a sustained result carries a much weaker claim.

**Audible under any is audible.** The verdict over a set of stimuli is a union,
never an average: one note hearing the change is proof the parameter reaches the
signal path, and the others failing to hear it says only that they asked the
wrong question. The reverse does not hold, which is why the list of what was
tried has to travel with a null.

Each entry names what it can see and, more usefully, what it cannot.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stimulus:
    name: str
    program: int
    note: int
    velocity: int
    hold: float
    """Seconds the key is held down."""

    seconds: float
    """Seconds captured, from the first block of audio rather than from the call."""

    lead: float
    """Silence before the note. The noise floor is measured in it."""

    sees: str
    blind_to: str

    def describe(self) -> str:
        return (
            f"program {self.program}, note {self.note}, velocity {self.velocity}, "
            f"held {self.hold:.2f} s, captured {self.seconds:.1f} s"
        )

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "program": self.program,
            "note": self.note,
            "velocity": self.velocity,
            "hold_s": self.hold,
            "captured_s": self.seconds,
            "lead_s": self.lead,
            "sees": self.sees,
            "blind_to": self.blind_to,
        }


CATALOGUE: dict[str, Stimulus] = {
    "struck": Stimulus(
        name="struck",
        program=0,
        note=60,
        velocity=100,
        hold=1.0,
        seconds=3.0,
        lead=0.6,
        sees="level, timbre, and anything an effect does to a decaying note",
        blind_to="what happens after note-off, since the note is still held for most "
        "of the take, and anything needing a steady tone",
    ),
    "released": Stimulus(
        name="released",
        program=0,
        note=60,
        velocity=100,
        hold=0.15,
        seconds=3.0,
        lead=0.6,
        sees="release and the tail after note-off, which is nearly the whole take",
        blind_to="anything shaping a held note, and anything too quiet to clear the "
        "floor once the note has let go",
    ),
    "sustained": Stimulus(
        name="sustained",
        program=19,
        note=60,
        velocity=100,
        hold=2.5,
        seconds=4.0,
        lead=0.6,
        sees="filters, amplitude shaping and modulation on a note that does not decay",
        blind_to="attack shaping, and anything smaller than a yardstick 20 to 29 dB "
        "above the noise. No sustaining voice on this unit repeats the way a struck "
        "piano note does -- measured, holding one note for 2.5 s: church organ 21 to "
        "29 dB above its floor, drawbar organ 33 to 40, square lead 32 to 50, flute 57 "
        "to 58, against 0.1 to 2.3 for the piano. The church organ is the best of them "
        "and is what this uses",
    ),
    "soft": Stimulus(
        name="soft",
        program=0,
        note=60,
        velocity=30,
        hold=1.0,
        seconds=3.0,
        lead=0.6,
        sees="the quiet end of a velocity curve, and anything that only bites there",
        blind_to="everything the level costs it -- 20 dB less signal over the same "
        "converter noise, so the floor is 20 dB closer and small changes vanish",
    ),
    "loud": Stimulus(
        name="loud",
        program=0,
        note=60,
        velocity=127,
        hold=1.0,
        seconds=3.0,
        lead=0.6,
        sees="the loud end of a velocity curve, and anything that only bites there",
        blind_to="what a curve does between the ends. Two velocities cannot separate "
        "a curve from any other passing through both",
    ),
    "low": Stimulus(
        name="low",
        program=0,
        note=36,
        velocity=100,
        hold=1.5,
        seconds=3.5,
        lead=0.6,
        sees="anything keyed to pitch, and filters whose corner sits above middle C",
        blind_to="the top of the keyboard, and anything whose effect is proportional "
        "rather than absolute in frequency",
    ),
    "high": Stimulus(
        name="high",
        program=0,
        note=84,
        velocity=100,
        hold=1.0,
        seconds=2.5,
        lead=0.6,
        sees="anything keyed to pitch at the other end, and short decays",
        blind_to="the bottom of the keyboard; a high note also has less energy below "
        "the corner of most filters",
    ),
}

DEFAULT = ("struck",)

# Enough to answer most parameters without asking the wrong question, at the cost
# of five times the device time. Not the whole catalogue: 'low' and 'high' only
# earn their place for something already suspected of tracking pitch.
BROAD = ("struck", "released", "sustained", "soft", "loud")


def resolve(names) -> list[Stimulus]:
    """Look up names, refusing an unknown one rather than silently dropping it."""
    chosen = []
    for name in names:
        if name == "broad":
            chosen.extend(CATALOGUE[n] for n in BROAD)
        elif name == "all":
            chosen.extend(CATALOGUE.values())
        elif name in CATALOGUE:
            chosen.append(CATALOGUE[name])
        else:
            raise KeyError(f"no stimulus named {name!r}; have: {', '.join(CATALOGUE)}, broad, all")
    seen, unique = set(), []
    for s in chosen:
        if s.name not in seen:
            seen.add(s.name)
            unique.append(s)
    return unique


__all__ = ["BROAD", "CATALOGUE", "DEFAULT", "Stimulus", "resolve"]
