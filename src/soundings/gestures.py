"""Messages sent after a setting is written and before the note, for the switches.

A `contrast` run puts the unit in a state, changes one parameter and plays a
note. That is the whole method, and it has one blind spot: **a parameter that
does nothing but decide whether an incoming message is acted on has nothing to
act on.** The run sends its volume and its expression before it writes the
setting, so a switch turned off afterwards gates a message that has already
arrived, both settings sound alike, and the null is a fact about the order the
harness sends things in rather than about the unit.

The addresses this matters for are not a guess: the part block of this unit holds
eighteen bytes the write probe found to accept nothing but 0 and 1, and a switch
is what a byte with two values is. Which message each of them gates is exactly
what is not known, and is what a run under a gesture finds out.

**A gesture is deliberately several messages at once.** Asking one message per
run would need a guess about which address gates which -- that is the manual's
table, and reconstructing it from the manual would make the measurement a check
on the reading rather than a measurement. A gesture that moves many things at
once asks a weaker question of every switch and needs no such guess: something in
this list reached it. Which one is a second run, over the ones that answered.

**Each gesture carries at most one message that costs level.** Volume and
expression are each about 20 dB, and two of them stacked take the take 40 dB down
towards the noise the yardstick is measured from. So they go in separate
gestures, and what is in a gesture beside its one attenuator is chosen to leave
the voice repeating: nothing here replaces the struck piano note with a sustained
voice, which on this unit repeats 20 to 58 dB worse.

**A modulator cannot share a gesture.** Modulation is a free-running LFO at a
different phase every strike, so it does not change the sound so much as stop the
sound repeating: measured here, adding it took takes of one setting from 0.2 dB
apart to 10.6, and 10.6 dB is a yardstick nothing clears. It has a gesture to
itself, where that is the evidence rather than the ruin of it.

**A gesture must not write the address under test.** Volume, panpot, the filter,
a program change and a bank select each store into the part block, and a gesture
carrying one of them while that byte is the parameter puts both settings at the
gesture's value. Measured the first time these ran: the part level written as 0
came back sounding 31.8 dB over the lead-in, because the volume in the gesture
had put it back. `without` takes such a move out for that run, and what it took
out is reported, since the gesture is a weaker question one message short.

**A move can also be timed into the note.** Everything above is sent before the
note, which is the whole reason two kinds of message were long unaskable:
polyphonic pressure names a note and has nowhere to land before there is one,
and a pedal released after the note has to be released after something. A move
carried in a stimulus's `during` is sent at a stated time from the first
note-on instead, which is the same list of messages with a clock against it.

The gestures that are sent up front say so themselves, in `CANNOT_ASK`, and that
sentence is still true of each of them: it is a statement about one gesture, not
about the harness. A null taken under `MOVED` is still bounded by the fact that
nothing in `MOVED` arrives while the note sounds.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Move:
    """One message, or one indivisible group of them, sent before the note.

    A registered or non-registered parameter is a group: the selector is re-sent
    with every data entry rather than left standing, because a selector left over
    from something else is the classic way for a data entry to land at an address
    nobody named.
    """

    kind: str
    data: tuple[int, ...]
    means: str

    stores_at: tuple[int, ...] = ()
    """Offsets in the part block this message was measured to write, if any.

    A gesture sent while an address is under test must not be one that writes
    that address: it would put both settings at the same byte and the run would
    measure the gesture. Seen the first time the gestures were used, on the part
    level -- the volume in the gesture aliases it, so the setting written as 0
    came back sounding 31.8 dB over the lead-in.

    Offsets rather than addresses, because the block moves with the part while
    the offset does not. Each one is from this archive's own alias scan, which
    diffed the address space around the message rather than reading a manual.
    """

    def messages(self, channel: int) -> list[list[int]]:
        status = channel & 0x0F
        if self.kind == "cc":
            controller, value = self.data
            return [[0xB0 | status, controller & 0x7F, value & 0x7F]]
        if self.kind == "bend":
            (value,) = self.data
            bend = max(0, min(0x3FFF, value + 0x2000))
            return [[0xE0 | status, bend & 0x7F, (bend >> 7) & 0x7F]]
        if self.kind == "pressure":
            (value,) = self.data
            return [[0xD0 | status, value & 0x7F]]
        if self.kind == "polypressure":
            note, value = self.data
            return [[0xA0 | status, note & 0x7F, value & 0x7F]]
        if self.kind == "program":
            (program,) = self.data
            return [[0xC0 | status, program & 0x7F]]
        if self.kind == "bank":
            msb, lsb, program = self.data
            return [
                [0xB0 | status, 0, msb & 0x7F],
                [0xB0 | status, 32, lsb & 0x7F],
                [0xC0 | status, program & 0x7F],
            ]
        if self.kind in ("rpn", "nrpn"):
            select = (101, 100) if self.kind == "rpn" else (99, 98)
            msb, lsb, value = self.data
            return [
                [0xB0 | status, select[0], msb & 0x7F],
                [0xB0 | status, select[1], lsb & 0x7F],
                [0xB0 | status, 6, value & 0x7F],
                # Park the selector. A data entry sent later by anything else
                # would otherwise land on whatever this left standing.
                [0xB0 | status, select[0], 0x7F],
                [0xB0 | status, select[1], 0x7F],
            ]
        raise KeyError(f"no move of kind {self.kind!r}")

    def describe(self) -> str:
        shown = " ".join(f"{v:02X}" for v in self.data)
        return f"{self.kind} {shown} ({self.means})"

    def to_json(self) -> dict:
        return {
            "kind": self.kind,
            "data": list(self.data),
            "means": self.means,
            "stores_at_part_offset": list(self.stores_at),
        }


def cc(controller: int, value: int, means: str, stores_at: tuple[int, ...] = ()) -> Move:
    return Move("cc", (controller, value), means, stores_at)


# Moved after the setting and before the note, so a switch that gates one of them
# has something to gate. Level cost: CC7 alone.
MOVED: tuple[Move, ...] = (
    cc(7, 40, "volume, about 20 dB down", (0x19,)),
    cc(10, 0, "panpot hard to one side", (0x1C,)),
    cc(64, 127, "hold, which outlasts the note on a decaying voice"),
    Move("pressure", (127,), "channel pressure"),
)

# Modulation on its own, and the reason it is on its own is that it cannot share
# a gesture with anything. A free-running LFO is at a different phase every time
# the note is struck, so it does not change the sound so much as stop the sound
# repeating -- measured here, takes of one setting went from -0.2 dB apart to
# -10.6, which is a yardstick no change of any size clears. Every other switch
# asked under it therefore answers inconclusive, correctly: the run had no power
# to find anything. What it can answer is the switch that gates modulation
# itself, which shows as one setting repeating far worse than the other.
VIBRATO: tuple[Move, ...] = (cc(1, 127, "modulation"),)

# The other half, and the reason there are two: expression costs the same 20 dB
# volume does, and the two stacked would take the take down to where the
# yardstick is most of it. Level cost: CC11 alone.
#
# The program change goes to a struck voice rather than a sustained one. Every
# sustained voice on this unit repeats far worse than the piano, so a gesture
# that left the part on one would raise the yardstick for every switch asked
# under it and answer inconclusive rather than no.
RETUNED: tuple[Move, ...] = (
    Move(
        "bank",
        (8, 0, 12),
        "a bank select and a program change together, to a tone the "
        "tone map records this unit having in bank 8",
        (0x00, 0x01),
    ),
    cc(11, 60, "expression"),
    cc(74, 0x30, "the filter", (0x32,)),
    Move("nrpn", (0x01, 0x21, 0x60), "a non-registered parameter", (0x33,)),
    Move("rpn", (0x00, 0x00, 12), "a registered parameter, the range the bend below is read in"),
    Move("bend", (8191,), "pitch bend to the top of its range"),
)

# The pedals nothing else sends, with the second note the first of them needs.
#
# Portamento and the soft pedal were named as unaskable together with polyphonic
# pressure and a pedal released mid-note, and they do not belong with those two.
# A glide needs a second note to reach rather than a message sent while the first
# sounds, and the catalogue already plays a second note; the soft pedal shapes
# whatever is struck after it, so sending it before the note is when it works.
# Only the other two need a message timed into a sounding note, which nothing
# here can do.
#
# Portamento time goes first and is not decoration: with the time at zero the
# glide is instantaneous, which is what no glide sounds like, so the switch would
# read as inaudible for want of a second controller rather than for want of
# reaching the voice.
#
# None of the three carries `stores_at`. That is from this archive's own alias
# scan rather than an assumption: it attributed twelve controllers on this
# channel and none of these is among them, so no part-block byte follows them and
# no run has to drop one to ask its address.
PEDALLED: tuple[Move, ...] = (
    cc(5, 64, "portamento time, without which the glide is instant and sounds like no glide"),
    cc(65, 127, "portamento on, which the note below glides under"),
    cc(67, 127, "the soft pedal, which shapes what is struck after it"),
)

# The soft pedal on its own, which is the half of `PEDALLED` the glide did not
# ruin. `PEDALLED` measured with a yardstick that was the whole signal and the
# portamento was the suspect, being a pitch sweep landing at a different place on
# every strike; whether a soft pedal alone repeats was left unasked, and this is
# what asks it. One message, so a verdict under it names the message.
SOFTENED: tuple[Move, ...] = (cc(67, 127, "the soft pedal, which shapes what is struck after it"),)

# The two below are timed into the note rather than sent before it. Each entry is
# `(seconds from the first note-on, the move)`.
#
# **One message each, against the practice above.** A gesture carries several
# messages at once because asking one per run would need a guess about which
# address gates which, and a guess reconstructed from the manual makes the
# measurement a check on the reading. Neither of these needs such a guess: the
# question is already one message wide -- whether the part still acts on
# polyphonic pressure, whether it still acts on a pedal lifted after the key --
# and a second message in the gesture would take that back for nothing.

PRESSED_AT_S = 0.35
"""Where in the note the pressure lands: past the attack, inside the hold."""


def pressed_on(note: int, value: int = 127) -> tuple[tuple[float, Move], ...]:
    """Polyphonic pressure on a note, sent while that note is sounding.

    The note is passed in rather than assumed, because polyphonic pressure names
    one: a pressure addressed to a note the stimulus does not play is a run with
    no pressure in it, and it would answer inaudible at every address without
    anything saying why.
    """
    return (
        (
            PRESSED_AT_S,
            Move("polypressure", (note, value), "polyphonic pressure on the sounding note"),
        ),
    )


RELEASED_AFTER_S = 0.15
"""How long after the key is lifted the pedal is, which is what makes it a pedal.

Lifted before the note-off it is a pedal that never held anything and the take is
the plain note. Kept short because the voice is decaying: the further past the
key the release goes, the less tail is left for the damper to take, and the tail
that stops is the whole of what a comparison can see.
"""


def released_after(hold: float) -> tuple[tuple[float, Move], ...]:
    """Hold down before the note, up a moment after the key, so a damper falls."""
    return ((hold + RELEASED_AFTER_S, cc(64, 0, "the hold pedal lifted, after the key")),)


CANNOT_ASK = (
    "polyphonic pressure and a pedal released after the note -- each needs a message timed into "
    "a note already sounding, and everything here is sent before the note"
)

WHY_TIMED = (
    "This move was sent at a stated time from the first note-on rather than before the note. Two "
    "kinds of message have nowhere to land otherwise: polyphonic pressure names a note, and a "
    "pedal released after the key has to be released after something. A gesture sent up front "
    "answers inaudible for either of them whatever the address does, and the null is a fact about "
    "when the harness sends things."
)

WHY_DROPPED = (
    "A message the alias scan measured to store at the address under test is left out of the "
    "gesture for that run. Sending it would put both settings at the byte the gesture writes "
    "rather than at the two the run asked for, so the takes would differ by nothing and the "
    "null would be a fact about the gesture. What is left out is named per run, since the "
    "gesture it was dropped from is a weaker question than the whole one."
)


def without(moves: tuple[Move, ...], offset: int | None) -> tuple[tuple[Move, ...], list[Move]]:
    """The gesture with any move that writes `offset` taken out, and what was taken.

    An offset of None -- a run against a controller rather than an address, or
    against an address outside the part block -- takes nothing out.
    """
    if offset is None:
        return moves, []
    kept = tuple(m for m in moves if offset not in m.stores_at)
    return kept, [m for m in moves if offset in m.stores_at]


def describe(moves: tuple[Move, ...]) -> str:
    return ", ".join(m.describe() for m in moves)


__all__ = [
    "CANNOT_ASK",
    "MOVED",
    "PEDALLED",
    "PRESSED_AT_S",
    "RELEASED_AFTER_S",
    "RETUNED",
    "SOFTENED",
    "VIBRATO",
    "WHY_DROPPED",
    "WHY_TIMED",
    "Move",
    "cc",
    "describe",
    "pressed_on",
    "released_after",
    "without",
]
