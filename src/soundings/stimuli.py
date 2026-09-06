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

from . import gestures


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

    channel: int | None = None
    """Zero-based MIDI channel, when the stimulus needs one of its own.

    None means the channel the command was given. A drum stimulus overrides it,
    because an unpitched sound lives on channel 10 and nowhere else -- the same
    note on any other channel is a pitched voice, which is the thing it exists
    not to be.
    """

    writes: tuple[tuple[str, int], ...] = ()
    """Addresses set after the reset and before the note, for a stimulus that needs them.

    A stimulus is normally a note under power-on defaults, and that is the right
    default: the fewer things a run changes, the fewer it has to put back. But a
    part's *kind* is set by an address rather than by a note, and putting an
    unpitched voice on a melodic part is the only way to ask an effect a question
    a pitched note cannot carry. What is written here is undone by the reset the
    next stimulus begins with.
    """

    moves: tuple[gestures.Move, ...] = ()
    """Messages sent after the setting is written and before each take.

    The other side of `writes`, and the difference is the order. A stimulus's
    writes go in before the parameter under test, so they are the state it is
    asked in; these go in after it, so they are something for it to act on. A
    parameter that decides whether an incoming message is received has nothing
    to receive without them, and reads as inaudible.
    """

    also: tuple[tuple[int, int, float, float], ...] = ()
    """Further notes, each `(note, velocity, seconds after the first, hold)`.

    One note cannot ask a parameter about polyphony. Whether a part is
    monophonic, and what it does when the same voice is asked for twice, sound
    identical under a single note however the address is set -- so the address
    answers inaudible and the null is a fact about the stimulus.
    """

    def on(self, default: int) -> int:
        return default if self.channel is None else self.channel

    def played(self) -> tuple[tuple[int, int, float, float], ...]:
        """Every note this stimulus plays, its own first."""
        return ((self.note, self.velocity, 0.0, self.hold), *self.also)

    def describe(self) -> str:
        # The writes belong in the description, not only in the JSON. A stimulus
        # that turns a melodic part into a rhythm part on one drum map is asking
        # its question in a state the note numbers alone do not name, and two
        # stimuli that differ only by which map they select would otherwise read
        # as the same condition in every verdict either of them produced.
        where = "" if self.channel is None else f"channel {self.channel + 1}, "
        prepared = "".join(f", {a} = {v}" for a, v in self.writes)
        # The moves belong in the description for the same reason: a note played
        # after six controllers have been moved is a different question from the
        # same note played under power-on defaults, and a verdict that did not
        # say so would read as the plain note's.
        moved = f", moved: {gestures.describe(self.moves)}" if self.moves else ""
        beside = "".join(
            f", with note {n} at velocity {v} {a:.2f} s later held {h:.2f} s"
            for n, v, a, h in self.also
        )
        return (
            f"{where}program {self.program}, note {self.note}, velocity {self.velocity}, "
            f"held {self.hold:.2f} s, captured {self.seconds:.1f} s{prepared}{beside}{moved}"
        )

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "program": self.program,
            "note": self.note,
            "velocity": self.velocity,
            "channel": self.channel,
            "writes": [[a, v] for a, v in self.writes],
            "also": [list(n) for n in self.also],
            "moves": [m.to_json() for m in self.moves],
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
    # The two below are what an *effect* has to be asked with. Every stimulus
    # above is a pitched note, and a pitched note repeats: middle C repeats every
    # 3.8 ms, so a delay of 3.8 ms correlates exactly as well as no delay, and a
    # modulated delay wider than half of that folds. Measured on this unit, the
    # chorus at CC93 100: the track swung 59 ms across a 3.81 ms period and no
    # rate could be read. The same harmonic structure starves a decay
    # measurement, since only the partials falling inside a band excite it, and
    # the reverb's octaves scattered 1.41 to 2.44 s against bounds of 5 to 11
    # percent -- a spread that is the piano's line spectrum, not the reverb.
    "unpitched": Stimulus(
        name="unpitched",
        program=0,
        note=38,
        velocity=100,
        hold=0.15,
        seconds=3.0,
        lead=0.6,
        channel=9,
        sees="a decay, because a snare is broadband and dies in a fifth of a second, "
        "so a tail measured after it is the effect's rather than the note's",
        blind_to="anything keyed to pitch, and anything needing the sound to still be "
        "there after a moment",
    ),
    "wash": Stimulus(
        name="wash",
        program=0,
        note=49,
        velocity=100,
        hold=0.15,
        seconds=4.0,
        lead=0.6,
        channel=9,
        sees="a modulation, being broadband and lasting seconds: its autocorrelation "
        "has one peak, so a delay measured against it is a delay rather than a delay "
        "modulo something",
        blind_to="anything keyed to pitch, and a decay -- a crash rings longer than "
        "most rooms, and the slower of the two is what a tail measures",
    ),
    # A low note is the third way out of the folding problem, and the only one that
    # keeps a melodic part: the ambiguity is the note's own period, so dropping two
    # octaves doubles then doubles again what a delay can swing before it wraps.
    # Note 24 repeats every 30.6 ms against middle C's 3.8. The voice that has to
    # supply it is the piano, since it is the only one on this unit that repeats.
    #
    # The two obvious alternatives were tried and do not work. The rhythm part is
    # unpitched but its chorus send stores without sounding, and GM 126 applause is
    # unpitched on a melodic part but is noise-driven and does not repeat at all --
    # measured, two takes of it differ by 0.1 dB, so the yardstick is the whole
    # signal and nothing could ever clear it.
    "deep": Stimulus(
        name="deep",
        program=0,
        note=24,
        velocity=110,
        hold=2.0,
        seconds=4.0,
        lead=0.6,
        sees="a modulated delay on a melodic part, by being slow enough not to fold "
        "one: 30.6 ms between repeats, against 3.8 at middle C",
        blind_to="anything the bottom two octaves do not excite, and -- measured -- a "
        "delay of any length. Dropping to 32.7 Hz stops a delay folding, and stops it "
        "being locatable in the same move: tens of milliseconds is a fraction of one "
        "cycle down here, so the correlation peak is broad enough to fill whatever "
        "range it is given. It is here for what a long slow note can carry, not for a "
        "delay",
    ),
    # The crash a melodic part can play. `wash` is the same sound on the rhythm
    # part, where this unit's chorus send stores without sounding; putting part 2
    # into rhythm mode gives the same unpitched broadband source on a part whose
    # sends do work, and the difference between the two says whether the exclusion
    # is about part 10 or about being a rhythm part at all.
    #
    # It exists because nothing else on this unit has all three properties a
    # modulated delay has to be asked with. A pitched note folds the delay into
    # its own period; a low note has a long enough period but at 32.7 Hz a delay
    # of tens of milliseconds is a fraction of a cycle, so the correlation peak
    # is too broad to place -- measured, it filled whatever range it was given.
    # Applause is unpitched on a melodic part and does not repeat at all.
    "struck_kit": Stimulus(
        name="struck_kit",
        program=0,
        note=49,
        velocity=100,
        hold=0.15,
        seconds=4.0,
        lead=0.6,
        channel=1,
        writes=(("40 12 15", 1),),
        sees="a modulated delay, being broadband, aperiodic and on a melodic part all "
        "at once, which nothing else here manages",
        blind_to="anything keyed to pitch, and anything a rhythm part is excluded from "
        "-- which is the thing it is partly there to find out",
    ),
    # The same note on the same part, differing only in which of the two drum
    # maps the part is told to use. That difference is what separates a claim
    # about a drum block from a claim about the half of it a part happens to
    # read: an address in the other half is unreachable however live it is, and
    # reads as storage nothing consults.
    "struck_kit_map2": Stimulus(
        name="struck_kit_map2",
        program=0,
        note=49,
        velocity=100,
        hold=0.15,
        seconds=4.0,
        lead=0.6,
        channel=1,
        writes=(("40 12 15", 2),),
        sees="whether an address in the second half of a drum block is read, which the "
        "first-map form cannot ask at all",
        blind_to="everything the first-map form is blind to, and additionally anything "
        "the two maps happen to agree on, which at power-on is most of them",
    ),
    # The same part and map as struck_kit, at the note the drum block's own
    # controls address. A drum parameter is stored per note -- 41 04 24 is the
    # panpot of note 36 and of nothing else -- so a verdict taken while a
    # different note sounds is a fact about that other note. The catalogue had
    # note 49 and note 38 on the rhythm part and note 36 only on a melodic one,
    # where the same number is a pitched voice, so the four addresses the NRPN
    # scan attributed at note 36 could not be asked at all.
    "struck_kit_36": Stimulus(
        name="struck_kit_36",
        program=0,
        note=36,
        velocity=100,
        hold=0.15,
        seconds=4.0,
        lead=0.6,
        channel=1,
        writes=(("40 12 15", 1),),
        sees="a parameter stored against note 36 of the first drum map, which every "
        "other rhythm-part stimulus here sounds a different note than",
        blind_to="anything stored against another note, and anything needing the sound "
        "to still be there after a moment",
    ),
    # The two below are what a *switch* has to be asked with. Everything above
    # plays its note under whatever the setting left, which is the right question
    # for a parameter that shapes a voice and no question at all for one that
    # decides whether an incoming message is acted on: the harness sends its
    # volume and its expression before it writes the setting, so a switch turned
    # off afterwards has nothing left to gate. These play the same struck piano
    # note with a handful of messages sent after the setting instead.
    "struck_moved": Stimulus(
        name="struck_moved",
        program=0,
        note=60,
        velocity=100,
        hold=1.0,
        seconds=3.0,
        lead=0.6,
        moves=gestures.MOVED,
        sees="whether the part still acted on a controller, a pedal or channel pressure "
        "after the setting was written -- something in the list reached the voice, or "
        "nothing did",
        blind_to="which of them it was, since they move together; " + gestures.CANNOT_ASK,
    ),
    "struck_retuned": Stimulus(
        name="struck_retuned",
        program=0,
        note=60,
        velocity=100,
        hold=1.0,
        seconds=3.0,
        lead=0.6,
        moves=gestures.RETUNED,
        sees="whether the part still acted on a bank select, a program change, a "
        "registered or non-registered parameter or a bend after the setting was written",
        blind_to="which of them it was; " + gestures.CANNOT_ASK + ". The program change "
        "also decides what voice everything else here is heard through, so a switch that "
        "gates it hides the rest behind itself",
    ),
    # Modulation on its own, because it cannot share a gesture with anything: it
    # is a free-running LFO, so it stops the takes repeating rather than changing
    # what they hold, and the yardstick it leaves is one no change clears. Under
    # it every switch but the one gating modulation answers inconclusive, which
    # is the right answer -- the run had no power to find anything.
    "struck_vibrato": Stimulus(
        name="struck_vibrato",
        program=0,
        note=60,
        velocity=100,
        hold=1.0,
        seconds=3.0,
        lead=0.6,
        moves=gestures.VIBRATO,
        sees="a switch that gates modulation, which shows as one setting's takes "
        "repeating far worse than the other's rather than as a difference between them",
        blind_to="everything else, and not by omission: the modulation raises the "
        "yardstick to where no change of any size clears it, so anything not gating the "
        "modulation itself answers inconclusive here",
    ),
    # The pedals the three gestures above never send, with the second note one of
    # them needs. Two of part 1's five remaining nulls came back not because the
    # switch was silent but because nothing in the run moved what it gates: no
    # gesture carries a portamento or a soft pedal at all.
    #
    # A glide is why this plays two notes and why they are an octave apart. The
    # second is what the first slides to, and the further it travels the longer it
    # spends somewhere neither note is, which is the only part of it a comparison
    # against an unglided pair can see.
    #
    # **Measured, it does not reach the voice, and nothing taken under it alone
    # is a verdict.** Takes of one setting came back 0.66 to 0.84 dB apart on
    # every address tried -- a yardstick that is the whole signal, which no change
    # of any size clears. The control that says it is the gesture rather than the
    # switches: 40 11 06 gates control changes and is audible under all three of
    # the others, and under this one it cannot be measured at all. The glide is
    # the suspect, being a pitch sweep whose phase falls differently on each
    # strike, which is how the modulator ruins a take a few entries above.
    # Whether a soft pedal alone repeats has not been asked.
    "struck_pedalled": Stimulus(
        name="struck_pedalled",
        program=0,
        note=60,
        velocity=100,
        hold=1.0,
        seconds=3.5,
        lead=0.6,
        moves=gestures.PEDALLED,
        also=((72, 100, 0.5, 1.0),),
        sees="nothing, measured. It was built to ask whether the part still acted on a "
        "portamento or a soft pedal, and takes of one setting came back under a decibel "
        "apart, so it has no power to answer that or anything else",
        blind_to="everything, and not by omission: the yardstick it leaves is the whole "
        "signal. A verdict taken under it alone is inconclusive rather than null, and the "
        "run that showed this is the one where 40 11 06 -- audible under all three other "
        "gestures -- could not be measured under this one. " + gestures.CANNOT_ASK,
    ),
    # The two below are what a parameter about *polyphony* has to be asked with,
    # and nothing above can ask one at all. Whether a part is monophonic, and
    # what it does when the same voice is asked for a second time, sound
    # identical under a single note however the address is set.
    "struck_pair": Stimulus(
        name="struck_pair",
        program=0,
        note=60,
        velocity=100,
        hold=1.0,
        seconds=3.0,
        lead=0.6,
        also=((67, 100, 0.0, 1.0),),
        sees="whether the part sounds two notes at once or takes the second in place of the first",
        blind_to="anything one note already answers, which it answers with a worse "
        "yardstick: two notes have two attacks to scatter instead of one",
    ),
    "struck_again": Stimulus(
        name="struck_again",
        program=0,
        note=60,
        velocity=100,
        hold=1.2,
        seconds=3.5,
        lead=0.6,
        also=((60, 100, 0.4, 1.2),),
        sees="what the part does when the voice already sounding is asked for again -- "
        "the second strike cuts the first off, or the two ring together",
        blind_to="everything a pair of different notes answers, since one voice being "
        "asked for twice and two voices being asked for are different questions",
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

# What an effect is asked with, as opposed to a parameter. Neither repeats, so a
# delay measured against them is unambiguous and a band is excited across its
# width rather than at a few partials.
#
# Measured, on the block of system effect addresses: asked with `struck` and both
# sends raised, takes of one setting came back 0.55 dB apart and 35 of 40
# addresses could not be measured at all. Asked with `unpitched` and the reverb
# send alone, the same addresses came back 34 dB apart. The 33 dB is the whole
# difference between a sweep that answers and a sweep that spends its device time
# and reports nothing, and it is two separate mistakes: a pitched note against an
# effect, and a chorus send raised over a take that has to repeat, which puts a
# free-running LFO at a new phase on every strike exactly as the modulator does.
#
# **Three of the four cannot ask a chorus question at all**, and it is not the
# stimulus so much as where it sounds: unpitched and wash are on the rhythm part,
# struck_kit puts a melodic part into rhythm mode, and a rhythm part on this unit
# is excluded from the chorus however its send is set. Measured four ways round,
# on the same address and the same part, so it is a fact about the mode and not
# about the crash --
#
#   40 12 21  part 2, rhythm mode, struck_kit   not audible, yardstick -40.0 dB
#   40 12 21  part 2, melodic,     deep         audible, +9.71 dB
#   40 11 21  part 1, melodic,     deep         audible, +9.39 dB
#   40 12 22  part 2, rhythm mode, struck_kit   audible, +30.4 dB, yardstick -40.2 dB
#
# The last row is what closes it: the same stimulus on the same part in the same
# mode hears the *reverb* send move by 30 dB. So `deep` is the whole of what can
# be asked through the chorus, and what that costs is repeatability -- the chorus
# is a free-running LFO, so raising its send takes takes of one setting from
# -53.6 dB apart to -4.5, and every null under it comes back inconclusive rather
# than answered. Lowering the send is a real lever on that and not enough of one:
# measured across the same address, 7F gives -3.6 dB, 40 gives -4.8 and 20 gives
# -9.0, and the chorus grows fainter as fast as the yardstick improves.
EFFECT = ("unpitched", "wash", "deep", "struck_kit")

# What a byte with two values is asked with. The plain note answers a parameter
# that shapes the voice; the other two answer one that gates a message, which the
# plain note cannot ask at all.
SWITCH = ("struck", "struck_moved", "struck_retuned", "struck_vibrato")

# The three that move, without the plain note. A gesture is a rescue for a null:
# it asks a much narrower question, in a state the verdict then only holds in, so
# it is worth its device time on an address the plain note could not hear and
# nothing on one it could. Asking a block in two passes rather than one costs
# only what the second pass covers, which is the addresses that came back
# inaudible -- measured on this unit, 2 minutes 39 seconds an address against 40
# seconds.
# struck_pedalled is deliberately not here. A gesture set is what a sweep spends
# its device time on, and this one measured as unable to answer anything: adding
# it to a block pass would buy a column of inconclusives at the same price as a
# column of verdicts. It stays in the catalogue, askable by name, because the
# thing to do next is find out whether the glide is what ruins it.
GESTURE = ("struck_moved", "struck_retuned", "struck_vibrato")

# What a parameter about polyphony has to be asked with, and the only two notes
# in the catalogue that play more than one note. Nothing else can ask one at all,
# so a null from anything else is a fact about the stimulus.
POLYPHONY = ("struck_pair", "struck_again")


def resolve(names) -> list[Stimulus]:
    """Look up names, refusing an unknown one rather than silently dropping it."""
    chosen = []
    for name in names:
        if name == "broad":
            chosen.extend(CATALOGUE[n] for n in BROAD)
        elif name == "effect":
            chosen.extend(CATALOGUE[n] for n in EFFECT)
        elif name == "switch":
            chosen.extend(CATALOGUE[n] for n in SWITCH)
        elif name == "gesture":
            chosen.extend(CATALOGUE[n] for n in GESTURE)
        elif name == "polyphony":
            chosen.extend(CATALOGUE[n] for n in POLYPHONY)
        elif name == "all":
            chosen.extend(CATALOGUE.values())
        elif name in CATALOGUE:
            chosen.append(CATALOGUE[name])
        else:
            raise KeyError(
                f"no stimulus named {name!r}; have: {', '.join(CATALOGUE)}, "
                "broad, effect, switch, gesture, polyphony, all"
            )
    seen, unique = set(), []
    for s in chosen:
        if s.name not in seen:
            seen.add(s.name)
            unique.append(s)
    return unique


__all__ = [
    "BROAD",
    "CATALOGUE",
    "DEFAULT",
    "EFFECT",
    "GESTURE",
    "POLYPHONY",
    "SWITCH",
    "Stimulus",
    "resolve",
]
