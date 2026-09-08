"""Reading a block's worth of contrast records into one verdict per address.

A part block is asked one address at a time, so what comes back is a directory of
records rather than an answer about the block. Three things have to happen to it
before it is one, and none of them is arithmetic anyone should do by hand.

**The block has to account for itself.** An address is audible, inaudible under
what was tried, unmeasurable, or was never asked -- and the last of those has to
be visible, because a directory holding forty-five records of a forty-seven
address block reads exactly like a complete answer. The plan says how many there
should be and which one cannot be asked at all, and the count is checked against
it rather than against the files that happen to be there.

**A gesture is a rescue for a null, so the two passes are not peers.** The plain
note asks the address in the state the unit powers up in; a gesture asks it with
half a dozen messages moved, which is a narrower question whose verdict holds
only in that state. So an address the plain note heard is answered by the plain
note, and the gesture's job is the addresses it could not.

**The repeatability channel is meaningless wherever both settings scatter.** That
channel exists for a parameter that switches something moving on: one setting
repeats, the other does not. But a stimulus in which *both* settings scatter has
nothing for it to detect, and the gap between them is a free-running phase.
Measured on the first address asked under the modulation gesture: 29.2 dB against
14.5, which clears the bar comfortably, on an address the plain note found no
difference in at all -- across 51.92 dB against a yardstick of 51.9 and a level
difference of 0.001 dB. The gap is read as evidence only when the steadier
setting is steady in absolute terms, which the plain pass measured for that same
address.

The rule was first written naming the modulation gesture, and that was too
narrow. Any stimulus sounding more than one voice has the same problem for the
same reason: the trigger cannot fix the relative phase of two voices, and this
unit's trigger scatter is wider than a period of the note being played. Asked
across three part blocks holding the same parameters, the name-based rule passed
sixteen verdicts, each witnessed by a single stimulus, on a different set of
addresses in every block. Parameters do not do that. So the guard is keyed on the
channel a verdict came from rather than on which stimulus produced it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

METHOD = (
    "Each address in the block was asked on its own, at the pair the write probe measured it "
    "to accept, and the records are read back together here. An address the plain note heard "
    "is answered by the plain note; one it could not was asked again with messages moved after "
    "the setting, since a parameter that only decides whether a message is received has nothing "
    "to receive otherwise. The block is counted against the plan rather than against the files "
    "present, so an address that was never asked cannot read as one that answered."
)

# The same account with its middle sentence gone, written out rather than
# assembled from shared pieces. The archive gate holds every published method
# against the source as one verbatim literal, so a sentence joined at run time
# is pinned to nothing and orphans the prose of every record already carrying
# it. Two sentences are duplicated to keep both forms pinned; changing one and
# not the other is what the gate then catches.
METHOD_ONE_PASS = (
    "Each address in the block was asked on its own, at the pair the write probe measured it "
    "to accept, and the records are read back together here. The block is counted against the "
    "plan rather than against the files present, so an address that was never asked cannot read "
    "as one that answered."
)


# A third form, for a block nothing could be asked of. Written out in full for
# the reason above, and separate because the other two describe listening: a
# block that was never sounded must not publish an account of having been.
METHOD_NOTHING_TO_ASK = (
    "Nothing in this block was sounded. The write probe wrote to each of its addresses and read "
    "each one back holding the value it started at, so there is no pair of settings to compare "
    "and the audible question cannot be put here at all. What the block is counted against is "
    "the same plan every other block is, and the plan asked for nothing -- which is a bound on "
    "the addresses rather than a run that found nothing."
)


def method_for(gesture: bool, *, asked: bool = True) -> str:
    """The method sentence, saying only what this fold was actually given.

    The gesture half used to be part of one fixed string, so a block folded from
    a single pass published a method describing a second pass that never ran.
    Nothing downstream compares a method against the directories it was handed,
    so the sentence is the run's own account of itself and there was nothing to
    catch it -- which is the whole reason a record is allowed to make claims.

    The same holds one step further out. A block whose addresses accept a single
    value each is folded from no passes at all, and either of the sentences above
    would tell a reader it had been listened to.
    """
    if not asked:
        return METHOD_NOTHING_TO_ASK
    return METHOD if gesture else METHOD_ONE_PASS


WHY_TWO_PASSES = (
    "A gesture is a rescue for a null rather than a better question. It asks the address with "
    "half a dozen messages moved, which is a state the verdict then holds only in, so it is "
    "spent on the addresses the plain note could not hear and on nothing else. Where both "
    "passes spoke, the plain one is the verdict and the gesture is what it took to hear it."
)

WHY_POLYPHONY_PASS = (
    "A third pass asked the addresses the first two could not hear with two notes instead of "
    "one. It is not a narrower question the way a gesture is: the notes are played in the state "
    "the unit powers up in, so a verdict from it is as broad as the plain note's. What it adds "
    "is the only question a single note cannot put at all -- whether the part sounds two voices "
    "at once, and what it does when the voice already sounding is asked for again. A null from "
    "any one-note stimulus on such a parameter is a fact about the stimulus rather than about "
    "the address, so until this pass has run those addresses are unasked rather than silent."
)

WHY_SCATTER_NOT_A_GATE = (
    "This verdict rested on the channel that reads one setting's takes agreeing worse than the "
    "other's, and on nothing else. That channel is evidence only where the steadier setting is "
    "steady in absolute terms, against what the same address measured under the plain note: a "
    "gate leaves the setting it blocks repeating as a plain note does, while two draws from a "
    "free-running phase can sit any distance apart and mean nothing. Here both settings "
    "scattered far worse than a plain note on this unit, so the gap between them is the phase "
    "and not the parameter. Any stimulus sounding more than one voice can do this, because the "
    "trigger cannot fix their relative phase and the scatter it does leave is wider than a "
    "period of the note being played."
)

WHY_LEFT_OUT = (
    "A gesture cannot carry a message that writes the address under test: it would put both "
    "settings at the byte the gesture sends rather than at the two the run asked for. So the "
    "gesture asked here was one message short, and a null under it is narrower than a null "
    "under the whole one. For an address whose only route to being heard is the message that "
    "stores into it, that is narrower to the point of being unanswerable this way."
)

STEADY_WITHIN_DB = 12.0
"""How near the plain note's own repeatability the steadier setting has to come.

Twice the margin, the same span the repeatability channel already asks a gap to
clear before it is a gap at all. The measured case it has to reject sits 28 dB
outside it, and a real gate would leave the blocked setting repeating as the
plain note does, so the two are not near each other.
"""


WHY_MEASURED = (
    "The quantities each verdict was decided from, so a reader can redo the decision instead of "
    "taking it. A difference is audible when the two settings differ by more than either differs "
    "from itself, by the stated margin: compare across_setting_residual_db with "
    "same_setting_residual_db, and across_setting_level_db with same_setting_level_db. The floor "
    "says what the chain could have heard at all. Without these a null states that nothing was "
    "found and gives no way to ask how hard it was looked for, which is a claim this archive "
    "treats as unfinished."
)

WHY_DECIDED_BY = (
    "Which of the three channels carried each stimulus's verdict -- a change of shape, a change "
    "of level, or one setting repeating far worse than the other. The third is not the same kind "
    "of evidence as the first two: it fires on something having started moving rather than on "
    "the two settings sounding different, so a verdict resting on it alone is read differently "
    "from one a shape or a level change also carries. An empty list beside an audible verdict "
    "would be a contradiction and does not occur; an empty list beside a null is the ordinary case."
)

WHY_NO_REPEATABILITY_FIGURE = (
    "A missing repeatability figure is one that could not be formed, not one that was not "
    "measured. The figure is the median over the pairs of takes of a setting, and fewer than "
    "three pairs leaves no median to take -- so a setting whose takes could not all be compared "
    "reports nothing here rather than a number computed from too little. Read it as unknown for "
    "that setting, and read the other setting's figure as still standing."
)


@dataclass
class AddressVerdict:
    """One address, over every stimulus it was asked under."""

    address: str
    values: list[int]
    audible: bool
    heard_by: list[str] = field(default_factory=list)
    not_heard_by: list[str] = field(default_factory=list)
    inconclusive_under: list[str] = field(default_factory=list)
    unrepeatable_db: dict[str, list] = field(default_factory=dict)
    discounted: list[str] = field(default_factory=list)
    """Stimuli whose audible verdict was set aside, with the reason in the record."""

    grounds: dict[str, list[str]] = field(default_factory=dict)
    """Which of the three channels carried each stimulus's verdict.

    Kept because the repeatability channel is the one that can fire on nothing,
    and a verdict resting on it alone has to be treated differently from the same
    verdict backed by a change of shape or of level.
    """

    measured: dict[str, dict] = field(default_factory=dict)
    """The quantities each stimulus's verdict was decided from.

    Carried rather than summarised into the verdict word. A null is a claim that
    nothing was found, and this archive's own bar is that such a claim is worth
    reading only when the record says what could have been found instead -- which
    is the margin, the floor, and the two residuals the margin was applied to. A
    reader with the verdict alone cannot tell a null that cleared the bar by a
    decibel from one that cleared it by forty.
    """

    read_back: list = field(default_factory=list)
    """What each setting read as after it was written.

    The strongest control the run has: a null taken at an address that never took
    the value is a fact about the write, not about the sound. Kept per address
    because it is measured per address, and left empty rather than assumed for
    the records taken before the contrast stage read settings back at all.
    """

    read_back_why: str = ""
    """The contrast stage's own sentence about what a read-back means.

    Carried from the record rather than restated here, so the block record and
    the per-address record it was folded from cannot drift into two accounts of
    the same control.
    """

    caveat: str = ""
    """What the plan said about how this address had to be asked, if anything.

    Carried through rather than left in the plan. A plan is what a run was aimed
    by and a block record is what anyone reads afterwards, so a caveat that stays
    behind is one nobody sees: the addresses asked over a guessed span, because
    the write probe could not read what they held and so never wrote to them, are
    the ones whose null means least and they arrive looking like every other null.
    """

    superseded_by: dict = field(default_factory=dict)
    """Another record in this archive that asked this address and answers for it.

    The verdict here is kept rather than removed, because it was measured and
    removing it would leave the archive claiming the address was never asked this
    way. What it cannot do is stand unqualified beside a record that disagrees
    with it: a reader landing on this file would take a null as the archive's
    answer when the archive has a better one.
    """

    did_not_reproduce: list[str] = field(default_factory=list)
    """Stimuli asked more than once at this address that did not answer the same way.

    An archive that asked a question twice and got two answers has to say so. The
    verdict is withdrawn rather than settled by a majority or by whichever run is
    newer: what the runs establish together is that the answer did not hold, and
    every run's figures stay in the record so a reader can see how far apart they
    were.
    """

    left_out: dict = field(default_factory=dict)
    """Moves the gesture could not carry here, because they write this address.

    A null under a gesture one message short is narrower than a null under the
    whole one, and for some addresses it is narrower to the point of being
    unanswerable: the message that would reveal the address is the message that
    stores into it. Measured here on the two bytes a bank select and a program
    change land in, which is the only gesture that could have moved them.
    """

    @property
    def still_open(self) -> bool:
        """Whether anything is left to try: nothing heard it, or nothing could."""
        return not self.audible

    @property
    def verdict(self) -> str:
        if self.audible:
            return "audible"
        if self.not_heard_by:
            return "not audible under what was tried"
        return "could not be measured"

    def to_json(self) -> dict:
        out = {
            "address": self.address,
            "values": self.values,
            "verdict": self.verdict,
            "audible": self.audible,
            "heard_by": self.heard_by,
            "not_heard_by": self.not_heard_by,
            "inconclusive_under": self.inconclusive_under,
            "each_setting_unrepeatable_db": self.unrepeatable_db,
        }
        if any(v is None for pair in self.unrepeatable_db.values() for v in pair):
            out["why_a_repeatability_figure_can_be_absent"] = WHY_NO_REPEATABILITY_FIGURE
        if self.grounds:
            out["decided_by"] = self.grounds
            out["why_decided_by"] = WHY_DECIDED_BY
        if self.measured:
            out["measured"] = self.measured
            out["why_measured"] = WHY_MEASURED
        if self.read_back:
            out["setting_read_back"] = self.read_back
            if self.read_back_why:
                out["why_read_back"] = self.read_back_why
        if any(" with " in n for n in self.heard_by + self.not_heard_by):
            out["why_the_state_is_in_the_name"] = WHY_STATE_IN_THE_NAME
        if self.superseded_by:
            out["superseded_by"] = self.superseded_by
            out["why_superseded"] = WHY_SUPERSEDED
        if self.did_not_reproduce:
            out["did_not_reproduce"] = self.did_not_reproduce
            out["why_did_not_reproduce"] = WHY_DID_NOT_REPRODUCE
        if self.discounted:
            out["set_aside"] = self.discounted
            out["why_set_aside"] = WHY_SCATTER_NOT_A_GATE
        if self.left_out:
            out["left_out_of_the_gesture"] = self.left_out
            out["why_left_out"] = WHY_LEFT_OUT
        if self.caveat:
            out["how_it_had_to_be_asked"] = self.caveat
        return out


WHY_STATE_IN_THE_NAME = (
    "A stimulus asked in a prepared state is named with that state, because it is not the same "
    "question as the same stimulus asked as the unit powers up. Two runs differing only in what "
    "was written first would otherwise share one name, and the record would either drop one of "
    "them or report the pair as a verdict that failed to reproduce -- when what they are is two "
    "verdicts, each holding in its own state."
)


def _state(record: dict) -> str:
    """What was written before the run, as the suffix that keeps its name apart."""
    prepared = record.get("prepared") or []
    if not prepared:
        return ""
    return " with " + ", ".join(f"{e.get('address')} = {e.get('bytes')}" for e in prepared)


def _entries(record: dict) -> dict[str, dict]:
    state = _state(record)
    return {str(e.get("stimulus_name", "")) + state: e for e in record.get("by_stimulus") or []}


def read_one(record: dict) -> AddressVerdict | None:
    """One record into one verdict. None when it asked nothing."""
    entries = _entries(record)
    if not entries:
        return None
    return AddressVerdict(
        address=str(record.get("address", "")),
        values=list(record.get("values") or []),
        audible=bool(record.get("audible")),
        heard_by=[n + _state(record) for n in record.get("heard_by") or []],
        not_heard_by=[n + _state(record) for n in record.get("not_heard_by") or []],
        inconclusive_under=[n + _state(record) for n in record.get("inconclusive_under") or []],
        unrepeatable_db={
            name: list(e.get("each_setting_unrepeatable_db") or []) for name, e in entries.items()
        },
        grounds={name: _grounds(e) for name, e in entries.items()},
        measured={name: _measured(e) for name, e in entries.items()},
        read_back=list(record.get("setting_read_back") or []),
        read_back_why=str(record.get("why_read_back") or ""),
        left_out=dict(record.get("left_out_of_the_gesture") or {}),
    )


_CHANNELS = (
    ("shape", "changed_the_shape"),
    ("level", "changed_the_level"),
    ("repeatability", "changed_the_repeatability"),
)


def _grounds(entry: dict) -> list[str]:
    """Which of the three channels carried this stimulus's verdict."""
    return [name for name, key in _CHANNELS if entry.get(key)]


#: What a verdict was decided from, in the order the decision reads them.
_MEASURED = (
    "takes_per_setting",
    "same_setting_residual_db",
    "across_setting_residual_db",
    "same_setting_level_db",
    "across_setting_level_db",
    "across_setting_level_is_a_lower_bound",
    "margin_db",
    "noise_floor_db",
)


def _measured(entry: dict) -> dict:
    """The quantities behind one stimulus's verdict, absent keys left out.

    Left out rather than filled with a placeholder: a record written before a
    figure existed did not measure it, and a null standing in for that is a
    number a reader would compare against the others.
    """
    return {key: entry[key] for key in _MEASURED if key in entry}


WHY_DID_NOT_REPRODUCE = (
    "This address was asked more than once under this stimulus and did not answer the same way "
    "each time, so the verdict under it is withdrawn rather than carried. It is not settled by "
    "which run was later or by which answer occurred more often: a difference that appears in "
    "one asking and not in another has not been shown to be a property of the parameter, and "
    "the figures of every run stay in the record because how far apart they were is the whole "
    "of what the disagreement says. An address audible under some other stimulus is still "
    "audible; what is withdrawn is this stimulus's answer."
)


WHY_SUPERSEDED = (
    "Another record in this archive asked this address and its verdict is the one to read. This "
    "one is kept because it was measured and because what a run found is not undone by a later "
    "run finding more -- but the two disagree, and a reader landing here would otherwise take "
    "this answer for the archive's. Which method separates them is stated in the method of each "
    "record rather than asserted here: the named record says how it asked, and the difference "
    "between that and the sentence above is the whole of why it answers and this does not."
)


def superseded(found: list, by: dict[str, dict]) -> list:
    """Name, on each address, the record that answers for it instead.

    `by` maps an address to the naming record and its verdict there. Applied
    after the passes are joined, since what supersedes an address is a fact about
    the archive rather than about any one of this record's passes.
    """
    for verdict in found:
        if verdict is not None and verdict.address in by:
            verdict.superseded_by = by[verdict.address]
    return found


def _combine(first: AddressVerdict, second: AddressVerdict) -> AddressVerdict:
    """Two askings of one address into one verdict, disagreements named.

    A directory holding two records for an address holds two askings of it, and
    keeping whichever sorted first would drop a measurement without anything
    failing. Stimuli the two runs do not share simply join. A stimulus they both
    asked has to agree with itself: where it does not, it is withdrawn from both
    the heard and the unheard lists and named, since a verdict that did not
    reproduce is neither.
    """
    heard = set(first.heard_by) | set(second.heard_by)
    unheard = set(first.not_heard_by) | set(second.not_heard_by)
    disagreed = sorted(
        (heard & unheard) | set(first.did_not_reproduce) | set(second.did_not_reproduce)
    )
    keep = lambda names: [n for n in names if n not in disagreed]  # noqa: E731
    joined = AddressVerdict(
        address=first.address,
        values=first.values or second.values,
        audible=False,
        heard_by=keep(sorted(heard)),
        not_heard_by=keep(sorted(unheard)),
        inconclusive_under=sorted(set(first.inconclusive_under) | set(second.inconclusive_under)),
        # The later asking's figures do not replace the earlier one's: both are
        # indexed by stimulus, and where the same stimulus was asked twice the
        # run that disagreed is exactly the one a reader needs to see.
        unrepeatable_db={**first.unrepeatable_db, **second.unrepeatable_db},
        grounds={**first.grounds, **second.grounds},
        measured={**first.measured, **second.measured},
        read_back=first.read_back + second.read_back,
        read_back_why=first.read_back_why or second.read_back_why,
        did_not_reproduce=disagreed,
        left_out={**first.left_out, **second.left_out},
    )
    joined.audible = bool(joined.heard_by)
    return joined


def survey(root: str | Path) -> dict[str, AddressVerdict]:
    """Every record under a directory, keyed by the address it holds.

    Two records naming one address are two askings of it and are combined, not
    deduplicated: a run asked again writes a second file, and keeping only the
    first would answer with half the evidence and say nothing about the half it
    dropped.
    """
    out: dict[str, AddressVerdict] = {}
    for path in sorted(Path(root).glob("*.json")):
        found = read_one(json.loads(path.read_text()))
        if found is None or not found.address:
            continue
        prior = out.get(found.address)
        out[found.address] = _combine(prior, found) if prior is not None else found
    return out


WHY_ASKED_IN = (
    "A system effect is reached only through a part's send to it, so a block of effect "
    "parameters asked as the unit powers up answers about the send rather than about the "
    "addresses. What each pass raised first is therefore part of every verdict it produced, and "
    "a null taken without it is bounded by a state the record would otherwise not carry. Read "
    "back from the contrast records rather than written beside them, so it says what the runs "
    "did and not what the driver meant to do -- listed per distinct state, since a pass whose "
    "records disagree about it is two passes wearing one name."
)


def states(root: str | Path) -> list[dict]:
    """The distinct states a directory's records were asked in.

    One entry per distinct preparation, carrying the stimuli asked under it and
    how many addresses it covered. A pass that prepared nothing is a state too --
    the one the unit powers up in -- and says so rather than being left out.
    """
    seen: dict[tuple, dict] = {}
    for path in sorted(Path(root).glob("*.json")):
        record = json.loads(path.read_text())
        if not record.get("by_stimulus"):
            continue
        prepared = tuple(
            f"{e.get('address')} = {e.get('bytes')}" for e in record.get("prepared") or []
        )
        stimuli = tuple(str(e.get("stimulus_name", "")) for e in record["by_stimulus"])
        entry = seen.setdefault(
            (prepared, stimuli),
            {"prepared": list(prepared), "stimuli": list(stimuli), "addresses": 0},
        )
        entry["addresses"] += 1
    return list(seen.values())


#: Why a record under the directory can be one the plan does not name.
WHY_OUTSIDE_THE_PLAN = (
    "A run keeps its control beside its verdicts, which is what makes its nulls readable, and "
    "the control is usually an address in another block. Folding one into the count would make "
    "the block a byte wider than it is and one verdict better than it earned. Named rather than "
    "dropped: a record the plan does not ask for is either a control or a mistake, and which of "
    "those it is belongs to the reader."
)


def split_by_plan(found: dict[str, AddressVerdict], planned: dict) -> tuple[dict, list[str]]:
    """The records the plan asked for, and the addresses of the ones it did not."""
    asked = {a["address"] for a in planned.get("ask", [])}
    for entry in planned.get("cannot_be_asked", []):
        # Written either as the address with its reason beside it or as the two
        # in one string, depending on which stage wrote the plan.
        asked.add(entry["address"] if isinstance(entry, dict) else str(entry).split(" (")[0])
    kept = {address: v for address, v in found.items() if address in asked}
    return kept, sorted(set(found) - set(kept))


def _steadier(pair: list) -> float | None:
    """The better of a setting pair's two repeatability figures, if it has two.

    Better is the more negative: these are how far what fails to repeat sits
    below the setting's own signal, so a smaller number is a steadier take.
    """
    usable = [v for v in pair if isinstance(v, int | float)]
    return min(usable) if len(usable) == 2 else None


def scatter_not_a_gate(name: str, rescue: AddressVerdict, plain: AddressVerdict | None) -> bool:
    """Whether a verdict resting on the repeatability channel alone is scatter.

    A gate leaves the blocked setting repeating as the plain note's takes do. Two
    draws from a free-running phase do not, however wide the gap between them, so
    the plain pass's own figure for the same address is what separates the two.

    Keyed on the channel the verdict came from rather than on which stimulus
    produced it. Naming the modulation gesture was the narrower rule and it was
    wrong: any stimulus that plays more than one voice has a relative phase the
    trigger cannot fix, and this unit's trigger scatter is wider than a period of
    the note being played. Asked across three part blocks that hold the same
    parameters, the name-based rule let through sixteen verdicts, every one of
    them witnessed by a single stimulus, on a different set of addresses in each
    block -- which is what a parameter cannot do and a coin can.
    """
    if name not in rescue.heard_by:
        return False
    # A change of shape or of level is a measurement of the sound itself and
    # stands on its own. Only a verdict with nothing but the repeatability
    # channel behind it needs the steadier setting to be steady in absolute
    # terms before the gap between the two means anything.
    if rescue.grounds.get(name) != ["repeatability"]:
        return False
    if plain is None:
        return True
    here = _steadier(rescue.unrepeatable_db.get(name) or [])
    there = _steadier(plain.unrepeatable_db.get("struck") or [])
    if here is None or there is None:
        return True
    return bool(here > there + STEADY_WITHIN_DB)


def join(
    plain: dict[str, AddressVerdict],
    gesture: dict[str, AddressVerdict],
    polyphony: dict[str, AddressVerdict] | None = None,
) -> list:
    """One verdict per address over every pass, the plain note answering first.

    The two rescues are peers of each other and neither is a peer of the plain
    note: both are spent only where it heard nothing, and an address either of
    them hears is audible. They are kept apart here rather than merged into one
    pass because only the gesture's verdict is narrowed by the state it was
    asked in, and only the gesture carries a modulator to be discounted.
    """
    polyphony = polyphony or {}
    out = []
    for address in sorted(set(plain) | set(gesture) | set(polyphony)):
        first = plain.get(address)
        second, third = gesture.get(address), polyphony.get(address)
        rescued = [r for r in (second, third) if r is not None]
        if first is not None and first.audible:
            out.append(first)
            continue
        if not rescued:
            out.append(first)
            continue
        merged = AddressVerdict(
            address=address,
            values=next((r.values for r in rescued if r.values), first.values if first else []),
            audible=any(r.audible for r in rescued),
            heard_by=[n for r in rescued for n in r.heard_by],
            not_heard_by=(first.not_heard_by if first else [])
            + [n for r in rescued for n in r.not_heard_by],
            inconclusive_under=(first.inconclusive_under if first else [])
            + [n for r in rescued for n in r.inconclusive_under],
            unrepeatable_db={
                **(first.unrepeatable_db if first else {}),
                **{k: v for r in rescued for k, v in r.unrepeatable_db.items()},
            },
            # Merged the same way as the figures they explain. Left behind, a
            # rescued address would carry its stimuli's numbers and none of the
            # grounds for them, which reads as a verdict with no channel.
            grounds={
                **(first.grounds if first else {}),
                **{k: v for r in rescued for k, v in r.grounds.items()},
            },
            measured={
                **(first.measured if first else {}),
                **{k: v for r in rescued for k, v in r.measured.items()},
            },
            # The plain pass and each rescue read their own settings back, and a
            # rescue that found nothing still proves its writes took. Keeping the
            # plain pass's is not enough: an address answered only by a gesture
            # would carry a control taken in a different run from its verdict.
            read_back=(first.read_back if first else [])
            + [b for r in rescued for b in r.read_back],
            superseded_by=(first.superseded_by if first else {})
            or next((r.superseded_by for r in rescued if r.superseded_by), {}),
            did_not_reproduce=sorted(
                set(first.did_not_reproduce if first else [])
                | {n for r in rescued for n in r.did_not_reproduce}
            ),
            read_back_why=next(
                (r.read_back_why for r in (first, *rescued) if r is not None and r.read_back_why),
                "",
            ),
            left_out={
                **(first.left_out if first else {}),
                **{k: v for r in rescued for k, v in r.left_out.items()},
            },
        )
        for rescue in rescued:
            for name in list(rescue.heard_by):
                if not scatter_not_a_gate(name, rescue, first):
                    continue
                merged.discounted.append(name)
                merged.heard_by = [n for n in merged.heard_by if n != name]
                merged.inconclusive_under = merged.inconclusive_under + [name]
        merged.audible = bool(merged.heard_by)
        out.append(merged)
    return out


WHY_BALANCE_COUNTS = (
    "A parameter that moves signal between the two channels is invisible to a comparison made "
    "in one of them, and does not read as nothing there: a balance landing somewhere new on "
    "each take reads as the unit failing to repeat itself. So a balance the same run measured "
    "moving counts as the parameter having reached the signal path, which is what audible means "
    "here, and the stimulus it was found under carries the route it was found by."
)


def with_balance(found: list, measured: dict) -> list:
    """Fold a balance record's verdicts into the addresses they were taken on.

    Keyed by the name the takes were saved under, which is the address with its
    spaces turned to dashes -- the same name the driver gave the directory and
    the record.
    """
    by_address = {}
    for entry in measured.get("runs", []):
        if entry.get("moved_between_settings") or entry.get("did_not_repeat_within_a_setting"):
            address = str(entry.get("name", "")).replace("-", " ").upper()
            by_address.setdefault(address, []).append(str(entry.get("stimulus_name", "")))
    for verdict in found:
        if verdict is None or verdict.address not in by_address:
            continue
        for stimulus in by_address[verdict.address]:
            name = f"{stimulus} (balance)"
            if name not in verdict.heard_by:
                verdict.heard_by.append(name)
        verdict.audible = True
        verdict.not_heard_by = [
            n for n in verdict.not_heard_by if n not in by_address[verdict.address]
        ]
        verdict.inconclusive_under = [
            n for n in verdict.inconclusive_under if n not in by_address[verdict.address]
        ]
    return found


def with_the_plans_caveats(found: list, planned: dict) -> list:
    """Put each address's caveat from the plan onto its verdict.

    The plan carries one where an address could not be asked in the ordinary
    way -- over a guessed span, because the write probe never established what it
    accepts. Nothing else in the record says so, and such a null is the weakest
    one in the block: it cannot separate an address that ignored the write from
    one that took it and reached nothing.
    """
    caveats = {a["address"]: a.get("caveat", "") for a in planned.get("ask", [])}
    for verdict in found:
        if verdict is not None and caveats.get(verdict.address):
            verdict.caveat = caveats[verdict.address]
    return found


def against_plan(found: list, planned: dict) -> dict:
    """What the block holds against what the plan said it should.

    A directory of records answers about the records. The plan is what says how
    many addresses the block has, so an address that was never asked shows as
    missing rather than as absent.
    """
    asked = {a["address"] for a in planned.get("ask", [])}
    answered = {f.address for f in found if f is not None}
    return {
        "planned": len(asked),
        "answered": len(answered),
        "never_asked": sorted(asked - answered),
        "cannot_be_asked": planned.get("cannot_be_asked", []),
    }


def summarise(found: list, coverage: dict) -> str:
    piles: dict[str, list[str]] = {}
    lines = []
    for entry in found:
        if entry is None:
            continue
        piles.setdefault(entry.verdict, []).append(entry.address)
        heard = ", ".join(entry.heard_by) if entry.heard_by else "-"
        lines.append(f"  {entry.address}  {entry.verdict:32} {heard}")
    head = [
        f"{coverage['answered']} of {coverage['planned']} addresses answered, "
        f"{len(coverage['cannot_be_asked'])} that cannot be asked"
    ]
    tail = [f"  => {len(v)} {k}" for k, v in sorted(piles.items())]
    if coverage["never_asked"]:
        tail.append(f"  => {len(coverage['never_asked'])} never asked")
    return "\n".join(head + lines + tail)


__all__ = [
    "METHOD",
    "METHOD_ONE_PASS",
    "WHY_ASKED_IN",
    "WHY_BALANCE_COUNTS",
    "WHY_LEFT_OUT",
    "STEADY_WITHIN_DB",
    "WHY_SCATTER_NOT_A_GATE",
    "WHY_POLYPHONY_PASS",
    "WHY_TWO_PASSES",
    "AddressVerdict",
    "against_plan",
    "method_for",
    "states",
    "with_balance",
    "join",
    "scatter_not_a_gate",
    "read_one",
    "summarise",
    "survey",
    "with_the_plans_caveats",
]
