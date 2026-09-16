"""Reading every insertion effect's modulator off one held tone, one take each.

The sort beside this one needs a pair of takes per type and reads a delay between
them. This needs one take, reads the partials of the tone inside it, and answers
three questions the pair cannot: at what rate, of what kind, and how far.

**The rate is the take's own, not one that was written before it.** A projection
is walked across a grid and the largest is reported, so no byte's value enters the
reading and nothing is predicted. That matters here more than usual, because what
a rate byte means on this family is a claim made elsewhere in the archive, and a
record that took its rate from that claim could not then be used to test it.

**A take holding two cycles is enough, and a take holding none is not.** A line
fitted to a series needs cycles to count; a projection at a named rate does not,
because it is a matched filter. What it needs instead is separation, and two rates
closer together than one over the length read are not separated. That figure is
reported with every reading, and it is the floor a null here has to be read
against.

**Three gates, and each is drawn where a control stopped working.** A peak has to
stand over what the same projection returns on the take with nothing in the path,
rate by rate -- a held tone is not steady, and a statistic with nothing to subtract
the voice's own drift with reads that drift as a modulator. A level swing has to be
a comb rather than the whole voice rising and falling together, which the fit
cannot tell for itself: a plain level modulation is fitted as a comb explaining
more of its own series than any real comb does. And a fitted comb has to explain
its series, which is what fails when the rate is wrong.

**Nothing here is named and nothing is a delay unless it is one.** The comb's
excursion is the delay a two path comb would need to move its notches that far. On
a type whose notches come from an all-pass section rather than from a delay line
that is an equivalent, not a length, and a level series cannot say which a type is.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import partials, rates, takes

METHOD = (
    "One take per type of a held tone, with the type written and nothing else, so every "
    "parameter sat at the value the unit powers it up holding. Each order of the tone that "
    "cleared the threshold on the bypassed take was demodulated by its own frequency, leaving "
    "that partial's phase and its level over the take. A projection was walked across a grid of "
    "rates and the largest reported, separately for the phase and for the level. Where the level "
    "swung, the partials were compared with each other to tell a comb being swept from the whole "
    "voice rising and falling, and a two path comb was fitted forward to that level series with "
    "its rate free, started from every whole submultiple of both peaks."
)

WHY_ONE_CHANNEL = (
    "Which channel of the interface every take was read on, and the highest each channel reached "
    "across them. One channel for the run rather than the loudest of each take, because on this "
    "stage the input decides the answer twice over. An interface carries inputs the unit is not "
    "on which are not silent, and an idle input carries no modulation -- which is what a type "
    "without one also returns. And the unit's own two outputs carry a stereo modulator in "
    "antiphase, so a level series read on one is not the level series read on the other, and two "
    "types read on different ones are not comparable however loud both were. `loudest_elsewhere` "
    "names every take whose own loudest channel was not the one read."
)

WHY_THE_ORDERS_COME_FROM_THE_BYPASSED_TAKE = (
    "Which orders are read is decided by the take with nothing in the path and never by the take "
    "being read. An effect that adds sidebands raises its own take's quiet orders over the "
    "threshold and brings a different set of partials into the answer, so two takes compared "
    "against each other stop being comparable. The same confusion picked the wrong carrier once "
    "already."
)

WHY_A_GRID_AND_NOT_A_LINE = (
    "The rate is where a projection across a grid is largest, not where a line was fitted. A line "
    "is not read under three cycles of the series it is found in, and every type on a unit whose "
    "modulators power up under a hertz puts a nine second take under that floor. A projection at "
    "a named rate is a matched filter and needs no cycles counted; what it needs is separation, "
    "and two rates closer together than one over the length read are one peak here. That figure "
    "is `rates_are_separated_by` and it is what bounds the rates the projections report. The comb "
    "fit is not bounded by it, because it does not read a peak: it fits the rate, and the model "
    "is far sharper in it than the projection is -- a fiftieth of a hertz out at half a hertz, it "
    "explains 0.16 of a series it explains 0.91 of when placed. Where a peak and a fitted rate "
    "are both given, they are two readings and not one."
)

WHY_TWO_QUANTITIES = (
    "A swept delay turns a partial's phase in proportion to that partial's own frequency. An "
    "all-pass section turns every partial by the same angle. A level modulation turns none of "
    "them and moves the level instead. Reading the two quantities apart is what tells the three "
    "kinds apart; reading either alone returns a number for all three."
)

WHY_THE_PHASE_EXCURSION_IS_A_FLOOR = (
    "Where a delayed path is mixed against a direct one the composite's phase turns by far less "
    "than the delayed path's own, and not in proportion. Measured on this carrier by injection: a "
    "fifth wet turns a two millisecond sweep into five thousandths of a millisecond, while "
    "detection survives it. So an excursion read off the phase is a lower bound wherever the "
    "effect is not fully wet, and what it bounds is not stated by the number itself. The comb "
    "reading is the one that answers the mixed case, and a type answering to neither is reported "
    "as answering to neither."
)

WHY_THE_PARTIALS_MUST_DISAGREE = (
    "How well a forward comb fit explains its series is a gate against the wrong rate -- told a "
    "rate nothing is running at it falls from 0.91 to 0.05 -- and it is no gate at all against "
    "the wrong mechanism. A plain level modulation with no delay anywhere in it is fitted as a "
    "comb explaining 0.97 of its own series, which is more than any real comb injected here "
    "returns. What separates them is not a fit: a comb sweeps a notch, so a partial near it "
    "swings far more than one on a peak and the partials disagree in shape, while one envelope "
    "over the whole voice moves every partial by the same decibels at the same moment."
)

WHAT_THE_GATES_WERE_SET_AGAINST = {
    "stands_over_bypassed": (
        "What the same projection returns on the same material with nothing in the path, rate by "
        "rate. The record's second bypassed take came back at 1.1 in the phase and 1.3 in the "
        "level, against injected sweeps standing 7 to 3000, so the line sits in a gap both "
        "controls clear by their own margin. It is not a figure over the grid's own middle: this "
        "carrier drifts enough on its own to stand seventeen times its grid median at half a "
        "hertz, which any statistic with nothing to subtract reads as a modulator."
    ),
    "agrees_as_one_voice": (
        "Not a round number but where correlation changes sign. Every comb injected here came "
        "back between -0.05 and -0.37 and every level swing between 0.81 and 0.97."
    ),
    "comb_explains": (
        "Injected combs that returned their excursion exactly explained 0.628 and above, and the "
        "same fit told a rate nothing is running at explains 0.042 to 0.055. The line's place "
        "inside that gap is a choice, and the gap is narrower than those two figures suggest: "
        "swept over the sizes a chorus lives at, the fit returns the excursion to within a few "
        "per cent while what it explains falls to 0.64 wherever the two paths are mixed evenly. "
        "So a fit can be right and still be withheld here, and what the withholding means is "
        "that this reading could not tell a right answer from a wrong rate -- not that the type "
        "has no comb in it."
    ),
}
"""What each gate's number was set against, carried into the record beside the number.

A threshold published as a bare figure cannot be argued with, and these three decide
which types give up a quantity. Assembled here so a run that states the number also
states what would have moved it.

How much room the gate's own run left around it is not in here. It is measured by
`how_far_the_line_could_have_moved` off the run being written, because a figure typed
into a constant is right for the run it was read on and silently wrong for the next one.
"""


def how_far_the_line_could_have_moved(found: list[dict]) -> str:
    """How much room this run left around the gate, read off the run itself.

    Which side of the line a reading fell is not worth reporting: the line is what
    puts it there, so every withheld fit is below it by construction. What the run
    decides and the gate does not is the empty band between the highest withheld
    reading and the lowest published one. A wide band means the figure could have
    been put anywhere across it and this record would be identical; a narrow one
    means the choice carried types.
    """
    fits = [row["comb"] for row in found if (row.get("comb") or {}).get("explains") is not None]
    published = [f["explains"] for f in fits if f.get("excursion_ms") is not None]
    withheld = [f["explains"] for f in fits if f.get("excursion_ms") is None]
    if not published or not withheld:
        return (
            "Every fit on this run came out on one side of the line, so the run says nothing "
            "about how much room there was around it."
        )
    top, bottom = max(withheld), min(published)
    room = min(COMB_EXPLAINS - top, bottom - COMB_EXPLAINS)
    return (
        f"On this unit the most any withheld fit explained was {top:.3f} and the least any "
        f"published one did was {bottom:.3f}, so the line could have been put anywhere between "
        f"those two and this record would read the same. It is {room:.3f} from the nearer of "
        f"them. That is where the figures fell, not a property of the gate: a run whose "
        f"readings crowded the line would have to be read with the figure's exact place in mind."
    )


WHY_THE_EXCURSION_CARRIES_A_SPAN = (
    "A comb fit returns one excursion and the surface it was found on has a minimum wherever the "
    "notches line up, so the number alone says which minimum was deepest and not how much deeper. "
    "Beside each excursion is the range of excursions that explain the series to within a "
    "twentieth of its own spread, measured over every candidate this fit refined. A wide span "
    "settles the reading the one way that matters: the type has not been measured to a number, "
    "however much of its series the fit explains. "
    "What the span was read against is twenty-four injected combs of known excursion, over "
    "centres of three to thirteen milliseconds, excursions of half a millisecond to five, and "
    "both a third wet and an even mix, with no feedback anywhere so the model is exactly right "
    "and nothing in the material is outside it. The span held the injected excursion in seventeen "
    "of the twenty-four. Where it did not, the sweep is shallow and the two paths are mixed "
    "evenly -- a millisecond at half wet came back half again too large at two centres -- and one "
    "of those closed its span and still sat a fifth away from the answer. "
    "So a closed span is not a warrant. It bounds what this fit's own candidates could not tell "
    "apart, which is a smaller thing than how far the answer could be from the truth, and the two "
    "come apart exactly where a complete notch makes the level series a cusp rather than a curve."
)

WHY_AN_EQUIVALENT_AND_NOT_A_LENGTH = (
    "A comb's phase is the delay times the partial's frequency, so a phase swing converts to a "
    "delay excursion outright. That conversion is arithmetic and the interpretation is not: a "
    "phaser and a wah move notches with all-pass sections and have no delay line to be long, and "
    "a level series cannot tell that from a delay. The figure is what a two path comb would need; "
    "which types it is a delay for is not decided here."
)

STANDS_OVER_BYPASSED = 3.0
"""How far a peak must stand over the bypassed take at the same rate.

Against the bypassed take rather than against the middle of the reading's own grid.
A held tone is not steady, and this unit's carrier drifts enough on its own to put
a peak seventeen times its own grid median at half a hertz -- a fact about the voice
that any statistic with nothing to subtract it with reads as a modulator. The
figure a peak has to beat is what the same projection returns on the same material
with nothing in the path, and the record carries a second bypassed take so that
figure is measured rather than assumed to be one. On this unit it came back at 1.1
in the phase and 1.3 in the level, against injected sweeps standing 7 to 3000.
"""

AGREES_AS_ONE_VOICE = 0.0
"""Where a level swing stops being a comb and becomes the whole voice.

The reading is the least-agreeing pair of partials. Zero is not a round number
here, it is where correlation changes sign: partials on opposite sides of a
sweeping notch move against each other and cannot agree, while partials under one
envelope cannot disagree. Over the injections here every comb came back between
-0.05 and -0.37 and every level swing between 0.81 and 0.97, so both controls clear
the line by their own margin and neither approaches it.
"""

COMB_EXPLAINS = 0.60
"""How much of its level series a comb fit must explain before its excursion is read.

Every injected comb that returned its excursion exactly explained at least 0.628,
and the same fit told a rate nothing is running at explains 0.042 to 0.055. No
control sits between those, so the line's place inside the gap is a choice.
"""


def limits(floor: Floor, found: list[dict]) -> dict:
    """What bounds this reading, assembled beside the verdicts it bounds.

    Kept in one function with the record it goes into rather than written at each
    call site: a limitation left out of the next run's file does not fail, and a
    negative without a stated bound is not a result.

    Takes the run's own rows because one of the gates reports where this run's
    readings fell against it, which no constant can hold.
    """
    gates = {
        **WHAT_THE_GATES_WERE_SET_AGAINST,
        "comb_explains": (
            WHAT_THE_GATES_WERE_SET_AGAINST["comb_explains"]
            + " "
            + how_far_the_line_could_have_moved(found)
        ),
    }
    return {
        "why_the_orders_come_from_the_bypassed_take": WHY_THE_ORDERS_COME_FROM_THE_BYPASSED_TAKE,
        "why_a_grid_and_not_a_line": WHY_A_GRID_AND_NOT_A_LINE,
        "why_a_rate_at_the_edge_of_the_search_is_the_search": WHY_THE_EDGE_IS_NOT_A_RATE,
        "why_the_phase_and_the_level_are_read_apart": WHY_TWO_QUANTITIES,
        "why_the_excursion_off_the_phase_is_a_floor": WHY_THE_PHASE_EXCURSION_IS_A_FLOOR,
        "why_the_comb_needs_the_partials_to_disagree": WHY_THE_PARTIALS_MUST_DISAGREE,
        "why_an_equivalent_and_not_a_length": WHY_AN_EQUIVALENT_AND_NOT_A_LENGTH,
        "why_the_excursion_carries_a_span": WHY_THE_EXCURSION_CARRIES_A_SPAN,
        "stands_over_bypassed": STANDS_OVER_BYPASSED,
        "agrees_as_one_voice": AGREES_AS_ONE_VOICE,
        "comb_explains": COMB_EXPLAINS,
        "what_the_gates_were_set_against": gates,
        "orders_read": floor.held.orders,
        "rates_are_separated_by": round(floor.held.rates_are_separated_by, 4),
    }


def rows_of(root: str | Path, setting: re.Pattern, bypassed: str) -> dict:
    """The takes under a directory, split into the bypassed ones and the types.

    The files are the subject. A take store rewrites its manifest when it closes,
    so a directory captured in more than one pass keeps manifest rows only for the
    last of them -- and a record built from the short list reads exactly like a
    complete one. Which files the manifest had forgotten is reported.
    """
    root = Path(root)
    listed, files = takes.listing(root)
    controls, typed, skipped, collided = [], {}, [], {}
    for name in files:
        entry = listed.get(name, {})
        if bypassed in name or bypassed in str(entry.get("setting", "")):
            controls.append(name)
            continue
        _, found = takes.named_by(setting, entry, name)
        if found is None:
            skipped.append(str(entry.get("setting") or name))
            continue
        type_id = found.group("type").replace("-", " ").upper()
        # Two files naming one type is a pattern that is reading something other
        # than the type, and it is silent: the extra takes vanish into the first
        # one's key and the survey reports a short list as the whole set. It
        # happened here with an unanchored default, which read a stimulus's own
        # program number out of the file names the manifest had forgotten.
        collided.setdefault(type_id, []).append(name)
        typed.setdefault(type_id, name)
    return {
        "controls": controls,
        "types": typed,
        "manifest": takes.manifest_note(listed, files),
        "not_matching": takes.not_matching(skipped),
        "more_than_one_take_named": {
            type_id: names for type_id, names in sorted(collided.items()) if len(names) > 1
        },
    }


def body_of(path: str | Path, lead_s: float, hold_s: float, guard_s: float = 0.5, on: int = -1):
    """The part of a take the note is sounding through, on one named channel.

    A guard is cut from each end of the held stretch: the note's attack is not the
    steady tone the partials are read out of, and its release is not either.

    `on` names the channel, and a run names one for all its takes rather than
    letting each take answer from whichever of its own is loudest. This stage reads
    a modulation out of a take, so the input it is read from decides the answer
    twice over: an idle input carries no modulation, which is what a type without
    one also returns, and the unit's own two outputs carry a stereo modulator in
    antiphase, so a series read on one is not the series read on the other. Sixty
    seven takes of one survey answered on two different channels that way, and the
    record it produced said nothing about any of it.

    A negative `on` keeps the loudest channel of this take alone, which is what a
    single take asked on its own has to do -- there is no run to pick from.

    Returns the body, the rate, and which channel this take's own loudest was, so a
    caller can say which takes answered somewhere other than where it read them.
    """
    samples, rate = takes.read(path)
    frames = np.asarray(samples, dtype=np.float64)
    own = 0
    if frames.ndim > 1:
        own = int(np.argmax(np.sqrt(np.mean(np.square(frames), axis=0))))
    channels = takes.loudest(samples) if on < 0 else takes.channel(samples, on)
    whole = np.asarray(channels, dtype=np.float64)
    first = int((lead_s + guard_s) * rate)
    last = int((lead_s + hold_s - guard_s) * rate)
    return whole[first:last], rate, own


class Floor:
    """The bypassed take's own projections, which every other take is read against.

    Held as an object rather than recomputed because it is the same two arrays for
    every type in a survey, and because what a peak means here is entirely a
    question of what this take does at the same rate.
    """

    def __init__(self, body: np.ndarray, rate: int, *, carrier_hz: float, grid: np.ndarray):
        held = partials.read(body, rate, carrier_hz=carrier_hz)
        if held is None:
            raise ValueError(
                f"no order of {carrier_hz} Hz cleared the threshold in the bypassed take"
            )
        self.held = held
        self.keep = held.kept
        self.grid = grid
        self.phase = partials.project(held.phase, held.at, grid)
        self.level = partials.project(held.level_db, held.at, grid)


SEED_SUBMULTIPLES = 4
"""How many submultiples of each peak are offered to the comb fit as a start.

A peak in either series sits at a whole multiple of the modulator's rate and never
between two of them: a comb dragging several notches past a partial in one cycle
puts its largest at twice or three times the rate, and both were seen here on
injections of a known one. So the candidates are enumerated rather than searched
for, and which is right is settled by the fit's own residual. That is the whole
reason the seeds are a list: a threshold on the projection would be a second
invented number, and the model already has a criterion.
"""


def seeds_from(*peaks: dict) -> tuple[float, ...]:
    """Every rate a peak could be a whole multiple of, largest first."""
    found = {
        round(peak["hz"] / k, 4)
        for peak in peaks
        for k in range(1, SEED_SUBMULTIPLES + 1)
        if peak["hz"] / k > 0.0
    }
    return tuple(sorted(found, reverse=True))


WHY_THE_EDGE_IS_NOT_A_RATE = (
    "Two ways a rate here can be the reading's own limit rather than the unit's, and both are "
    "stated per reading rather than left for a reader to work out. A projection walked across a "
    "grid reports where it was largest, and largest *at an end of the grid* means the largest "
    "thing seen was at the boundary -- the maximum may be outside it, and a held tone drifting "
    "slowly over the take puts its energy exactly there. `rates_are_separated_by` bounds how "
    "close two rates can be and says nothing about this. Separately, a take holds only so many "
    "cycles of a slow rate: `slowest_measurable_hz` is the rate two cycles of which fill the "
    "part of the take that sounded, and the same figure computed the same way elsewhere in this "
    "harness was measured to matter -- an injected sweep at 0.15 Hz, one cycle of which does not "
    "fit in seven seconds, came back as 0.28. A reading on either count is reported with the "
    "figure that bounds it and is not withheld, because which of them disqualifies a reading is "
    "a judgement and this record does not make judgements."
)


def _on_the_edge(at_hz: float | None, grid: np.ndarray, slowest: float) -> list[str] | None:
    """Whether this rate is the search's own limit, and by which of the two counts."""
    if at_hz is None:
        return None
    against = []
    if abs(at_hz - float(grid[0])) <= 1e-9:
        against.append("the lowest rate searched")
    if abs(at_hz - float(grid[-1])) <= 1e-9:
        against.append("the highest rate searched")
    if at_hz <= slowest:
        against.append("at or under the slowest rate this take could carry two cycles of")
    return against or None


def measure_one(
    body: np.ndarray,
    rate: int,
    *,
    carrier_hz: float,
    floor: Floor,
    fit_comb: bool = True,
) -> dict | None:
    """Everything one take says about what is moving under its tone."""
    held = partials.read(body, rate, carrier_hz=carrier_hz, keep=floor.keep)
    if held is None:
        return None
    grid = floor.grid
    phase_peak = partials.peak(held.phase, held.at, grid, floor.phase)
    level_peak = partials.peak(held.level_db, held.at, grid, floor.level)
    phase_moves = phase_peak["stands_over_bypassed"] >= STANDS_OVER_BYPASSED
    level_moves = level_peak["stands_over_bypassed"] >= STANDS_OVER_BYPASSED
    # Each quantity is read at the rate its own series shows, never at the other's.
    # The comb fit is sharp in the rate -- a fiftieth of a hertz out, it explains
    # 0.16 of a series it explains 0.91 of when placed -- and the two peaks do not
    # have to land in the same grid bin. Where they differ by more than this take
    # can separate, that is a finding and it is reported rather than averaged away.
    at_hz = phase_peak["hz"] if phase_moves else level_peak["hz"]
    apart = abs(phase_peak["hz"] - level_peak["hz"])
    slowest = round(rates.CYCLES_WANTED / held.sounded_s, 4)
    out = {
        "orders": held.orders,
        "sounded_s": round(held.sounded_s, 4),
        "rates_are_separated_by": round(held.rates_are_separated_by, 4),
        "slowest_measurable_hz": slowest,
        "read_at_hz": at_hz,
        "stands_on_the_edge_of_the_search": _on_the_edge(at_hz, grid, slowest),
        "the_two_peaks_are_separated": bool(
            phase_moves and level_moves and apart > held.rates_are_separated_by
        ),
        "phase": {"peak": phase_peak, **partials.excursion(held, phase_peak["hz"])},
        "level": {"peak": level_peak, "partials": partials.together(held)},
        "moves": bool(phase_moves or level_moves),
    }
    agreement = (out["level"]["partials"] or {}).get("lowest_agreement")
    out["the_level_swing_is"] = (
        None
        if agreement is None or not level_moves
        else ("a comb being swept" if agreement <= AGREES_AS_ONE_VOICE else "the whole voice")
    )
    if fit_comb and out["the_level_swing_is"] == "a comb being swept":
        fitted = partials.comb(body, rate, *seeds_from(phase_peak, level_peak))
        # The excursion is withheld rather than reported small when the fit does
        # not account for the series it was taken on: a fit at a rate the take
        # does not show still returns a number.
        if (fitted.get("explains") or 0.0) < COMB_EXPLAINS:
            fitted = {
                **fitted,
                "excursion_ms": None,
                "why": "the fit does not account for the series it was taken on, so what it "
                "returned is not read",
            }
        out["comb"] = fitted
    return out


def controls_from(
    body: np.ndarray,
    rate: int,
    *,
    carrier_hz: float,
    floor: Floor,
    others: list[tuple[str, np.ndarray]],
    at_hz: float,
    ladder: tuple[tuple[float, float], ...],
) -> dict:
    """What this material returns with nothing in it, and with a known sweep in it.

    Three things, and the record needs all of them. The bypassed take against
    itself, which is one by construction and says so. A second bypassed take,
    which is the only figure here that is not arithmetic -- it is what two takes of
    the same nothing actually return. And a ladder of sweeps of a size somebody
    chose, including the fully wet rung, which has no comb in it and which the comb
    fit must refuse rather than answer.
    """
    said = {
        "at_hz": at_hz,
        "other_takes_with_nothing_in_the_path": [
            {
                "take": name,
                **measure_one(other, rate, carrier_hz=carrier_hz, floor=floor, fit_comb=False),
            }
            for name, other in others
        ],
        "a_level_swing_with_no_delay_in_it": [],
        "injected": [],
    }
    # A level modulation and nothing else, which is the mechanism the comb fit
    # cannot refuse on its own. Injected at the depth this unit's own level-moving
    # types show, so the control is the case that actually arises.
    t = np.arange(body.size, dtype=np.float64) / rate
    for depth in (0.25, 0.55):
        shaken = body * (1.0 + depth * np.sin(2.0 * np.pi * at_hz * t))
        reading = measure_one(shaken, rate, carrier_hz=carrier_hz, floor=floor)
        said["a_level_swing_with_no_delay_in_it"].append({"depth": depth, **reading})
    for mix, excursion in ladder:
        wet = partials.swept(body, rate, 2.0 + excursion / 2.0, excursion, at_hz, mix)
        reading = measure_one(wet, rate, carrier_hz=carrier_hz, floor=floor)
        said["injected"].append({"wet": mix, "asked_ms": excursion, **reading})
    return said


LADDER = ((1.00, 2.00), (0.50, 2.00), (0.20, 2.00), (0.20, 0.50), (0.05, 2.00))
"""Wet proportion and excursion peak to peak, in ms, injected into the unit's own
bypassed take. The fully wet rung is the one with no comb in it at all: what the
comb fit does there is what this route says when the other one is the one that
applies, and it has to be a refusal."""


WHY_THE_RATE_IS_READ_ON_BOTH_OUTPUTS = (
    "One channel decides a level series and it does not decide a rate, so the rate is read on "
    "both of the unit's outputs and both are reported. A stereo modulator runs the two outputs "
    "against each other, which makes a level read on one of them incomparable with a level read "
    "on the other -- that is why everything else here is read on the single channel named above. "
    "A rate is a different quantity and the two outputs were measured to disagree about it: over "
    "one survey of sixty-five types, five returned the lowest rate anybody searched on the "
    "channel the run reads and a rate on the other, and four returned one output at twice the "
    "other, which is the projection's own largest landing on a different multiple of the same "
    "modulator rather than two modulators. Reported side by side rather than resolved here, "
    "because choosing between them is a reading and this record does not make readings. The "
    "second output is measured for its rates alone: no comb is fitted to it and no level series "
    "is taken from it, both of which belong to the one channel."
)


PAIRED_WITHIN_DB = 30.0
"""How far below the channel that was read the unit's other output may reach.

Not "the loudest channel that is not the one read": on this rig the interface's
unused inputs sit around thirty decibels under the unit's own pair and are not
silent, so that rule names an input nothing is plugged into as the second output
and reads a rate out of its noise. The unit's two outputs reached within 1.4 dB
of each other across the survey, so any bar between those two figures separates
them; thirty is set where it is because it is the gap that was measured and not
the margin that was wanted.
"""


def _the_other_output(on: int, reached: list[float]) -> int | None:
    """The unit's second output, named by which channel of the interface it reached on.

    Returned as None where no other channel is close enough to be one, so a mono
    rig -- or a stereo one with a lead out -- produces a record that says the
    question was not asked rather than one that answers it off an idle input.
    """
    order = sorted(range(len(reached)), key=lambda i: -reached[i])
    for index in order:
        if index != on and reached[index] >= reached[on] - PAIRED_WITHIN_DB:
            return index
    return None


def _rate_on(take, *, carrier_hz: float, floor: Floor, channel: int) -> dict:
    """What the other output says about the rate, and nothing else.

    The comb is not fitted here and the level series is not published from here.
    What this answers is whether the channel the run reads could carry the rate at
    all, which is a question the one channel cannot be asked about itself.
    """
    body, at = take
    reading = measure_one(body, at, carrier_hz=carrier_hz, floor=floor, fit_comb=False)
    if reading is None:
        return {"channel": channel, "why": "nothing to read on this output"}
    return {
        "channel": channel,
        "read_at_hz": reading["read_at_hz"],
        "moves": reading["moves"],
        "slowest_measurable_hz": reading["slowest_measurable_hz"],
        "stands_on_the_edge_of_the_search": reading["stands_on_the_edge_of_the_search"],
        "phase": {"peak": reading["phase"]["peak"]},
        "level": {"peak": reading["level"]["peak"]},
    }


def survey(
    root: str | Path,
    *,
    carrier_hz: float,
    setting: re.Pattern,
    bypassed: str = "bypassed",
    lead_s: float = 0.6,
    hold_s: float = 8.0,
    grid_hz: tuple[float, float] = (0.20, 8.00),
    step_hz: float = 0.01,
    control_at_hz: float = 0.45,
    progress=None,
) -> dict:
    """Read every type whose take is under this directory, against one bypassed one."""
    where = rows_of(root, setting, bypassed)
    if not where["controls"]:
        raise FileNotFoundError(f"no take under {root} named by --bypassed {bypassed!r}")
    grid = np.arange(grid_hz[0], grid_hz[1] + step_hz / 2, step_hz)
    root = Path(root)

    every = sorted({*where["controls"], *where["types"].values()})
    on, reached = takes.channel_reaching(root, every)
    beside = _the_other_output(on, reached)
    elsewhere: list[str] = []

    def read(name: str, channel: int | None = None):
        here = on if channel is None else channel
        body, rate, own = body_of(root / name, lead_s, hold_s, on=here)
        if channel is None and own != on:
            elsewhere.append(name)
        return body, rate

    body, rate = read(where["controls"][0])
    floor = Floor(body, rate, carrier_hz=carrier_hz, grid=grid)
    others = [(name, read(name)[0]) for name in where["controls"][1:]]

    # The unit's other output gets its own floor, because a floor built on one
    # channel of the bypassed take does not bound a reading made on the other.
    aside = None
    if beside is not None:
        alongside, at = read(where["controls"][0], beside)
        aside = Floor(alongside, at, carrier_hz=carrier_hz, grid=grid)

    found = []
    for type_id, name in sorted(where["types"].items()):
        take, _ = read(name)
        reading = measure_one(take, rate, carrier_hz=carrier_hz, floor=floor)
        row = {"type": type_id, "take": name, **(reading or {"why": "nothing to read"})}
        if aside is not None:
            row["also_on_the_other_output"] = _rate_on(
                read(name, beside), carrier_hz=carrier_hz, floor=aside, channel=beside
            )
        found.append(row)
        if progress:
            progress(row)

    return {
        "carrier_hz": carrier_hz,
        "channel": {
            "read": on,
            "reached_db": [round(value, 1) for value in reached],
            "loudest_elsewhere": sorted(elsewhere),
            "why": WHY_ONE_CHANNEL,
            "the_other_output": beside,
            "why_the_rate_is_read_on_both": WHY_THE_RATE_IS_READ_ON_BOTH_OUTPUTS,
            "paired_within_db": PAIRED_WITHIN_DB,
        },
        "grid_hz": list(grid_hz),
        "grid_step_hz": step_hz,
        "bypassed_take": where["controls"][0],
        "method": METHOD,
        "limits": limits(floor, found),
        "controls": controls_from(
            body,
            rate,
            carrier_hz=carrier_hz,
            floor=floor,
            others=others,
            at_hz=control_at_hz,
            ladder=LADDER,
        ),
        "types": found,
        "takes": {
            **where["manifest"],
            "not_matching": where["not_matching"],
            "more_than_one_take_named": where["more_than_one_take_named"],
        },
    }


def moving(found: list[dict]) -> list[str]:
    return [row["type"] for row in found if row.get("moves")]


def named_an_excursion(found: list[dict]) -> list[dict]:
    """The types whose comb fit survived every gate, which is what this stage adds.

    Each carries the span of excursions that explain the series about as well as
    the winner does. Reading the winner without it is reading a best fit as though
    it were the only one, and on this material that is usually wrong: the span
    opens to a factor of two or more wherever the notches do not sweep far enough
    to shape the series.

    `settled_to_one_excursion` says the span closed and does not say the answer is
    right. On injections where the model is exactly right it closed on a figure a
    fifth away from the truth once, in the corner where a shallow sweep meets an
    even mix. What the span bounds is this fit's own ambiguity.
    """
    out = []
    for row in found:
        fitted = row.get("comb") or {}
        if fitted.get("excursion_ms") is not None:
            span = fitted.get("excursions_that_explain_it_about_as_well_ms")
            out.append(
                {
                    "type": row["type"],
                    "hz": fitted["hz"],
                    "excursion_ms": fitted["excursion_ms"],
                    "excursions_that_explain_it_about_as_well_ms": span,
                    "settled_to_one_excursion": bool(
                        span is not None and span[1] <= 1.1 * max(span[0], 1e-9)
                    ),
                    "mix": fitted.get("mix"),
                    "explains": fitted.get("explains"),
                    "closer_to": fitted.get("closer_to"),
                    "lowest_agreement": (row["level"]["partials"] or {}).get("lowest_agreement"),
                }
            )
    return out
