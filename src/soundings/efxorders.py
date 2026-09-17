"""Reading what one setting did to the orders of a held tone, order by order.

The band reading beside this one measures how much energy came back in each third
of an octave. That is the right question for a filter and the wrong one for a
nonlinearity: a stage that adds harmonics puts them at whole multiples of the
carrier, and a band set wide enough to hold a whole profile puts several orders in
one band and the skirts of one order in the band beside it. What comes out is a
tilt, and a tilt is what a filter and a curve both look like once the orders have
been averaged together.

So this reads the orders themselves, and reads each of them **under the first**.

**A ratio and not a level, which is the whole point.** What a memoryless curve does
to one partial is fixed by the shape of the curve and by how far up it the signal
sits, and it is the same whatever happens to the output afterwards. So the level of
the orders under the first survives any gain after the stage, and two settings that
came back at different loudnesses are still comparable. That is measured on this
family rather than assumed: the output level byte of a drive type was swept
twenty-six decibels and the orders under the first did not move.

**What it cannot separate is what got in.** A stimulus that is not one partial
arrives with orders of its own, and a curve fed two partials returns products
between them as well as harmonics of each. A product landing on a whole multiple of
the carrier is counted here as that order, and nothing in these figures says which
it was. How much of the stimulus is its own fundamental is what bounds that, and it
belongs beside the stimulus in the record rather than being inferred from it.

**No curve, no order of polynomial, no clipper.** Which nonlinearity these orders
belong to, whether it is odd or even, where it bends -- that is a fit across
settings and types, and the fit is not made here.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import partials, takes

VALUE = "value"
"""The named group a `--setting` pattern has to capture.

The same contract the band reading sets: a run names its takes however its own
question needed, so the pattern that pulls the setting back out is in the
invocation and therefore in the record, where it can be checked rather than
trusted.
"""

ORDERS = 8
"""How many orders are read, counting the first.

A count and not a bound on the stage: energy the curve put above the eighth order
is not in these figures, so a reading of a stage whose products are mostly high
reads low here and says so through `settled_at_the_top_db` rather than through a
smaller number nobody can see the edge of.
"""

WITHIN = 0.03
"""How far from where a note is nominally printed the carrier is looked for.

The carrier is measured rather than taken from the note number: a unit's own tuning
moves it, and a boxcar aimed a few tenths of a hertz off reads every order through
the skirt of its own window. Three per cent is wider than any tuning offset a GS
unit reaches and narrower than half the distance to the next order.
"""

QUESTION = (
    "What one setting did to the harmonic orders of a held tone -- the level of each "
    "order under the first -- at each setting swept."
)

#: The two methods, whole rather than a shared opening and two endings. A record
#: carries the sentence it was written with, and a sentence assembled from pieces
#: is one no reader can find in the source that produced it -- which is the thing
#: `tests/test_report_prose.py` is there to stop.
METHOD_OVER_THE_STRETCH = (
    "A tone was held through the effect at one setting and each of the first orders of "
    "its carrier was demodulated by that order's own frequency. The carrier itself was "
    "measured from each take rather than taken from the note that was played, because a "
    "demodulation aimed off a partial reads it through the skirt of its own filter. "
    "The filter after the mixer is the whole of the held stretch, windowed -- a matched "
    "filter for a partial that stands still, and the narrowest one these takes can give. "
    "The level reported per order is that filter's output, in decibels under the first "
    "order of the same take."
)

METHOD_OVER_A_BOXCAR = (
    "A tone was held through the effect at one setting and each of the first orders of "
    "its carrier was demodulated by that order's own frequency. The carrier itself was "
    "measured from each take rather than taken from the note that was played, because a "
    "demodulation aimed off a partial reads it through the skirt of its own filter. "
    "The filter after the mixer is a boxcar a named number of carrier periods long, "
    "which puts its nulls on the neighbouring orders and passes what sits between them. "
    "The level reported per order is the median of that filter's output over the held "
    "stretch, in decibels under the first order of the same take."
)

LIMITS = (
    "Every figure is the level of an order under the first order **of the same take**, "
    "so nothing here is a level and nothing here moves when something after the stage "
    "scales the output. That is what makes settings of different loudness comparable "
    "and it is also what this record cannot report: how loud the setting came back is "
    "in `heard_db` and nowhere in the orders. "
    "`floor_db` is the spread of the run's own repeats of one setting, order by order, "
    "and a difference inside it is not a reading. It is a range over a handful of takes "
    "rather than a bound, so a difference just outside it is not thereby a reading "
    "either, and how many takes drew it is in `reference`. A run with no repeats in it "
    "carries no floor at all: its figures still stand, because each is read against its "
    "own take's first order, but nothing says how small a difference between two of them "
    "can be told from the same setting asked twice. "
    "Only whole multiples of the carrier are read. A stimulus that is not one partial "
    "arrives with orders of its own and returns products between them, and a product on "
    "a whole multiple is counted as that order; what bounds this is how much of the "
    "stimulus is its own fundamental, which is stated with the stimulus and is not a "
    "figure this reading can produce. Energy the stage put between the orders is not "
    "here at all. "
    "An order quieter than what the same chain recorded with nothing played is the floor "
    "and not the order. That comparison is between levels and cannot be made between two "
    "ratios, which is why `orders_db` is beside `under_the_first_db` on every reading "
    "here and on every take in `silence`: the check is each order against the same order "
    "of the silence, and it is left for a reader to make rather than folded into a "
    "verdict. `floor_between_orders_db` is the same check made without another take at "
    "all -- the noise halfway to the next order, read through the same filter, in this "
    "take and at this moment -- and it is the tighter of the two wherever the chain was "
    "quieter than the silence take happened to be. "
    "`orders_turn_deg` is which way each order points against the carrier, with the "
    "take's own clock taken out by subtracting `k` times the first order's own angle. "
    "Nothing is fitted in it and it does not depend on when the note was struck. Fed one "
    "steady tone, a stage with no state in it returns every order either along the "
    "carrier or exactly against it, so a turn that is neither nought nor a hundred and "
    "eighty is a reading -- but of what, this record does not say: a state inside the "
    "stage and a filter after it that is not flat across the orders both do it, and "
    "separating those needs a second carrier at another frequency. "
    "How far from nought-or-against counts is `floor_turn_deg` in `reference`, which is "
    "the run's own repeats of one setting read the same way. It is an arc and not a "
    "difference of two numbers: two takes at plus and minus a hundred and seventy-nine "
    "degrees are two degrees apart, and the reading beside it for the sizes is a "
    "subtraction because a level is on a line and an angle is not. "
    "The whole series is "
    "determined only up to which way round the carrier was fed in, which adds `k` times "
    "a hundred and eighty to every turn and leaves the distance from nought-or-against "
    "unchanged -- so that distance is the reading and the value is not. A turn on an "
    "order that is at its own floor is the floor's angle and means nothing. "
    "`orders` is a resolution. Where `settled_at_the_top_db` is not near nothing the "
    "series was still carrying level at the last order read, so the total below it is a "
    "lower bound on what the stage returned rather than the whole of it, and the answer "
    "is to read the same takes again with more orders asked for. "
    "The orders are read where they were put, which holds only while they stay under "
    "half the rate the stage runs at. Where the last one lands is `top_order_hz`, the "
    "carrier asked for times the number of orders asked for, and it is printed rather "
    "than judged: this record does not know what rate the stage runs at, so whether the "
    "top of this series is where the reading says it is has to be decided against that "
    "figure by a reader who does. Raising `orders` and raising the carrier both move it, "
    "and neither announces itself in the numbers."
)

NOT_HERE = (
    "No curve, no order of polynomial and no clipper. Which nonlinearity these orders "
    "belong to, whether it is symmetric, and where along it the signal sat is a fit "
    "across settings and types, and the fit is not made here. Neither is what set the "
    "orders: a stage reached through the part rather than through the type is a stage "
    "the type's own bytes also reach, and which of them a reader is looking at is what "
    "the subject of this record says and what these figures cannot."
)

WHY_REFERENCE = (
    "The setting the run repeated, and the spread of those repeats is the floor every "
    "difference here has to clear. Taken in the same session rather than carried from "
    "another run: what a held voice fails to repeat belongs to the sitting it was "
    "recorded in. "
    "Unlike the band reading beside this one, nothing below is reported against these "
    "takes -- each reading is under its own take's first order -- so a run without them "
    "still produces its figures. What it loses is the floor, and a record that lost it "
    "says so here rather than leaving a reader to notice."
)

WHY_CONTROL = (
    "The takes made with the part routed past the effect. They are what says the orders "
    "below belong to the stage rather than to the voice: a carrier is not one partial, "
    "and the orders it arrives with are counted by this reading exactly as the ones the "
    "stage added are. Without them every order in the record would imply a stage that "
    "put it there. "
    "Where one of these carries a `value` the run made a control at every setting rather "
    "than at one, which it does when what was swept sits in front of the effect: its "
    "`heard_db` is then the measurement of what the stage was being given, and the only "
    "one in the record, because nothing inside the type reads its own input."
)

WHY_SILENCE = (
    "The takes made with the same chain and nothing played, read as orders the same way. "
    "A setting that turns the output down far enough returns the room and the converter, "
    "and those have a shape at every frequency asked for -- so an order that has fallen "
    "into this is the floor being reported as a harmonic."
)

WHY_CHANNEL = (
    "Which channel of the interface every take was read from, and how loud the reference "
    "takes were in each. One channel for the run rather than the loudest of each take: an "
    "interface carries inputs the unit is not on and they are not silent, so a setting "
    "that turns the output down is otherwise read from whichever input happened to be "
    "noisiest, and the orders of that are a complete and entirely wrong answer."
)

WHY_OTHER = (
    "The same orders read on the next loudest channel, as a control on the first. It is "
    "named rather than assumed to be the unit's other output; what says it carried the "
    "unit is that it follows the setting, and how far the two channels disagree is "
    "reported with every reading."
)

WHY_READ_OVER = (
    "How long the filter after the mixer is, which decides what counts as an order. "
    "Over the whole held stretch it is a matched filter for a steady partial and the "
    "narrowest this take can give, so what it returns is the order and very little "
    "else; over a boxcar a few carrier periods long the lobe is a fraction of a "
    "fundamental wide and whatever the stage put beside the order is read as part of "
    "it. Both are readings of these takes and neither is the better one in general -- "
    "a partial that moves is read by the short filter and lost by the long one. Which "
    "was used is stated because the same takes read the other way are another record, "
    "and because two records that disagree are saying where the energy sits rather "
    "than that one of them is wrong."
)

WHY_FUNDAMENTAL = (
    "The carrier each take was actually read at, and the note it was asked for. Measured "
    "per take because a boxcar aimed off the partial reads every order through the skirt "
    "of its own window, and because a take whose carrier is not where it was asked for is "
    "a take of something else -- which is a reading and not a nuisance."
)

WHY_HELD = (
    "The addresses the run had written while it read, and what they held. Orders are the "
    "whole chain's, so a setting read with another of the type's stages moved is a "
    "reading of something else and nothing in the numbers says so."
)


def fundamental(
    body: np.ndarray, rate: int, *, near_hz: float, within: float = WITHIN
) -> float | None:
    """Where the carrier really is, found near where it was asked for.

    A transform first and a boxcar afterwards, which is not two readings of the same
    thing: the transform is asked only where the peak is and the boxcar is asked how
    big it is. Interpolated across the three bins around the peak, because the
    transform's own spacing is coarser than the offset being looked for.
    """
    body = np.asarray(body, dtype=np.float64)
    if body.size < 2 or near_hz <= 0:
        return None
    line = np.abs(np.fft.rfft(body * np.hanning(body.size)))
    freq = np.fft.rfftfreq(body.size, 1.0 / rate)
    near = np.where(np.abs(freq - near_hz) < near_hz * within)[0]
    if near.size < 3:
        return None
    peak = int(near[int(np.argmax(line[near]))])
    if peak <= 0 or peak + 1 >= line.size:
        return None
    left, middle, right = (float(np.log(max(line[i], 1e-30))) for i in (peak - 1, peak, peak + 1))
    under = left - 2.0 * middle + right
    offset = 0.0 if under == 0.0 else 0.5 * (left - right) / under
    return float(freq[peak] + offset * (freq[1] - freq[0]))


def under_the_first(sizes: list[float]) -> list[float]:
    """Every order but the first, in decibels under it."""
    first = max(sizes[0], 1e-30)
    return [round(float(20.0 * np.log10(max(size, 1e-30) / first)), 2) for size in sizes[1:]]


def all_of_them_db(under: list[float]) -> float:
    """The orders above the first taken together, in decibels under it.

    One number for a series of them, and a summary rather than a reading: it is what
    orders the settings of a sweep against each other when the whole series moves the
    same way, and it says nothing at all when two settings move different orders in
    opposite directions. The series is beside it for that.
    """
    total = float(np.sum([10.0 ** (value / 10.0) for value in under]))
    return round(float(10.0 * np.log10(max(total, 1e-30))), 2)


def _settled(under: list[float]) -> float:
    """How much level the series still carried at the last order read.

    The last order under the largest of the ones above the first. Near nothing says
    the series had fallen away inside the count asked for; anything else says the
    total is a lower bound and the count is what ended it.
    """
    if not under:
        return 0.0
    return round(float(under[-1] - max(under)), 2)


def _body(samples, rate: int, *, index: int, lead_s: float, hold_s: float, trim_s: float):
    first = int((lead_s + trim_s) * rate)
    last = int((lead_s + hold_s - trim_s) * rate)
    return takes.channel(samples, index)[first:last]


def _loudness_db(samples, index: int) -> float:
    body = takes.channel(samples, index)
    return float(20.0 * np.log10(max(float(np.sqrt((body**2).mean())), 1e-12)))


def _read(
    where: Path,
    name: str,
    entry: dict,
    *,
    channels: tuple[int, ...],
    carrier_hz: float,
    orders: int,
    periods: int | None,
    lead_s: float,
    trim_s: float,
    hold_s: float | None,
) -> tuple[dict[int, list[float] | None], dict[int, float], dict[int, float | None], float, int]:
    """Every channel asked for, out of one read of the take.

    Read once and measured twice for the same reason the band reading gives: the
    second channel is a control over the first, and a control that costs another
    pass over the whole directory is one that gets dropped from the next run.
    """
    samples, rate = takes.read(where / name)
    seconds = float(entry.get("seconds") or samples.shape[0] / rate)
    hold = hold_s if hold_s is not None else seconds - 1.0
    found: dict[int, list[float] | None] = {}
    heard: dict[int, float] = {}
    at: dict[int, float | None] = {}
    turned: dict[int, list[float] | None] = {}
    floors: dict[int, list[float] | None] = {}
    for index in channels:
        body = _body(samples, rate, index=index, lead_s=lead_s, hold_s=hold, trim_s=trim_s)
        here = fundamental(body, rate, near_hz=carrier_hz)
        at[index] = None if here is None else round(here, 3)
        found[index] = (
            None
            if here is None
            else partials.levels(body, rate, carrier_hz=here, count=orders, periods=periods)
        )
        # The angles and the floor beside each order come from the whole-stretch
        # reading whatever `periods` says, because neither means anything through a
        # boxcar one carrier period wide: that lobe is a whole fundamental across, so
        # it collects what sits between the orders along with the orders.
        turned[index] = None if here is None else _turns(body, rate, here, orders)
        floors[index] = (
            None
            if here is None
            else partials.between_orders(body, rate, carrier_hz=here, count=orders)
        )
        heard[index] = round(_loudness_db(samples, index), 2)
    own = int(np.argmax(takes.channel_levels(samples)))
    return found, heard, at, round(hold, 3), own, turned, floors


def _middle_turn(rows: list[list[float]]) -> list[float]:
    """Where the repeats of one setting put each order's turn, on the circle.

    A mean of degrees is not a mean of angles: a hundred and seventy-nine and minus a
    hundred and seventy-nine average to nought, which is the opposite side of the
    circle from both of them. Average the unit vectors instead.
    """
    return [
        round(
            float(
                np.degrees(
                    np.angle(np.mean(np.exp(1j * np.radians([row[i] for row in rows]))))
                )
            ),
            1,
        )
        for i in range(len(rows[0]))
    ]


def _turn_spread(rows: list[list[float]]) -> list[float]:
    """The shortest arc the repeats of one order's turn all fit inside.

    The size beside this is spread as the largest repeat minus the smallest, which is
    the right reading for a quantity on a line and the wrong one for a quantity on a
    circle: two takes at plus and minus a hundred and seventy-nine degrees differ by
    two degrees and subtracting says three hundred and fifty-eight. So the repeats are
    sorted around the circle, the widest gap between neighbours is found, and what is
    left over is the arc they occupy.
    """
    spread: list[float] = []
    for i in range(len(rows[0])):
        here = sorted(float(row[i]) % 360.0 for row in rows)
        gaps = [b - a for a, b in zip(here, here[1:], strict=False)]
        gaps.append(here[0] + 360.0 - here[-1])
        spread.append(round(360.0 - max(gaps), 1))
    return spread


def _turns(body, rate, carrier_hz: float, orders: int) -> list[float] | None:
    """Each order's angle against the carrier's, with the take's own clock taken out.

    The note started whenever it started, and that rotates order `k` by `k` times one
    angle. So does every delay between the stage and the converter. Neither is a fact
    about the effect, and

        turn_k = angle_k - k * angle_1

    removes both exactly. There is nothing fitted in it: the first order fixes the
    angle, every branch of it gives the same answer because the branches differ by `k`
    times a whole turn, and the figure is the same whenever the note was struck.

    What it is for: fed one steady tone, a stage with no state in it returns each order
    either along the carrier or exactly against it, so **every turn is nought or a
    hundred and eighty degrees and none is between**. A degree between them is a
    reading, and what it is a reading of -- a state in the stage, or a filter after it
    that is not flat across the orders -- this cannot separate and does not claim to.

    The series is determined up to one thing, and it is the same thing a reading of
    sizes is already up to: whether the curve was fed the carrier or its negative. That
    adds `k` times a hundred and eighty to every turn, which leaves the distance from
    nought-or-against unchanged and is why the distance is the test rather than the
    value.
    """
    standing = partials.standing_at(body, rate, carrier_hz=carrier_hz, count=orders)
    if standing is None:
        return None
    first = float(np.angle(standing[0]))
    out = []
    for k, value in enumerate(standing, start=1):
        turn = float(np.angle(value)) - k * first
        out.append(round(float(np.degrees((turn + np.pi) % (2.0 * np.pi) - np.pi)), 1))
    return out


def _matched(pattern, listed: dict, files: list[str]) -> list[tuple[str, dict, object]]:
    out = []
    for name in files:
        entry = listed.get(name, {})
        source, got = takes.named_by(pattern, entry, name)
        if got is not None:
            out.append((name, entry, (source, got)))
    return out


def read_directory(
    where: str | Path,
    *,
    type_id: str,
    address: str | None = None,
    controller: int | None = None,
    setting: str,
    carrier_hz: float,
    orders: int = ORDERS,
    periods: int | None = None,
    reference: str | None = None,
    control: str | None = None,
    silence: str | None = None,
    stimulus: str | None = None,
    held: list[dict] | None = None,
    channel: int | None = None,
    lead_s: float = 0.6,
    trim_s: float = 0.5,
    hold_s: float | None = None,
    progress=None,
) -> dict:
    """Every take under `where` whose setting matches, read into one record.

    Exactly one of `address` and `controller`: a run swept one thing, and a record
    naming both would not say which of them the figures belong to.

    `reference` is optional here and required on the band reading beside this one,
    and the difference is in what each reads. A band profile is a difference from
    the reference and does not exist without it; an order is under its own take's
    first order and does exist. What the repeats add is the floor, so a run without
    them is short of a bound rather than short of a reading, and the record says
    which it is.
    """
    if (address is None) == (controller is None):
        raise ValueError("a record is about one address or one controller, not both or neither")
    where = Path(where)
    listed, files = takes.listing(where)

    swept = takes.capturing(setting, VALUE)
    flat = re.compile(reference) if reference else None
    bypassed = re.compile(control) if control else None
    quiet = re.compile(silence) if silence else None

    flats = _matched(flat, listed, files) if flat is not None else []
    if flat is not None and not flats:
        raise ValueError(f"no take under {where} matched the reference {reference!r}")

    # The channel from the reference takes where there are any, and from the swept
    # takes where there are not -- in both cases from takes the unit is certainly
    # sounding in, and once for the run rather than per take.
    naming = [name for name, _, _ in flats] or [
        name for name, _, _ in _matched(swept, listed, files)
    ]
    if not naming:
        raise ValueError(f"no take under {where} matched the setting {setting!r}")
    reached, reference_levels = takes.channel_reaching(where, naming)
    used = reached if channel is None else int(channel)
    ranked = sorted(range(len(reference_levels)), key=lambda i: -reference_levels[i])
    beside = next((i for i in ranked if i != used), None)
    wanted = (used,) if beside is None else (used, beside)
    elsewhere: list[str] = []

    turns_at: dict[str, list[float] | None] = {}
    floor_at: dict[str, list[float] | None] = {}

    def one(name: str, entry: dict):
        found, heard, at, hold, own, turned, floors = _read(
            where, name, entry,
            channels=wanted, carrier_hz=carrier_hz, orders=orders, periods=periods,
            lead_s=lead_s, trim_s=trim_s, hold_s=hold_s,
        )
        if own != used and name not in elsewhere:
            elsewhere.append(name)
        turns_at[name] = turned[used]
        floor_at[name] = floors[used]
        return found, heard, at, hold

    def said(sizes: list[float] | None, name: str | None = None) -> dict:
        if sizes is None:
            return {"under_the_first_db": None, "all_of_them_db": None, "orders_db": None}
        under = under_the_first(sizes)
        beside = floor_at.get(name)
        return {
            "under_the_first_db": under,
            "all_of_them_db": all_of_them_db(under),
            "settled_at_the_top_db": _settled(under),
            # Every order's own level as well as its level under the first. The
            # ratios are the reading and these are what the limit about the floor is
            # checked with: an order is the floor rather than an order when it is
            # not above what the same chain returned there with nothing played, and
            # that comparison cannot be made between two ratios.
            "orders_db": [round(float(20.0 * np.log10(max(size, 1e-30))), 2) for size in sizes],
            # Which way each order points, against the carrier, with the take's own
            # clock removed. Nought or a hundred and eighty on every order is what a
            # stage with no state in it returns when it is fed one steady tone.
            "orders_turn_deg": turns_at.get(name),
            # And the floor each order is standing in, read halfway to the next order
            # in the same take. This is what says an order is an order: a figure, in
            # this take, at this moment, rather than a number of decibels chosen under
            # the first order or a floor carried in from the silence of another take.
            "floor_between_orders_db": (
                None
                if beside is None
                else [round(float(20.0 * np.log10(max(v, 1e-30))), 2) for v in beside]
            ),
        }

    quiets: list[str] = []
    floor_heard: float | None = None
    silent: list[dict] = []
    if quiet is not None:
        heard_at = []
        for name, entry, _ in _matched(quiet, listed, files):
            quiets.append(name)
            found, heard, at, _hold = one(name, entry)
            heard_at.append(heard[used])
            silent.append({**said(found[used], name), "heard_db": heard[used], "take": name})
        if heard_at:
            floor_heard = round(float(np.mean(heard_at)), 1)

    def above(heard: float) -> float | None:
        return None if floor_heard is None else round(heard - floor_heard, 1)

    middle: list[float] | None = None
    beside_middle: list[float] | None = None
    floor: list[float] | None = None
    middle_turn: list[float] | None = None
    turn_floor: list[float] | None = None
    flat_heard: float | None = None
    flat_at: float | None = None
    if flats:
        read = [one(name, entry) for name, entry, _ in flats]
        turns = [turns_at[name] for name, _, _ in flats if turns_at.get(name)]
        if turns:
            middle_turn = _middle_turn(turns)
            turn_floor = _turn_spread(turns)
        rows = [under_the_first(found[used]) for found, _, _, _ in read if found[used]]
        middle = [round(float(np.mean([row[i] for row in rows])), 2) for i in range(len(rows[0]))]
        floor = [
            round(max(row[i] for row in rows) - min(row[i] for row in rows), 2)
            for i in range(len(rows[0]))
        ]
        if beside is not None:
            theirs = [under_the_first(found[beside]) for found, _, _, _ in read if found[beside]]
            if theirs:
                beside_middle = [
                    round(float(np.mean([row[i] for row in theirs])), 2)
                    for i in range(len(theirs[0]))
                ]
        flat_heard = round(float(np.mean([heard[used] for _, heard, _, _ in read])), 2)
        seen = [at[used] for _, _, at, _ in read if at[used] is not None]
        flat_at = round(float(np.mean(seen)), 3) if seen else None

    def against(under: list[float] | None) -> dict:
        """How far one reading's orders are from the repeats', order by order."""
        if under is None or middle is None or floor is None:
            return {}
        apart = [round(a - b, 2) for a, b in zip(under, middle, strict=True)]
        outside = [
            order
            for order, value, edge in zip(range(2, 2 + len(apart)), apart, floor, strict=True)
            if abs(value) > edge
        ]
        largest = max(
            (
                (value, order)
                for order, value, edge in zip(range(2, 2 + len(apart)), apart, floor, strict=True)
                if abs(value) > edge
            ),
            key=lambda pair: abs(pair[0]),
            default=(None, None),
        )
        return {
            "against_the_reference_db": apart,
            "outside_the_floor_orders": outside,
            "largest_db": largest[0],
            "largest_at_order": largest[1],
        }

    def apart(found: dict[int, list[float] | None]) -> dict:
        """How far the second channel's own orders are from the first channel's.

        Each channel under its own first order, so a standing level difference
        between the two is not counted as a disagreement about the orders.
        """
        if beside is None or found[used] is None or found[beside] is None:
            return {}
        mine, theirs = under_the_first(found[used]), under_the_first(found[beside])
        at = range(2, 2 + len(mine))
        gap = max(
            (
                (round(a - b, 2), order)
                for a, b, order in zip(mine, theirs, at, strict=True)
            ),
            key=lambda pair: abs(pair[0]),
        )
        return {"other_channel_apart_db": gap[0], "other_channel_apart_at_order": gap[1]}

    claimed = {name for name, _, _ in flats} | set(quiets)
    controls: list[dict] = []
    if bypassed is not None:
        for name, entry, (_, got) in _matched(bypassed, listed, files):
            claimed.add(name)
            found, heard, at, _hold = one(name, entry)
            # A control pattern that captures the same group the sweep does says
            # which setting each control take was made at, and a control taken at
            # every setting is a ladder rather than one state: on a run that swept
            # something in front of the effect, what these takes came back at is
            # the only measurement of what the effect was being given. Left out
            # where the pattern does not capture it, which is the ordinary case of
            # a control made at one state.
            at_value = (
                {VALUE: int(got.group(VALUE))}
                if VALUE in (bypassed.groupindex or {})
                else {}
            )
            controls.append(
                {
                    **at_value,
                    **said(found[used], name),
                    **against(under_the_first(found[used]) if found[used] else None),
                    **apart(found),
                    "fundamental_hz": at[used],
                    "heard_db": heard[used],
                    "above_the_silence_db": above(heard[used]),
                    "take": name,
                }
            )

    readings: list[dict] = []
    for name, entry, (source, got) in _matched(swept, listed, files):
        if name in claimed:
            continue
        claimed.add(name)
        found, heard, at, hold = one(name, entry)
        reading = {
            VALUE: int(got.group(VALUE)),
            **said(found[used], name),
            **against(under_the_first(found[used]) if found[used] else None),
            **apart(found),
            "fundamental_hz": at[used],
            "heard_db": heard[used],
            "above_the_silence_db": above(heard[used]),
            "hold_s": hold,
            "take": name,
            "named_by": source,
        }
        readings.append(reading)
        if progress:
            progress(reading)

    readings.sort(key=lambda row: row[VALUE])
    looked_at = len(files)
    return {
        "question": QUESTION,
        "type": type_id,
        **({"address": address} if address is not None else {"controller": controller}),
        "method": METHOD_OVER_THE_STRETCH if periods is None else METHOD_OVER_A_BOXCAR,
        "limits": LIMITS,
        "not_in_this_record": NOT_HERE,
        "orders": orders,
        "carrier_asked_hz": carrier_hz,
        "top_order_hz": round(carrier_hz * orders, 1),
        "read_over": (
            "the whole of the held stretch"
            if periods is None
            else f"a boxcar {periods} carrier periods long"
        ),
        "why_read_over": WHY_READ_OVER,
        "why_fundamental": WHY_FUNDAMENTAL,
        "channel": {
            "read": used,
            "chosen_by": "given" if channel is not None else "loudest in the takes named below",
            "named_by": "the reference takes" if flats else "the swept takes",
            "reference_db": reference_levels,
            "loudest_elsewhere": sorted(elsewhere),
            "why": WHY_CHANNEL,
        },
        "other_channel": {
            "read": beside,
            "under_the_first_db": beside_middle,
            "why": WHY_OTHER,
        },
        "silence": {"takes": sorted(quiets), "readings": silent, "why": WHY_SILENCE},
        "reference": {
            "takes": sorted(name for name, _, _ in flats),
            "under_the_first_db": middle,
            "floor_db": floor,
            "turn_deg": middle_turn,
            "floor_turn_deg": turn_floor,
            "heard_db": flat_heard,
            "fundamental_hz": flat_at,
            "why": WHY_REFERENCE,
        },
        "control": {
            "takes": [row["take"] for row in controls],
            "readings": controls,
            "why": WHY_CONTROL,
        },
        "held": held or [],
        "why_held": WHY_HELD,
        **({"stimulus": stimulus} if stimulus else {}),
        "readings": readings,
        "takes_not_matching": {
            "count": looked_at - len(claimed),
            "why": "a take none of the patterns named is counted rather than dropped: a "
            "pattern that matches nothing and a directory that holds nothing return the "
            "same empty record otherwise, and they are different mistakes",
        },
    }
