"""What a parameter did to the balance between the channels, which one channel cannot say.

Every other measurement here reduces a take to one channel, and that is what
makes a residual mean anything: two takes are compared sample against sample, so
they have to be the same signal. `perform.loudest_channel` picks the input the
unit arrived on and the rest of the harness works in it.

**A parameter that moves signal between the channels is invisible to that.** What
it does is not a change in the chosen channel, it is a change in the relation
between two, and reducing to one channel measures whichever side of the relation
was picked. Worse, it does not read as nothing: a pan that lands somewhere new
each time the note is struck reads as the unit failing to repeat, which is a
verdict about the machine wearing the parameter's name.

Measured on this unit, four takes at each setting of the part panpot, the two
channels the unit arrived on:

- at 127 the two sit at -58.6 and -48.4 dB, on every take, to a tenth of a dB
- at 0 they land at -54.5/-49.3, -47.3/-62.9, -52.2/-50.6 and -63.7/-47.4 --
  a balance swinging over 31 dB while the two together stay within 3

So the two facts this reports are separate: whether the balance *moved between
the settings*, and whether it *failed to repeat inside one*. The second is the
one nothing else here can see, and it is a fact about the parameter rather than
about the unit only because the other setting repeats to a tenth of a dB in the
same session.

**Both are judged against a yardstick measured in the same run**, as everything
else here is: a chain with a channel imbalance of its own, or a voice that is not
centred, would otherwise read as a parameter that pans.

Two settings is the screening question and not the shape of the reading. The same
pair of numbers per take -- the difference between the channels and the two of them
together -- is what a byte printed as a pan has to be read out of, so a run that
asks a byte at a dozen settings is read here too and the verdicts are the widest
gap and the widest difference in scatter over all of them. At two settings that is
the same arithmetic; what it stops is a sweep publishing a screening verdict that
was defined for a pair and quietly returns false everywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .efxbands import BAND_SETS, energies
from .takes import (
    SECOND_CHANNEL_ABOVE_DB,
    WHY_THE_FLOOR_IS_THE_PAIRS_OWN,
    grouped,
    level_db,
    pair_of_channels,
    read,
    rms,
    spread,
)

METHOD = (
    "The two channels the unit arrived on were measured separately on every take, and what is "
    "reported is the level difference between them. A parameter that moves signal between the "
    "channels does not change either take in the way a residual measures, so it is invisible to "
    "a comparison made in one channel -- and a balance that lands somewhere new on each take "
    "reads there as the unit failing to repeat itself. Both the movement between the settings "
    "and the scatter within one are judged against the same run's own spread, since a chain "
    "with an imbalance of its own would otherwise read as a parameter that pans."
)

WHY_NOT_A_LEVEL = (
    "Reported beside the two channels together, because a balance that moves while the pair's "
    "total stays put is a pan and one that moves with it is a level change reaching one channel. "
    "The difference is not visible in either channel on its own."
)

MONO_SOURCE = (
    "Only one channel carried the unit, so there is no balance to measure and no null here "
    "either: what a parameter does between two channels cannot be asked of one."
)


@dataclass
class SettingBalance:
    """One setting's takes, as a balance and a total."""

    setting: str
    balance_db: list[float] = field(default_factory=list)
    """Per take, the first channel's level less the second's."""

    total_db: list[float] = field(default_factory=list)
    """Per take, the two channels together. A pan moves the first and not this."""

    @property
    def spread_db(self) -> float:
        return spread(self.balance_db)

    @property
    def total_spread_db(self) -> float:
        return spread(self.total_db)

    @property
    def typical_db(self) -> float:
        return float(np.median(self.balance_db)) if self.balance_db else float("nan")

    def to_json(self) -> dict:
        return {
            "setting": self.setting,
            "balance_db": [round(v, 2) for v in self.balance_db],
            "together_db": [round(v, 2) for v in self.total_db],
            "balance_spread_db": round(self.spread_db, 2),
            "together_spread_db": round(self.total_spread_db, 2),
        }


@dataclass
class Verdict:
    stimulus: str
    settings: list[SettingBalance]
    channels: tuple[int, int] | None
    margin_db: float = 6.0
    not_measured: str | None = None
    """Why this stimulus has no verdict, when it has none.

    Carried rather than dropped. A stimulus the run asked and this could not
    answer is absent from a list of verdicts, and absent reads as never asked --
    so a count of what moved would be taken over a denominator that quietly
    shrank.
    """

    @property
    def yardstick_db(self) -> float:
        """The steadiest setting's own scatter, which is what a claim has to clear.

        The steadiest rather than the worst: the question a wide setting is being
        asked is whether it is wide, so taking it into the yardstick would be
        measuring it against itself.

        Over more than two settings the minimum is drawn from more settings and so
        runs lower than the run's typical scatter. That is a bound on the yardstick
        and not a finding about the parameter, and what absorbs it is the margin
        rather than a different rule: a sweep whose gap does not clear the steadiest
        setting's scatter by the margin was not going to clear the worst either.
        """
        spreads = [s.spread_db for s in self.settings]
        return min(spreads) if spreads else float("nan")

    @property
    def moved_between_settings(self) -> bool:
        """Whether any two of the settings sit further apart than the run resolves.

        The widest gap in the run rather than the gap between the ends, because a
        parameter need not be monotonic in its byte and a table that turns back on
        itself would put its two ends in the same place.
        """
        if len(self.settings) < 2:
            return False
        typical = [s.typical_db for s in self.settings]
        return bool(max(typical) - min(typical) > self.yardstick_db + self.margin_db)

    @property
    def did_not_repeat(self) -> bool:
        """Whether one setting's balance landed somewhere new on each take.

        The asymmetry is the evidence and it has to be an asymmetry: a chain that
        wanders wanders at every setting, and a parameter is only implicated when
        some setting is steady in the same session that another is not.
        """
        if len(self.settings) < 2:
            return False
        spreads = [s.spread_db for s in self.settings]
        return bool(max(spreads) - min(spreads) > self.margin_db * 2)

    @property
    def while_the_total_stayed(self) -> bool:
        """Whether the pair's total held still while the balance did not.

        What separates a pan from a level change that reached one channel, and
        neither channel on its own can tell them apart.
        """
        widest = max((s.total_spread_db for s in self.settings), default=float("nan"))
        return bool(self.did_not_repeat and widest < max(s.spread_db for s in self.settings) / 2)

    def describe(self) -> str:
        if self.not_measured:
            return f"{self.stimulus}: not measured -- {self.not_measured}"
        shown = " against ".join(
            f"{s.setting}: {s.typical_db:+.1f} dB, spread {s.spread_db:.1f}" for s in self.settings
        )
        if not self.settings:
            return f"{self.stimulus}: nothing to measure"
        how = []
        if self.moved_between_settings:
            how.append("the balance moved between the settings")
        if self.did_not_repeat:
            how.append(
                "one setting's balance landed somewhere new on each take"
                + (
                    ", while the two channels together held still"
                    if self.while_the_total_stayed
                    else ""
                )
            )
        found = "; ".join(how) if how else "nothing the balance could show"
        where = f"channels {self.channels[0]} and {self.channels[1]}"
        return f"{self.stimulus} ({where}): {found}. {shown}"

    def to_json(self) -> dict:
        if self.not_measured:
            return {
                "stimulus_name": self.stimulus,
                "channels": None,
                "measured": False,
                "not_measured": self.not_measured,
            }
        return {
            "stimulus_name": self.stimulus,
            "channels": list(self.channels) if self.channels else None,
            "measured": True,
            "margin_db": self.margin_db,
            "yardstick_db": round(self.yardstick_db, 2),
            "moved_between_settings": self.moved_between_settings,
            "did_not_repeat_within_a_setting": self.did_not_repeat,
            "while_the_two_together_held_still": self.while_the_total_stayed,
            "by_setting": [s.to_json() for s in self.settings],
        }


WHY_THE_PAIR_IS_IN_INPUT_ORDER = (
    "Which two inputs the unit arrived on is decided by level; which of the two is subtracted "
    "from the other is decided by the input's own number. Ordering the pair by level as well "
    "leaves the sign of every balance in the hands of whichever channel was louder by a hair: "
    "two readings of one saved run returned +10.18 dB and -10.18 dB, each self-consistent with "
    "the pair it also reported, and nothing in the figure said which convention it was under. "
    "A parameter printed as a pan is a direction before it is a size, so the direction has to "
    "survive being read twice. Which of the unit's outputs is on the lower-numbered input is a "
    "fact about the cabling and this does not establish it -- the pair is reported beside the "
    "figure so a reader can map it."
)


WHY_A_SEPARATION_NEEDS_A_CEILING = (
    "A separation that stops has to be shown to be the unit's and not the chain's, so what is "
    "reported beside it is how far apart the same two inputs have been measured to put a signal "
    "on some other address of the same unit. A reading that stops short of that is stopping "
    "somewhere the chain could have seen past; one that stops at it is a bound and not a figure "
    "about the parameter. The quieter channel's distance from the take's own lead is reported "
    "for the same reason and answers a different question -- whether the reading is a level at "
    "all."
)

WHY_A_PATH_AROUND_THE_EFFECT_IS_ASKED = (
    "A separation between two channels is the separation of whatever reached them, so a second "
    "path to the output that the parameter does not act on would set a floor under it that has "
    "nothing to do with the parameter. What is reported is what the two channels hold with the "
    "effect's own output level written to zero: if that is the take's own lead, there is no "
    "second path for the reading to be the sum of."
)


def measure(root: str | Path, *, margin_db: float = 6.0) -> list[Verdict]:
    """Read a saved run and say what each setting did to the balance.

    One verdict per stimulus the run holds takes for, so a parameter asked under
    several notes carries what each of them saw, as everywhere else here.
    """
    root = Path(root)
    leads, by_setting, order = grouped(root)

    out = []
    for stimulus, settings in by_setting.items():
        loaded = {k: [read(p) for p in v] for k, v in settings.items()}
        # The rate and the noise floor come from the setting that sounded
        # loudest, not from the first one: a parameter that silences its part at
        # one of its two values would otherwise have its floor read out of noise.
        # Measured here, one address in the part block silences the part at 127,
        # which is the setting the plan asks first. The channels are a separate
        # question and are asked of every take, per `takes.WHY_ACROSS_THE_SETTINGS`.
        loudest = max(
            (loaded[s][0] for s in order[stimulus]), key=lambda t: float(np.abs(t[0]).max())
        )
        rate = loudest[1]
        lead = int((leads.get(stimulus, 0.6) * 0.8) * rate)
        head = loudest[0][:lead] if lead > rate // 100 else None
        every = [frames for setting in order[stimulus] for frames, _ in loaded[setting]]
        channels, _floor = pair_of_channels(every, head)
        if channels is None:
            out.append(
                Verdict(stimulus=stimulus, settings=[], channels=None, not_measured=MONO_SOURCE)
            )
            continue
        left, right = channels
        measured = []
        for setting in order[stimulus]:
            found = SettingBalance(setting=setting)
            for frames, _ in loaded[setting]:
                a, b = level_db(rms(frames[:, left])), level_db(rms(frames[:, right]))
                found.balance_db.append(a - b)
                found.total_db.append(level_db(rms(frames[:, [left, right]])))
            measured.append(found)
        out.append(
            Verdict(stimulus=stimulus, settings=measured, channels=channels, margin_db=margin_db)
        )
    return out


BAND_ABOVE_THE_FLOOR_DB = 10.0
"""How far a band of the quieter channel has to stand over its own lead to be read.

Broadband the quieter channel can be thirty decibels clear of the lead while some
band of it is not clear at all, because neither the signal nor the floor is flat.
A separation computed in such a band is the distance from the louder channel to
the converter, which is a number about the rig and would be published as a
property of the byte.
"""

SEPARATION_BY_BAND = (
    "The two channels the unit arrived on were measured band by band on every take, and what "
    "is reported per band is the first channel's energy less the second's -- the same "
    "difference the balance reports, asked of one band at a time instead of the whole take. "
    "Both channels' bands are given against one number, the two of them together over the same "
    "window, so the two rows are shapes that can be read on their own and their difference is "
    "still exactly the separation."
)

WHY_THE_WINDOWS_ARE_ONE_LENGTH = (
    "Every window read here is the length of the take's own lead, the body's being the average "
    "of as many of them as fit between one lead's length after the note starts and one before "
    "the take ends. A band's energy does not scale with the window the same way for a tone as "
    "for noise -- summed over a band it grows with the window for one and with its square for "
    "the other -- so a body read over seconds and a lead read over a fraction of one cannot be "
    "subtracted without the answer depending on what was played. Matching the lengths removes "
    "the question rather than correcting for it."
)

BAND_REPEATS_WITHIN_DB = 2.0
"""How far a band's separation may move between a setting's own takes and be read.

A broadband balance repeats to a hundredth of a decibel on this rig, so nothing
there needs a rule of this kind. One band of a channel carrying almost nothing is
a different reading: measured here, two takes of one byte disagree by eight
decibels in bands where the quieter channel is thirty above the interface's floor
and at the unit's own, and a curve drawn through those bands would be a shape read
out of noise.

The number is the coarsest thing the run's own flat settings need. Where the two
channels carry the same signal every band repeats within a few tenths, so this
leaves those untouched and reaches only the bands where the reading has stopped
working.
"""

WHY_A_BAND_HAS_TO_REPEAT = (
    "A band is read only where it cleared the take's own lead and where the setting's two "
    "takes agreed about it. The second is not the first: a band of the quieter channel can sit "
    "thirty decibels over the interface's floor and still be at the unit's own, and there it "
    "returns a different figure on each take while never once looking like silence. Both "
    "reasons a band was left out are named against it, so a profile with holes in it is a "
    "statement about where the reading stopped rather than a curve that quietly got shorter."
)

WHY_A_FLAT_SEPARATION_IS_THE_YARDSTICK = (
    "What a separation being frequency dependent has to be judged against is the same pair of "
    "channels carrying the same signal, which is the setting whose separation varies least "
    "across the bands. That is the chain's own mismatch between two inputs plus what a band "
    "reading fails to repeat, and it is measured in the same run rather than assumed to be "
    "zero. A sweep with no such setting in it has no yardstick here and says so."
)


@dataclass
class BandSetting:
    """One setting's two channels, band by band."""

    setting: str
    reference_db: float = 0.0
    """The two channels together over the window, which every row here is against.

    Published rather than divided out and forgotten. Without it no two settings
    can be laid on one scale, and the question a band reading of a separation is
    asked -- whether the quieter channel stopped falling while the louder one went
    on -- is a question about two settings.
    """

    first_db: list[float] = field(default_factory=list)
    second_db: list[float] = field(default_factory=list)
    floor_db: list[float] = field(default_factory=list)
    """Per band, the louder of the two channels' leads, on this setting's own scale.

    Carried per setting rather than once for the run because every row here is
    against the two channels together over the same window, and that number is
    the setting's. A floor written once in the run's own units would be the one
    row in the record a reader had to convert before using.
    """

    scatter_db: list[float] = field(default_factory=list)
    """Per band, how far the separation moved between this setting's own takes.

    The reading's own floor, per WHY_A_BAND_HAS_TO_REPEAT, and the one a broadband
    balance does not need: over a whole take the two channels repeat to a hundredth
    of a decibel, while a single band of a channel carrying almost nothing does not
    repeat at all.
    """

    kept: list[bool] = field(default_factory=list)
    """Whether the band cleared its own lead and repeated between the takes."""

    @property
    def separation_db(self) -> list[float | None]:
        return [
            round(a - b, 2) if keep else None
            for a, b, keep in zip(self.first_db, self.second_db, self.kept, strict=True)
        ]

    @property
    def spread_over_bands_db(self) -> float:
        """How far the separation moved across the bands that were read."""
        held = [v for v in self.separation_db if v is not None]
        return float(max(held) - min(held)) if len(held) > 1 else 0.0

    @property
    def worst_scatter_db(self) -> float:
        """The widest take-to-take scatter among the bands this setting kept."""
        held = [s for s, keep in zip(self.scatter_db, self.kept, strict=True) if keep]
        return max(held) if held else 0.0

    def to_json(self) -> dict:
        return {
            "setting": self.setting,
            "reference_db": round(self.reference_db, 2),
            "separation_db": self.separation_db,
            "first_db": [round(v, 2) for v in self.first_db],
            "second_db": [round(v, 2) for v in self.second_db],
            "floor_db": [round(v, 2) for v in self.floor_db],
            "scatter_db": [round(v, 2) for v in self.scatter_db],
            "spread_over_bands_db": round(self.spread_over_bands_db, 2),
            "worst_scatter_db": round(self.worst_scatter_db, 2),
            "bands_read": int(sum(self.kept)),
        }


@dataclass
class BandVerdict:
    """What one stimulus' settings did to the separation, band by band."""

    stimulus: str
    channels: tuple[int, int] | None
    centres: list[float] = field(default_factory=list)
    width_octaves: float = 1 / 3
    settings: list[BandSetting] = field(default_factory=list)
    settings_left_out: list[str] = field(default_factory=list)
    """Settings the run holds one take of, which cannot say whether a band repeats."""

    margin_db: float = 6.0
    not_measured: str | None = None

    @property
    def flattest_db(self) -> float:
        return min((s.spread_over_bands_db for s in self.settings), default=float("nan"))

    @property
    def widest_db(self) -> float:
        return max((s.spread_over_bands_db for s in self.settings), default=float("nan"))

    @property
    def depends_on_frequency(self) -> bool:
        """Whether some setting's separation varies across the bands and another's does not.

        The asymmetry is the evidence, per WHY_A_FLAT_SEPARATION_IS_THE_YARDSTICK.
        A pair of channels that never agree across the bands at any setting is a
        chain with a response mismatch, and calling that the byte's would be
        reporting the rig.
        """
        if len(self.settings) < 2:
            return False
        return bool(self.widest_db - self.flattest_db > self.margin_db)

    def describe(self) -> str:
        if self.not_measured:
            return f"{self.stimulus}: not measured -- {self.not_measured}"
        found = (
            "the separation depends on which band it is read in"
            if self.depends_on_frequency
            else "the separation is the same in every band the run could read"
        )
        return (
            f"{self.stimulus} (channels {self.channels[0]} and {self.channels[1]}): {found}. "
            f"flattest setting spreads {self.flattest_db:.1f} dB over the bands, "
            f"widest {self.widest_db:.1f}"
        )

    def to_json(self) -> dict:
        if self.not_measured:
            return {
                "stimulus_name": self.stimulus,
                "channels": None,
                "measured": False,
                "not_measured": self.not_measured,
            }
        return {
            "stimulus_name": self.stimulus,
            "channels": list(self.channels) if self.channels else None,
            "measured": True,
            "margin_db": self.margin_db,
            "centres_hz": list(self.centres),
            "width_octaves": round(self.width_octaves, 4),
            "flattest_setting_spreads_db": round(self.flattest_db, 2),
            "widest_setting_spreads_db": round(self.widest_db, 2),
            "separation_depends_on_frequency": self.depends_on_frequency,
            "settings_left_out_for_want_of_a_second_take": list(self.settings_left_out),
            "by_setting": [s.to_json() for s in self.settings],
        }


def _one_length_windows(frames: np.ndarray, *, lead_n: int, start: int) -> list[np.ndarray]:
    """The body as whole windows of the lead's own length, per WHY_THE_WINDOWS_ARE_ONE_LENGTH.

    One window of the lead's length is dropped at each end of the body: the first
    covers the note's onset, and the last would reach into whatever follows the
    hold. What is left is averaged, so a band's figure is the mean of several
    windows rather than one.
    """
    first = start + lead_n
    last = frames.shape[0] - lead_n
    return [frames[i : i + lead_n] for i in range(first, last - lead_n + 1, lead_n)]


def _averaged_bands(windows: list[np.ndarray], channel: int, rate: int, centres, width) -> list:
    """Band energies averaged over the windows, in power and then put back into dB."""
    rows = [np.asarray(energies(w[:, channel], rate, centres, width)) for w in windows]
    mean = np.mean([10.0 ** (row / 10.0) for row in rows], axis=0)
    return [float(10.0 * np.log10(max(v, 1e-30))) for v in mean]


def by_band(
    root: str | Path,
    *,
    margin_db: float = 6.0,
    band_set: str = "third-octave",
) -> list[BandVerdict]:
    """Read a saved run and say what each setting did to the separation, band by band.

    The question a broadband balance cannot answer: whether the quieter channel is
    the louder one scaled, which is what a pair of multipliers does, or something
    with a shape of its own.
    """
    root = Path(root)
    centres, width = BAND_SETS[band_set]
    centres = list(centres)
    leads, by_setting, order = grouped(root)

    out = []
    for stimulus, settings in by_setting.items():
        loaded = {k: [read(p) for p in v] for k, v in settings.items()}
        loudest = max(
            (loaded[s][0] for s in order[stimulus]), key=lambda t: float(np.abs(t[0]).max())
        )
        rate = loudest[1]
        lead_s = leads.get(stimulus, 0.6)
        lead_n = int(lead_s * 0.8 * rate)
        head = loudest[0][:lead_n] if lead_n > rate // 100 else None
        every = [frames for setting in order[stimulus] for frames, _ in loaded[setting]]
        channels, _floor = pair_of_channels(every, head)
        if channels is None or head is None:
            out.append(BandVerdict(stimulus=stimulus, channels=None, not_measured=MONO_SOURCE))
            continue
        left, right = channels
        floor = [
            max(pair)
            for pair in zip(
                _averaged_bands([head], left, rate, centres, width),
                _averaged_bands([head], right, rate, centres, width),
                strict=True,
            )
        ]

        measured: list[BandSetting] = []
        left_out: list[str] = []
        for setting in order[stimulus]:
            rows: list[tuple[list[float], list[float]]] = []
            for frames, _ in loaded[setting]:
                windows = _one_length_windows(frames, lead_n=lead_n, start=int(lead_s * rate))
                if not windows:
                    continue
                rows.append(
                    (
                        _averaged_bands(windows, left, rate, centres, width),
                        _averaged_bands(windows, right, rate, centres, width),
                    )
                )
            if len(rows) < 2:
                # One take cannot say whether a band repeats, and a setting whose
                # scatter is unknown would be read as a setting whose scatter is
                # zero -- every band kept, on the one evidence this reading needs.
                left_out.append(setting)
                continue
            firsts = np.mean([r[0] for r in rows], axis=0)
            seconds = np.mean([r[1] for r in rows], axis=0)
            per_take = np.asarray([np.asarray(r[0]) - np.asarray(r[1]) for r in rows])
            scatter = (
                per_take.max(axis=0) - per_take.min(axis=0)
                if len(rows) > 1
                else np.zeros(len(centres))
            )
            # One normaliser for the pair, so that the two rows are shapes a reader
            # can use and their difference is still exactly the separation.
            together = float(
                10.0 * np.log10(np.mean(10.0 ** (firsts / 10.0) + 10.0 ** (seconds / 10.0)))
            )
            measured.append(
                BandSetting(
                    setting=setting,
                    reference_db=together,
                    first_db=[float(v) - together for v in firsts],
                    second_db=[float(v) - together for v in seconds],
                    floor_db=[v - together for v in floor],
                    scatter_db=[float(v) for v in scatter],
                    kept=[
                        bool(min(a, b) > f + BAND_ABOVE_THE_FLOOR_DB)
                        and bool(s <= BAND_REPEATS_WITHIN_DB)
                        for a, b, f, s in zip(firsts, seconds, floor, scatter, strict=True)
                    ],
                )
            )
        out.append(
            BandVerdict(
                stimulus=stimulus,
                channels=channels,
                centres=centres,
                width_octaves=width,
                settings=measured,
                settings_left_out=left_out,
                margin_db=margin_db,
            )
        )
    return out


__all__ = [
    "BAND_ABOVE_THE_FLOOR_DB",
    "BAND_REPEATS_WITHIN_DB",
    "METHOD",
    "MONO_SOURCE",
    "SECOND_CHANNEL_ABOVE_DB",
    "SEPARATION_BY_BAND",
    "WHY_A_BAND_HAS_TO_REPEAT",
    "WHY_A_FLAT_SEPARATION_IS_THE_YARDSTICK",
    "WHY_THE_WINDOWS_ARE_ONE_LENGTH",
    "BandSetting",
    "BandVerdict",
    "by_band",
    "WHY_A_PATH_AROUND_THE_EFFECT_IS_ASKED",
    "WHY_A_SEPARATION_NEEDS_A_CEILING",
    "WHY_NOT_A_LEVEL",
    "WHY_THE_FLOOR_IS_THE_PAIRS_OWN",
    "WHY_THE_PAIR_IS_IN_INPUT_ORDER",
    "SettingBalance",
    "Verdict",
    "measure",
]
