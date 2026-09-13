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

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .takes import read

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

SECOND_CHANNEL_ABOVE_DB = 20.0
"""How far over the noise the second channel has to sit to be a channel at all.

Below this the pair is one channel and a floor, and a balance computed from it is
the floor's own wander. The inputs this ran on sit near -94 dB unloaded against a
unit arriving at -48, so the separation is not a fine judgement.
"""


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
        return _spread(self.balance_db)

    @property
    def total_spread_db(self) -> float:
        return _spread(self.total_db)

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


def _db(x: float) -> float:
    return float(20.0 * np.log10(x + 1e-15))


def _rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(frame * frame)))


def _spread(values: list[float]) -> float:
    return float(max(values) - min(values)) if len(values) > 1 else 0.0


#: Why the channels are chosen from every take rather than from one of them.
WHY_ACROSS_THE_SETTINGS = (
    "A channel the unit arrived on is one that carried signal at some point in the run, so "
    "each channel is taken at the loudest it ever reached. Asking a single take instead "
    "cannot see a parameter that moves signal from one channel to the other: taken to its "
    "two ends a panpot leaves every take with one live channel and one empty one, so "
    "whichever take is asked answers mono -- and the one parameter this measurement exists "
    "for is the one it would refuse."
)


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


WHY_THE_FLOOR_IS_THE_PAIRS_OWN = (
    "The floor the quieter channel has to clear is read from the lead of the channels the unit "
    "arrived on, and not from the lead of every input the interface has. An input the unit is "
    "not on is not silent: measured here, two unused inputs sat at -70 and -75 dBFS through "
    "every take while the two carrying the unit sat at -110, so a floor taken over all of them "
    "was thirty decibels above the one the reading is against. A run whose quieter channel "
    "reached -59.7 was refused as a mono source by three tenths of a decibel on that account -- "
    "which is not a bound being reported, it is a statement that the unit arrived on one "
    "channel, and it was false."
)


def _pair_of_channels(
    takes: list[np.ndarray], lead: np.ndarray | None
) -> tuple[tuple[int, int] | None, float]:
    """The two channels the unit arrived on, in the interface's own order, and their floor.

    Each channel is taken at the loudest it reached across every take of every
    setting, per WHY_ACROSS_THE_SETTINGS. The pair is picked on level alone, which
    needs no floor, and the floor is then read from those two channels' own lead,
    per WHY_THE_FLOOR_IS_THE_PAIRS_OWN. None when the quieter of the two does not
    clear it. The pair is put back into input order, per
    WHY_THE_PAIR_IS_IN_INPUT_ORDER.
    """
    if not takes:
        return None, -120.0
    width = min(frames.shape[1] for frames in takes)
    levels = [max(_db(_rms(frames[:, c])) for frames in takes) for c in range(width)]
    order = sorted(range(len(levels)), key=lambda c: levels[c], reverse=True)
    if len(order) < 2:
        return None, -120.0
    pair = sorted(order[:2])
    floor = _db(_rms(lead[:, pair])) if lead is not None and lead.shape[0] else -120.0
    if levels[order[1]] < floor + SECOND_CHANNEL_ABOVE_DB:
        return None, floor
    return (pair[0], pair[1]), floor


def measure(root: str | Path, *, margin_db: float = 6.0) -> list[Verdict]:
    """Read a saved run and say what each setting did to the balance.

    One verdict per stimulus the run holds takes for, so a parameter asked under
    several notes carries what each of them saw, as everywhere else here.
    """
    root = Path(root)
    manifest = json.loads((root / "takes-manifest.json").read_text())
    leads = {s.get("name"): s.get("lead_s") or 0.6 for s in manifest.get("stimuli", [])}

    grouped: dict[str, dict[str, list[Path]]] = {}
    order: dict[str, list[str]] = {}
    for entry in manifest.get("takes", []):
        setting = str(entry.get("setting"))
        stimulus = str(entry.get("stimulus"))
        grouped.setdefault(stimulus, {}).setdefault(setting, []).append(root / entry["file"])
        seen = order.setdefault(stimulus, [])
        if setting not in seen:
            seen.append(setting)

    out = []
    for stimulus, settings in grouped.items():
        loaded = {k: [read(p) for p in v] for k, v in settings.items()}
        # The rate and the noise floor come from the setting that sounded
        # loudest, not from the first one: a parameter that silences its part at
        # one of its two values would otherwise have its floor read out of noise.
        # Measured here, one address in the part block silences the part at 127,
        # which is the setting the plan asks first. The channels are a separate
        # question and are asked of every take, per WHY_ACROSS_THE_SETTINGS.
        loudest = max(
            (loaded[s][0] for s in order[stimulus]), key=lambda t: float(np.abs(t[0]).max())
        )
        rate = loudest[1]
        lead = int((leads.get(stimulus, 0.6) * 0.8) * rate)
        head = loudest[0][:lead] if lead > rate // 100 else None
        every = [frames for setting in order[stimulus] for frames, _ in loaded[setting]]
        channels, _floor = _pair_of_channels(every, head)
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
                a, b = _db(_rms(frames[:, left])), _db(_rms(frames[:, right]))
                found.balance_db.append(a - b)
                found.total_db.append(_db(_rms(frames[:, [left, right]])))
            measured.append(found)
        out.append(
            Verdict(stimulus=stimulus, settings=measured, channels=channels, margin_db=margin_db)
        )
    return out


__all__ = [
    "METHOD",
    "MONO_SOURCE",
    "SECOND_CHANNEL_ABOVE_DB",
    "WHY_NOT_A_LEVEL",
    "WHY_THE_FLOOR_IS_THE_PAIRS_OWN",
    "WHY_THE_PAIR_IS_IN_INPUT_ORDER",
    "SettingBalance",
    "Verdict",
    "measure",
]
