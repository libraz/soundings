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
    channels: tuple[int, int]
    margin_db: float = 6.0

    @property
    def yardstick_db(self) -> float:
        """The steadier setting's own scatter, which is what a claim has to clear.

        The steadier rather than the worse: the question the wider one is being
        asked is whether it is wide, so taking it into the yardstick would be
        measuring it against itself.
        """
        spreads = [s.spread_db for s in self.settings]
        return min(spreads) if spreads else float("nan")

    @property
    def moved_between_settings(self) -> bool:
        if len(self.settings) != 2:
            return False
        gap = abs(self.settings[0].typical_db - self.settings[1].typical_db)
        return bool(gap > self.yardstick_db + self.margin_db)

    @property
    def did_not_repeat(self) -> bool:
        """Whether one setting's balance landed somewhere new on each take.

        The asymmetry is the evidence and it has to be an asymmetry: a chain that
        wanders wanders at both settings, and a parameter is only implicated when
        one setting is steady in the same session that the other is not.
        """
        if len(self.settings) != 2:
            return False
        first, second = (s.spread_db for s in self.settings)
        return bool(abs(first - second) > self.margin_db * 2)

    @property
    def while_the_total_stayed(self) -> bool:
        """Whether the pair's total held still while the balance did not.

        What separates a pan from a level change that reached one channel, and
        neither channel on its own can tell them apart.
        """
        widest = max((s.total_spread_db for s in self.settings), default=float("nan"))
        return bool(self.did_not_repeat and widest < max(s.spread_db for s in self.settings) / 2)

    def describe(self) -> str:
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
        return {
            "stimulus_name": self.stimulus,
            "channels": list(self.channels),
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


def _pair_of_channels(frames: np.ndarray, floor_db: float) -> tuple[int, int] | None:
    """The two channels the unit arrived on, loudest first.

    None when only one of them is over the floor: a mono source has no balance,
    and computing one from a channel holding nothing but noise would report the
    noise's own wander as a pan.
    """
    levels = [_db(_rms(frames[:, c])) for c in range(frames.shape[1])]
    order = sorted(range(len(levels)), key=lambda c: levels[c], reverse=True)
    if len(order) < 2 or levels[order[1]] < floor_db + SECOND_CHANNEL_ABOVE_DB:
        return None
    return order[0], order[1]


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
        # Which channels the unit arrived on is asked of the setting that sounded
        # loudest, not of the first one. A parameter that silences its part at
        # one of its two values leaves the other setting's takes holding nothing
        # but noise, and choosing the pair from those picks whichever input
        # carried the most of it -- then refuses the run for being mono. Measured
        # here: one address in the part block silences the part at 127, which is
        # the setting the plan asks first, and it was the one run of forty-six
        # this returned nothing for.
        loudest = max(
            (loaded[s][0] for s in order[stimulus]), key=lambda t: float(np.abs(t[0]).max())
        )
        rate = loudest[1]
        lead = int((leads.get(stimulus, 0.6) * 0.8) * rate)
        floor = _db(_rms(loudest[0][:lead])) if lead > rate // 100 else -120.0
        channels = _pair_of_channels(loudest[0], floor)
        if channels is None:
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
    "SettingBalance",
    "Verdict",
    "measure",
]
