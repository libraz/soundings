"""What of a take arrived at once and what arrived late, which one window cannot say.

A byte printed `D> 0E - D 0<E` sets how much of what leaves an insertion effect is
the signal that went through it and how much is the signal that went past it.
Sixty-five rows over forty-six types carry one, and what has been measured about
them is which end is which -- not what the settings between the ends do.

**The reading that separated a pan cannot separate these.** A pan puts its two
halves on two channels, so measuring the channels apart measures the halves apart.
A balance puts both halves on both channels, and no reading of the output tells
them apart: what comes back is one level per channel, and one level is one number
however many things went into it.

**What separates them is time.** Where the effect is a delay and its feedback is at
nothing, its return is one copy of the note arriving later, so a take of a gated
note holds the direct sound in one window and the return in another and each can be
read on its own. That is the whole of this stage. It is not a general reading -- a
type whose return begins while the note is still sounding has no window to be read
in and is refused by name rather than read approximately.

What is reported per setting is three levels of one length: the take's own lead,
the window the direct sound arrives in, and the window the return arrives in. The
two figures a byte of this kind turns on come out of the second and third -- the
difference between them, and the two of them together -- and the first is what says
either of them is a level at all.

**Nothing here says which window is dry.** Early and late are where the reading
looked; which of them the page calls the effect is a separate claim resting on a
separate measurement, and putting that word in these rows would be publishing it
twice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .takes import channel_across, grouped, level_db, pair_of_channels, read, rms, spread

METHOD = (
    "Each take was reduced to three levels over windows of one length: the silence before the "
    "note, the window the direct sound arrives in, and the window the effect's return arrives "
    "in. Where a delay's feedback is at nothing the return is one copy of the note arriving "
    "later, so the two halves a balance mixes are in different parts of the take and each can "
    "be read on its own -- which no reading of the output at one instant can do, since both "
    "halves leave on the same channels and one level is one number however many things went "
    "into it."
)

WHY_THE_WINDOWS_ARE_ONE_LENGTH = (
    "The lead, the direct sound and the return are read over windows of the same length, so the "
    "three levels are comparable without a correction being applied to any of them. A floor "
    "measured over a long window and a signal measured over a short one differ by the ratio of "
    "the two before either has said anything about the unit, and a reader would have to know "
    "which window each figure came from to subtract them. Matching the lengths removes the "
    "question instead of answering it."
)

WHY_THE_WINDOWS_ARE_MEASURED_AND_NOT_PRINTED = (
    "Where the return lands was measured on this unit before any setting was swept, and the "
    "windows are placed from that. The printed range says the delay reaches five hundred "
    "milliseconds, which is a statement about a page: a window placed on the strength of it "
    "would read the direct sound's own tail as the return at every setting, and nothing in the "
    "figures would say so. The two settings that bound the reading -- the one where the return "
    "window holds only the direct sound's tail, and the one where the direct window holds only "
    "the floor -- are in the sweep and are reported beside it."
)

WHY_THE_LEAD_IS_THE_FLOOR = (
    "How far each window stood over the same take's own lead is reported per setting, because a "
    "window that has fallen to the lead is not a quiet reading of the unit, it is the interface. "
    "A balance taken to one of its ends is expected to empty one of the two windows, so the "
    "distance from the lead is the figure that says whether the end was reached or only "
    "approached, and it is a bound in one direction rather than a level in either."
)

WHY_A_RETURN_THAT_OVERLAPS_IS_REFUSED = (
    "A type whose return begins before the direct sound has ended has no window this reading can "
    "use, and it is named rather than read: the two would be summed inside one window and the "
    "figure would move with the setting exactly as a separated reading does, while meaning "
    "something else entirely. Measured here, a reverb's return begins a tenth of a second after "
    "the note and is refused on that account; two delays return half a second and more later and "
    "are read."
)

NO_WINDOWS = (
    "The run does not say where its windows go. Where a return lands is a measurement and not a "
    "thing this can work out from a take, so a directory whose manifest carries no windows is "
    "left unread rather than read against a guess."
)

WINDOWS_OVERLAP = (
    "The windows the run names overlap, so the direct sound and the return would be summed "
    "inside one of them, per WHY_A_RETURN_THAT_OVERLAPS_IS_REFUSED."
)


@dataclass
class ArrivalSetting:
    """One setting's takes, as three levels each."""

    setting: str
    lead_db: list[float] = field(default_factory=list)
    early_db: list[float] = field(default_factory=list)
    late_db: list[float] = field(default_factory=list)

    @property
    def typical_lead_db(self) -> float:
        return float(np.median(self.lead_db)) if self.lead_db else float("nan")

    @property
    def typical_early_db(self) -> float:
        return float(np.median(self.early_db)) if self.early_db else float("nan")

    @property
    def typical_late_db(self) -> float:
        return float(np.median(self.late_db)) if self.late_db else float("nan")

    @property
    def late_less_early_db(self) -> list[float]:
        """Per take, the late window's level less the early one's.

        The same arithmetic a balance between two channels reports, asked between
        two windows instead. A setting that moves this and not the total below is
        moving signal from one window to the other rather than changing how much
        there is.
        """
        return [b - a for a, b in zip(self.early_db, self.late_db, strict=True)]

    @property
    def together_db(self) -> list[float]:
        """Per take, the two windows together, which is the figure that separates the laws."""
        return [
            level_db(np.sqrt(10.0 ** (a / 10.0) + 10.0 ** (b / 10.0)))
            for a, b in zip(self.early_db, self.late_db, strict=True)
        ]

    @property
    def typical_together_db(self) -> float:
        held = self.together_db
        return float(np.median(held)) if held else float("nan")

    @property
    def spread_db(self) -> float:
        return spread(self.late_less_early_db)

    @property
    def together_spread_db(self) -> float:
        return spread(self.together_db)

    @property
    def early_over_the_lead_db(self) -> float:
        return self.typical_early_db - self.typical_lead_db

    @property
    def late_over_the_lead_db(self) -> float:
        return self.typical_late_db - self.typical_lead_db

    def to_json(self) -> dict:
        return {
            "setting": self.setting,
            "lead_db": [round(v, 2) for v in self.lead_db],
            "early_db": [round(v, 2) for v in self.early_db],
            "late_db": [round(v, 2) for v in self.late_db],
            "late_less_early_db": [round(v, 2) for v in self.late_less_early_db],
            "together_db": [round(v, 2) for v in self.together_db],
            "late_less_early_spread_db": round(self.spread_db, 2),
            "together_spread_db": round(self.together_spread_db, 2),
            "early_over_the_lead_db": round(self.early_over_the_lead_db, 2),
            "late_over_the_lead_db": round(self.late_over_the_lead_db, 2),
        }


@dataclass
class ArrivalVerdict:
    """What one stimulus' settings did to the two windows."""

    stimulus: str
    channels: list[int] | None
    windows_s: dict | None = None
    settings: list[ArrivalSetting] = field(default_factory=list)
    margin_db: float = 6.0
    not_measured: str | None = None

    @property
    def yardstick_db(self) -> float:
        """The steadiest setting's own scatter, which is what a claim has to clear.

        The steadiest rather than the worst, for the reason the balance stage gives:
        the question a wide setting is being asked is whether it is wide, so taking
        it into the yardstick would be measuring it against itself.
        """
        spreads = [s.spread_db for s in self.settings]
        return min(spreads) if spreads else float("nan")

    @property
    def moved_between_settings(self) -> bool:
        """Whether the two windows' difference moved further than the run resolves."""
        if len(self.settings) < 2:
            return False
        typical = [s.typical_late_db - s.typical_early_db for s in self.settings]
        return bool(max(typical) - min(typical) > self.yardstick_db + self.margin_db)

    @property
    def moved_in_opposite_directions(self) -> bool:
        """Whether the setting that fills one window is not the one that fills the other.

        What says the byte moves signal between the two rather than turning one of
        them up. Both windows have to have moved by more than the margin for the
        comparison to mean anything: a window that never moved has no loudest
        setting worth naming.
        """
        if len(self.settings) < 2:
            return False
        early = [s.typical_early_db for s in self.settings]
        late = [s.typical_late_db for s in self.settings]
        if max(early) - min(early) <= self.margin_db or max(late) - min(late) <= self.margin_db:
            return False
        return int(np.argmax(early)) != int(np.argmax(late))

    @property
    def together_spread_db(self) -> float:
        """How far the two windows together moved over the whole sweep.

        The figure the candidates for a byte of this kind differ in: a pair of
        multipliers whose squares sum to one holds this flat, a pair that sums in
        amplitude puts it three decibels down in the middle, and a table does
        whatever it was written to do.
        """
        held = [s.typical_together_db for s in self.settings]
        return float(max(held) - min(held)) if len(held) > 1 else 0.0

    @property
    def quietest_early_over_the_lead_db(self) -> float:
        return min((s.early_over_the_lead_db for s in self.settings), default=float("nan"))

    @property
    def quietest_late_over_the_lead_db(self) -> float:
        return min((s.late_over_the_lead_db for s in self.settings), default=float("nan"))

    def describe(self) -> str:
        if self.not_measured:
            return f"{self.stimulus}: not measured -- {self.not_measured}"
        if not self.settings:
            return f"{self.stimulus}: nothing to measure"
        how = []
        if self.moved_between_settings:
            how.append("the two windows' difference moved between the settings")
        if self.moved_in_opposite_directions:
            how.append("the window one setting fills is not the window another fills")
        found = "; ".join(how) if how else "nothing the two windows could show"
        where = "channels " + " and ".join(str(c) for c in self.channels or ())
        return (
            f"{self.stimulus} ({where}): {found}. the two together move "
            f"{self.together_spread_db:.1f} dB over the sweep; the quieter window reaches "
            f"{self.quietest_early_over_the_lead_db:.1f} and "
            f"{self.quietest_late_over_the_lead_db:.1f} dB over the lead at its emptiest"
        )

    def to_json(self) -> dict:
        if self.not_measured:
            return {
                "stimulus_name": self.stimulus,
                "channels": list(self.channels) if self.channels else None,
                "measured": False,
                "not_measured": self.not_measured,
            }
        return {
            "stimulus_name": self.stimulus,
            "channels": list(self.channels) if self.channels else None,
            "measured": True,
            "margin_db": self.margin_db,
            "windows_s": dict(self.windows_s or {}),
            "yardstick_db": round(self.yardstick_db, 2),
            "moved_between_settings": self.moved_between_settings,
            "moved_in_opposite_directions": self.moved_in_opposite_directions,
            "the_two_windows_together_spread_db": round(self.together_spread_db, 2),
            "quietest_early_over_the_lead_db": round(self.quietest_early_over_the_lead_db, 2),
            "quietest_late_over_the_lead_db": round(self.quietest_late_over_the_lead_db, 2),
            "by_setting": [s.to_json() for s in self.settings],
        }


def _windows_of(root: Path) -> dict | None:
    """Where the run says its three windows go, in seconds from the start of a take."""
    manifest = json.loads((root / "takes-manifest.json").read_text())
    asked = manifest.get("windows_s")
    if not isinstance(asked, dict):
        return None
    if not {"length", "lead", "early", "late"} <= set(asked):
        return None
    return {k: float(asked[k]) for k in ("length", "lead", "early", "late")}


ROUNDING_S = 1e-6
"""Slack on a window boundary, which is arithmetic and not a tolerance on the unit.

Three windows placed end to end at a tenth, three tenths and a half of a second do
not satisfy `0.3 - 0.1 >= 0.2` in binary floating point, and a run whose windows
exactly abut is the ordinary case rather than a marginal one. A microsecond is
under a sample at every rate this records at, so nothing that overlaps by anything
a take could hold passes through it.
"""


def _apart(windows: dict) -> bool:
    """Whether the three windows run in order and none of them reaches into the next."""
    length = windows["length"]
    starts = [windows["lead"], windows["early"], windows["late"]]
    return length > 0 and all(
        b - a >= length - ROUNDING_S for a, b in zip(starts[:-1], starts[1:], strict=True)
    )


def _level(
    frames: np.ndarray, channels: list[int], *, at: float, length: float, rate: int
) -> float:
    """One window of one take over the channels the unit arrived on, in dBFS."""
    first = int(round(at * rate))
    last = first + int(round(length * rate))
    if first < 0 or last > frames.shape[0]:
        return float("nan")
    return level_db(rms(frames[first:last, channels]))


def measure(root: str | Path, *, margin_db: float = 6.0) -> list[ArrivalVerdict]:
    """Read a saved run and say what each setting put in each of the two windows."""
    root = Path(root)
    leads, by_setting, order = grouped(root)
    windows = _windows_of(root)

    out: list[ArrivalVerdict] = []
    for stimulus, settings in by_setting.items():
        if windows is None:
            out.append(ArrivalVerdict(stimulus=stimulus, channels=None, not_measured=NO_WINDOWS))
            continue
        if not _apart(windows):
            out.append(
                ArrivalVerdict(
                    stimulus=stimulus,
                    channels=None,
                    windows_s=windows,
                    not_measured=WINDOWS_OVERLAP,
                )
            )
            continue
        loaded = {k: [read(p) for p in v] for k, v in settings.items()}
        loudest = max(
            (loaded[s][0] for s in order[stimulus]), key=lambda t: float(np.abs(t[0]).max())
        )
        rate = loudest[1]
        lead_n = int(leads.get(stimulus, 0.6) * 0.8 * rate)
        head = loudest[0][:lead_n] if lead_n > rate // 100 else None
        every = [frames for setting in order[stimulus] for frames, _ in loaded[setting]]
        # Two channels where the unit arrived on two and one where it arrived on
        # one. Unlike a balance between the channels, this reading has an answer
        # either way: what it separates is two parts of a take, not two legs of a
        # cable, and a mono source has both parts in it.
        pair, _floor = pair_of_channels(every, head)
        channels = list(pair) if pair is not None else [channel_across(*every)[0]]

        measured: list[ArrivalSetting] = []
        for setting in order[stimulus]:
            found = ArrivalSetting(setting=setting)
            for frames, _ in loaded[setting]:
                found.lead_db.append(
                    _level(
                        frames, channels, at=windows["lead"], length=windows["length"], rate=rate
                    )
                )
                found.early_db.append(
                    _level(
                        frames, channels, at=windows["early"], length=windows["length"], rate=rate
                    )
                )
                found.late_db.append(
                    _level(
                        frames, channels, at=windows["late"], length=windows["length"], rate=rate
                    )
                )
            measured.append(found)
        out.append(
            ArrivalVerdict(
                stimulus=stimulus,
                channels=channels,
                windows_s=windows,
                settings=measured,
                margin_db=margin_db,
            )
        )
    return out


__all__ = [
    "METHOD",
    "NO_WINDOWS",
    "WHY_A_RETURN_THAT_OVERLAPS_IS_REFUSED",
    "WHY_THE_LEAD_IS_THE_FLOOR",
    "WHY_THE_WINDOWS_ARE_ONE_LENGTH",
    "WHY_THE_WINDOWS_ARE_MEASURED_AND_NOT_PRINTED",
    "WINDOWS_OVERLAP",
    "ArrivalSetting",
    "ArrivalVerdict",
    "measure",
]
