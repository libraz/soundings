"""Whether changing a parameter changes the sound, against the unit's own repeatability.

The address map says what a parameter stores. It cannot say whether storing it
does anything, and the four levels this archive classifies an address into --
audible, state, accept, ignore -- turn on exactly that. A parameter that is
stored and read back faithfully and reaches nothing in the signal path is
indistinguishable, from the MIDI side alone, from one that reshapes the whole
voice.

The comparison has to be made against a yardstick rather than against zero.
Every pair of takes differs a little: by the noise, and by whatever in the unit
does not repeat. So a difference between two settings only means something if it
is bigger than the difference between two takes of the *same* setting, and that
is measured here in the same session rather than assumed -- it moves with the
patch, the note and the room the unit is sitting in.

**Two ways to be audible, and one of them survives the alignment.** A parameter
that only changes level leaves no residual at all once the best-fitting gain is
divided out, which is what the comparator does to keep a level drift from
masquerading as a spectral change. So level is judged on its own yardstick,
beside the residual, and either one clearing its own bar makes the parameter
audible.

**Turning on a modulator makes the difference method blind, so that counts as
its own evidence.** A parameter that starts a free-running LFO -- a chorus, a
rotary, a modulated delay -- gives takes that no longer repeat, because the
modulator is at a different phase each time the note is struck. The yardstick
then swallows everything, and the difference between the settings can never
clear it however large it is. Measured on this unit: with the chorus on, two
takes of the same note differ by 3 dB where dry they differ by 55. So each
setting's repeatability is kept separately, and a setting that repeats far worse
than the other is audible on that ground alone -- something was switched on that
moves, which is a fact about the signal path even though no residual can show it.

**A null here is about this stimulus, not about the parameter.** A filter
resonance heard only on a sustained note, a velocity curve heard only at the
extremes, a release parameter heard only after note-off: each reads as inaudible
if the note asked the wrong question. What the verdict carries is the stimulus
it was reached with.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .stability import Comparison, compare

UNUSABLE_ABOVE_DB = -12.0
"""Above this, takes of one setting differ so much that no change could clear the yardstick.

Set below the worst yardstick a healthy stimulus has produced on this chain: the
quiet 'soft' note, at -16.4 dB, whose verdict was sound.
"""


@dataclass
class Verdict:
    label: str
    stimulus: str

    within_db: float
    """Worst residual between two takes of the same setting. The yardstick."""

    across_db: float
    """Worst residual between the settings, aligned and levelled."""

    within_level_db: float
    """Widest level difference between two takes of the same setting."""

    across_level_db: float
    """Level difference between the settings."""

    stimulus_name: str = ""

    within_each_db: tuple[float, float] = (float("nan"), float("nan"))
    """Each setting's repeatability as dB above its own noise floor.

    Above its own floor, rather than below its own signal. A setting 20 dB
    quieter has 20 dB less signal over the same converter noise, so its residual
    is 20 dB worse for a reason that has nothing to do with the unit -- measured
    here on CC7, where 40 against 127 differed by 19 dB of apparent
    repeatability and by 20.09 dB of level. Distance to the floor cancels that
    and leaves only what actually failed to repeat.
    """

    margin_db: float = 6.0

    floor_db: float = float("nan")
    takes: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def changed_the_shape(self) -> bool:
        return bool(self.across_db > self.within_db + self.margin_db)

    @property
    def changed_the_level(self) -> bool:
        return bool(abs(self.across_level_db) > abs(self.within_level_db) + self.margin_db)

    @property
    def changed_the_repeatability(self) -> bool:
        """One setting repeats far worse than the other, so something that moves came on.

        Needs at least three takes per setting. With two there is one pair per
        group, and a single pair cannot be told from an outlier.
        """
        first, second = self.within_each_db
        if np.isnan(first) or np.isnan(second):
            return False
        return bool(abs(first - second) > self.margin_db * 2)

    @property
    def audible(self) -> bool:
        return bool(
            self.changed_the_shape or self.changed_the_level or self.changed_the_repeatability
        )

    @property
    def inconclusive(self) -> bool:
        """A null nothing could have cleared, which is not the same as a null.

        When takes of one setting barely resemble each other, the yardstick is
        most of the signal and no change however large can get over it. Reporting
        that as "inaudible" states a fact about the parameter on the strength of
        a measurement that had no power to find one. Seen once on hardware: a
        disturbed take left a stimulus with a 0.0 dB yardstick, and the parameter
        it was asked about was a volume control.

        An audible verdict is kept regardless -- something was found, so the
        measurement plainly had the power to find it.
        """
        return bool(not self.audible and self.within_db > UNUSABLE_ABOVE_DB)

    def describe(self) -> str:
        if self.inconclusive:
            return (
                f"{self.label}: inconclusive. Takes of one setting differ by "
                f"{self.within_db:.1f} dB, so the yardstick is most of the signal and no "
                "change could have cleared it. This says nothing about the parameter."
            )
        if not self.audible:
            return (
                f"{self.label}: nothing above the noise. Takes of one setting differ by "
                f"{self.within_db:.1f} dB and the two settings differ by {self.across_db:.1f} dB, "
                f"which is not {self.margin_db:.0f} dB clear of it. "
                f"Inaudible under {self.stimulus}."
            )
        how = []
        if self.changed_the_shape:
            how.append(f"shape, {self.across_db - self.within_db:+.1f} dB over the yardstick")
        if self.changed_the_level:
            how.append(f"level, {self.across_level_db:+.2f} dB")
        if self.changed_the_repeatability:
            first, second = self.within_each_db
            how.append(
                f"repeatability, typically {first:.1f} dB against {second:.1f} dB over their "
                "own noise -- something that moves came on, and no residual can measure it"
            )
        return (
            f"{self.label}: audible -- {' and '.join(how)}. "
            f"Takes of one setting differ by {self.within_db:.1f} dB, "
            f"the settings by {self.across_db:.1f} dB."
        )

    def to_json(self) -> dict:
        return {
            "label": self.label,
            "stimulus": self.stimulus,
            "stimulus_name": self.stimulus_name,
            "takes_per_setting": self.takes,
            "same_setting_residual_db": round(self.within_db, 2),
            "each_setting_above_its_floor_db": [
                None if np.isnan(v) else round(v, 2) for v in self.within_each_db
            ],
            "across_setting_residual_db": round(self.across_db, 2),
            "same_setting_level_db": round(self.within_level_db, 3),
            "across_setting_level_db": round(self.across_level_db, 3),
            "margin_db": self.margin_db,
            "noise_floor_db": None if np.isnan(self.floor_db) else round(self.floor_db, 2),
            "changed_the_shape": self.changed_the_shape,
            "changed_the_repeatability": self.changed_the_repeatability,
            "changed_the_level": self.changed_the_level,
            "audible": self.audible,
            "inconclusive": self.inconclusive,
            "caveat": "A null is about this stimulus. A parameter heard only on a longer "
            "note, at another velocity, or after note-off would read as inaudible here.",
        }


def _worst(comparisons: list[Comparison]) -> tuple[float, float]:
    """Worst residual and widest level departure across a set of comparisons."""
    if not comparisons:
        return float("-inf"), 0.0
    return (
        max(c.residual_db for c in comparisons),
        max((c.gain_db for c in comparisons), key=abs),
    )


def _typical_headroom(comparisons: list[Comparison]) -> float:
    """Median residual above the noise floor, for the modulator channel.

    Median rather than worst, and the direction is the reason. Everywhere else
    the worst pair is the conservative choice, because it makes a change harder
    to claim. Here it is the opposite: one disturbed take raises the worst pair
    of its own group and that alone reads as a modulator switching on. Measured:
    the same reverb contrast ran at 15.5 dB over three takes and 1.4 dB over
    four, on a unit that does repeat with the reverb on.

    A real modulator raises every pair in the group, so the median moves with it
    and a single bad take does not.
    """
    if len(comparisons) < 2:
        # One pair cannot say whether a value is typical or an outlier.
        return float("nan")
    return float(np.median([c.headroom_db for c in comparisons]))


def judge(
    first: list[np.ndarray],
    second: list[np.ndarray],
    sample_rate: int,
    *,
    label: str,
    stimulus: str,
    silence_before: float,
    stimulus_name: str = "",
    margin_db: float = 6.0,
) -> Verdict:
    """Decide whether two settings sound different, using their own repeatability.

    Every take of a setting is compared with the first take of that setting to
    fix the yardstick, and every take of the second setting with the first take
    of the first, to measure the change. Both use the same alignment and the same
    level fit, so the two numbers are on one scale and can be subtracted.
    """
    within_first = [
        compare(first[0], t, sample_rate, silence_before=silence_before) for t in first[1:]
    ]
    within_second = [
        compare(second[0], t, sample_rate, silence_before=silence_before) for t in second[1:]
    ]
    across = [compare(first[0], t, sample_rate, silence_before=silence_before) for t in second]

    within_db, within_level = _worst(within_first + within_second)
    across_db, across_level = _worst(across)
    return Verdict(
        label=label,
        stimulus=stimulus,
        stimulus_name=stimulus_name,
        within_db=within_db,
        within_each_db=(
            _typical_headroom(within_first),
            _typical_headroom(within_second),
        ),
        across_db=across_db,
        within_level_db=within_level,
        across_level_db=across_level,
        margin_db=margin_db,
        floor_db=across[0].floor_db if across else float("nan"),
        takes=min(len(first), len(second)),
    )


@dataclass
class Overall:
    """One parameter's verdict over every stimulus it was asked under."""

    label: str
    verdicts: list[Verdict] = field(default_factory=list)

    @property
    def audible(self) -> bool:
        return any(v.audible for v in self.verdicts)

    @property
    def heard_by(self) -> list[str]:
        return [v.stimulus_name for v in self.verdicts if v.audible]

    @property
    def deaf_to(self) -> list[str]:
        return [v.stimulus_name for v in self.verdicts if not v.audible and not v.inconclusive]

    @property
    def inconclusive_under(self) -> list[str]:
        return [v.stimulus_name for v in self.verdicts if v.inconclusive]

    def describe(self) -> str:
        if not self.verdicts:
            return f"{self.label}: nothing was asked"
        unusable = (
            f"; {', '.join(self.inconclusive_under)} could not measure it"
            if self.inconclusive_under
            else ""
        )
        if self.audible:
            missed = f"; {', '.join(self.deaf_to)} did not hear it" if self.deaf_to else ""
            return f"{self.label}: AUDIBLE, heard by {', '.join(self.heard_by)}{missed}{unusable}"
        if not self.deaf_to:
            return (
                f"{self.label}: INCONCLUSIVE. Every stimulus tried "
                f"({', '.join(self.inconclusive_under)}) had a yardstick nothing could clear, "
                "so this is not a null and must be asked again."
            )
        return (
            f"{self.label}: not audible under {', '.join(self.deaf_to)}{unusable}. "
            "That is a statement about these notes, not about the parameter -- another "
            "stimulus may still hear it."
        )

    def to_json(self) -> dict:
        return {
            "label": self.label,
            "audible": self.audible,
            "heard_by": self.heard_by,
            "not_heard_by": self.deaf_to,
            "inconclusive_under": self.inconclusive_under,
            "conclusive": bool(self.audible or self.deaf_to),
            "verdict_rule": "Audible under any stimulus is audible: one note hearing the "
            "change proves the parameter reaches the signal path, and the others failing "
            "to hear it says only that they asked the wrong question. The reverse does not "
            "hold, so a null carries the list of what was tried. A stimulus whose takes of "
            "one setting barely resemble each other is reported as inconclusive rather than "
            "as a null, since no change could have cleared that yardstick.",
            "by_stimulus": [v.to_json() for v in self.verdicts],
        }


__all__ = ["UNUSABLE_ABOVE_DB", "Overall", "Verdict", "judge"]
