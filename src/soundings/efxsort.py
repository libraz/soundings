"""Sorting insertion effects by what a modulator does to a unit's repeatability.

`efxmotion` answers the same question by tracking the delay between the two
takes, which is the direct measurement and also the fragile one: it needs a
broadband aperiodic source, enough of the take to stand over the noise floor, and
a swing inside the band a 10 ms frame can follow. Measured on this unit, real
takes of a real insertion effect meet the last of those rarely.

This is the other route, and it needs none of that. A free-running modulator is
at a different phase every time the note is struck, so **takes of the setting
with the effect on stop resembling each other** while takes with it bypassed
still do. That asymmetry is already measured by every `contrast` run, as its own
yardstick, and it is what `changed_the_repeatability` reports.

**A type that is not audible at all is sorted as neither.** Nothing reached the
signal path under that note, so it holds no evidence about a modulator -- the
same failure the delay route's injected control exists to catch, arriving here
as the audible verdict rather than as a separate measurement.

**The two routes are kept apart and reported apart.** They rest on different
properties, so a type they disagree about is a finding rather than a number to
be reconciled: the delay route can miss a modulator it cannot follow, and this
one can be fooled by anything else that stops a take repeating.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .audible import LEVEL_ARTEFACT_REACHED_DB, MODULATOR_ABOVE_DB

METHOD = (
    "Each insertion effect type was recorded four times with the part routed through it and four "
    "times with it bypassed, and the two sets were compared against each other and against "
    "themselves. A free-running modulator is at a different phase each time the note is struck, "
    "so the takes made with it on stop resembling one another while the bypassed takes still do. "
    "A type is sorted as moving on that asymmetry, as static when the effect was audible and both "
    "settings repeated alike, and as neither when nothing was audible under the note at all."
)

WHY_NOT_AUDIBLE = (
    "These types changed nothing this note could hear, so they carry no evidence either way about "
    "a modulator. An effect that never reached the signal path repeats exactly as well as one "
    "standing still, and reading the second from the first would be a verdict on the routing, or "
    "on the note, wearing the effect's name."
)

WHY_ASYMMETRY = (
    "Moving is read from the two settings repeating differently, not from the two settings "
    "sounding different. A modulator makes the difference method blind -- with it on, two takes "
    "of one note disagree about nearly everything -- so the yardstick swallows the change and no "
    "residual can show it. The blindness is the evidence."
)

WHY_INSIDE_THE_GAP = (
    "These types opened a gap between the two settings' repeatability, but the setting that "
    "repeated worse landed between the worst a level change alone has produced on this chain and "
    "the bar a modulator is claimed above. Inside that band the calibration does not separate the "
    "two explanations: a shallow modulator and a level change at a poor signal to noise both "
    "reach it. They are reported here rather than in either pile, because the bar has to fall "
    "somewhere and a reading just under it is not the same fact as a reading far under it."
)

WHY_TWO_ROUTES = (
    "This sort and the delay-tracked one rest on different properties and are reported apart. "
    "Tracking measures the motion itself and can miss one it cannot follow: too deep for a frame, "
    "too shallow to find, or folded by the note's own period. This one needs none of that and is "
    "correspondingly less specific -- it says a modulator is there, never its rate or its depth, "
    "and anything else that stops a take repeating would read the same. A type the two disagree "
    "about is a finding, not a number to be averaged."
)


@dataclass
class TypeSort:
    """What one type's own repeatability said about it."""

    type_id: str
    stimulus: str
    audible: bool
    inconclusive: bool
    moves: bool
    unrepeatable_db: list[float]
    """What fails to repeat at each setting, bypassed first."""

    heard_by: str | None = None
    """Which of the audible module's three grounds carried the verdict."""

    @property
    def inside_the_gap(self) -> bool:
        """Whether the asymmetry is real but lands where the calibration cannot read it.

        The modulator bar was set between a real chorus and the worst a pure level
        change reached, and a setting inside that band is evidence for neither. It
        clears the asymmetry test and fails the bar, so without this it would be
        filed as standing still on the strength of a number nobody can interpret.
        """
        if len(self.unrepeatable_db) != 2 or self.moves:
            return False
        first, second = self.unrepeatable_db
        worse = max(first, second)
        return bool(
            abs(first - second) > 12.0 and LEVEL_ARTEFACT_REACHED_DB < worse <= MODULATOR_ABOVE_DB
        )

    @property
    def verdict(self) -> str:
        if self.inconclusive or not self.audible or self.inside_the_gap:
            return "could not say"
        return "moves" if self.moves else "static"

    def to_json(self) -> dict:
        return {
            "type": self.type_id,
            "stimulus": self.stimulus,
            "verdict": self.verdict,
            "audible": self.audible,
            "inconclusive": self.inconclusive,
            "unrepeatable_db": self.unrepeatable_db,
            "inside_the_calibration_gap": self.inside_the_gap,
            "audible_by": self.heard_by,
        }


def _ground(entry: dict) -> str | None:
    for name, key in (
        ("repeatability", "changed_the_repeatability"),
        ("shape", "changed_the_shape"),
        ("level", "changed_the_level"),
    ):
        if entry.get(key):
            return name
    return None


def sort_one(record: dict, type_id: str) -> TypeSort | None:
    """Read one contrast record. None when it asked no stimulus."""
    entries = record.get("by_stimulus") or []
    if not entries:
        return None
    # The stimulus that heard most, rather than the first. A type asked under
    # several notes is moving if any of them saw it move, on the same reasoning
    # that makes audible-under-any audible.
    entry = max(entries, key=lambda e: (bool(e.get("changed_the_repeatability")), e.get("audible")))
    return TypeSort(
        type_id=type_id,
        stimulus=str(entry.get("stimulus_name", "")),
        audible=bool(entry.get("audible")),
        inconclusive=bool(entry.get("inconclusive")),
        moves=bool(entry.get("changed_the_repeatability")),
        unrepeatable_db=list(entry.get("each_setting_unrepeatable_db") or []),
        heard_by=_ground(entry),
    )


def survey(root: str | Path, progress=None) -> list[TypeSort]:
    """Sort every type with a contrast record under this directory, one file each."""
    out = []
    for path in sorted(Path(root).glob("*.json")):
        found = sort_one(json.loads(path.read_text()), path.stem.replace("-", " ").upper())
        if found is None:
            continue
        out.append(found)
        if progress:
            progress(found)
    return out


def partition(found: list[TypeSort]) -> dict[str, list[str]]:
    return {
        "moving": [f.type_id for f in found if f.verdict == "moves"],
        "static": [f.type_id for f in found if f.verdict == "static"],
        "could_not_say": [f.type_id for f in found if f.verdict == "could not say"],
    }


def inside_the_gap(found: list[TypeSort]) -> list[str]:
    """Which of the undecided types are undecided for the calibration's own reason.

    Not a fourth pile: these are inside `could_not_say`, and this says which of
    the ways of being undecided they took. A type here differs from one that
    reached nothing -- it plainly did something, and the number it did it by is
    one the bar cannot read.
    """
    return [f.type_id for f in found if f.inside_the_gap]


def disagreements(sorted_here: list[TypeSort], tracked: list) -> list[dict]:
    """Types the two routes reached different verdicts for, neither overruled.

    A verdict of "could not say" from either side is not a disagreement: it is
    one route declining, which the other is free to answer.
    """
    by_id = {f.type_id: f.verdict for f in tracked}
    out = []
    for entry in sorted_here:
        other = by_id.get(entry.type_id)
        if other is None or other == entry.verdict:
            continue
        if "could not say" in (other, entry.verdict):
            continue
        out.append(
            {"type": entry.type_id, "by_repeatability": entry.verdict, "by_delay_track": other}
        )
    return out


def summarise(found: list[TypeSort]) -> str:
    piles = partition(found)
    lines = [f"{len(found)} types sorted by their own repeatability"]
    for entry in found:
        detail = ""
        if entry.unrepeatable_db and len(entry.unrepeatable_db) == 2:
            first, second = entry.unrepeatable_db
            detail = (
                f" -- what fails to repeat, {first:.1f} dB bypassed against {second:.1f} routed"
            )
        lines.append(f"  {entry.type_id:8} {entry.verdict}{detail}")
    lines.append(
        f"  => {len(piles['moving'])} move, {len(piles['static'])} stand still, "
        f"{len(piles['could_not_say'])} could not be told apart"
    )
    return "\n".join(lines)


__all__ = [
    "METHOD",
    "WHY_ASYMMETRY",
    "WHY_INSIDE_THE_GAP",
    "WHY_NOT_AUDIBLE",
    "WHY_TWO_ROUTES",
    "TypeSort",
    "disagreements",
    "inside_the_gap",
    "partition",
    "sort_one",
    "summarise",
    "survey",
]
