"""Which of a unit's insertion effects move, and which stand still.

The type map says how many insertion effects a unit has and what each loads. It
does not say what any of them is, and the first division that matters for
identifying one is whether it has a free-running modulator: a chorus, a flanger,
a phaser, a tremolo, a rotary and an auto-pan are all *motions*, and everything
else -- a filter, a distortion, a compressor, an equaliser, a fixed delay -- is a
response that can be measured by averaging. The two need different measurements,
and running the wrong one produces a number either way.

So this is the sort, done once over every type the unit accepts, from a pair of
takes per type: the same note with the part routed through the effect and with it
bypassed.

**A type that reports no motion is only sorted if the pair could have shown
one.** Each pair carries its own injected control, so a type lands in `static`
only when a known modulation at the real return's level was recovered from that
same material, and in `could_not_say` otherwise. Without that split every type
whose takes were unusable would be filed as standing still, which is the answer
that costs nothing to produce and is wrong for exactly the effects this is trying
to find.

**Nothing here is named.** A type that moves at 0.9 Hz over 3 ms of delay is
reported as that, not as a chorus. What the unit calls it is not this archive's
question, and a label would survive longer than the measurement that suggested it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import motion
from .takes import channel, channel_across, read

METHOD = (
    "For each insertion effect the unit accepts, the same note was recorded with the part routed "
    "through the effect and with it bypassed, and the pair was put through the delay tracker. A "
    "type is sorted as moving on a periodic line in its delay or level track, and as static only "
    "when a known modulation injected into that same pair at the real return's level was "
    "recovered. A pair that could not recover its own control sorts as neither."
)

WHY_COULD_NOT_SAY = (
    "These types were neither shown to move nor shown to stand still. Their takes did not recover "
    "an injected modulation at any depth, so a modulator and the absence of one would have "
    "produced the same empty track. Filing them as static is the error this category exists to "
    "prevent: it is the cheapest answer available and it is wrong for precisely the types worth "
    "finding. Read the floor each pair carries beside this -- `slowest_rate_hz` under every "
    "type's track, and `slowest_shown_hz` over the survey. A modulation under that floor would "
    "have been reported as absent whatever the control did, because a line is not read under "
    "three cycles and these takes are as long as they are."
)

WHY_SLOWEST_SHOWN = (
    "The slowest modulation the takes in this survey could have carried, which is a fact about "
    "how long the note sounded and not about the band the run was asked for. The two are far "
    "apart here and only the asked-for one used to be written down, which left every null in "
    "this survey reading as a type that does not modulate. What a null means is bounded by this "
    "number: under it, a type that modulates and a type that does not return the same empty "
    "track, and no control can tell them apart because the control is injected above the floor "
    "as well."
)

NOT_NAMED = (
    "A type that moves is reported by its rate, depth and shape, not by what an effect doing that "
    "is usually called. The measurement is what this archive holds; a name is an identification "
    "made from it, and one written down here would outlive the evidence for it."
)

WHY_ONE_PAIR = (
    "One pair of takes per type, not several. A modulator is free-running, so its phase differs "
    "between takes and averaging them flattens the very motion under test. Repetition buys "
    "nothing here that it buys elsewhere, and the control is what stands in for it."
)

WHY_CHANNEL = (
    "Which channel of the interface each pair of takes was read from, and the highest each "
    "channel of that pair reached. One channel for the pair rather than the loudest of each take: "
    "the measurement subtracts the dry take from the wet one, and an interface carries inputs the "
    "unit is not on which are not silent, so a take whose output fell below one of them would be "
    "answered from that input and subtracted from a different one. What came back would be a "
    "residual of nothing, and the control injected into the same pair is the only thing that "
    "would have caught it."
)

WHY_MISSING = (
    "These are types the unit accepts that no pair of takes was found for, so the survey below "
    "says nothing about them at all. They are listed because a survey reports on what it was "
    "handed, and a capture that dropped a type would otherwise leave a shorter list of verdicts "
    "reading as a complete one."
)


@dataclass
class TypeMotion:
    """What one insertion effect type did between its two takes."""

    type_id: str
    stimulus: str
    moves: bool
    conclusive: bool
    rate_hz: float | None = None
    depth: str | None = None
    shape: str | None = None
    where: str | None = None
    """Which track the line was found in: the delay, or the level."""

    detectable_ms: list[float] | None = None
    return_level_db: float | None = None
    channel: int = 0
    channel_db: list[float] = field(default_factory=list)
    """Which interface channel both takes were read from, and what each reached."""

    track: dict = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        if not self.conclusive:
            return "could not say"
        return "moves" if self.moves else "static"

    def to_json(self) -> dict:
        return {
            "type": self.type_id,
            "stimulus": self.stimulus,
            "verdict": self.verdict,
            "rate_hz": self.rate_hz,
            "depth": self.depth,
            "shape": self.shape,
            "found_in": self.where,
            "control_detectable_ms": self.detectable_ms,
            "return_level_db": self.return_level_db,
            "channel": {"read": self.channel, "reached_db": self.channel_db},
            "track": self.track,
        }


def pair_from(manifest: dict, root: Path, dry: str, wet: str) -> tuple[Path, Path] | None:
    """The first take at each of the two settings, as file paths.

    The first rather than a chosen one: every take of a setting is the same
    measurement, and picking by any property of the audio would be selecting the
    pair that answers best.
    """
    found: dict[str, Path] = {}
    for entry in manifest.get("takes", []):
        setting = str(entry.get("setting"))
        if setting in (dry, wet) and setting not in found:
            found[setting] = root / entry["file"]
    if dry not in found or wet not in found:
        return None
    return found[dry], found[wet]


def measure_type(
    type_id: str,
    dry_path: Path,
    wet_path: Path,
    *,
    stimulus: str,
    lead_s: float = 0.0,
    search_ms: tuple[float, float] = (0.0, 60.0),
    rate_range: tuple[float, float] = (0.05, 20.0),
    depths_ms: tuple[float, ...] = motion.CONTROL_DEPTHS_MS,
    window: float = 0.010,
) -> TypeMotion:
    """Sort one type, from its own pair of takes and its own control."""
    dry, dry_rate = read(dry_path)
    wet, wet_rate = read(wet_path)
    if dry_rate != wet_rate:
        raise ValueError(f"{type_id}: takes at {dry_rate} and {wet_rate} Hz")
    # One channel for the pair, not one for each. The measurement below subtracts
    # the dry take from the wet one, so two takes read from different inputs
    # subtract one input from another and leave a residual of nothing.
    picked, reached = channel_across(dry, wet)
    dry, wet = channel(dry, picked), channel(wet, picked)

    found = motion.measure(
        dry,
        wet,
        dry_rate,
        search_ms=search_ms,
        rate_range=rate_range,
        lead_s=lead_s,
        window=window,
    )
    # Only where the type stood still. A control sweeps the take's own return, so
    # on a type that already moves it adds a motion to a motion and recovers a
    # line either way -- an answer that means nothing, bought at seven more
    # passes over the audio than the measurement itself cost.
    vouched = (
        {"detectable_ms": None, "return_level_db": None}
        if found.moves
        else motion.control(
            dry,
            wet,
            dry_rate,
            depths_ms=depths_ms,
            search_ms=search_ms,
            rate_range=rate_range,
            lead_s=lead_s,
            window=window,
        )
    )

    fit = found.delay if (found.delay is not None and found.delay_answered) else found.level
    where = None
    if found.delay is not None and found.delay_answered:
        where = "delay"
    elif found.level is not None:
        where = "level"

    return TypeMotion(
        type_id=type_id,
        stimulus=stimulus,
        moves=found.moves,
        # The track as well as the control, and it is the wrapped track this
        # guards. A swing wider than half the input's own period folds, and a
        # folded line can vanish -- which is the shape a static effect has. The
        # control cannot stand in for that: its shallowest rungs are too small to
        # fold, so they come back recovered and would sort a wrapped modulator as
        # standing still, with a detectable band that excludes the depth it
        # actually swung.
        conclusive=motion.is_conclusive(found, vouched),
        rate_hz=round(fit.rate_hz, 4) if fit and where else None,
        depth=f"{fit.depth:.4f} {fit.unit}" if fit and where else None,
        shape=("sine" if fit.sinusoidal else f"shape error {fit.shape_error:.2f}")
        if fit and where
        else None,
        where=where,
        detectable_ms=vouched["detectable_ms"],
        return_level_db=vouched["return_level_db"],
        channel=picked,
        channel_db=reached,
        track=found.track.to_json(),
    )


def survey(
    root: str | Path,
    *,
    dry: str = "0",
    wet: str = "1",
    search_ms: tuple[float, float] = (0.0, 60.0),
    rate_range: tuple[float, float] = (0.05, 20.0),
    depths_ms: tuple[float, ...] = motion.CONTROL_DEPTHS_MS,
    lead_s: float = 0.0,
    window: float = 0.010,
    progress=None,
) -> list[TypeMotion]:
    """Sort every type whose takes are under this directory, one subdirectory each."""
    out = []
    for directory in sorted(Path(root).iterdir()):
        manifest_path = directory / "takes-manifest.json"
        if not directory.is_dir() or not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        stimulus = next((e["stimulus"] for e in manifest.get("takes", [])), "")
        paths = pair_from(manifest, directory, dry, wet)
        if paths is None:
            continue
        found = measure_type(
            directory.name.replace("-", " ").upper(),
            *paths,
            stimulus=stimulus,
            lead_s=lead_from(manifest, stimulus, lead_s),
            search_ms=search_ms,
            rate_range=rate_range,
            depths_ms=depths_ms,
            window=window,
        )
        out.append(found)
        if progress:
            progress(found)
    return out


def lead_from(manifest: dict, stimulus: str, fallback: float = 0.0) -> float:
    """How much silence the takes were recorded with, from the capture's own record.

    The lead is what the noise floor is measured in, and every stimulus declares
    its own. Reading it from the manifest rather than taking it as an argument
    keeps a directory of takes captured under several stimuli readable, and stops
    a figure typed on the command line from standing in for one the capture knows.
    """
    for entry in manifest.get("stimuli", []):
        if entry.get("name") == stimulus and entry.get("lead_s") is not None:
            return float(entry["lead_s"])
    return fallback


def accepted_types(path: str | Path) -> list[str]:
    """The type ids a unit answered to, read from its own type map record.

    The list the survey is measured against comes from the unit rather than from
    a count written here, so a unit accepting a different set is compared with
    its own.
    """
    record = json.loads(Path(path).read_text())
    return [str(entry["type"]) for entry in record.get("effects", [])]


def missing(found: list[TypeMotion], accepted: list[str]) -> list[str]:
    """Types the unit accepts that the survey never saw a pair of takes for."""
    sorted_ids = {f.type_id for f in found}
    return [t for t in accepted if t not in sorted_ids]


def partition(found: list[TypeMotion]) -> dict[str, list[str]]:
    """The three piles, which is what the survey is for."""
    return {
        "moving": [f.type_id for f in found if f.verdict == "moves"],
        "static": [f.type_id for f in found if f.verdict == "static"],
        "could_not_say": [f.type_id for f in found if f.verdict == "could not say"],
    }


def slowest_shown(found: list[TypeMotion]) -> dict:
    """The floor the takes in this survey carried, over every type that was sorted.

    Per type it is already under each track. Here it is over the survey, because the
    band the invocation asked for is written at this level too and the two would
    otherwise sit a record apart -- one of them true and the other the one a reader
    takes a null against.
    """
    floors = [
        f.track["slowest_rate_hz"]
        for f in found
        if (f.track or {}).get("slowest_rate_hz") is not None
    ]
    if not floors:
        return {"hz": None, "why": WHY_SLOWEST_SHOWN}
    return {
        "hz": [round(min(floors), 3), round(max(floors), 3)],
        "sounded_s": [
            round(min(f.track["sounded_s"] for f in found), 3),
            round(max(f.track["sounded_s"] for f in found), 3),
        ],
        "why": WHY_SLOWEST_SHOWN,
    }


def summarise(found: list[TypeMotion]) -> str:
    piles = partition(found)
    lines = [f"{len(found)} types sorted"]
    for entry in found:
        detail = ""
        if entry.verdict == "moves":
            detail = f" -- {entry.rate_hz} Hz, {entry.depth} in the {entry.where}, {entry.shape}"
        elif entry.verdict == "static" and entry.detectable_ms:
            deep, shallow = entry.detectable_ms
            detail = f" -- nothing between {shallow} and {deep} ms would have been missed"
        lines.append(f"  {entry.type_id:8} {entry.verdict}{detail}")
    lines.append(
        f"  => {len(piles['moving'])} move, {len(piles['static'])} stand still, "
        f"{len(piles['could_not_say'])} could not be told apart"
    )
    return "\n".join(lines)


__all__ = [
    "METHOD",
    "NOT_NAMED",
    "WHY_COULD_NOT_SAY",
    "WHY_MISSING",
    "WHY_ONE_PAIR",
    "WHY_SLOWEST_SHOWN",
    "TypeMotion",
    "accepted_types",
    "lead_from",
    "measure_type",
    "missing",
    "pair_from",
    "partition",
    "slowest_shown",
    "summarise",
    "survey",
]
