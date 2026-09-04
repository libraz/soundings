"""Ask an audible verdict of takes that were already recorded.

`contrast` reaches its verdict at capture time and keeps the takes. That is the
right way round -- device time is the scarce thing -- but it left the verdict
reachable only from the machine, so a question raised after the session ended
had to be paid for again in device time even when the audio to answer it was
already on disk.

The takes carry everything the judgement needs. A `--save` run writes the
settings, the stimuli with their leads, and the label, so the only thing not in
the directory is the machine, and the machine is not consulted after the last
take is recorded.

**A verdict reached here is the same verdict, not a weaker one.** It runs the
same comparator over the same takes with the same yardstick, so it is worth as
much as one printed during the session. What it cannot do is choose a stimulus
or a margin the session did not record takes for.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import audible
from .takes import read

METHOD_SUFFIX = (
    " The takes were recorded in an earlier session and judged afterwards with no machine "
    "attached. The comparator, the yardstick and the takes are the ones the session captured; "
    "only the moment of asking is later."
)


class NothingToJudge(RuntimeError):
    pass


def settings_in(manifest: dict) -> list[str]:
    """Every setting the directory holds takes at, in the order first recorded."""
    seen: list[str] = []
    for entry in manifest.get("takes", []):
        value = str(entry.get("setting"))
        if value not in seen:
            seen.append(value)
    return seen


def _loudest_index(frames: np.ndarray) -> int:
    return int(np.argmax([np.abs(frames[:, c]).max() for c in range(frames.shape[1])]))


def judge_directory(
    root: str | Path,
    *,
    first: str | None = None,
    second: str | None = None,
    margin_db: float = 6.0,
) -> tuple[audible.Overall, dict]:
    """Re-judge one saved run, one verdict per stimulus it holds takes for.

    The two settings default to the first and last the manifest records, which is
    what a two-valued `contrast` run leaves behind. A run that swept more than two
    has to be told which pair to compare, since any choice made here would be a
    measurement decision wearing a default's clothes.
    """
    root = Path(root)
    manifest = json.loads((root / "takes-manifest.json").read_text())
    values = settings_in(manifest)
    if len(values) < 2:
        raise NothingToJudge(f"{root} holds takes at {len(values)} setting(s); a pair is needed")
    first = first if first is not None else values[0]
    second = second if second is not None else values[-1]
    if first not in values or second not in values:
        raise NothingToJudge(f"{root} holds no takes at {first} or {second}; it has {values}")

    leads = {s.get("name"): s.get("lead_s") for s in manifest.get("stimuli", [])}
    described = {s.get("name"): s for s in manifest.get("stimuli", [])}
    overall = audible.Overall(label=str(manifest.get("label", root.name)))

    grouped: dict[str, dict[str, list[Path]]] = {}
    for entry in manifest.get("takes", []):
        setting = str(entry.get("setting"))
        if setting in (first, second):
            grouped.setdefault(entry["stimulus"], {}).setdefault(setting, []).append(
                root / entry["file"]
            )

    for stimulus, settings in grouped.items():
        if first not in settings or second not in settings:
            continue
        loaded = {k: [read(p) for p in v] for k, v in settings.items()}
        rate = loaded[first][0][1]
        index = _loudest_index(loaded[first][0][0])
        lead = leads.get(stimulus) or 0.6
        overall.verdicts.append(
            audible.judge(
                [frames[:, index] for frames, _ in loaded[first]],
                [frames[:, index] for frames, _ in loaded[second]],
                rate,
                label=f"{overall.label} ({stimulus})",
                stimulus=_describe(described.get(stimulus), stimulus),
                stimulus_name=stimulus,
                silence_before=lead * 0.8,
                margin_db=margin_db,
            )
        )
    if not overall.verdicts:
        raise NothingToJudge(f"{root} holds no stimulus with takes at both {first} and {second}")
    return overall, manifest


def _describe(entry: dict | None, name: str) -> str:
    if not entry:
        return name
    return (
        f"program {entry.get('program')}, note {entry.get('note')}, "
        f"velocity {entry.get('velocity')}, held {entry.get('hold_s')} s, "
        f"captured {entry.get('captured_s')} s"
    )


__all__ = ["METHOD_SUFFIX", "NothingToJudge", "judge_directory", "settings_in"]
