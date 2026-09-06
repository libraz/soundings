"""Reading one insertion effect type's parameters into a verdict apiece.

The type-level sort asks whether an effect does anything. This asks the same
question of each of the parameters it loads, which is the question that bounds
what comes after: a parameter no note hears needs no response measured across its
range, and on a unit where every type is audible the type-level sort turns
nothing away.

Assembled here rather than beside the run, so that the verdicts and the
limitation that qualifies them are written by one function. A record whose
caveats are added afterwards is a record whose next run does not have them.
"""

from __future__ import annotations

import json
from pathlib import Path

#: How the parameters were asked. Stated in the record because a verdict holds in
#: the state it was taken in, and this state is most of what makes it readable.
METHOD = (
    "Each of the type's parameter addresses was written at two settings, four takes each, "
    "and the takes of one setting compared with each other to measure what the unit fails "
    "to repeat before the two settings were compared with each other. The change has to "
    "clear that by the margin. Level is judged apart from shape, because the alignment "
    "divides out the best fitting gain and a parameter that only moves the level would "
    "otherwise read as no change at all. The accepted range for each address was measured "
    "with this type loaded rather than carried from a probe taken with no effect loaded, "
    "since what an address accepts can depend on what is in it."
)

#: Why a run of nulls is readable at all.
WHY_THE_CONTROL = (
    "Routing the stimulus's part through this type against bypassing it, under the same "
    "type, the same note and the same protocol. Without it a run of nulls says only that "
    "something in the chain was doing nothing, and which thing would be open."
)

#: An address whose two settings differ by less than its own takes do.
WHY_UNREADABLE = (
    "The change does not clear this address's own repeatability, so nothing here is a "
    "difference between the settings. What separates them is that one setting repeats and "
    "the other does not. A free-running modulator makes the difference method blind, and "
    "the asymmetry is what is left to read."
)

#: An address heard as a difference, one of whose settings also stopped repeating.
WHY_ALSO_MOVING = (
    "The two settings differ by more than the takes of either do, so the verdict stands. "
    "The setting away from the default also repeats far worse than the one at it, which is "
    "what a free-running modulator does, so this address started something moving as well "
    "as changing the sound."
)

#: An address the alignment heard only after it had divided out the gain.
WHY_LEVEL_ONLY = (
    "Heard on the level and not on the shape: once the best fitting gain is divided out the "
    "two settings are the same sound. What this address moves is how much of it there is."
)

#: An address one of whose settings had no signal to repeat.
WHY_SILENT_PAIR = (
    "One of the settings left the routed part silent, so it had nothing to repeat and the "
    "comparison was between sound and silence rather than between two settings. Asked "
    "again at a pair where both settings sound."
)

#: What a pass at two settings does not establish.
LIMITS = (
    "What any audible parameter is, or by how much it moves anything. Two settings were "
    "compared and no curve was measured between them.",
    "Anything about the types not asked. The parameter addresses are shared across types "
    "and what each of them holds is not, so this is one type's answer at those addresses.",
    "That a null does nothing. A parameter heard only on a longer note, at another "
    "velocity, under another type's settings, or between two values neither of which is an "
    "end of its range would read as inaudible here.",
    "That a parameter which broke its own yardstick does nothing besides. Whatever else it "
    "does sits behind the thing that removed the yardstick.",
)

AUDIBLE = "audible"
NULL = "not audible under the note asked"
UNREADABLE = "started something that does not repeat"

#: Repeatability worse than this leaves a setting with no usable yardstick of its
#: own, so a verdict resting on it is reported with the asymmetry rather than as a
#: difference. Not a threshold on the unit -- a threshold on what can be read.
NO_YARDSTICK_DB = -25.0


def _first(record: dict) -> dict:
    return record["by_stimulus"][0]


def _verdict(record: dict, stimulus: dict) -> tuple[str, str | None]:
    """What this address answered, and the sentence that qualifies it.

    The comparison already decided whether the sound changed and on which
    channel. What is added here is the reading that decision needs when the
    channel it was heard on was repeatability: a setting that stops repeating is
    not a setting that sounds different, and reporting it as one would put a
    modulator's name on whatever parameter started it.
    """
    if not record["audible"]:
        return NULL, None
    heard_as_a_difference = stimulus.get("changed_the_shape") or stimulus.get("changed_the_level")
    if not heard_as_a_difference:
        return UNREADABLE, WHY_UNREADABLE
    worse = max(v for v in stimulus["each_setting_unrepeatable_db"] if v is not None)
    if worse > NO_YARDSTICK_DB:
        return AUDIBLE, WHY_ALSO_MOVING
    if stimulus.get("changed_the_level") and not stimulus.get("changed_the_shape"):
        return AUDIBLE, WHY_LEVEL_ONLY
    return AUDIBLE, None


def row(type_id: str, slot: int, default: int, record: dict, withdrawn: dict | None) -> dict:
    """One parameter's verdict, with the figures a reader needs to check it."""
    stimulus = _first(record)
    verdict, why = _verdict(record, stimulus)
    out = {
        "type": type_id,
        "parameter": slot,
        "address": record["address"],
        "default": default,
        "asked_at": record["values"],
        "audible": verdict == AUDIBLE,
        "verdict": verdict,
        "takes_per_setting": stimulus["takes_per_setting"],
        "each_setting_unrepeatable_db": stimulus["each_setting_unrepeatable_db"],
        "same_setting_residual_db": stimulus["same_setting_residual_db"],
        "across_setting_residual_db": stimulus["across_setting_residual_db"],
        "across_setting_level_db": stimulus.get("across_setting_level_db"),
        "noise_floor_db": stimulus.get("noise_floor_db"),
        "changed_the_shape": stimulus.get("changed_the_shape"),
        "changed_the_level": stimulus.get("changed_the_level"),
    }
    if why:
        out["why"] = why
    if withdrawn:
        out["withdrawn_pair"] = withdrawn
        out["why_asked_again"] = WHY_SILENT_PAIR
    return out


def assemble(type_id: str, rows: list[dict], control: dict, prepared: list[dict]) -> dict:
    """The type's parameters, the control that makes their nulls readable, and the limits."""
    stimulus = _first(control)
    counted: dict[str, int] = {}
    for entry in rows:
        counted[entry["verdict"]] = counted.get(entry["verdict"], 0) + 1
    return {
        "question": (
            "Which of one insertion effect type's parameters change the sound, asked at the "
            "parameter rather than at the type."
        ),
        "type": type_id,
        "prepared": prepared,
        "stimulus": stimulus["stimulus"],
        "method": METHOD,
        "positive_control": {
            "address": control["address"],
            "values": control["values"],
            "audible": control["audible"],
            "shape_db": stimulus["across_setting_residual_db"],
            "repeatability_db": stimulus["same_setting_residual_db"],
            "why": WHY_THE_CONTROL,
        },
        "results": {**counted, "asked": len(rows)},
        "parameters": rows,
        "not_established": list(LIMITS),
        "reproduced": (
            "Every address carries the take-to-take figure of each of its two settings beside "
            "the comparison, so a verdict resting on a broken yardstick can be seen to be one. "
            "An address asked a second time names the pair that was withdrawn and why."
        ),
        "found_by": (
            "Asking the parameters of one type, after the sort that was meant to bound them "
            "passed every type the unit has."
        ),
    }


def read_directory(
    where: str | Path,
    type_id: str,
    defaults: list[int],
    control: str | Path,
    prepared: list[dict],
    supersede: dict[str, str] | None = None,
) -> dict:
    """Assemble from a directory of per-address contrast records.

    `supersede` names an address whose first pair was withdrawn and the record
    that replaced it, so the reason travels with the row rather than being
    remembered by whoever reads the directory.
    """
    where = Path(where)
    supersede = supersede or {}
    superseding = {str(Path(p).resolve()) for p in supersede.values()}
    # Keyed by the address the record states rather than by the file name, so a
    # run asked again writes a second file without becoming a second parameter.
    found: dict[str, dict] = {}
    for path in sorted(where.glob("*.json")):
        if str(path.resolve()) in superseding:
            continue
        record = json.loads(path.read_text())
        if not isinstance(record, dict) or "address" not in record:
            continue
        found.setdefault(record["address"], record)
    # The control is a routing byte kept in the same directory as the run it
    # belongs to. Which address that is comes from the control record itself,
    # since counting it as a parameter would shift every later slot onto the
    # wrong default -- and no address may be written down here, because an
    # address written beside the code is a finding about one unit.
    watching = json.loads(Path(control).read_text())
    found.pop(watching["address"], None)
    rows = []
    for slot, address in enumerate(sorted(found)):
        record, withdrawn = found[address], None
        if address in supersede:
            withdrawn = {"values": record["values"], "why": WHY_SILENT_PAIR}
            record = json.loads(Path(supersede[address]).read_text())
        rows.append(row(type_id, slot, defaults[slot], record, withdrawn))
    return assemble(type_id, rows, watching, prepared)
