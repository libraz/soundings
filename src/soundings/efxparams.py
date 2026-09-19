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

from . import documents

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

#: An address whose first pair the run would not answer, because the takes had no
#: silence in front of the note to be judged against.
WHY_THE_TAKES_HAD_NO_FLOOR = (
    "The run would not answer this pair. At one of its settings the take carried no silence "
    "before the note, so the noise floor -- the yardstick every number here is judged against "
    "-- was the parameter's own output, and nothing measured from those takes would have meant "
    "anything. The refusal beside this names the tail of the take before it as the cause; the "
    "takes do not bear that out. A tail falls take over take and these are flat across every "
    "take of the run, with the first one silent and every one after it at the same level. What "
    "does not stop is what the parameter is: a delay at either end of its feedback range loses "
    "a sixth of a decibel per repeat, and a hum at its own level does not decay at all. Asked "
    "again at a pair whose settings die away between takes."
)

#: An address whose pair named a value the page gives its parameter none of.
WHY_OUTSIDE_ITS_PRINTED_VALUES = (
    "The pair was the two ends of what this address was measured to accept, and this address "
    "accepts every seven-bit value while the page gives its parameter fewer -- a list of "
    "states, or a run between two ends that are not nought and 127. So one of the two settings "
    "names a value the parameter is not printed as having, and whatever the engine did with it "
    "is not one of this parameter's settings. Read as a null it would say the parameter does "
    "nothing; what it says is that the pair was not two of its settings. Two such parameters "
    "asked again at two of their own, under the same note and the same routing that had "
    "returned the null, were both audible at once -- a reverb type at twenty-nine decibels "
    "over its own yardstick and a vowel at forty-four."
)

#: An address whose two settings are one setting of its parameter, twice.
WHY_ASKED_AT_ONE_SETTING = (
    "Both settings name the same setting of the parameter. The page's value column refers this "
    "address to a column of the conversion grid, that column comes back to where it started, and "
    "the grid prints each of the two ends as equal to the other -- which is the page saying the "
    "byte's two ends are one place. So the two sets of takes are takes of one setting and they "
    "differ by what a setting differs from itself by, which is the number a null is. Read as a "
    "null it would say the parameter does nothing; what it says is that it was asked twice in the "
    "same place. Asked again at the value the unit powers up holding against one end."
)

#: An address asked while the stage it belongs to was not in the signal path.
WHY_ITS_STAGE_WAS_SHUT = (
    "Every address but the one being asked sat where this run left it, and for this parameter "
    "that was its own stage taken out of the output: a byte the page prints as that stage's "
    "level, mix or switch stood at nought while the pair was compared. A parameter behind it "
    "reaches the output at neither of its settings, so the two sets of takes are two takes of "
    "the same sound without that stage in it, and they differ by what a sound differs from "
    "itself by -- which is the number a null is. Read as a null it would say the parameter "
    "does nothing; what it says is that the run could not have heard it either way. Which byte "
    "it was, and whether this run wrote it down or the unit powers up holding it there, is "
    "beside the verdict."
)

#: An address describing a modulation that was not running while it was asked.
WHY_ITS_MODULATION_WAS_STILL = (
    "The stage was in the output and its modulation was at nothing: a byte the page prints as "
    "that stage's depth or sensitivity stood at nought while the pair was compared. What this "
    "parameter describes is how that modulation runs, and a modulation at no depth runs the "
    "same however it is described -- a rate scales nothing, a waveform shapes nothing. So the "
    "two sets of takes are two takes of one sound. Read as a null it would say the parameter "
    "does nothing; what it says is that there was nothing for it to do. The byte that decides "
    "how much of the stage reaches the output is not read this way, because a stage at no "
    "depth is still a stage -- a chorus at none of it is a fixed delay, and mixing that in is "
    "audible."
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
    "Which values a parameter has, where a row carries `printed_values`. Those were read off "
    "this unit's own effect list and never measured here, and a page can be wrong about a "
    "part. They decide nothing about what was heard: they only separate a slot whose pair "
    "named two of the parameter's own values from one whose pair named a value the page gives "
    "it none of, and a slot of the second kind is reported as unasked rather than as "
    "answering nothing.",
)

#: An address whose settings were never separable, so nothing was asked of it.
WHY_NO_YARDSTICK = (
    "Takes of one setting already differed by most of the signal, so no change could have "
    "cleared the bar and none did. The comparison refused a verdict rather than returning a "
    "null, and this carries that refusal: the address is unasked, not inaudible. It is what "
    "a free-running modulator does to every parameter of a type asked at the values it "
    "powers up holding, and it is answered by asking again with the modulator parked."
)

#: Why a slot with no record is named rather than left out of the count.
WHY_NEVER_ASKED = (
    "A parameter slot the run set out to ask and has no record for. The comparison was "
    "attempted and did not return one -- so this slot is unasked, and a record that simply "
    "omitted it would report the type as having one parameter fewer than it has. Counted "
    "against the slots the type map gives the type rather than against the files present, "
    "since a directory holding nineteen records of a twenty slot type reads exactly like a "
    "complete answer."
)

#: Why a slot the run held still is counted apart from one it failed to reach.
WHY_HELD = (
    "A parameter slot the run set to a fixed value and deliberately did not ask, because the "
    "rest of the type could not be asked while it moved. A free-running modulator puts every "
    "take of one setting at a different phase, and the yardstick a change is judged against "
    "becomes the modulator's own swing, so every other parameter answers that there was no "
    "yardstick. Asking the held slot means writing it to the top of its range, which starts the "
    "modulator again part way through the run and measures the slots after it against a "
    "yardstick that had quietly come back. It is separate from the slots with no record because "
    "only one of the two is work outstanding, and it carries no verdict: what this run says "
    "about the held parameter is nothing at all. The value it was held at is in `prepared`."
)

#: Why a null measured beside a held byte is a narrower null than the others.
WHY_NULL_WHILE_HELD = (
    "That a null on this run is a null on the type. One of the type's own bytes was held at "
    "zero for every ask, and a parameter whose whole effect is to scale what that byte moves "
    "has nothing here to scale -- it is inaudible because of what this run did, not because of "
    "what the unit does. The held address is in `prepared` and the run says so; which of the "
    "other slots stand behind it is not decided here, because that is a reading of what each "
    "parameter is for and no take on this run separates the two. It is the price of the hold: "
    "unparked, every slot of this type answered that there was no yardstick at all."
)

#: Why a slot whose run refused to answer is counted apart from one nobody asked.
WHY_REFUSED = (
    "A parameter slot whose comparison ran, would not answer, and said which of its own "
    "conditions the takes failed. It is kept apart from the slots with no record at all "
    "because only one of the two is work outstanding: asking this one again the same way "
    "gets the same refusal, and reading it as unasked would put it in a queue it can never "
    "leave. It carries no verdict either way -- what was refused was the takes, and nothing "
    "here says whether the parameter does anything."
)

#: Why the slots are taken from the caller in order rather than from the directory.
WHY_SLOTS_IN_ORDER = (
    "Which slot each address is was given to the fold in order, not inferred from the records "
    "that happened to be present. A slot read off the sorted file names moves every later "
    "parameter up by one whenever a record is missing, and each of them is then reported "
    "against the default belonging to the slot before it -- a wrong number in the archive "
    "rather than a missing one."
)

AUDIBLE = "audible"
NULL = "not audible under the note asked"
UNREADABLE = "started something that does not repeat"
NO_YARDSTICK = "no yardstick under the note asked, so not asked"
OUTSIDE_ITS_PRINTED_VALUES = "asked at a value its parameter is not printed as having, so not asked"

ASKED_AT_ONE_SETTING = "asked twice at one setting of its parameter, so not asked"

ITS_STAGE_WAS_SHUT = "asked with its own stage out of the output, so not asked"
ITS_MODULATION_WAS_STILL = "asked with its own stage's modulation at nothing, so not asked"

#: Repeatability worse than this leaves a setting with no usable yardstick of its
#: own, so a verdict resting on it is reported with the asymmetry rather than as a
#: difference. Not a threshold on the unit -- a threshold on what can be read.
NO_YARDSTICK_DB = -25.0


def _first(record: dict) -> dict:
    return record["by_stimulus"][0]


def asked_outside_its_printed_values(record: dict, printed_values: str | None) -> bool:
    """Whether either setting names a value the page gives this parameter none of.

    Only the pair is read, not the verdict: a parameter heard at such a pair was
    still heard, and only a null taken there is uninterpretable.

    Read off the value column the page prints rather than off a count of the names
    in its setting column, which is what separates three kinds of parameter a count
    gets wrong. A rotor's two speeds are the bytes 0 and 127, so the pair the stage
    asked is exactly its two states. A tone gain is a run from 52 to 76, so neither
    end of the pair is a value it has, and no count of names would have said so
    because the setting column prints a range of decibels and no names at all. And
    a damping frequency printed `315-8k/Bypass` has a slash in its setting column
    and is not a list of anything -- its value column refers the reader to a table
    of 128 entries.
    """
    if not printed_values:
        return False
    here = documents.values_printed(printed_values)
    if here is None or here == documents.EVERY_VALUE:
        return False
    return any(int(value) not in here for value in record.get("values", []))


def asked_at_one_setting(record: dict, settings: dict[int, str] | None) -> bool:
    """Whether the pair names one setting of the parameter rather than two.

    A value column referring to the conversion grid narrows nothing -- the grid
    gives a setting at all 128 -- so `asked_outside_its_printed_values` returns
    false here however the pair was chosen, and correctly. The other question is
    whether the two values the pair names are the same setting, which one column
    of the grid answers yes to: it comes back to where it began, and the page
    prints each of its ends as equal to the other.

    Only the pair is read, not the verdict. A parameter heard at such a pair was
    heard, and would be the page being wrong about the unit rather than the reverse.
    """
    if not settings:
        return False
    values = [int(value) for value in record.get("values", [])]
    if len(values) != 2:
        return False
    return documents.one_setting(settings.get(values[0]), settings.get(values[1]))


#: Where a byte that shut a stage got the value it was shut at. The two are not
#: one finding: a run that parks a modulator's depth says so and the type's other
#: record answers the parameters behind it, where a value nobody chose is a state
#: no record mentions at all.
BY_THIS_RUN, BY_THE_UNIT = "this run's prepare block", "the value the unit powers up holding"

#: What a gate at nought leaves behind it. The stage out of the output, or the
#: stage in it and not moving -- which silence different parameters and are
#: therefore two verdicts rather than one.
STAGE_ABSENT = "its stage out of the output"
MODULATION_STILL = "its stage in the output and its modulation at nothing"


def gates_at_nought(
    address: str,
    names: dict[str, str] | None,
    held: dict[str, int] | None,
    resting: dict[str, int | None] | None,
) -> list[dict]:
    """The gates of this address's own stage that sat at nought while it was asked.

    `names` is what the page calls each of the type's parameters, `held` is what
    the run overrode and at what value, and `resting` is what the unit powers each
    address up holding. Together they are what every other address stood at while
    this one was being compared.

    Two kinds are reported and they do not silence the same things. A level, a mix
    or a switch at nought leaves the stage out of the output, so nothing in it can
    be heard. A depth or a sensitivity at nought leaves the stage in the output and
    still, so what cannot be heard is what the modulation does -- which is every
    parameter describing it, and not the byte that decides how much of the stage
    reaches the output. A byte of the second kind is therefore only ever reported
    against the first.

    The address itself is never among them: what it rests at is not what it was
    asked at, and a gate asked at nought against full is asked properly.
    """
    if not names:
        return []
    held, resting = held or {}, resting or {}
    here = names.get(address)
    if here is None:
        return []
    # A byte that decides how much of its stage is in the output is not silenced by
    # one that decides how far the stage's modulation travels: a chorus at no depth
    # is a fixed delay, and turning its mix up is still audible.
    only_absence = documents.takes_the_stage_out(here)
    stage = documents.stage_named(here)
    out = []
    for other, name in sorted(names.items()):
        if other == address or not documents.names_a_gate(name):
            continue
        if documents.stage_named(name) != stage:
            continue
        absent = documents.takes_the_stage_out(name)
        if only_absence and not absent:
            continue
        standing = held[other] if other in held else resting.get(other)
        if standing == 0:
            out.append(
                {
                    "address": other,
                    "parameter": name,
                    "leaves": STAGE_ABSENT if absent else MODULATION_STILL,
                    "at_nought_by": BY_THIS_RUN if other in held else BY_THE_UNIT,
                }
            )
    return out


def _verdict(
    record: dict,
    stimulus: dict,
    outside_its_printed_values: bool = False,
    at_one_setting: bool = False,
    shut_gates: list[dict] | None = None,
) -> tuple[str, str | None]:
    """What this address answered, and the sentence that qualifies it.

    The comparison already decided whether the sound changed and on which
    channel. What is added here is the reading that decision needs when the
    channel it was heard on was repeatability: a setting that stops repeating is
    not a setting that sounds different, and reporting it as one would put a
    modulator's name on whatever parameter started it.

    A refusal is read before a null, because the two arrive the same way. A
    comparison that could not separate its settings reports `audible` false and
    `conclusive` false, and reading the first without the second turns "this was
    not asked" into "this does nothing" -- which is the one thing an archive of
    negatives cannot afford to get wrong.
    """
    if not record.get("conclusive", True):
        return NO_YARDSTICK, WHY_NO_YARDSTICK
    if not record["audible"]:
        # Read before the null for the same reason the refusal above is: the two
        # arrive the same way, and only one of them is a fact about the parameter.
        if outside_its_printed_values:
            return OUTSIDE_ITS_PRINTED_VALUES, WHY_OUTSIDE_ITS_PRINTED_VALUES
        if at_one_setting:
            return ASKED_AT_ONE_SETTING, WHY_ASKED_AT_ONE_SETTING
        # A stage out of the output is read before a stage that is in it and
        # still, because it is the stronger statement and both can be true at once.
        if any(g["leaves"] == STAGE_ABSENT for g in shut_gates or ()):
            return ITS_STAGE_WAS_SHUT, WHY_ITS_STAGE_WAS_SHUT
        if shut_gates:
            return ITS_MODULATION_WAS_STILL, WHY_ITS_MODULATION_WAS_STILL
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


def row(
    type_id: str,
    slot: int,
    default: int,
    record: dict,
    withdrawn: dict | None,
    printed_values: str | None = None,
    printed_settings: dict[int, str] | None = None,
    shut_gates: list[dict] | None = None,
) -> dict:
    """One parameter's verdict, with the figures a reader needs to check it."""
    stimulus = _first(record)
    outside = asked_outside_its_printed_values(record, printed_values)
    one_place = asked_at_one_setting(record, printed_settings)
    verdict, why = _verdict(record, stimulus, outside, one_place, shut_gates)
    out = {
        "type": type_id,
        "parameter": slot,
        "address": record["address"],
        "default": default,
        "asked_at": record["values"],
        "audible": verdict == AUDIBLE,
        "verdict": verdict,
        "conclusive": record.get("conclusive", True),
        "takes_per_setting": stimulus["takes_per_setting"],
        "each_setting_unrepeatable_db": stimulus["each_setting_unrepeatable_db"],
        "same_setting_residual_db": stimulus["same_setting_residual_db"],
        "across_setting_residual_db": stimulus["across_setting_residual_db"],
        "across_setting_level_db": stimulus.get("across_setting_level_db"),
        "noise_floor_db": stimulus.get("noise_floor_db"),
        "changed_the_shape": stimulus.get("changed_the_shape"),
        "changed_the_level": stimulus.get("changed_the_level"),
    }
    if printed_values is not None:
        # Kept as the page sets it rather than as the values it expands to. What a
        # reader has to be able to check is the cell, and a set of numbers here would
        # be this module's reading of it standing where the reading's input belongs.
        out["printed_values"] = printed_values
        out["asked_inside_its_printed_values"] = not outside
        if printed_settings:
            # Two different questions and a row that carries only the first reads as
            # a pair that was asked properly. Inside its printed values it was: the
            # referral gives every value a setting. Both of them the same setting is
            # the other thing, and it is stated here as the page states it.
            out["the_pair_names_one_setting"] = one_place
            out["printed_settings"] = [
                printed_settings.get(int(value)) for value in record.get("values", [])
            ]
    if shut_gates:
        # Named whatever the verdict came out as, so a reader can check the reading
        # rather than take it: which byte, what the page calls it, what it leaves
        # behind it, and whether this run wrote it down or the unit powers up there.
        out["at_nought_while_this_was_asked"] = shut_gates
    if why:
        out["why"] = why
    if withdrawn:
        out["withdrawn_pair"] = withdrawn
        out["why_asked_again"] = withdrawn.get("why", WHY_SILENT_PAIR)
    return out


def assemble(
    type_id: str,
    rows: list[dict],
    control: dict,
    prepared: list[dict],
    coverage: dict | None = None,
) -> dict:
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
        **(
            {
                "coverage": coverage,
                "why_never_asked": WHY_NEVER_ASKED,
                "why_slots_in_order": WHY_SLOTS_IN_ORDER,
            }
            if coverage is not None
            else {}
        ),
        "parameters": rows,
        # The held byte narrows every null in the run, so the limit is carried by
        # the record that has one rather than standing over the sixty-odd that do
        # not. Read off the coverage the fold built, which is where the hold is
        # already established, rather than by inspecting `prepared` a second time.
        "not_established": [
            *LIMITS,
            *([WHY_NULL_WHILE_HELD] if (coverage or {}).get("held_still") else []),
        ],
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
    slots: list[str] | None = None,
    printed: dict[str, str] | None = None,
    settings: dict[str, dict[int, str]] | None = None,
    names: dict[str, str] | None = None,
) -> dict:
    """Assemble from a directory of per-address contrast records.

    `supersede` names an address whose first pair was withdrawn and the record
    that replaced it, so the reason travels with the row rather than being
    remembered by whoever reads the directory.

    `slots` is the type's parameter addresses in slot order, and it is what makes
    a missing record readable. Without it the slot is the position in the sorted
    file names, so an address the run failed to measure silently renumbers every
    parameter after it and pairs each with the default of the slot before -- and
    the count says the type has one parameter fewer than it has. The addresses
    are passed in rather than derived here because which address is which slot is
    a fact about the unit being measured.

    `printed` names, by address, the value column an effect list prints against a
    parameter. It
    decides nothing about what was heard: a null taken at a value the parameter
    has no state for is reported as a slot nobody asked rather than as a slot
    that answered nothing.

    `settings` is what a value column that refers to the conversion grid says, by
    address: the setting printed at each of the 128. It answers the question the
    referral leaves open -- not how many values are settings, which is all of them,
    but whether the two the pair names are two of them or one of them twice.

    `names` is what the page calls each parameter. It is what says which addresses
    are one stage and which of them gate it, so a null taken with that stage not in
    the signal path is reported as a slot the run could not have heard rather than
    as a slot that answered nothing.
    """
    where = Path(where)
    supersede = supersede or {}
    printed = printed or {}
    settings = settings or {}
    names = names or {}
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
    ordered = list(slots) if slots else sorted(found)
    # A slot the run was told to hold is not a slot it failed to reach, and the
    # two arrive identically -- as an address with no record under the directory.
    # Read off `prepared` rather than passed separately, because the value it was
    # held at is published there and a second list would be one to keep in step.
    holding = [entry["address"] for entry in prepared if entry["address"] in set(ordered)]
    # What every address stood at while any one of them was being asked. The run's
    # own prepare block where it overrode a value; the unit's power-on value, which
    # `defaults` is, everywhere else. A byte the prepare block writes as more than
    # one value is not a setting of one parameter and is left out rather than cut.
    held = {
        entry["address"]: int(entry["bytes"], 16)
        for entry in prepared
        if len(entry.get("bytes", "").split()) == 1
    }
    resting = {
        address: defaults[slot]
        for slot, address in enumerate(ordered)
        if slot < len(defaults)
    }
    rows = []
    missing = []
    refused = []
    for slot, address in enumerate(ordered):
        if address not in found:
            if address not in holding:
                missing.append(address)
            continue
        record, withdrawn = found[address], None
        if address in supersede:
            # Why the first pair was withdrawn is read off that pair rather than
            # assumed. Four things send an address round again and they are not
            # the same finding: a pair the run would not answer at all, a setting
            # that left the part silent, a setting the page gives the parameter
            # none of, and a pair whose two values are one setting of it. Naming
            # one of them for all four would put the wrong reason on the row, and
            # the reason is the whole of what a withdrawn pair carries.
            #
            # The refusal comes first because it is the one that says the pair was
            # never read. The others describe a verdict that cannot be believed;
            # this one is the absence of a verdict, and a row saying the pair was
            # outside its printed values implies a null that was never taken.
            if record.get("refused"):
                why_again = WHY_THE_TAKES_HAD_NO_FLOOR
            elif asked_outside_its_printed_values(record, printed.get(address)):
                why_again = WHY_OUTSIDE_ITS_PRINTED_VALUES
            elif asked_at_one_setting(record, settings.get(address)):
                why_again = WHY_ASKED_AT_ONE_SETTING
            else:
                why_again = WHY_SILENT_PAIR
            withdrawn = {"values": record["values"], "why": why_again}
            # The refusal's own measurement travels with the row it was withdrawn
            # from. Superseding replaces the record, so what the first takes were
            # refused for is otherwise readable nowhere: the published file would
            # show a parameter asked twice and nothing about the first attempt
            # beyond the pair.
            if record.get("refused"):
                withdrawn["refused"] = record["refused"]
            record = json.loads(Path(supersede[address]).read_text())
        # A refused run holds no takes to read a verdict out of, so it cannot
        # become a row. It is not missing either, and the difference is the
        # whole point of it having been written down.
        if record.get("refused"):
            refused.append({"parameter": slot, "address": address, **record["refused"]})
            continue
        rows.append(
            row(
                type_id,
                slot,
                defaults[slot],
                record,
                withdrawn,
                printed.get(address),
                settings.get(address),
                gates_at_nought(address, names, held, resting),
            )
        )
    coverage = None
    if slots:
        coverage = {
            "slots": len(ordered),
            "answered": len(rows),
            "never_asked": missing,
            "held_still": holding,
            **({"why_held_still": WHY_HELD} if holding else {}),
            "refused": refused,
            **({"why_refused": WHY_REFUSED} if refused else {}),
            # A record under the directory that no slot claims. It is not folded
            # in, because a slot number is what pairs a parameter with its
            # default and this address has none -- but dropping it unnamed is how
            # a directory comes to disagree with its own record.
            "outside_the_slots": sorted(set(found) - set(ordered)),
        }
    return assemble(type_id, rows, watching, prepared, coverage)
