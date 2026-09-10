"""Counting a directory against the bar for a finished unit.

The one failure this must not have is a false pass. A stage nobody ran and a
stage the machine cannot answer look identical in a listing, and a report that
called either of them met would retire it silently.

No hardware, and no unit: every directory here is built in the test.
"""

from __future__ import annotations

import json
from pathlib import Path

from soundings import completion


def _write(unit: Path, name: str, payload: dict) -> None:
    """A record at its path under the unit, which is a stage's directory and a name."""
    (unit / name).parent.mkdir(parents=True, exist_ok=True)
    (unit / name).write_text(json.dumps(payload))


def _regions(*blocks: tuple[str, list[int]]) -> list[dict]:
    """A map's regions for the named blocks, each with the sizes given."""
    out = []
    for block, sizes in blocks:
        for offset, size in enumerate(sizes):
            out.append({"address": f"{block} {offset:02X}", "size": size, "data": "00"})
    return out


def test_a_directory_with_nothing_in_it_is_not_complete(tmp_path) -> None:
    found = completion.survey(tmp_path)
    assert found["complete"] is False
    assert all(s["verdict"] != completion.MET for s in found["stages"])


def test_a_stage_the_unit_cannot_answer_is_not_counted_against_it(tmp_path) -> None:
    """'Not run' and 'does not apply' are opposite, and only the unit can say which."""
    _write(
        tmp_path,
        "meta.json",
        {
            "unit_id": "somebody-01",
            completion.EXCLUSIONS_KEY: {"resets": "this machine accepts no reset"},
        },
    )
    stages = {s["stage"]: s for s in completion.survey(tmp_path)["stages"]}
    assert stages["resets"]["verdict"] == completion.EXCLUDED
    assert stages["resets"]["evidence"] == "this machine accepts no reset"


def test_a_stage_whose_bar_is_about_controls_is_not_passed_by_a_file_being_there(
    tmp_path,
) -> None:
    """A file listing cannot see whether a null carried its bound."""
    _write(tmp_path, "reset-probe/whole-map.json", {"resets": []})
    stages = {s["stage"]: s for s in completion.survey(tmp_path)["stages"]}
    assert stages["resets"]["verdict"] == completion.UNDECIDED


def _offsets(*blocks) -> dict:
    """An offsets record, given (block, [offsets that answered]) pairs."""
    return {
        "blocks_asked": len(blocks),
        "stopped": None,
        "blocks": [
            {
                "address": f"{block} 00",
                # Every offset of the block, which is what the stage asks. A row
                # without it reads as a block nobody asked at all, so leaving it
                # out of the fixture would let a stage that cannot see coverage
                # pass for one that correctly found none.
                "offsets_asked": 128,
                "answered": {f"{block} {off}": "00" for off in answering},
            }
            for block, answering in blocks
        ],
    }


def test_a_whole_space_stage_is_short_until_it_covers_every_shape(tmp_path) -> None:
    _write(tmp_path, "offsets/whole-map.json", _offsets(("40 11", ["00", "2A"]), ("40 21", ["00"])))
    _write(
        tmp_path,
        "write-probe/whole-map.json",
        {"regions": [{"start": "40 11 00", "bytes": [{"address": "40 11 00"}]}]},
    )
    stages = {s["stage"]: s for s in completion.survey(tmp_path)["stages"]}
    assert stages["accepted values"]["verdict"] == completion.UNMET
    assert stages["accepted values"]["remaining"] == {
        "offsets not covered, by representative": {"40 11": 1, "40 21": 1}
    }


def test_a_stage_counts_every_record_it_filed_not_only_the_whole_map_one(tmp_path) -> None:
    """The offsets stage adds addresses a later run probes into its own file."""
    _write(tmp_path, "offsets/whole-map.json", _offsets(("40 11", ["00", "2A"])))
    _write(
        tmp_path,
        "write-probe/whole-map.json",
        {"regions": [{"start": "40 11 00", "bytes": [{"address": "40 11 00"}]}]},
    )
    _write(
        tmp_path,
        "write-probe/offsets-gaps.json",
        {"regions": [{"start": "40 11 2A", "bytes": [{"address": "40 11 2A"}]}]},
    )
    stages = {s["stage"]: s for s in completion.survey(tmp_path)["stages"]}
    assert stages["accepted values"]["verdict"] == completion.MET


def test_a_stage_that_records_a_whole_region_at_once_is_read_as_coverage(tmp_path) -> None:
    """The hold probe names its addresses in `given`, not in a `bytes` entry each.

    Read for one spelling only, the other stage reports as having covered
    nothing -- which reads as a stage nobody ran, rather than as a reader that
    cannot see it.
    """
    _write(tmp_path, "offsets/whole-map.json", _offsets(("40 11", ["00", "01"])))
    _write(
        tmp_path,
        "hold-probe/whole-map.json",
        {
            "regions": [
                {
                    "start": "40 11 00",
                    "given": {"40 11 00": "00", "40 11 01": "7F"},
                    "read_back": {"40 11 00": "00", "40 11 01": "7F"},
                }
            ]
        },
    )
    stages = {s["stage"]: s for s in completion.survey(tmp_path)["stages"]}
    assert stages["independent storage"]["verdict"] == completion.MET


def test_a_stage_counted_over_shapes_does_not_grow_with_the_number_of_parts(tmp_path) -> None:
    """Sixteen parts of one shape are one representative's worth of coverage."""
    parts = [(f"40 1{n:X}", ["00", "2A"]) for n in range(16)]
    _write(tmp_path, "offsets/whole-map.json", _offsets(*parts))
    _write(
        tmp_path,
        "write-probe/whole-map.json",
        {
            "regions": [
                {"start": "40 10 00", "bytes": [{"address": "40 10 00"}, {"address": "40 10 2A"}]}
            ]
        },
    )
    stages = {s["stage"]: s for s in completion.survey(tmp_path)["stages"]}
    assert stages["accepted values"]["verdict"] == completion.MET
    assert "all 1 shapes" in stages["accepted values"]["evidence"]


def test_a_block_measured_to_be_a_window_is_not_counted_as_one_left_to_ask(tmp_path) -> None:
    """Asking it would ask the store it points at a second time."""
    _write(
        tmp_path,
        "sweep/whole-map.json",
        {
            "complete": True,
            "trustworthy": True,
            "regions": _regions(("40 11", [2]), ("42 11", [2])),
            "findings": [{"kind": "blocks-that-are-a-window", "blocks": ["42"]}],
        },
    )
    _write(tmp_path, "offsets/whole-map.json", _offsets(("40 11", ["00"])))
    stages = {s["stage"]: s for s in completion.survey(tmp_path)["stages"]}
    # Counting the window would leave one block short, and the only way to make
    # that up would be to measure the store it points at a second time.
    assert stages["offsets and shapes"]["verdict"] != completion.UNMET


def test_an_offsets_run_that_lost_the_unit_does_not_stand_as_coverage(tmp_path) -> None:
    record = _offsets(("40 11", ["00"]))
    record["stopped"] = "the canary stopped answering"
    _write(tmp_path, "offsets/whole-map.json", record)
    stages = {s["stage"]: s for s in completion.survey(tmp_path)["stages"]}
    assert stages["offsets and shapes"]["verdict"] == completion.UNMET


def test_blocks_of_one_shape_are_one_kind_so_a_window_is_not_counted_twelve_times(
    tmp_path,
) -> None:
    """Twelve blocks of the same shape are one block's worth of work, not twelve."""
    blocks = [(f"4{n:X} 01", [2, 64]) for n in range(1, 13)]
    _write(
        tmp_path,
        "sweep/whole-map.json",
        {"complete": True, "trustworthy": True, "regions": _regions(*blocks)},
    )
    kinds = completion.block_kinds(tmp_path)
    assert len(kinds) == 1
    assert len(next(iter(kinds.values()))) == 12
    stage = completion.whole_blocks(tmp_path)
    assert stage.remaining["addresses in one block of each"] == 66


def test_a_swept_block_answers_for_every_block_of_its_shape(tmp_path) -> None:
    _write(
        tmp_path,
        "sweep/whole-map.json",
        {
            "complete": True,
            "trustworthy": True,
            "regions": _regions(("40 11", [47, 9]), ("40 12", [47, 9]), ("40 01", [16, 24])),
        },
    )
    _write(tmp_path, "block/40-11.json", {"block": "40 11"})
    stage = completion.whole_blocks(tmp_path)
    assert stage.verdict == completion.UNMET
    assert stage.remaining["addresses in one block of each"] == 40
    assert stage.remaining["kinds not swept"] == ["40 01, sizes [16, 24]"]


def test_an_effect_screen_that_admits_every_type_has_bounded_nothing(tmp_path) -> None:
    """The count of parameters left to ask must not fall because a type was audible."""
    _write(
        tmp_path,
        "efx-map/types.json",
        {"accepted": 2, "effects": [{"parameters": [0] * 20}, {"parameters": [0] * 20}]},
    )
    _write(tmp_path, "transfer/analogue-input.json", {"answer": "no"})
    _write(tmp_path, "efx-motion/tracked.json", {"types": [{"type": "00 00", "audible": True}]})
    stage = completion.effect_response(tmp_path)
    assert stage.verdict == completion.UNMET
    assert stage.remaining == {"effect parameters to screen": 40}


def test_a_parameter_carrying_its_own_verdict_is_counted(tmp_path) -> None:
    _write(tmp_path, "efx-map/types.json", {"accepted": 1, "effects": [{"parameters": [0, 0]}]})
    _write(tmp_path, "transfer/analogue-input.json", {"answer": "no"})
    _write(
        tmp_path,
        "efx-params/00-00.json",
        {
            "parameters": [
                {"type": "00 00", "parameter": 0, "audible": True},
                {"type": "00 00", "parameter": 1, "audible": False},
            ]
        },
    )
    stage = completion.effect_response(tmp_path)
    assert stage.verdict == completion.MET


def test_the_route_has_to_be_established_before_the_stage_can_be_short_of_it(
    tmp_path,
) -> None:
    """Without it, nothing says whether a known signal can be swept through at all."""
    _write(tmp_path, "efx-map/types.json", {"accepted": 1, "effects": [{"parameters": [0]}]})
    stage = completion.effect_response(tmp_path)
    assert stage.verdict == completion.UNMET
    assert "unasked" in stage.evidence


def _documented(root, unit, *rows: str) -> None:
    """A unit naming a document, and that document's address map rows."""
    _write(unit, "meta.json", {"unit_id": "somebody-01", "documents": ["a-manual"]})
    (root / "documents" / "a-manual").mkdir(parents=True, exist_ok=True)
    (root / "documents" / "a-manual" / "address-map.json").write_text(
        json.dumps({"rows": [{"address": a} for a in rows]})
    )


def test_a_unit_naming_no_document_has_nothing_outside_itself_to_be_counted_against(
    tmp_path,
) -> None:
    _write(tmp_path, "meta.json", {"unit_id": "somebody-01"})
    stages = {s["stage"]: s for s in completion.survey(tmp_path)["stages"]}
    assert stages["documented addresses"]["verdict"] == completion.UNDECIDED


def test_a_block_a_document_names_and_nobody_asked_is_what_is_left(tmp_path) -> None:
    """It has no shape to be folded into, because nothing was ever measured there."""
    unit = tmp_path / "data" / "units" / "somebody-01"
    unit.mkdir(parents=True)
    _documented(tmp_path, unit, "40 11 00", "40 40 20")
    _write(unit, "offsets/whole-map.json", _offsets(("40 11", ["00"])))
    _write(
        unit,
        "sweep/whole-map.json",
        {
            "complete": True,
            "trustworthy": True,
            "regions": _regions(("40 11", [1]), ("40 40", [1])),
        },
    )
    stage = completion.documented_addresses(unit)
    assert stage.verdict == completion.UNMET
    assert stage.remaining["blocks the documents name that were never asked"] == ["40 40"]


def test_a_row_naming_a_family_is_not_counted_once_per_part(tmp_path) -> None:
    """`40 1x 00` is one statement about sixteen parts, not sixteen things to ask."""
    unit = tmp_path / "data" / "units" / "somebody-01"
    unit.mkdir(parents=True)
    _documented(tmp_path, unit, "40 1x 00")
    parts = [(f"40 1{n:X}", ["00"]) for n in range(16)]
    _write(unit, "offsets/whole-map.json", _offsets(*parts))
    _write(
        unit,
        "sweep/whole-map.json",
        {"complete": True, "trustworthy": True, "regions": _regions(*[(b, [1]) for b, _ in parts])},
    )
    stage = completion.documented_addresses(unit)
    assert stage.verdict == completion.MET


def test_a_later_run_asking_named_offsets_counts_as_having_asked_them(tmp_path) -> None:
    """The first run skipped the block; a second asked the three offsets named.

    A record names the range it asked by its first address, so `40 40 20` for
    three is offsets 20 to 22. Read as a count from the start of the block it
    would be 00 to 02, and the offsets actually asked would report as unasked.
    """
    unit = tmp_path / "data" / "units" / "somebody-01"
    unit.mkdir(parents=True)
    _documented(tmp_path, unit, "40 40 20", "40 40 21", "40 40 22")
    _write(unit, "offsets/whole-map.json", _offsets(("40 11", ["00"])))
    _write(
        unit,
        "offsets/documented.json",
        {
            "blocks": [
                {
                    "address": "40 40 20",
                    "offsets_asked": 3,
                    "answered": {f"40 40 {o}": "00" for o in ("20", "21", "22")},
                }
            ]
        },
    )
    _write(
        unit,
        "sweep/whole-map.json",
        {
            "complete": True,
            "trustworthy": True,
            "regions": _regions(("40 11", [1]), ("40 40", [1])),
        },
    )
    assert completion.documented_addresses(unit).verdict == completion.MET


def test_a_block_asked_at_three_offsets_is_not_a_shape_for_others_to_fold_into(
    tmp_path,
) -> None:
    """Its answers reply to a narrower question than a whole-block sweep asked."""
    unit = tmp_path / "data" / "units" / "somebody-01"
    unit.mkdir(parents=True)
    _write(unit, "offsets/whole-map.json", _offsets(("40 11", ["00"])))
    _write(
        unit,
        "offsets/documented.json",
        {"blocks": [{"address": "40 40 20", "offsets_asked": 3, "answered": {"40 40 20": "00"}}]},
    )
    shapes = completion._shapes(unit)
    assert [b for blocks in shapes.values() for b in blocks] == ["40 11 00"]


def test_a_letter_in_a_template_does_not_spell_an_eighth_bit(tmp_path) -> None:
    """An address byte carries seven bits, so 80 to FF are not addresses to ask."""
    unit = tmp_path / "data" / "units" / "somebody-01"
    unit.mkdir(parents=True)
    _documented(tmp_path, unit, "40 11 xx")
    _write(unit, "offsets/whole-map.json", _offsets(("40 11", ["00"])))
    _write(
        unit,
        "sweep/whole-map.json",
        {"complete": True, "trustworthy": True, "regions": _regions(("40 11", [1]))},
    )
    named = completion._named_by_documents(unit, {"documents": ["a-manual"]})
    assert max(named["40 11"]) == 0x7F
    assert len(named["40 11"]) == 128


def _capture(*answered: str, unread: int = 0) -> dict:
    """A power-on capture holding a value for each address named."""
    return {
        "regions_read": len(answered),
        "regions_unread": unread,
        "values": dict.fromkeys(answered, "00"),
    }


def test_a_capture_that_left_no_region_unread_can_still_be_short_of_an_address(
    tmp_path,
) -> None:
    """A region read can come back shorter than it was asked for.

    One did: thirty bytes for a thirty-two byte request, and the two it left out
    answer when they are asked alone. Counting regions cannot see that, so the
    capture reports nothing unread while two of its addresses have no baseline.
    """
    _write(tmp_path, "offsets/whole-map.json", _offsets(("40 03", ["00", "1E"])))
    _write(tmp_path, "power-on/whole-map.json", _capture("40 03 00", unread=0))
    stages = {s["stage"]: s for s in completion.survey(tmp_path)["stages"]}
    assert stages["power-on state"]["verdict"] == completion.UNMET
    assert stages["power-on state"]["remaining"] == {
        "addresses answering with no power-on value": 1
    }


def test_a_later_capture_over_a_wider_map_fills_what_the_first_one_never_read(
    tmp_path,
) -> None:
    """A second power cycle does not unmake the first capture, so they union."""
    _write(tmp_path, "offsets/whole-map.json", _offsets(("40 03", ["00", "1E"])))
    _write(tmp_path, "power-on/whole-map.json", _capture("40 03 00"))
    _write(tmp_path, "power-on/whole-map-wider.json", _capture("40 03 00", "40 03 1E"))
    stages = {s["stage"]: s for s in completion.survey(tmp_path)["stages"]}
    assert stages["power-on state"]["verdict"] == completion.MET


def test_a_capture_with_nothing_to_hold_it_against_is_not_passed(tmp_path) -> None:
    """Without a list of what answers, a full capture and an empty one look alike."""
    _write(tmp_path, "power-on/whole-map.json", _capture("40 03 00"))
    stages = {s["stage"]: s for s in completion.survey(tmp_path)["stages"]}
    assert stages["power-on state"]["verdict"] == completion.UNDECIDED


def test_the_survey_reports_counts_and_never_hours(tmp_path) -> None:
    """A rate belongs to a chain, and one carried across units is a wrong number."""
    _write(tmp_path, "meta.json", {"unit_id": "somebody-01"})
    found = completion.survey(tmp_path)
    assert "hour" not in json.dumps(found["remaining"]).lower()
    assert "counts_not_hours" in found
