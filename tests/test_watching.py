"""The set of addresses a stage watches, assembled from what the unit answered.

This is the fix for the defect that cost this archive most. Three stages take a
map and bound every negative they make by it, and all three were given the
sweep's map -- which asks a named set of third bytes under each block, and so
says which blocks exist rather than which addresses do. A message landing outside
every region it named was recorded as landing nowhere, in five stages at once,
and each was found by stepping on it rather than by counting.

So what is checked here is that the set is a union and not a substitution, that
nothing is carried between units, and that each half is present: the addresses
only a region read reaches, and the addresses only a single read reaches.

No hardware, and no unit: every directory here is built in the test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soundings import watching


def _unit(tmp_path: Path, *, sweep: dict, offsets: dict | None = None) -> Path:
    unit = tmp_path / "data" / "units" / "somebody-01"
    (unit / "sweep").mkdir(parents=True)
    (unit / "sweep" / "whole-map.json").write_text(json.dumps(sweep))
    if offsets is not None:
        (unit / "offsets").mkdir()
        (unit / "offsets" / "whole-map.json").write_text(json.dumps(offsets))
    return unit


def _answered(*addresses: str) -> dict:
    return {
        "blocks": [
            {"address": f"{a} 00", "answered": {a: "00"}} for a in ()
        ] + [{"address": "40 11 00", "answered": dict.fromkeys(addresses, "00")}]
    }


def test_the_set_is_the_union_of_both_reads_and_not_either_one(tmp_path) -> None:
    """Neither contains the other, so watching one leaves a place nothing looks.

    A region read reaches addresses that answer no single read; a single read
    reaches addresses that begin where no region does. A set built from one of
    them was measured to lose a store the other found.
    """
    unit = _unit(
        tmp_path,
        sweep={"regions": [{"address": "40 11 00", "size": 2}]},
        offsets=_answered("40 11 00", "40 11 2A"),
    )
    built = watching.build(unit)
    held = {
        f"{r['address'][:5]} {int(r['address'][6:8], 16) + i:02X}"
        for r in built["regions"]
        for i in range(r["size"])
    }
    # 40 11 01 comes only from the sweep, 40 11 2A only from the offsets record.
    assert held == {"40 11 00", "40 11 01", "40 11 2A"}
    assert built["addresses"] == 3


def test_a_block_measured_to_be_a_window_is_left_out_unless_it_is_asked_for(tmp_path) -> None:
    """A value landing in a window lands in the store it points at.

    Watching both asks one store twice and reports it as two. A capture of what
    each address held wants it anyway, since a window holds what it points at.
    """
    sweep = {
        "regions": [{"address": "40 11 00", "size": 1}, {"address": "42 11 00", "size": 1}],
        "findings": [{"kind": "blocks-that-are-a-window", "blocks": ["42"]}],
    }
    unit = _unit(tmp_path, sweep=sweep, offsets=_answered("40 11 00"))
    assert watching.build(unit)["addresses"] == 1
    assert watching.build(unit, keep_windows=True)["addresses"] == 2


def test_asking_one_at_a_time_holds_only_what_a_single_read_answers(tmp_path) -> None:
    """A reply to a single read cannot land a byte on the address below its own.

    It reaches less: an address that comes back only inside a longer read is not
    in it, and that is what the other kind of map is for.
    """
    unit = _unit(
        tmp_path,
        sweep={"regions": [{"address": "40 11 00", "size": 4}]},
        offsets=_answered("40 11 00", "40 11 02"),
    )
    built = watching.build(unit, one_at_a_time=True)
    assert [r["address"] for r in built["regions"]] == ["40 11 00", "40 11 02"]
    assert all(r["size"] == 1 for r in built["regions"])


def test_consecutive_addresses_become_one_region(tmp_path) -> None:
    """A run asked in one request is one request, and a gap is where a run ends."""
    unit = _unit(
        tmp_path,
        sweep={"regions": []},
        offsets=_answered("40 11 00", "40 11 01", "40 11 02", "40 11 10"),
    )
    assert [(r["address"], r["size"]) for r in watching.build(unit)["regions"]] == [
        ("40 11 00", 3),
        ("40 11 10", 1),
    ]


def test_a_unit_with_no_offsets_record_is_refused_rather_than_given_the_sweep(tmp_path) -> None:
    """That substitution is the defect. Falling back to it would reintroduce it."""
    unit = _unit(tmp_path, sweep={"regions": [{"address": "40 11 00", "size": 2}]})
    with pytest.raises(FileNotFoundError, match="which addresses answer"):
        watching.build(unit)


def test_a_unit_with_no_sweep_is_refused_rather_than_given_another_units_map(tmp_path) -> None:
    """A set carried across is a finding about one machine arriving as an assumption."""
    unit = tmp_path / "data" / "units" / "somebody-01"
    (unit / "offsets").mkdir(parents=True)
    (unit / "offsets" / "whole-map.json").write_text(json.dumps(_answered("40 11 00")))
    with pytest.raises(FileNotFoundError, match="no default map"):
        watching.build(unit)


def test_every_offsets_record_is_read_not_only_the_whole_map_one(tmp_path) -> None:
    """A later run asks the offsets a document names in blocks the first skipped."""
    unit = _unit(
        tmp_path,
        sweep={"regions": []},
        offsets=_answered("40 11 00"),
    )
    (unit / "offsets" / "documented.json").write_text(
        json.dumps({"blocks": [{"address": "40 40 20", "answered": {"40 40 20": "01"}}]})
    )
    held = {r["address"] for r in watching.build(unit)["regions"]}
    assert held == {"40 11 00", "40 40 20"}


def test_the_file_says_it_is_not_a_measurement_and_what_it_was_built_from(tmp_path) -> None:
    """It is a list assembled from records, and a reader has to be able to rebuild it."""
    unit = _unit(
        tmp_path,
        sweep={"regions": [{"address": "40 11 00", "size": 1}]},
        offsets=_answered("40 11 00"),
    )
    built = watching.build(unit)
    assert "not a measurement" in built["is_not_a_measurement"]
    assert any("sweep/whole-map.json" in p for p in built["built_from"])
    assert any("offsets/whole-map.json" in p for p in built["built_from"])


def test_a_block_a_record_names_is_asked_one_address_at_a_time(tmp_path) -> None:
    """A region read of such a block carries its first byte and zeros after it.

    Length and checksum are both what was asked for, so nothing refuses the reply
    and every byte of it is published. A stage given a map that asks the block as
    a region therefore cannot see a value land in it, and its negatives there are
    about the reply rather than about the unit.
    """
    unit = _unit(
        tmp_path,
        sweep={"regions": [{"address": "21 0C 00", "size": 4}]},
        offsets={
            "blocks": [
                {
                    "address": "21 0C 00",
                    "answered": {f"21 0C {i:02X}": f"{i:02X}" for i in range(4)},
                }
            ]
        },
    )
    (unit / "power-on").mkdir()
    (unit / "power-on" / "whole-map.json").write_text(
        json.dumps(
            {
                "findings": [
                    {
                        "kind": watching.FIRST_BYTE_ONLY,
                        "blocks": ["21 0C"],
                    }
                ]
            }
        )
    )
    built = watching.build(unit)

    assert [r["size"] for r in built["regions"]] == [1, 1, 1, 1]
    assert built["addresses"] == 4
    assert built["first_byte_only"]["blocks"] == ["21 0C"]
    assert built["first_byte_only"]["addresses"] == 4
    assert any("power-on/whole-map.json" in p for p in built["built_from"])


def test_a_unit_whose_records_name_no_such_block_is_asked_as_runs(tmp_path) -> None:
    """The blocks are one unit's answer, so the next unit starts with none of them."""
    unit = _unit(
        tmp_path,
        sweep={"regions": [{"address": "21 0C 00", "size": 4}]},
        offsets={
            "blocks": [
                {
                    "address": "21 0C 00",
                    "answered": {f"21 0C {i:02X}": f"{i:02X}" for i in range(4)},
                }
            ]
        },
    )
    built = watching.build(unit)

    assert [r["size"] for r in built["regions"]] == [4]
    assert built["first_byte_only"]["blocks"] == []


def test_a_map_asked_for_a_prefix_holds_only_that_and_says_so(tmp_path) -> None:
    """A run aimed at part of the space says nothing about the rest, and the map is the bound."""
    unit = _unit(
        tmp_path,
        sweep={"regions": [{"address": "21 0C 00", "size": 2}, {"address": "40 11 00", "size": 2}]},
        offsets=_answered("40 11 00"),
    )
    built = watching.build(unit, prefixes=["21 0C"])

    assert {r["address"] for r in built["regions"]} == {"21 0C 00"}
    assert built["prefixes"]["asked_for"] == ["21 0C"]
    assert "says nothing about the rest" in built["prefixes"]["why"]

    whole = watching.build(unit)
    assert whole["prefixes"]["asked_for"] == []
    assert len(whole["regions"]) == 2
