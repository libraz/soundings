"""What is published under `documents/`, held against itself.

These records are read by hand, a page at a time, over however long it takes.
That is the point of them and it is also what makes them easy to get quietly
wrong: a row citing a page nobody registered, a hand-kept row the parser has
since learnt to read, a note restated against an address no table holds. None of
those look like failures. Each of them puts something on the site that no page
of any document states.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soundings import documents

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ROOT / "documents"
IDS = sorted(p.name for p in DOCUMENTS.iterdir() if p.is_dir()) if DOCUMENTS.exists() else []


def load(document_id: str, name: str):
    path = DOCUMENTS / document_id / name
    return json.loads(path.read_text()) if path.is_file() else None


def rows_of(document_id: str) -> list[dict]:
    parsed = load(document_id, "address-map.json") or {"rows": []}
    hand = load(document_id, "by-hand.json") or {"rows": []}
    return [*parsed["rows"], *hand["rows"]]


@pytest.mark.parametrize("document_id", IDS)
def test_a_document_says_which_file_it_was_read_from(document_id: str) -> None:
    """The file is not committed, so the record has to be enough to find it."""
    meta = load(document_id, "document.json")
    assert meta is not None, f"{document_id} has no document.json"
    source = meta["source_file"]
    assert len(source["sha256"]) == 64
    assert source["pages"] >= 1
    assert isinstance(source["page_offset"], int)
    assert meta["language"]
    assert meta["publisher"] and meta["copyright"]


@pytest.mark.parametrize("document_id", IDS)
def test_every_page_of_the_document_is_in_the_ledger(document_id: str) -> None:
    """An unread page is not an empty one, so every page has to be in there."""
    meta = load(document_id, "document.json")
    assert sorted(int(page) for page in meta["pages"]) == list(
        range(1, meta["source_file"]["pages"] + 1)
    )


@pytest.mark.parametrize("document_id", IDS)
def test_every_row_cites_a_page_somebody_read(document_id: str) -> None:
    """A citation of a page nobody registered is a citation of nothing."""
    meta = load(document_id, "document.json")
    offset = meta["source_file"].get("page_offset", 0)
    unread = []
    for row in rows_of(document_id):
        state = meta["pages"].get(str(row["page"] + offset))
        if state not in {"read", "nothing readable"}:
            unread.append((row["address"], row["page"], state))
    assert not unread, (
        f"{document_id}: rows cite pages the ledger does not have as read: {unread[:5]}"
    )


@pytest.mark.parametrize("document_id", IDS)
def test_no_row_is_both_parsed_and_hand_kept(document_id: str) -> None:
    """A row read both ways is a mistake in one of them.

    The hand-kept file is for what the parser refused. A row it has since learnt
    to read belongs in one place, and whichever of the two is right, showing one
    and dropping the other would hide that they disagree.
    """
    parsed = {
        (row["page"], row["address"])
        for row in (load(document_id, "address-map.json") or {"rows": []})["rows"]
    }
    hand = {
        (row["page"], row["address"])
        for row in (load(document_id, "by-hand.json") or {"rows": []})["rows"]
    }
    assert not parsed & hand, f"{document_id}: in both files: {sorted(parsed & hand)[:5]}"


@pytest.mark.parametrize("document_id", IDS)
def test_the_figure_of_where_the_blocks_begin_is_about_addresses(document_id: str) -> None:
    """The block map is read by hand off a figure, so nothing checks it as it is read.

    It is two columns of a two-column page and the reading of it is easy to put
    an address in the wrong half of. What can be checked is that every entry is
    an address at all, and that no entry says the same thing twice.
    """
    figure = load(document_id, "block-map.json")
    if figure is None:
        pytest.skip(f"{document_id} has no block map")
    seen: set[tuple[str, str, str]] = set()
    for entry in figure["blocks"]:
        assert documents._ADDRESS.match(entry["address"]), (
            f"{document_id}: the block map holds {entry['address']!r}, which is not an address"
        )
        assert entry["block"] and entry["model_id"]
        key = (entry["model_id"], entry.get("port", ""), entry["address"])
        assert key not in seen, f"{document_id}: the block map states {key} twice"
        seen.add(key)


@pytest.mark.parametrize("document_id", IDS)
def test_the_figure_of_where_the_blocks_begin_cites_a_page_somebody_looked_at(
    document_id: str,
) -> None:
    """The block map is the one file no `document add` writes, so nothing marks its page.

    It was read off a page the ledger still had as unread, which says nobody has
    looked at a page a file in this directory was read from. The figure is not a
    parameter address map and the page holding it can hold no rows, so what the
    ledger has to say is that somebody looked -- not that the page was read.
    """
    figure = load(document_id, "block-map.json")
    if figure is None:
        pytest.skip(f"{document_id} has no block map")
    meta = load(document_id, "document.json")
    offset = meta["source_file"].get("page_offset", 0)
    state = meta["pages"].get(str(figure["page"] + offset))
    assert state != documents.UNREAD, (
        f"{document_id}: the block map was read off page {figure['page']}, which the ledger "
        "has as a page nobody has looked at"
    )


@pytest.mark.parametrize("document_id", IDS)
def test_every_address_a_read_page_prints_is_held_somewhere(document_id: str) -> None:
    """A page counted as read has to have given up everything it prints.

    The parser refuses what it cannot cut and says so, and the residue is read by
    hand -- but a row nobody noticed in the residue is not refused, it is absent,
    and an absent row is one the site would show as a thing no document states.
    Checked against the document itself rather than against the residue, which is
    the list of what somebody already knew was missing.

    Skipped where the document is not on this machine: it is not committed.
    """
    meta = load(document_id, "document.json")
    pdf = Path(meta["source_file"]["path_when_read"]).expanduser()
    if not pdf.is_file() or documents.sha256(pdf) != meta["source_file"]["sha256"]:
        pytest.skip(f"{document_id} is not on this machine")

    held = {(row["page"], row["address"]) for row in rows_of(document_id)}
    figure = load(document_id, "block-map.json")
    if figure:
        held |= {(figure["page"], entry["address"]) for entry in figure["blocks"]}

    offset = meta["source_file"].get("page_offset", 0)
    missing: list[tuple[int, str]] = []
    for page, state in meta["pages"].items():
        if state != documents.READ:
            continue
        printed = int(page) - offset
        for line in documents.page_text(pdf, int(page)).splitlines():
            for cell in documents._CELL.finditer(line):
                value = cell.group().strip()
                # A size is three literal bytes and so is shaped like an address.
                if not documents._ADDRESS.match(value) or documents._SIZE.match(value):
                    continue
                if (printed, value) not in held:
                    missing.append((printed, value))
    assert not missing, f"{document_id}: printed and held nowhere: {sorted(set(missing))[:8]}"


@pytest.mark.parametrize("document_id", IDS)
def test_every_restated_note_is_about_a_row_the_document_holds(document_id: str) -> None:
    notes = load(document_id, "qualifications.json")
    if notes is None:
        pytest.skip(f"{document_id} has no restated notes")
    addresses = {row["address"] for row in rows_of(document_id)}
    for note in notes["qualifications"]:
        assert note["address"] in addresses, (
            f"{document_id}: a note is restated against {note['address']}, which no table holds"
        )
        assert note["qualifies"] in {"initial", "behaviour"}, (
            f"{document_id}: {note['address']} does not say what its note qualifies"
        )
        assert note["restated"] and note["page"]


@pytest.mark.parametrize("document_id", IDS)
def test_every_table_note_names_rows_the_document_holds(document_id: str) -> None:
    """A note about a table is recorded as the rows it reaches.

    Which rows those are is a reading, so the reading is written down beside
    them. What cannot be a reading is a template no table holds: that is a
    typo, and it would take the note off the site without taking it out of the
    record, which is the failure mode this file exists to catch.
    """
    statements = load(document_id, "statements.json")
    if statements is None:
        pytest.skip(f"{document_id} has no table notes")
    addresses = {row["address"] for row in rows_of(document_id)}
    seen: set[str] = set()
    for statement in statements["statements"]:
        assert statement["id"] not in seen, (
            f"{document_id}: two notes call themselves {statement['id']}"
        )
        seen.add(statement["id"])
        assert statement["restated"] and statement["read_as"] and statement["open"]
        assert statement["covers"], f"{document_id}: {statement['id']} reaches nothing"
        missing = sorted(set(statement["covers"]) - addresses)
        assert not missing, (
            f"{document_id}: {statement['id']} names rows no table holds: {missing[:5]}"
        )


@pytest.mark.parametrize("document_id", IDS)
def test_a_table_note_is_not_also_a_qualification(document_id: str) -> None:
    """The two files answer different questions and a note belongs in one.

    A qualification says a stated value is not the whole of what the document
    states there, and stops a verdict being put on it. A table note states
    behaviour and stops nothing. A note in both would suppress a comparison
    for a reason the note does not give.
    """
    statements = load(document_id, "statements.json")
    notes = load(document_id, "qualifications.json")
    if statements is None or notes is None:
        pytest.skip(f"{document_id} does not have both kinds of note")
    qualified = {(note["address"], note["page"]) for note in notes["qualifications"]}
    for statement in statements["statements"]:
        clash = sorted(
            address for address in statement["covers"] if (address, statement["page"]) in qualified
        )
        assert not clash, (
            f"{document_id}: {statement['id']} covers {clash[:3]}, which page "
            f"{statement['page']} is also restated against as a qualification"
        )


@pytest.mark.parametrize("document_id", IDS)
def test_a_note_naming_what_a_reset_leaves_says_it_for_a_reset(document_id: str) -> None:
    """`leaves` is compared against a measured reset, so it has to be values."""
    notes = load(document_id, "qualifications.json")
    if notes is None:
        pytest.skip(f"{document_id} has no restated notes")
    for note in notes["qualifications"]:
        for reset, value in (note.get("leaves") or {}).items():
            assert reset.strip(), f"{document_id}: {note['address']} names an empty reset"
            assert len(value) == 2 and value.upper() == value, (
                f"{document_id}: {note['address']} says {reset} leaves {value!r}, "
                "which is not a byte"
            )
