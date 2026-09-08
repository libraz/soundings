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
