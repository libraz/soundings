"""Reading a table out of a page, and refusing the rows that cannot be read.

Every case here is one the parser got wrong first. A row cut at the wrong column
is the failure that matters: it does not look like a failure, it looks like a
document disagreeing with a unit, and that is a claim the archive has no
business publishing.
"""

from __future__ import annotations

from soundings import documents

HEADER = (
    "   Address(H)          Size(H)        Data(H)        Parameter        "
    "Description        Default Value(H)      Description"
)

#: Three rows as the printing actually sets them: flush on the left, and set
#: against their own contents on the right, so the initial value begins in a
#: different column in every one. A fixture that aligned all seven columns would
#: be a table this parser was never written for, and would pass while the real
#: pages failed.
ROWS = [
    "   40 1x 0B     00 00 01     00 - 01     Rx. MODULATION     OFF/ON     01     ON",
    "   40 1x 0C     00 00 01     00 - 01     Rx. VOLUME         OFF/ON             01       ON",
    "   40 1x 0D     00 00 01     00 - 01     Rx. PANPOT         OFF/ON"
    "                     01       ON",
]


def read(lines: list[str], page: int = 238, carried: list[str] | None = None):
    return documents.read_address_map("\n".join(lines), page, carried)


def test_a_page_of_rows_is_read_under_the_header_that_describes_them() -> None:
    out = read([HEADER, *ROWS])
    assert [row["address"] for row in out.rows] == ["40 1x 0B", "40 1x 0C", "40 1x 0D"]
    assert out.rows[0]["parameter"] == "Rx. MODULATION"
    assert out.rows[0]["default"] == "01"
    assert out.rows[0]["page"] == 238
    assert out.rows[0]["read_by"] == "parser"


def test_a_page_continuing_a_table_is_read_under_the_header_it_does_not_carry() -> None:
    """The second page of a table opens with rows and no header of its own.

    Without the carried columns those rows belong to no table and are read as
    nothing at all, which is what happened to every row on two of the twelve
    pages of the first document read.
    """
    assert read(ROWS).rows == []
    carried = documents.last_header("\n".join([HEADER, *ROWS]))
    assert [row["address"] for row in read(ROWS, carried=carried).rows] == [
        "40 1x 0B",
        "40 1x 0C",
        "40 1x 0D",
    ]


def test_the_columns_are_learnt_from_the_rows_rather_than_from_the_header() -> None:
    """The header's labels sit nowhere near the values beneath them.

    Cutting at the header put the first character of one cell at the end of the
    one before it, which read as an initial value of `1 P` -- and so as a unit
    disagreeing with its manual.
    """
    out = read([HEADER, *ROWS])
    assert all(row["default"] == "01" for row in out.rows)
    assert all(row["default_description"] == "ON" for row in out.rows)


def test_a_row_off_the_tables_grid_is_refused_rather_than_cut() -> None:
    stray = "   40 1x 18#                                       Use nibblized data."
    out = read([HEADER, *ROWS, stray])
    assert [row["address"] for row in out.rows] == ["40 1x 0B", "40 1x 0C", "40 1x 0D"]
    assert any(documents.OFF_THE_GRID == missed["why"] for missed in out.not_extracted)


def test_a_line_continuing_a_row_is_folded_into_the_column_it_continues() -> None:
    # Three rows, because the grid is learnt from the rows and one row is not a
    # grid: a position a single row happens to sit at says nothing about where
    # the table's columns are.
    out = read(
        [
            HEADER,
            *ROWS,
            "   40 1x 30     00 00 01     00 - 7F     TONE MODIFY1       -64 - +63     40     0",
            "                                         Vibrato Rate",
        ]
    )
    modify = next(row for row in out.rows if row["address"] == "40 1x 30")
    assert modify["parameter"] == "TONE MODIFY1 Vibrato Rate"
    assert modify["default"] == "40"


def test_a_note_under_a_table_is_not_a_row_and_closes_the_one_above() -> None:
    out = read([HEADER, *ROWS, "   Single: the same note played twice is cut off."])
    assert len(out.rows) == 3
    assert all("Single" not in str(value) for row in out.rows for value in row.values())


def test_a_row_the_page_marks_is_flagged_and_the_note_is_not_copied() -> None:
    marked = (
        "   40 1x 23     00 00 01     00 - 01     Rx.BANK SELECT     OFF/ON"
        "         01(00*)        ON(OFF*)"
    )
    out = read([HEADER, *ROWS, marked])
    bank = next(row for row in out.rows if row["address"] == "40 1x 23")
    assert bank["needs_review"] == documents.QUALIFIED
    assert bank["default"] == "01(00*)"
    assert bank["default_description"] == "ON(OFF*)"
    # The mark is recorded; what it points at is prose printed elsewhere on
    # the page, and the row carries the cells and nothing else.
    assert set(bank) == {
        "address",
        "size",
        "data",
        "parameter",
        "description",
        "default",
        "default_description",
        "page",
        "read_by",
        "needs_review",
    }


def test_an_address_keeps_the_placeholder_letters_the_page_printed() -> None:
    """`40 1x 0A` is one statement about sixteen parts, and stays one row."""
    assert documents._ADDRESS.match("40 1x 0A")
    assert documents._ADDRESS.match("20 b0 pp")
    assert documents._ADDRESS.match("40 1x 18#")
    assert not documents._ADDRESS.match("Displayed Letter")


def test_every_page_of_a_document_starts_unread() -> None:
    ledger = documents.ledger(4)
    assert ledger == dict.fromkeys(("1", "2", "3", "4"), documents.UNREAD)
