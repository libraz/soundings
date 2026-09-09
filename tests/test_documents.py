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


def test_a_row_whose_size_cell_swallowed_the_column_beside_it_is_refused() -> None:
    """A page set in two columns can join two printed cells with a single space.

    Nothing about the row's position says so -- the cells sit where the grid says,
    and the grid was learnt from rows joined the same way. The size is the one
    cell whose shape says it, and it is the cell the join lands in. Read as a row,
    it put the parameter name in the data column and stated a range no page did.
    """
    joined = [
        "   10 00 00   00 00 20 20-7F      Displayed Letter 32-127(ASCII)     ---",
        "   10 00 01   00 00 20 20-7F      Displayed Letter 32-127(ASCII)     ---",
        "   10 00 02   00 00 20 20-7F      Displayed Letter 32-127(ASCII)     ---",
    ]
    out = read([HEADER, *joined])
    assert out.rows == []
    assert {missed["why"] for missed in out.not_extracted} == {documents.SIZE_SPANS_COLUMNS}


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


def test_a_column_the_header_does_not_name_leaves_the_named_one_settled() -> None:
    """A table can print a column its header never names.

    Refusing the whole tail because the count did not match left two hundred rows
    of one document stating no parameter at all -- and the parameter was never in
    doubt, because there is one named column left and the cell in it begins where
    the grid says it does. What sits further right is kept verbatim, under no
    heading, because the page printed none.

    A header naming nothing past the parameter is also the one an extra column
    cannot shift, which is why these rows are read rather than refused.
    """
    header = "   Address(H)     Size(H)      Data(H)           Parameter"
    rows = [
        "   23 pp 20       00 00 01     00 - 07           CHORUS MACRO     [Pro]   Chorus 1",
        "   23 pp 21       00 00 01     00 - 07           CHORUS PRE-LPF   [Pro]   0-7",
        "   23 pp 22       00 00 01     00 - 7F           CHORUS LEVEL     [Pro]   0-127",
    ]
    out = read([header, *rows])
    assert [row["parameter"] for row in out.rows] == [
        "CHORUS MACRO",
        "CHORUS PRE-LPF",
        "CHORUS LEVEL",
    ]
    assert out.rows[1]["unresolved"] == ["[Pro]", "0-7"]
    # Not filed as a description: the header names no such column here, and the
    # row would then state one the page never printed.
    assert "description" not in out.rows[1]
    assert out.not_extracted == []


def test_a_table_stating_no_description_is_not_read_as_stating_one() -> None:
    """Both columns a table may leave out are printed `Description`.

    Searching the line for each label in turn takes the first one it finds, so a
    table printing `Parameter | Default Value (H) | Description` had the trailing
    label read as its description column and no `Default` left after it. Every row
    under such a header then filed the marker beside its parameter as a
    description -- a heading the page states nothing under, and nothing flagged it.
    """
    header = "   Address(H)  Size(H)  Data(H)  Parameter  Default Value (H)  Description"
    assert documents.header_columns(header) == [
        "address",
        "size",
        "data",
        "parameter",
        "default",
        "default_description",
    ]


def test_a_header_label_broken_across_lines_still_names_its_column() -> None:
    """`Descrip-tion` is the printing's hyphenation, not a column of another name.

    Read as unnamed it left the table with one column more than its header, which
    is the shape this parser refuses -- so a page of rows the header describes
    perfectly well went unread over a hyphen.
    """
    header = (
        "  Address(H)  Size(H)  Data(H)  Parameter  Description  Default Value (H)  Descrip-tion"
    )
    assert documents.header_columns(header)[-1] == "default_description"


def test_a_row_holding_a_column_the_header_does_not_name_is_refused() -> None:
    """The extra column may sit between two the header does name.

    Reading the cells in the header's order then puts every value after it under
    the heading of the column to its left: a range under `default`, an initial
    value under `default_description`. Nothing about the row says so, which is why
    it is refused and read by hand rather than cut.
    """
    marked = [
        "   40 03 17   00 00 01   00 - 7F   EFX SEND LEVEL TO REVERB   [Pro]   0-127     28   40",
        "   40 03 18   00 00 01   00 - 7F   EFX SEND LEVEL TO CHORUS   [Pro]   0-127     00   0",
        "   40 03 1F   00 00 01   00 - 01   EFX SEND EQ SWITCH         [Pro]   OFF/ON    01   ON",
    ]
    out = read([HEADER, *marked], page=196)
    assert out.rows == []
    assert {missed["why"] for missed in out.not_extracted} == {documents.UNNAMED_COLUMN_INSIDE}


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


#: A table stating a family of addresses whose first byte carries the placeholder.
#: The page prints what the letter stands for above the table: `2a : Patch part
#: number (Part1: a=4, Part2: a=6)`.
PART_ROWS = [
    "   2a pp 30     00 00 01     00 - 7F     TONE MODIFY1       -64 - +63     40     0",
    "   2a pp 31     00 00 01     00 - 7F     TONE MODIFY2       -64 - +63             40       0",
    "   2a pp 32     00 00 01     00 - 7F     TONE MODIFY3       -64 - +63"
    "                   40       0",
]


def test_a_placeholder_letter_is_read_in_the_first_byte_as_in_the_others() -> None:
    """A whole table family is entered at an address the part number decides.

    Requiring the first byte to be a literal made every one of these lines
    something other than a row, so the grid could not be learnt from them either
    and the pages carrying them read as tables that could not be read at all.
    """
    assert documents._ADDRESS.match("2a pp 30")
    out = read([HEADER, *PART_ROWS])
    assert [row["address"] for row in out.rows] == ["2a pp 30", "2a pp 31", "2a pp 32"]
    assert out.rows[0]["parameter"] == "TONE MODIFY1"


def test_a_header_is_carried_across_every_page_the_table_runs_over() -> None:
    """A table three pages long heads its third page from two pages back.

    Looking only at the page before found the header for the second page of every
    table and none for the third, which lost the last page of a fourteen-page map.
    """
    pages = {1: "\n".join([HEADER, *ROWS]), 2: "\n".join(PART_ROWS), 3: "\n".join(PART_ROWS)}
    assert documents.header_above(pages.get, 3) == documents.last_header(pages[1])


def test_the_walk_back_for_a_header_stops_where_the_table_did() -> None:
    """A page of prose ends a table, and no page after it is inside one.

    Carrying a heading across one would head a later table with an earlier one's
    columns, which is the failure this parser exists to refuse.
    """
    pages = {
        1: "\n".join([HEADER, *ROWS]),
        2: "Section 4. Bulk Dump\nBulk Dump allows you to transmit a large amount of data.",
        3: "\n".join(PART_ROWS),
    }
    assert documents.holds_rows(pages[2]) is False
    assert documents.header_above(pages.get, 3) is None


def test_every_page_of_a_document_starts_unread() -> None:
    ledger = documents.ledger(4)
    assert ledger == dict.fromkeys(("1", "2", "3", "4"), documents.UNREAD)
