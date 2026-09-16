"""Reading a table out of a page, and refusing the rows that cannot be read.

Every case here is one the parser got wrong first. A row cut at the wrong column
is the failure that matters: it does not look like a failure, it looks like a
document disagreeing with a unit, and that is a claim the archive has no
business publishing.
"""

from __future__ import annotations

from soundings import documents
from soundings.cli import documents as reading

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


def test_every_table_the_ledger_keeps_a_state_for_has_something_that_reads_it() -> None:
    """The ledger is written per table and the readers are registered separately.

    A table in one list and not the other is either a column of the ledger
    nothing will ever write, or a pass whose pages are recorded nowhere.
    """
    assert sorted(reading.TABLES) == sorted(documents.TABLES)


def test_every_page_of_a_document_starts_unread_for_every_table() -> None:
    ledger = documents.ledger(4)
    assert sorted(ledger) == ["1", "2", "3", "4"]
    assert all(
        page == dict.fromkeys(documents.TABLES, documents.UNREAD) for page in ledger.values()
    )
    assert documents.state_of({"pages": ledger}, 3, "effect-list") == documents.UNREAD


#: The effect list as the page actually sets it: two columns, each opening with a
#: type's number and name and the two bytes that select it, and the parameters
#: under it with their ranges right-aligned in the same column. The long line is
#: what the columns are found by -- it is the widest thing in the left column, so
#: nothing to the left of where it ends can be a gutter.
WAH = [
    "8: Auto Wah                          [01H, 21H]"
    "          9: Rotary                           [01H, 22H]",
    "The Auto Wah cyclically controls a filter to make a b"
    "        A Rotary effect simulates a rotary speaker.",
    "Fil Type (Filter Type)                  LPF/BPF [1]      Low Slow (Low frequency slow rate)",
    "  Select the filter type.                                                    0.05 - 10.0 [1]",
    "#Rate                               0.05 - 10.0 [5]"
    "        Hi Fast (High frequency fast rate) 0.05 - 10.0 [6]",
    "  Adjust the rate of modulation."
    "                                    Adjust the speed of the high-range rotor.",
]


def test_an_effect_types_parameters_are_read_under_the_type_they_are_printed_under() -> None:
    """A parameter number means nothing without the type it numbers a parameter of."""
    out = documents.read_effect_list("\n".join(WAH), 59)
    wah = [row for row in out.rows if row["lsb"] == "21"]
    assert wah[0] == {
        "type": "8",
        "effect": "Auto Wah",
        "msb": "01",
        "lsb": "21",
        "page": 59,
        "read_by": "parser",
    }
    assert [(row["parameter_number"], row["parameter"], row["data"]) for row in wah[1:]] == [
        ("1", "Fil Type (Filter Type)", "LPF/BPF"),
        ("5", "#Rate", "0.05 - 10.0"),
    ]
    assert not out.not_extracted


def test_a_second_column_is_read_as_a_second_column() -> None:
    """`pdftotext -layout` prints both halves of a line as one line.

    Read a line at a time, the right column's parameters fall under the left
    column's type -- silently, and with a name and a range that are both right.
    """
    out = documents.read_effect_list("\n".join(WAH), 59)
    rotary = [row for row in out.rows if row["lsb"] == "22"]
    assert rotary[0]["effect"] == "Rotary"
    assert [(row["parameter_number"], row["parameter"], row["data"]) for row in rotary[1:]] == [
        ("1", "Low Slow (Low frequency slow rate)", "0.05 - 10.0"),
        ("6", "Hi Fast (High frequency fast rate)", "0.05 - 10.0"),
    ]


def test_a_name_too_long_for_its_line_is_read_off_the_line_above_its_range() -> None:
    """Printed on two lines, and the range alone begins right of the column's edge."""
    out = documents.read_effect_list("\n".join(WAH), 59)
    wrapped = next(row for row in out.rows if row.get("parameter", "").startswith("Low Slow"))
    assert wrapped["data"] == "0.05 - 10.0"


def test_a_name_run_together_with_its_range_is_cut_at_the_parenthetical() -> None:
    """One space between them makes them one cell, and the parenthetical is the only
    boundary the printing marks in it."""
    out = documents.read_effect_list("\n".join(WAH), 59)
    joined = next(row for row in out.rows if row.get("parameter", "").startswith("Hi Fast"))
    assert joined["parameter"] == "Hi Fast (High frequency fast rate)"
    assert joined["data"] == "0.05 - 10.0"


def test_a_name_run_together_with_its_range_and_no_parenthetical_is_refused() -> None:
    """Nothing in the cell says where the name ends, and half a name filed as a
    range reads exactly like a range."""
    out = documents.read_effect_list("\n".join([WAH[0], "Mod Wave Tri/Sqr/Sin/Saw1/Saw2 [1]"]), 59)
    assert [row for row in out.rows if "parameter_number" in row] == []
    assert out.not_extracted[0]["why"] == documents.NAME_AND_RANGE_JOINED


def test_a_parameter_printed_under_no_type_is_refused() -> None:
    out = documents.read_effect_list("Fil Type (Filter Type)      LPF/BPF [1]", 59)
    assert out.rows == []
    assert out.not_extracted[0]["why"] == documents.PARAMETER_UNDER_NO_TYPE


def test_a_type_is_carried_over_the_page_it_runs_past_the_foot_of() -> None:
    """A long type fills the page after the one that names it, and there its
    parameters are printed under nothing at all."""
    pages = {1: "\n".join(WAH), 2: "Depth                                   0 - 127 [6]"}
    carried = documents.type_above(pages.get, 2)
    assert carried["effect"] == "Rotary"
    out = documents.read_effect_list(pages[2], 61, carried)
    assert out.rows[0]["lsb"] == "22"
    assert out.rows[0]["parameter_number"] == "6"


def test_the_type_carried_forward_is_the_last_one_the_page_actually_opened() -> None:
    """A heading printed on two lines is a heading, to both readings or neither.

    Found by one of them and not the other, a page that opens a type looks like a
    page that opens none: the type from the page before is carried over it, and
    every parameter on the page after is filed under an effect that ended two
    pages earlier.
    """
    wrapped = [
        "52: GTR Multi 5 (Guitar Multi5)" + " " * 34 + "Level (Output level)      0 - 127 [20]",
        " " * 20 + "[04H, 04H]" + " " * 25 + "  Adjust the output level.",
        "This effect connects four effects in series to make"
        "    OD Sel (OD Select)          Odrv/Dist [1]",
        "  the sound of a guitar amplifier."
        "                    Select either Overdrive or Distortion.",
        "OD Sw (Overdrive Switch)                 Off/On [2]"
        "    +OD Drive (OD Drive)          0 - 127 [3]",
        "  Turn the overdrive on and off.                      Adjust the degree of distortion.",
    ]
    out = documents.read_effect_list("\n".join(wrapped), 81)
    assert [row["effect"] for row in out.rows if "parameter_number" not in row] == [
        "GTR Multi 5 (Guitar Multi5)"
    ]
    assert documents.last_type("\n".join(wrapped))["lsb"] == "04"


def test_the_walk_back_for_a_type_stops_where_the_list_did() -> None:
    """A page holding no parameter ends the list, and no page after it is inside one."""
    pages = {
        1: "\n".join(WAH),
        2: "Chapter 5. Performing\nThe SC-88Pro can play sixteen Parts at once.",
        3: "Depth                                   0 - 127 [6]",
    }
    assert documents.holds_parameters(pages[2]) is False
    assert documents.type_above(pages.get, 3) is None


#: The conversion grid as the page sets it: the columns numbered across the top,
#: their names wrapping onto two lines over the numbers, and each row opening with
#: its own value in hexadecimal and in decimal. Two of the columns give the value
#: itself and are printed left of the first number, which is what has to be kept
#: out of the first quantity's name. The last column is headed with no unit.
GRID = [
    "                        1           2         3        4",
    "                    Pre Delay     Delay    Cutoff",
    " Value    Value       Time        Time 1    Freq      Accl",
    " (Hex.)   (Dec.)      (ms)         (ms)     (Hz)",
    "   00        0         0.0         200       315        0",
    "   01        1         0.1         205         “        “",
    "   02        2         0.2         210         “        1",
]


def test_the_grids_columns_are_named_from_the_numbers_printed_over_them() -> None:
    out = documents.read_value_conversion("\n".join(GRID), 224)
    headings = [row for row in out.rows if "setting" not in row and "type" not in row]
    assert [(row["column"], row["quantity"], row.get("unit")) for row in headings] == [
        (1, "Pre Delay Time", "ms"),
        (2, "Delay Time 1", "ms"),
        (3, "Cutoff Freq", "Hz"),
        (4, "Accl", None),
    ]


def test_a_settings_column_is_the_one_it_is_printed_in_and_not_the_one_it_sits_nearest() -> None:
    """A row of the grid holds one cell per column and says its own value twice.

    So the cells are the quantities in printed order and nothing has to be placed
    by geometry -- which is what keeps a column whose settings are narrower than
    its heading from being read as the column beside it.
    """
    out = documents.read_value_conversion("\n".join(GRID), 224)
    first = [row for row in out.rows if row.get("decimal") == "0"]
    assert [(row["quantity"], row["setting"]) for row in first] == [
        ("Pre Delay Time", "0.0"),
        ("Delay Time 1", "200"),
        ("Cutoff Freq", "315"),
        ("Accl", "0"),
    ]
    assert not out.not_extracted


def test_a_setting_printed_as_a_repeat_is_filed_with_the_one_it_repeats() -> None:
    out = documents.read_value_conversion("\n".join(GRID), 224)
    freq = [row for row in out.rows if row.get("quantity") == "Cutoff Freq" and "setting" in row]
    assert [(row["decimal"], row["setting"], row.get("repeats_above")) for row in freq] == [
        ("0", "315", None),
        ("1", "315", True),
        ("2", "315", True),
    ]


def test_a_repeat_opening_a_page_takes_the_setting_the_page_before_left_standing() -> None:
    """The grid runs over two pages and a column that has not changed for a while
    opens the next one repeating a value printed on the page before."""
    second = [
        "                        1           2         3        4",
        "                    Pre Delay     Delay    Cutoff",
        " Value    Value       Time        Time 1    Freq      Accl",
        " (Hex.)   (Dec.)      (ms)         (ms)     (Hz)",
        "   03        3         0.3         215         “        “",
    ]
    pages = {1: "\n".join(GRID), 2: "\n".join(second)}
    carried = documents.settings_above(pages.get, 2)
    assert carried[3] == "315"
    assert carried[4] == "1"
    out = documents.read_value_conversion(pages[2], 225, carried)
    assert [row["setting"] for row in out.rows if "setting" in row] == ["0.3", "215", "315", "1"]


def test_the_walk_back_for_a_repeat_stops_where_the_grid_did() -> None:
    pages = {
        1: "\n".join(WAH),
        2: "\n".join(GRID),
        3: "   03        3         0.3         215         “        “",
    }
    assert documents.settings_above(pages.get, 2) is None


def test_a_line_opening_like_a_grid_row_and_miscounting_its_cells_is_refused() -> None:
    """Cut by placing its cells at the columns they sit nearest, a short row files
    one quantity's setting under another and nothing looks wrong."""
    short = [*GRID, "   03        3         0.3         215"]
    out = documents.read_value_conversion("\n".join(short), 224)
    assert [missed["why"] for missed in out.not_extracted] == [documents.GRID_ROW_MISCOUNTS]
    assert not [row for row in out.rows if row.get("decimal") == "3"]


#: The index the grid opens with, set as two lists side by side. The first runs off
#: the foot of its own column and into the head of the next, which is why it is read
#: list by list rather than line by line.
INDEX = [
    "    1. Pre Delay Time        6. Rate1",
    "      10: Stereo Flanger       07: Phaser",
    "      11: Step Flanger         08: Auto Wah",
    "    2. Delay Time1           7. Rate 2",
    "      23: 3 Tap Delay          48: GTR Multi 1",
    "    3. Cutoff Freq",
    "      01: Stereo-EQ",
    "    4. Accl",
    "      04: Humanizer",
    "    5. Manual",
    "      07: Phaser",
]


def test_a_list_running_from_one_column_into_the_next_is_read_in_printed_order() -> None:
    out = documents.read_value_conversion("\n".join([*INDEX, *GRID]), 224)
    uses = [row for row in out.rows if "type" in row]
    assert [(row["column"], row["type"], row["effect"]) for row in uses] == [
        (1, "10", "Stereo Flanger"),
        (1, "11", "Step Flanger"),
        (2, "23", "3 Tap Delay"),
        (3, "1", "Stereo-EQ"),
        (4, "4", "Humanizer"),
        (5, "7", "Phaser"),
        (6, "7", "Phaser"),
        (6, "8", "Auto Wah"),
        (7, "48", "GTR Multi 1"),
    ]


def test_an_index_whose_headings_do_not_come_out_in_order_is_left_unread() -> None:
    """Read in the wrong order the types are filed under whichever heading the
    reading happened to put them, and every row of it is printed somewhere real."""
    scrambled = [line.replace("3. Cutoff Freq", "9. Cutoff Freq") for line in INDEX]
    out = documents.read_value_conversion("\n".join([*scrambled, *GRID]), 224)
    assert not [row for row in out.rows if "type" in row]
    assert [missed["why"] for missed in out.not_extracted] == [documents.INDEX_OUT_OF_ORDER]
