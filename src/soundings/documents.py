"""What a published document states, read out of it one page at a time.

The archive records what a unit was measured to do. A document states what its
publisher said the unit would do. These are different kinds of claim and the
archive is careful never to let the second stand in for the first, so what is
read out of a document is kept apart from every measurement: in its own
directory, under its own rights note, and never merged into a measured field.

**Nothing here is evidence about a unit.** A row saying an address accepts
`00 - 01` is evidence that a page said so. Whether the unit agrees is a question
the archive answers, and answering it is the only reason to keep these tables.

Read page by page rather than in one pass. A document is a few hundred pages and
an extraction is fallible; a pass that produced the whole thing at once would
produce a file nobody could review, and a wrong row in it would be indexed,
served and compared against a measurement without ever having been looked at. So
a page enters the record only when somebody has run `document show` on it and
then asked for it, and the ledger says which pages that has happened to.

**An unread page is not an empty one.** The ledger holds every page of the
document from the moment it is initialised, and a page nobody has read says so.
The archive's whole discipline about absence -- that a thing not measured is not
a thing measured to be zero -- would be worth little if the extraction that feeds
it quietly treated the pages nobody got to as pages with nothing on them.

The same discipline runs through the parser. A row it cannot cut with confidence
is refused and shown rather than cut approximately, because a value under the
wrong heading is worse than a value nobody has yet read: the first is wrong and
looks right, and it is the one that would be held against a measurement.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import record

UNREAD = "unread"
"""Nobody has looked at this page. It is not a page with no tables on it."""

READ = "read"
"""Rows from this page are in the tables, and the page's own entry says which."""

NO_TABLES = "no tables"
"""Somebody looked and found nothing this schema can hold. A human's judgement,
not a parser's silence: the parser finding no rows is how a page looks when the
parser is wrong about it."""

NOTHING_READABLE = "nothing readable"
"""A page with tables on it that gave up no row. What stopped it is in the table's
`not_extracted`, per page. Held apart from `read` for the same reason `unread` is
held apart from `no tables`: a page that yielded nothing is not a page that
yielded what it had, and calling both of them read would hide exactly the pages
worth going back to."""


def where_it_was(path: str | Path) -> str:
    """Where the file sat when it was read, with the operator's home out of it.

    The document is not committed, so the record has to hold enough to find the
    file again -- and `~/Downloads/x.pdf` is exactly as much help as the absolute
    path while naming nobody. An absolute one says who ran the extraction and how
    their disk is laid out, neither of which is a fact about the document.

    The inverse is `Path.expanduser`, which is what reads this field back.
    """
    home = str(Path.home())
    text = str(path)
    return "~" + text[len(home) :] if text.startswith(home) else text


def sha256(path: str | Path) -> str:
    """The document's fingerprint, so an extraction can be tied to one file.

    A document is reprinted, corrected and re-issued under the same title, and
    the tables move when it is. The extraction cites printed page numbers, which
    are only meaningful against the printing they were read from, so the record
    holds enough to tell a reader whether the file in their hands is that one.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def page_text(pdf: str | Path, page: int) -> str:
    """One page of the document as laid-out text.

    `-layout` is what makes the tables readable: it keeps each cell at the
    horizontal position it was printed at, which is the only thing that says
    which column a value is in. The tables carry no delimiters, so position is
    the whole of their structure.
    """
    try:
        done = subprocess.run(
            ["pdftotext", "-layout", "-f", str(page), "-l", str(page), str(pdf), "-"],
            capture_output=True,
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "pdftotext is not installed. It is poppler's, and it is the only thing here "
            "that reads a document: install poppler and run this again."
        ) from None
    return done.stdout.decode("utf-8")


def page_count(pdf: str | Path) -> int:
    done = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, check=True)
    for line in done.stdout.decode("utf-8").splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1])
    raise ValueError(f"{pdf} does not say how many pages it has")


def _width(text: str) -> int:
    """How many columns this text occupies when printed.

    A full-width character is one character and two columns. The headers are in
    Japanese and the rows under them are mostly not, so a position counted in
    characters puts the two out of step by the number of full-width characters to
    their left -- which is how a parser reads an address out of the middle of a
    range. Positions here are counted in printed columns for that reason.
    """
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _column_of(line: str, index: int) -> int:
    return _width(line[:index])


def _cut(line: str, start: int, end: int | None) -> str:
    """The text between two printed columns."""
    out = []
    column = 0
    for ch in line:
        if column >= start and (end is None or column < end):
            out.append(ch)
        column += 2 if unicodedata.east_asian_width(ch) in "WF" else 1
    return "".join(out).strip()


#: The columns a parameter address map's header names, in the order they appear.
#: Only the sequence is taken from the header: where the columns actually are is
#: learnt from the rows, per `grid_of`.
HEADINGS = (
    ("address", ("Address", "アドレス")),
    ("size", ("Size", "サイズ")),
    ("data", ("Data", "データ")),
    ("parameter", ("Parameter", "パラメーター")),
    ("description", ("Description", "説明")),
    ("default", ("Default", "初期設定値")),
    ("default_description", ("Description", "説明")),
)
"""Each column's own labels, in every language the document is published in.

One reader for both editions rather than one per language. The tables are the
same tables -- the same addresses, sizes and ranges, laid out the same way -- and
what differs between the editions is which words are printed over the columns and
which language the description column is written in. A second parser would be a
second thing to keep right about a structure that is not in fact different.
"""

OPENING = ("address", "size", "data", "parameter")
"""The columns a table opens with, and the point past which a miscount is silent.

A header naming no more than these can be read against a row holding more cells
than it names: the cells past the parameter are then under no heading at all and
are kept verbatim, which is a reading nothing can shift. A header naming more
than these cannot, because the extra column may sit between two of the ones it
names, and then every value after it reads as the column to its left -- an
initial value under `description`, a range under `default`. That row is refused
and read by hand instead.
"""

#: An address as the tables print it. Any of the three bytes may carry the
#: document's own placeholder letters instead of digits: `x` for a part, `b` for a
#: bank, `pp` for a program, and others per table. They are kept as printed rather
#: than expanded, so that one row stays one statement.
#:
#: The first byte was taken to be a literal at first, on the reasoning that every
#: region of the map is entered at a fixed one. One document states a whole table
#: family at `2a pp xx`, where the letter is the part number and the region it
#: names depends on it. Under the narrower pattern not one of those rows was an
#: address, so the grid could not be learnt from them either and two pages came
#: out as tables that could not be read at all.
_ADDRESS = re.compile(r"^[0-9A-Fa-z]{2} [0-9A-Fa-z]{2} [0-9A-Fa-z]{2}#?$")
#: A size as the tables print it: three bytes, always literal. Unlike an address
#: it never carries a placeholder, because a family of addresses is the same size
#: at every one of them.
_SIZE = re.compile(r"^[0-9A-F]{2} [0-9A-F]{2} [0-9A-F]{2}$")

_CELL = re.compile(r"\S(?:\s(?!\s)|\S)*")

#: How far apart two positions can be and still be one column. The grid is set in
#: whole characters and the rows keep to it within a character or two.
NEARBY = 3

#: How far to the right of the address column a line must begin before it is
#: taken to continue the row above rather than to start one. The notes printed
#: under a table begin where its addresses do; a cell carried onto a second line
#: begins tens of columns further along, so this is not a fine judgement.
INDENTED = 8

QUALIFIED = "the page marks this row with a reference to a note stated elsewhere on it"

#: The reference mark each edition uses for those notes: an asterisk in the
#: English tables, a reference mark in the Japanese ones.
_QUALIFIER = re.compile(r"[※*]")

CONTINUES_A_ROW = (
    "this continues a row, but no row was open above it -- so which row it belongs to is "
    "not settled by where it sits."
)

OFF_THE_GRID = (
    "this row's cells do not begin where the rest of the table's do, so cutting it at the "
    "table's columns would put values under the wrong headings."
)

SIZE_SPANS_COLUMNS = (
    "this row's size cell holds more than the three bytes a size is, so the cut that made "
    "it took in the column beside it -- and every value after it sits under the heading of "
    "the column to its left."
)

NO_GRID = (
    "the rows under this header do not agree on where their columns begin, so there is no "
    "grid to cut them at. A table this cannot read, rather than one to read approximately."
)

SPANS_COLUMNS = (
    "a line continuing this row carries text wider than any one of the table's columns, so "
    "which column it continues is not settled by where it begins. Kept verbatim under "
    "`unresolved` rather than cut at the columns, which would file its fragments under "
    "headings they have nothing to do with."
)

UNNAMED_COLUMN = (
    "{found} cells right of the `{named}` column, which is the last one this table's "
    "header names. The table has a column the header does not, so what these hold is not "
    "settled by any heading: they are kept verbatim under `unresolved`, in the order the "
    "page prints them."
)

UNNAMED_COLUMN_INSIDE = (
    "this row holds more cells than its header names columns, and the header names columns "
    "past the ones a table opens with -- so the extra column may sit anywhere among them, "
    "and reading the cells in the header's order would put every value after it under the "
    "heading of the column to its left. Which cell is which is a question for a reader."
)

UNRESOLVED_TAIL = (
    "{found} cells where the table has {wanted} columns after the parameter, so which "
    "column each holds is not settled by position. They are kept verbatim under "
    "`unresolved` and belong to none of the named columns until a reader says so."
)


def cell_starts(line: str) -> list[int]:
    """The printed column each of a line's cells begins at.

    A cell is a run of text with no gap of two spaces in it. The printing never
    puts a gap that wide inside one value nor a narrower one between two, so this
    is the table's structure rather than a guess about it.
    """
    return [_column_of(line, found.start()) for found in _CELL.finditer(line)]


@dataclass
class Grid:
    """Where a table's columns are, learnt from the table's own rows.

    A header cannot say. Its labels are Japanese, set to their own widths, and
    land nowhere near the values beneath them -- the label over a two-character
    initial value is six characters wide. The rows can say, because there are
    many of them set to one grid.

    Only the left of a table turns out to be a grid. The address, size, data and
    parameter columns are flush in every row; the description and what follows it
    are set against their own contents and start at a different column in nearly
    every row. So the two halves are read differently: the left by cutting at
    fixed columns, the right by taking the row's own cells in the order the
    header names them. Reading the right half by position was tried first and
    put initial values in the description column of half the rows on a page.
    """

    page: int
    names: list[str]
    head: list[int]
    """The columns the fixed part is cut at. The last is where the rest begins."""

    @property
    def fixed(self) -> list[str]:
        return self.names[: len(self.head) - 1]

    @property
    def rest(self) -> list[str]:
        return self.names[len(self.head) - 1 :]

    def agrees(self, starts: list[int]) -> bool:
        """Whether a line's cells begin where the fixed columns do."""
        if len(starts) < len(self.head):
            return False
        return all(abs(starts[i] - at) <= NEARBY for i, at in enumerate(self.head))

    def fixed_cells(self, line: str, starts: list[int]) -> dict[str, str]:
        """The fixed columns of one row, cut where that row's own cells begin.

        The grid says which cells a row has; the row says where they are. Cutting
        at the grid instead is off by the character or two a row is allowed to
        drift, and a cut one column late takes the first character of the value
        to its right -- which read as an initial value of `1 P` on one row, and
        so as a unit disagreeing with its manual.
        """
        out = {}
        for position, name in enumerate(self.fixed):
            value = _cut(line, starts[position], starts[position + 1])
            if value:
                out[name] = value
        return out

    def rest_cells(self, line: str, starts: list[int]) -> list[tuple[int, str]]:
        """The cells right of the fixed part, each with the column it begins at.

        The column comes back with the text because the first of these cells is
        the only one whose heading can be told from where it sits: it is in the
        last column the grid pinned down. Everything further right is set against
        its own contents.
        """
        edge = starts[len(self.head) - 1]
        return [
            (_column_of(line, found.start()), found.group().strip())
            for found in _CELL.finditer(line)
            if _column_of(line, found.start()) >= edge
        ]


def header_columns(line: str) -> list[str] | None:
    """The columns this line names, in order, or None if it names none.

    A header is recognised by naming an address column and a data column. The
    rest are optional and several tables do without them: a drum setup table
    states no initial value, so a parser requiring one would read its rows as
    malformed rather than as rows of a table that states less.

    Matched a printed cell at a time rather than by searching the line for each
    label in turn. The two columns a table can leave out are both called
    `Description`, and a search takes the first one it finds: on a table printing
    `Parameter | Default Value (H) | Description` the search read the trailing
    label as the description column, found no `Default` after it, and named five
    columns for a table that prints six. Every row under it then had the marker
    beside its parameter filed as a description -- a heading the page states
    nothing under, on rows nothing flagged.

    A cell may name more than one column, because a page set in two columns can
    put `Address(H) Size(H)` a single space apart and that is one cell; and a
    label is matched with the printing's own hyphenation taken out, because a
    header breaking `Descrip-tion` across two lines names the column its rows are
    set under just the same.
    """
    names: list[str] = []
    heading = 0
    for found in _CELL.finditer(line):
        cell = found.group().replace("-", "")
        cursor = 0
        while (hit := _labelled(cell, cursor, heading)) is not None:
            heading, name, cursor = hit[0] + 1, hit[1], hit[2] + 1
            names.append(name)
    return names if {"address", "data"} <= set(names) else None


def _labelled(cell: str, cursor: int, heading: int) -> tuple[int, str, int] | None:
    """The first heading from `heading` on whose label this cell carries.

    Headings with no label in the cell are passed over rather than consumed, so a
    cell holding none of them leaves the search where it was. Consuming them would
    let one line of prose use up the headings the header two lines below it names.
    """
    for position in range(heading, len(HEADINGS)):
        name, labels = HEADINGS[position]
        found = [at for label in labels if (at := cell.find(label, cursor)) >= 0]
        if found:
            return position, name, min(found)
    return None


def last_header(text: str) -> list[str] | None:
    """The columns of the last table a page starts, for the page after it.

    A table longer than a page is printed with its header once and its rows on
    both sides of the break, so the second page opens with rows belonging to a
    header it does not carry. Read on its own it looks like a page of rows under
    no table at all, and the rows are lost -- which is what happened to every row
    on two of these twelve pages before this existed.
    """
    found = None
    for line in text.splitlines():
        names = header_columns(line)
        if names:
            found = names
    return found


def header_above(page_text_of: Callable[[int], str], page: int) -> list[str] | None:
    """The columns of the table still open above this page, if any.

    A table carries its header once and its rows on every page after, so a page
    inside one opens with rows under nothing. Looking only at the page before
    finds the header for the second page of a table and nothing for the third,
    which is how the last page of a fourteen-page map came out as a page holding
    no table at all.

    So the walk goes back until a page names a header. It stops at any page
    holding no row this could be a continuation of: a page of prose ends a table,
    and carrying a heading across one would head a later table with the columns of
    an earlier one.

    @param page_text_of the document's pages, by the file's own numbering
    @param page the page being read, which is not itself looked at
    """
    while page > 1:
        page -= 1
        text = page_text_of(page)
        names = last_header(text)
        if names:
            return names
        if not holds_rows(text):
            return None
    return None


def holds_rows(text: str) -> bool:
    """Whether any line on this page opens with something shaped like an address.

    Asked of the pages between a table's header and the page being read, to tell a
    page still under that header from one that ended it. A page of prose holds no
    such line, and a table's rows are nothing but such lines.
    """
    for line in text.splitlines():
        starts = cell_starts(line)
        if not starts:
            continue
        if _ADDRESS.match(_cut(line, starts[0], starts[1] if len(starts) > 1 else None)):
            return True
    return False


def grid_of(lines: list[str], names: list[str], page: int) -> Grid | None:
    """The fixed part of a table's grid, or None if its rows do not agree on one.

    Tried longest first: a table whose rows are flush all the way across is cut
    entirely by position, and one whose right-hand columns wander is cut by
    position as far as they stay flush. Two columns is the least that is worth
    having, since the address and what follows it is the whole shape.
    """
    rows = [
        cell_starts(line)
        for line in lines
        if (starts := cell_starts(line))
        and _ADDRESS.match(_cut(line, starts[0], starts[1] if len(starts) > 1 else None))
    ]
    if len(rows) < 2:
        return None
    for length in range(len(names), 1, -1):
        candidates = [tuple(starts[:length]) for starts in rows if len(starts) >= length]
        best, support = _modal(candidates)
        if best and support * 2 >= len(rows):
            return Grid(page=page, names=names, head=list(best))
    return None


def _modal(candidates: list[tuple[int, ...]]) -> tuple[tuple[int, ...] | None, int]:
    """The position tuple most of the others sit within a character or two of."""
    best: tuple[int, ...] | None = None
    support = 0
    for candidate in candidates:
        agreeing = sum(
            1
            for other in candidates
            if len(other) == len(candidate)
            and all(abs(a - b) <= NEARBY for a, b in zip(candidate, other, strict=True))
        )
        if agreeing > support:
            best, support = candidate, agreeing
    return best, support


@dataclass
class Reading:
    """What one page gave up, and what on it could not be read.

    The two are one result. A page that yielded forty rows and refused two is not
    the same page as one that yielded forty, and a caller who only ever sees the
    rows cannot tell them apart.
    """

    rows: list[dict] = field(default_factory=list)
    not_extracted: list[dict] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.rows or self.not_extracted)


def read_address_map(text: str, page: int, carried: list[str] | None = None) -> Reading:
    """Every parameter address map row printed on one page.

    Rows are taken under the header that describes them, so a page carrying three
    tables is read as three tables. A line beginning where the addresses do but
    holding no address is one of the notes printed under a table: it is not a row
    and it closes the row above, so nothing after it is folded into that row.

    `carried` is the columns of the table the previous page left open, from
    `last_header`. A page continuing a table opens with rows and no header, and
    without it they belong to no table and are read as nothing at all.

    An address like `40 1x 0A` is kept as printed. The `x` is the document's own
    way of writing a family of addresses, and expanding it here would lose the
    fact that the document made one statement rather than sixteen.
    """
    out = Reading()
    blocks: list[tuple[list[str], list[str]]] = [(carried, [])] if carried else []
    for line in text.splitlines():
        if not line.strip():
            continue
        names = header_columns(line)
        if names:
            blocks.append((names, []))
        elif blocks:
            blocks[-1][1].append(line)

    for names, lines in blocks:
        grid = grid_of(lines, names, page)
        if grid is None:
            out.not_extracted.append({"page": page, "columns": names, "why": NO_GRID})
            continue
        row: dict | None = None
        for line in lines:
            starts = cell_starts(line)
            if not starts:
                continue

            if starts[0] > grid.head[0] + INDENTED:
                if row is None:
                    out.not_extracted.append(
                        {"page": page, "line": line.strip(), "why": CONTINUES_A_ROW}
                    )
                    continue
                _carry(row, grid, line)
                continue

            address = _cut(line, grid.head[0], grid.head[1])
            if not _ADDRESS.match(address):
                row = None
                continue
            if not grid.agrees(starts):
                out.not_extracted.append({"page": page, "line": line.strip(), "why": OFF_THE_GRID})
                row = None
                continue
            if len(starts) > len(grid.names) > len(OPENING):
                out.not_extracted.append(
                    {"page": page, "line": line.strip(), "why": UNNAMED_COLUMN_INSIDE}
                )
                row = None
                continue
            cells = grid.fixed_cells(line, starts)
            # A page set in two columns puts an unrelated line beside this one, and
            # the extraction can join two printed cells with a single space -- so
            # the grid agrees, the cut is one column wide, and it holds two values.
            # Nothing about the position says so; the size is the one cell whose
            # shape does, and it is the cell the join lands in.
            if "size" in cells and not _SIZE.match(cells["size"]):
                out.not_extracted.append(
                    {"page": page, "line": line.strip(), "why": SIZE_SPANS_COLUMNS}
                )
                row = None
                continue
            row = {**cells, "page": page, "read_by": "parser"}
            _fill(row, grid, grid.rest_cells(line, starts), starts[len(grid.head) - 1])
            out.rows.append(row)

    for row in out.rows:
        if any(_QUALIFIER.search(str(value)) for value in row.values()):
            row["needs_review"] = QUALIFIED
    return out


def _fill(row: dict, grid: Grid, cells: list[tuple[int, str]], edge: int) -> None:
    """Put a row's right-hand cells in the columns the header names for them.

    In order when there are as many as there are columns, which is the ordinary
    case. Otherwise nothing is assigned: the cells are kept verbatim and the row
    says which columns are unsettled, because a value put in the wrong one of
    three columns reads exactly like a value in the right one.

    The one exception is a table with a column its header does not name. There the
    header names a single column after the fixed part and the rows carry two or
    three cells, and which of them is the named one is not in doubt: the first one
    begins in the column the grid pinned down, and everything after it is set
    against its own contents further right. Refusing those left two hundred rows
    of one document stating no parameter at all -- not because the reading was
    unsettled, but because the count did not match.

    @param edge the column the fixed part ends at, which the named cell begins in
    """
    wanted = grid.rest
    if len(cells) == len(wanted):
        for name, (_, value) in zip(wanted, cells, strict=True):
            if value:
                row[name] = value
        return
    if len(wanted) == 1 and len(cells) > 1 and abs(cells[0][0] - edge) <= NEARBY:
        row[wanted[0]] = cells[0][1]
        row["unresolved"] = [value for _, value in cells[1:]]
        row["needs_review"] = UNNAMED_COLUMN.format(found=len(cells) - 1, named=wanted[0])
        return
    if cells:
        row["unresolved"] = [value for _, value in cells]
        row["needs_review"] = UNRESOLVED_TAIL.format(found=len(cells), wanted=len(wanted))


def _carry(row: dict, grid: Grid, line: str) -> None:
    """Fold a line continuing a row back into the columns it continues.

    A row is cut at the columns because every one of its cells is inside one. A
    continuation's are not: the text carried onto the second line is one long
    cell that begins in the description column and runs past where the initial
    value column starts, so cutting it at the columns tears it into fragments and
    files them under headings it has nothing to do with. That produced an initial
    value of `1 P ）` on one row here, which then read as a unit disagreeing with
    its manual -- a difference the archive would have published, and the reason
    this is the one thing the parser refuses on width rather than position.

    So a cell is only folded into a column that holds all of it. Anything wider
    stays unresolved, where a reader can see it.
    """
    edges = [*grid.head, None]
    for found in _CELL.finditer(line):
        start = _column_of(line, found.start())
        end = _column_of(line, found.end())
        value = found.group().strip()
        if not value:
            continue
        name = None
        for position, name_at in enumerate(grid.names[: len(edges) - 1]):
            stop = edges[position + 1]
            if start >= edges[position] - NEARBY and (stop is None or end <= stop):
                name = name_at
                break
        if name is None:
            row.setdefault("unresolved", []).append(value)
            row["needs_review"] = SPANS_COLUMNS
            continue
        row[name] = f"{row[name]} {value}" if name in row else value


#: A cell carrying the two bytes the effect list prints at the end of a type's
#: line. The type's number and name sit in the cell to its left when the printing
#: leaves a gap, and inside this one when it leaves a single space.
_EFFECT_TYPE = re.compile(r"^(?:(.*?)\s+)?\[([0-9A-F]{2})H,\s*([0-9A-F]{2})H\]$")

#: How the list numbers and names a type: `8: Auto Wah`, with whatever the
#: printing expands the name to in brackets after it kept as part of the name.
_NUMBERED = re.compile(r"^(\d+):\s*(.+)$")

#: A cell ending in the parameter number the list prints in brackets. What
#: precedes it is that parameter's printed range.
_PARAMETER = re.compile(r"^(.*?)\s*\[(\d+)\]$")

#: A parameter's name and its range in one cell, which is how the page comes out
#: when a long name leaves a single space before a right-aligned range. Cut at
#: the close of the parenthetical the name expands to, because that is the only
#: boundary in the cell that the printing marks. A cell with no parenthetical, or
#: one whose range carries brackets of its own, is not cut at all.
_NAME_AND_RANGE = re.compile(r"^([+#]?[^()]*\([^()]*\))\s+([^()]+)$")

#: How many cells may cross a candidate gutter before a page is read as one
#: column. A page set in two has a gutter nothing crosses but its running
#: footer; a page of prose has no column that can be drawn without cutting
#: lines in half, and the count says which kind of page this is.
ONE_COLUMN = 3

#: How many lines each side of a candidate gutter has to carry text on before the
#: page is read as being set in two columns. A gap down the middle of two lines is
#: the space between a parameter's name and its right-aligned range; a gap down
#: forty is a gutter. Without this, the widest gap on a page holding one line is a
#: gutter, and the name and range on it are read as two columns' worth of
#: unrelated text.
BOTH_COLUMNS = 4

NAMES_NO_TYPE = (
    "this line states an effect type's two bytes, and no number and name for it were "
    "printed beside or above them -- so which type the bytes select is not settled by "
    "where they sit."
)

NAME_AND_RANGE_JOINED = (
    "the printing left a single space between this parameter's name and its range, so the "
    "two came out as one cell, and nothing in the cell marks where the name ends. Cut by "
    "hand rather than at a guess, which would file half a name as a range."
)

PARAMETER_HAS_NO_NAME = (
    "this cell holds a parameter's range and number, and the line above it in the same "
    "column holds no name for it -- so which parameter the range belongs to is not settled "
    "by where it sits."
)

PARAMETER_UNDER_NO_TYPE = (
    "this parameter is printed under no effect type: none was open above it on this page, "
    "and none was left open by the pages before it. A parameter number means nothing "
    "without the type it numbers a parameter of."
)


@dataclass
class Column:
    """One column of a page, as the cells of each of its lines.

    The effect list is set in two columns and `pdftotext -layout` prints both
    halves of a line as one line, so a type's name and an unrelated parameter's
    range arrive side by side. Read as lines that would put the second under the
    first. Split into columns first, they are two streams that each read in
    order.
    """

    lines: list[list[tuple[int, str]]]
    flush: int
    """The column the labels begin at. A description is printed indented from it,
    and a right-aligned range begins well to the right of it, so a cell sitting at
    it is a label and a cell sitting right of it is a value."""


def gutter(text: str) -> int | None:
    """The column a two-column page divides at, or None if it is set in one.

    Chosen as the column fewest printed cells cross, which is a fact about the
    page rather than a guess about where a gutter usually is: the columns are set
    to different widths on different pages of this document, and the block
    diagrams inside them reach different distances across.

    A page of prose has no such column -- every candidate cuts lines in half --
    and `ONE_COLUMN` is where that stops being a gutter and starts being a
    reading imposed on the page. Neither is a gap that only a line or two reach
    across, per `BOTH_COLUMNS`.
    """
    lines = [cells_in(line) for line in text.splitlines()]
    spans = [(start, start + _width(value)) for cells in lines for start, value in cells]
    if not spans:
        return None
    width = max(end for _, end in spans)
    best, crossed = None, None
    for candidate in range(width // 3, width * 2 // 3):
        count = sum(1 for start, end in spans if start < candidate < end)
        if crossed is None or count < crossed:
            best, crossed = candidate, count
    if best is None or crossed > ONE_COLUMN:
        return None
    for side in (
        [cells for cells in lines if any(start < best for start, _ in cells)],
        [cells for cells in lines if any(start >= best for start, _ in cells)],
    ):
        if len(side) < BOTH_COLUMNS:
            return None
    return best


def in_columns(text: str) -> list[Column]:
    """The page as one or two streams of cells, in the order they are read in.

    The left column entire, then the right, which is how the page is read and so
    the order a type heading reaches the parameters printed under it.
    """
    lines = [cells_in(line) for line in text.splitlines()]
    divide = gutter(text)
    if divide is None:
        sides = [lines]
    else:
        sides = [
            [[cell for cell in cells if cell[0] < divide] for cells in lines],
            [[cell for cell in cells if cell[0] >= divide] for cells in lines],
        ]
    return [Column(lines=side, flush=_flush(side)) for side in sides]


def cells_in(line: str) -> list[tuple[int, str]]:
    """Each of a line's cells with the printed column it begins at."""
    return [
        (_column_of(line, found.start()), found.group().strip()) for found in _CELL.finditer(line)
    ]


def _flush(lines: list[list[tuple[int, str]]]) -> int:
    """The column this one's labels begin at, as the commonest line beginning.

    The commonest rather than the leftmost. A single stray cell -- a figure's
    caption, a footer -- reaching a character or two further left would move the
    edge with it, and then every parameter name sitting at the true edge would
    read as a value right of it and be looked for on the line above.
    """
    starts = [cells[0][0] for cells in lines if cells]
    return max(set(starts), key=starts.count) if starts else 0


def last_type(text: str) -> dict | None:
    """The last effect type this page opens, for the page after it.

    A type's parameters run past the foot of the page it is named on, and on the
    next page they are printed under nothing. Read on its own that page gives
    twenty parameter numbers belonging to no type, which is not a weaker reading
    of them -- a parameter number means nothing without its type.

    Read by the pass that reads the page's rows rather than by a scan of its own.
    A second scan has to find a heading the same way the first one does, and the
    one written here did not: a name printed on the line above its two bytes was
    a heading to the reader and nothing at all to this, so five pages carried the
    type from the page before them and filed forty parameters under an effect
    that had ended two pages earlier. Nothing failed, and both readings were of
    rows that are really printed.
    """
    opened = [row for row in read_effect_list(text, 0).rows if "parameter_number" not in row]
    if not opened:
        return None
    return {key: opened[-1][key] for key in ("type", "effect", "msb", "lsb")}


def holds_parameters(text: str) -> bool:
    """Whether any cell on this page ends in a parameter number.

    Asked of the pages between a type's heading and the page being read, to tell
    a page still under that type from one that ended it.
    """
    return any(
        (found := _PARAMETER.match(value)) and found.group(1)
        for column in in_columns(text)
        for cells in column.lines
        for _, value in cells
    )


def type_above(page_text_of: Callable[[int], str], page: int) -> dict | None:
    """The effect type still open above this page, if any.

    The same walk `header_above` makes over the address map, and for the same
    reason: a type's parameters can fill the page after the one that names it,
    and looking only at the page before finds the type for the second page of a
    long one and nothing for the third.

    @param page_text_of the document's pages, by the file's own numbering
    @param page the page being read, which is not itself looked at
    """
    while page > 1:
        page -= 1
        text = page_text_of(page)
        found = last_type(text)
        if found:
            return found
        if not holds_parameters(text):
            return None
    return None


def read_effect_list(text: str, page: int, carried: dict | None = None) -> Reading:
    """Every effect type and effect parameter the list prints on one page.

    Two kinds of row, and the second is meaningless without the first. A type row
    says the list numbers and names a type and which two bytes select it; a
    parameter row says that type's parameter number `n` is printed with a name
    and a range. So a parameter is only ever read under the type still open above
    it, and one printed under none is refused rather than filed under whichever
    type happens to be nearest.

    The list is not laid out as a table and the parser does not treat it as one.
    There is no grid to learn: a name is at the left of its column and its range
    is right-aligned in the same column, and the two are one row because they are
    on one line. Where the printing has put them on two lines, or run them
    together into one cell, position is what says so -- a range alone begins
    right of the column's edge, a name and range together begin at it.

    The prose printed under each parameter is not read. It describes what the
    parameter does, which is the publisher's writing rather than a table's
    content, and it is indented from the column's edge, which is what keeps it
    out of here.

    `carried` is the type the previous page left open, from `type_above`.
    """
    out = Reading()
    kind = carried
    for column in in_columns(text):
        for position, cells in enumerate(column.lines):
            for index, (at, value) in enumerate(cells):
                named = _EFFECT_TYPE.match(value)
                if named:
                    beside = cells[index - 1][1] if index else _label_above(column, position)
                    numbered = _NUMBERED.match(named.group(1) or beside or "")
                    if not numbered:
                        out.not_extracted.append(
                            {"page": page, "line": value, "why": NAMES_NO_TYPE}
                        )
                        continue
                    kind = {
                        "type": numbered.group(1),
                        "effect": numbered.group(2).strip(),
                        "msb": named.group(2),
                        "lsb": named.group(3),
                    }
                    out.rows.append({**kind, "page": page, "read_by": "parser"})
                    continue

                found = _PARAMETER.match(value)
                if not found or not found.group(1):
                    continue
                if index:
                    name, printed = cells[index - 1][1], found.group(1)
                elif at <= column.flush + NEARBY:
                    together = _NAME_AND_RANGE.match(found.group(1))
                    if not together:
                        out.not_extracted.append(
                            {"page": page, "line": value, "why": NAME_AND_RANGE_JOINED}
                        )
                        continue
                    name, printed = together.group(1), together.group(2)
                else:
                    name, printed = _label_above(column, position), found.group(1)
                    if name is None:
                        out.not_extracted.append(
                            {"page": page, "line": value, "why": PARAMETER_HAS_NO_NAME}
                        )
                        continue
                if kind is None:
                    out.not_extracted.append(
                        {"page": page, "line": value, "why": PARAMETER_UNDER_NO_TYPE}
                    )
                    continue
                out.rows.append(
                    {
                        **kind,
                        "parameter_number": found.group(2),
                        "parameter": name,
                        "data": printed,
                        "page": page,
                        "read_by": "parser",
                    }
                )
    return out


def _label_above(column: Column, position: int) -> str | None:
    """The name on the line above this one, where the printing put it there.

    A name too long to leave room for its range is printed on its own line with
    the range right-aligned under it. Taken only when the line above holds one
    cell, at the column's edge, and that cell is not itself a value -- a line
    holding two cells is two columns' worth of something else, and a line of
    prose ends in a full stop.
    """
    for earlier in range(position - 1, -1, -1):
        cells = column.lines[earlier]
        if not cells:
            continue
        if len(cells) != 1 or cells[0][0] > column.flush + NEARBY:
            return None
        return None if cells[0][1].endswith(".") else cells[0][1]
    return None


#: The heading over a column of the conversion grid, as `6. Rate1` in its index or
#: `*6` beside a parameter in the effect list. The number is what joins the three
#: printings; the name is spelt slightly differently between them.
_QUANTITY = re.compile(r"^(\d+)\.\s*(.+)$")

#: How the index names an effect type under a quantity: `07: Phaser`.
_USES = re.compile(r"^(\d+):\s*(.+)$")

#: The unit a grid column is headed with, printed under the quantity's name.
_UNIT = re.compile(r"^\((.+)\)$")

#: What the grid prints in a cell holding the same setting as the cell above it.
DITTO = "“"

#: How many cells a row of the grid holds besides one per numbered column: the
#: value itself in hexadecimal, and the same value in decimal beside it.
GRID_OPENS_WITH = 2

#: How far apart two cells' left edges may be and still be one column of the
#: index, whose headings are set flush and whose entries are indented from them.
SAME_LIST = 4

#: How many columns a line has to number before it is read as the head of a grid
#: rather than as a list of small numbers that happens to run across a line.
#:
#: Deliberately low, because it is not what keeps a wrong line from being read as
#: a grid. What does that is everything under it: a line numbering columns with no
#: names beneath them refuses its header, and a line with no rows beneath it giving
#: their own value twice yields nothing. Set high enough to say something, this
#: would instead decide which real grids can be read at all.
GRID_COLUMNS_AT_LEAST = 4

GRID_HEADING_INCOMPLETE = (
    "the columns of this grid are numbered across the top and one of those numbers has no "
    "name printed under it that could be read as belonging to it -- so what quantity that "
    "column gives the settings of is not settled by where its heading sits."
)

GRID_ROW_MISCOUNTS = (
    "this line begins with a value in hexadecimal and the same value in decimal, which is "
    "how a row of the grid begins, and does not then hold one cell per column. Read by "
    "hand rather than by placing its cells at the columns they sit nearest, which would "
    "file one quantity's setting under another."
)

INDEX_OUT_OF_ORDER = (
    "the quantities this page indexes did not come out as one run from the first to the "
    "last, so the order the lists were read in is not the order they are printed in. The "
    "whole index is left unread rather than filed with the types under whichever heading "
    "the reading happened to put them."
)


def _grid_anchors(lines: list[list[tuple[int, str]]]) -> list[tuple[int, int]] | None:
    """The line numbering the grid's columns, as each number and where it sits.

    Printed above the names so that a parameter carrying `*6` in the effect list
    can be looked up, and it is the one part of the header whose meaning does not
    depend on reading anything else: a run of integers from one, in order, across
    a line. Everything above the grid is the index and everything below it is the
    grid, so finding it also cuts the page in two.
    """
    for cells in lines:
        values = [value for _, value in cells]
        if len(values) >= GRID_COLUMNS_AT_LEAST and values == [
            str(n) for n in range(1, len(values) + 1)
        ]:
            return [(int(value), at) for at, value in cells]
    return None


def _grid_row(cells: list[tuple[int, str]]) -> tuple[str, str] | None:
    """The value a line of the grid is for, in hexadecimal and in decimal.

    None for a line that is not one. The page prints the value twice and the two
    have to agree, which is a check the printing supplies rather than one imposed
    on it: a line that opens with a byte and that byte's own decimal is a row of
    this grid and nothing else on the page is.
    """
    if len(cells) < 3:
        return None
    figure, decimal = cells[0][1], cells[1][1]
    if len(figure) != 2 or any(digit not in "0123456789ABCDEF" for digit in figure):
        return None
    if not decimal.isdigit() or int(figure, 16) != int(decimal):
        return None
    return figure, decimal


def _grid_headings(
    lines: list[list[tuple[int, str]]], anchors: list[tuple[int, int]]
) -> dict[int, dict[str, str]] | None:
    """What each numbered column of the grid is headed with, by its number.

    The names are set over the numbers and wrap onto two lines where they are too
    long, so a name is assembled from whatever sits nearest its own number and
    nearer to it than to any other. The two columns giving the value itself are
    printed left of the first number by more than the columns are spaced, which is
    what keeps `Value (Hex.)` out of the first quantity's name.

    A trailing parenthesis is the unit the column is in. The last column is headed
    with no unit at all, so a missing one is not a failure to read anything.
    """
    reach = min(right - left for (_, left), (_, right) in zip(anchors, anchors[1:], strict=False))
    words: dict[int, list[str]] = {number: [] for number, _ in anchors}
    started = False
    for cells in lines:
        if [value for _, value in cells] == [str(number) for number, _ in anchors]:
            started = True
            continue
        if not started:
            continue
        if _grid_row(cells):
            break
        for at, value in cells:
            number, column = min(anchors, key=lambda pair: abs(pair[1] - at))
            if abs(column - at) <= reach:
                words[number].append(value)
    if not all(words.values()):
        return None
    out = {}
    for number, found in words.items():
        unit = _UNIT.match(found[-1])
        name = " ".join(found[:-1] if unit else found)
        out[number] = {"quantity": name, "unit": unit.group(1)} if unit else {"quantity": name}
    return out


def _index_lists(lines: list[list[tuple[int, str]]], page: int) -> tuple[list[dict], list[dict]]:
    """The types the page says use each quantity, off the index above the grid.

    The index is set in several lists side by side and a long one runs from the
    foot of one into the head of the next, so it is read list by list and not line
    by line. Each list keeps to one left edge -- its headings flush and its entries
    indented a couple of characters from them -- and `SAME_LIST` is what holds
    those two edges together without joining the list beside them.

    Read in the order the lists are printed in, the headings come out numbered
    from the first to the last with nothing missing. That is the check on the
    reading and not a property of the page: if they do not, the lists were read in
    the wrong order and every type under every heading is in doubt, so none of
    them is filed.

    A type number is filed with its leading zero taken off, which is how the
    effect list prints the same number and so is what lets the two be held
    together. The zero is the typesetting of a number, not a different number.
    """
    edges: dict[int, list[tuple[int, str]]] = {}
    for cells in lines:
        for at, value in cells:
            if not (_QUANTITY.match(value) or _USES.match(value)):
                continue
            near = next((edge for edge in edges if abs(edge - at) <= SAME_LIST), at)
            edges.setdefault(near, []).append((at, value))
    if not edges:
        return [], []

    rows, quantities, standing = [], [], None
    for edge in sorted(edges):
        for _, value in edges[edge]:
            heading = _QUANTITY.match(value)
            if heading:
                standing = {"column": int(heading.group(1)), "printed_as": heading.group(2)}
                quantities.append(standing["column"])
                continue
            if standing is None:
                continue
            used = _USES.match(value)
            rows.append(
                {
                    "column": standing["column"],
                    "indexed_as": standing["printed_as"],
                    "type": str(int(used.group(1))),
                    "effect": used.group(2).strip(),
                    "page": page,
                    "read_by": "parser",
                }
            )
    if quantities != list(range(1, len(quantities) + 1)):
        return [], [{"page": page, "line": str(quantities), "why": INDEX_OUT_OF_ORDER}]
    return rows, []


def read_value_conversion(text: str, page: int, carried: dict | None = None) -> Reading:
    """Every setting the conversion grid prints, and what the index says uses it.

    The grid gives all 128 values of each of a handful of quantities that the
    effect parameters are not stored in directly -- a delay in milliseconds, a
    rate in hertz, a corner frequency -- so a parameter the effect list marks
    `*6` has a printed setting at every byte it can hold rather than a range with
    the inside of it left to be worked out.

    Three kinds of row, and they are held together by the column number printed
    over the grid, not by the quantity's name: the index, the grid's own header
    and the effect list's footnote each spell some of those names differently,
    and the number is the one thing all three print the same.

    - a quantity row says column `n` of the grid gives a named quantity in a unit
    - a setting row says that quantity's value at one byte
    - a use row says the index prints an effect type under that quantity

    A setting the page prints as a ditto is filed with the setting it repeats and
    marked as having been printed that way, so a reader can tell a value that was
    set from one that was carried down. `carried` holds what the page before left
    standing in each column, from `settings_above`: the grid runs over two pages
    and a ditto crosses the break, so a page read on its own opens with cells
    repeating nothing.

    Nothing here places a setting by the column it sits at. A row of the grid
    holds one cell per column and opens with its own value printed twice, so the
    cells are the quantities in order and the page checks the reading itself.
    """
    out = Reading()
    lines = [cells_in(line) for line in text.splitlines()]
    anchors = _grid_anchors(lines)
    if anchors is None:
        return out

    cut = next(index for index, cells in enumerate(lines) if _grid_anchors([cells]))
    uses, refused = _index_lists(lines[:cut], page)
    out.rows.extend(uses)
    out.not_extracted.extend(refused)

    headings = _grid_headings(lines, anchors)
    if headings is None:
        numbered = str([number for number, _ in anchors])
        out.not_extracted.append({"page": page, "line": numbered, "why": GRID_HEADING_INCOMPLETE})
        return out
    for number, heading in headings.items():
        out.rows.append({"column": number, **heading, "page": page, "read_by": "parser"})

    standing = dict(carried or {})
    for cells in lines[cut:]:
        found = _grid_row(cells)
        if not found:
            continue
        if len(cells) != len(anchors) + GRID_OPENS_WITH:
            out.not_extracted.append(
                {"page": page, "line": " ".join(v for _, v in cells), "why": GRID_ROW_MISCOUNTS}
            )
            continue
        figure, decimal = found
        for number, (_, printed) in enumerate(cells[2:], start=1):
            repeated = printed == DITTO
            setting = standing.get(number) if repeated else printed
            if setting is None:
                continue
            standing[number] = setting
            out.rows.append(
                {
                    "column": number,
                    **headings[number],
                    "value": figure,
                    "decimal": decimal,
                    "setting": setting,
                    **({"repeats_above": True} if repeated else {}),
                    "page": page,
                    "read_by": "parser",
                }
            )
    return out


def settings_above(page_text_of: Callable[[int], str], page: int) -> dict | None:
    """What each column of the grid was last printed a setting for, before this page.

    The grid runs over more than one page and repeats its own header on each, so a
    page is readable on its own except for one thing: a column whose setting has
    not changed for a while opens the next page as a ditto repeating a value that
    was printed on the page before. Without this those cells are dropped and the
    column comes back with holes in it exactly where the setting held longest.

    Read by the pass that reads the page's rows, so a ditto is resolved once and
    by one reading.

    The walk stops at the first page that is not part of the grid rather than at
    the front of the document. The grid runs over consecutive pages, so a page
    without it is the page before the grid began -- and walking past it would
    read every page of the document to answer a question about two of them.

    @param page_text_of the document's pages, by the file's own numbering
    @param page the page being read, which is not itself looked at
    """
    while page > 1:
        page -= 1
        reading = read_value_conversion(page_text_of(page), 0)
        rows = [row for row in reading.rows if "setting" in row]
        if not reading.rows:
            return None
        if rows:
            return {
                row["column"]: row["setting"]
                for row in sorted(rows, key=lambda row: (int(row["decimal"]), row["column"]))
            }
    return None


TABLES = ("address-map", "effect-list", "value-conversion")
"""The tables the archive files a document's rows under, by directory name.

Named here rather than only where they are read because the ledger is kept per
table: a page read for one of them is not a page read for the other, and a record
that could not say so would have one table's reading stand for the other's.
"""


def ledger(pages: int) -> dict[str, dict[str, str]]:
    """Every page of the document, unread for every table.

    Written out in full rather than left to be inferred from which pages have
    rows. A reader asking whether page 200 holds anything gets an answer either
    way, and the answer to "nobody has looked" is not the answer to "there is
    nothing there".

    Per table, because one document carries more than one kind of table and they
    are read in different passes. A page of the effect list holds no parameter
    address map, and a ledger with one entry per page would have the pass that
    read the list say so about the map as well -- which is a claim nobody made,
    on the one point this directory exists to keep straight.
    """
    return {str(page): dict.fromkeys(TABLES, UNREAD) for page in range(1, pages + 1)}


def state_of(meta: dict, page: int | str, table: str) -> str:
    """What the ledger says about one page under one table."""
    return (meta["pages"].get(str(page)) or {}).get(table, UNREAD)


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def save(path: str | Path, payload: dict) -> None:
    """A document record, with the envelope every published record carries.

    The envelope's `unit_id` is null and correctly so: this is not a measurement
    of a unit, and the walk that finds one finds no `meta.json` above this
    directory. What the record is about is the document, which the payload names.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record.envelope(payload, out_path=path), ensure_ascii=False, indent=2) + "\n"
    )
