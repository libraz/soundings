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

#: An address as the tables print it. The first byte is always a literal -- every
#: region of the map is entered at a fixed one -- and the two after it may carry
#: the document's own placeholder letters instead of digits: `x` for a part, `b`
#: for a bank, `pp` for a program, and others per table. They are kept as printed
#: rather than expanded, so that one row stays one statement.
_ADDRESS = re.compile(r"^[0-9A-F]{2} [0-9A-Fa-z]{2} [0-9A-Fa-z]{2}#?$")
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

    def rest_cells(self, line: str, starts: list[int]) -> list[str]:
        edge = starts[len(self.head) - 1]
        return [
            found.group().strip()
            for found in _CELL.finditer(line)
            if _column_of(line, found.start()) >= edge
        ]


def header_columns(line: str) -> list[str] | None:
    """The columns this line names, in order, or None if it names none.

    A header is recognised by naming an address column and a data column. The
    rest are optional and several tables do without them: a drum setup table
    states no initial value, so a parser requiring one would read its rows as
    malformed rather than as rows of a table that states less.
    """
    names: list[str] = []
    cursor = 0
    for name, labels in HEADINGS:
        for label in labels:
            index = line.find(label, cursor)
            if index >= 0:
                names.append(name)
                cursor = index + len(label)
                break
    return names if {"address", "data"} <= set(names) else None


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
            row = {**grid.fixed_cells(line, starts), "page": page, "read_by": "parser"}
            _fill(row, grid, grid.rest_cells(line, starts))
            out.rows.append(row)

    for row in out.rows:
        if any(_QUALIFIER.search(str(value)) for value in row.values()):
            row["needs_review"] = QUALIFIED
    return out


def _fill(row: dict, grid: Grid, cells: list[str]) -> None:
    """Put a row's right-hand cells in the columns the header names for them.

    In order when there are as many as there are columns, which is the ordinary
    case. Otherwise nothing is assigned: the cells are kept verbatim and the row
    says which columns are unsettled, because a value put in the wrong one of
    three columns reads exactly like a value in the right one.
    """
    wanted = grid.rest
    if len(cells) == len(wanted):
        for name, value in zip(wanted, cells, strict=True):
            if value:
                row[name] = value
        return
    if cells:
        row["unresolved"] = cells
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


def ledger(pages: int) -> dict[str, str]:
    """Every page of the document, all of them unread.

    Written out in full rather than left to be inferred from which pages have
    rows. A reader asking whether page 200 holds anything gets an answer either
    way, and the answer to "nobody has looked" is not the answer to "there is
    nothing there".
    """
    return {str(page): UNREAD for page in range(1, pages + 1)}


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
