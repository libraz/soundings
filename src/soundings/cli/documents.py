"""Reading a published document into the archive, one page at a time.

Four commands around one ledger. `init` writes down which document is being read
and marks every one of its pages unread; `show` prints a page beside what the
parser makes of it, which is the step where a human decides whether the parser is
right; `add` puts a page's rows in; `status` says how much of the document has
been through that.

None of them touches the unit, and none of them is a measurement. They are here
rather than in a separate tool because what they produce is only useful held
against the records this repository already holds, and a reader who has to find
a second program to check a claim will not check it.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from .. import documents

TABLES = {
    "address-map": {
        "read": documents.read_address_map,
        "carry": documents.header_above,
        "order": lambda row: (row["page"], row.get("address", "")),
        "line": lambda row: (
            f"{row.get('address', ''):<12} {row.get('size', ''):<10} "
            f"{row.get('data', ''):<14} {row.get('parameter', '')}"
        ),
        "headings": {
            "en": "Parameter Address Map",
            "ja": "パラメーター・アドレス・マップ",
        },
    },
    "effect-list": {
        "read": documents.read_effect_list,
        "carry": documents.type_above,
        "order": lambda row: (
            row["page"],
            row["msb"],
            row["lsb"],
            int(row.get("parameter_number", -1)),
        ),
        "line": lambda row: (
            f"{row['type']:>3}: {row['effect']:<24} "
            f"[{row['msb']}H, {row['lsb']}H] "
            + (
                f"[{row['parameter_number']:>2}] {row['parameter']:<34} {row['data']}"
                if "parameter_number" in row
                else ""
            )
        ),
        "headings": {"en": "Different effect types"},
    },
    "value-conversion": {
        "read": documents.read_value_conversion,
        "carry": documents.settings_above,
        "order": lambda row: (
            row["column"],
            0 if "setting" in row else 1 if "type" in row else 2,
            int(row.get("decimal", -1)),
            int(row.get("type", -1)),
        ),
        "line": lambda row: (
            f"{row['column']:>3}. {row['quantity'] if 'quantity' in row else row['indexed_as']:<16}"
            f"{'(' + row['unit'] + ')' if row.get('unit') else '':<7}"
            + (
                f" {row['value']} ({row['decimal']:>3}) = {row['setting']}"
                + (" ditto" if row.get("repeats_above") else "")
                if "setting" in row
                else f" used by {row['type']:>2}: {row['effect']}"
                if "type" in row
                else ""
            )
        ),
        "headings": {"en": "Effect Parameter Value Conversion Table"},
    },
}
"""The tables this can read, by the name the archive files them under.

Each is a table family whose rows share a shape. Beside it is the heading each
edition prints over that table, so a citation can name the table a reader is
meant to find in the words their own copy prints -- picked by the document's
language rather than fixed here, because one reader serves every edition and a
heading written into the code is the heading of whichever edition was read
first. A language whose copy nobody has held is not guessed at: it is missing
here, and asking for it says so.
"""


def register(sub) -> None:
    p = sub.add_parser(
        "document",
        help="read a published document's tables, so what it states can be held "
        "against what a unit was measured to do",
    )
    p.set_defaults(needs_unit=False, func=_dispatch)
    inner = p.add_subparsers(dest="action", required=True)

    started = inner.add_parser("init", help="write down a document and mark its pages unread")
    started.add_argument("pdf", help="the document file, which is not copied into the archive")
    started.add_argument("--id", required=True, dest="document_id")
    started.add_argument("--title", required=True, help="as the document prints it")
    started.add_argument("--publisher", required=True)
    started.add_argument("--copyright", required=True, help="the year the document asserts")
    started.add_argument(
        "--language",
        required=True,
        choices=sorted({tag for table in TABLES.values() for tag in table["headings"]}),
        help="the edition's language. One document is one edition, and an edition is in "
        "one language: it decides which heading a citation names, and the description "
        "columns are written in it",
    )
    started.add_argument("--printing", help="the printing or revision, if it names one")
    started.add_argument(
        "--page-offset",
        type=int,
        default=0,
        help="how far the file's pages run ahead of the printed ones: the page printed "
        "191 is the file's 193rd when this is 2. Everything is asked for and cited by "
        "the printed number, because that is the one a reader holding the document has",
    )
    started.add_argument("--root", default="documents", type=Path)

    shown = inner.add_parser("show", help="a page, beside what the parser makes of it")
    shown.add_argument("document_id")
    shown.add_argument("--page", required=True, type=int, help="the printed page number")
    shown.add_argument("--table", choices=sorted(TABLES), default="address-map")
    shown.add_argument("--pdf", help="override the file the document record names")
    shown.add_argument("--root", default="documents", type=Path)

    added = inner.add_parser("add", help="put a page's rows in, or declare it holds none")
    added.add_argument("document_id")
    added.add_argument("--page", required=True, type=int)
    added.add_argument("--table", choices=sorted(TABLES), default="address-map")
    added.add_argument(
        "--no-tables",
        action="store_true",
        help="record that a human looked and found nothing this schema holds. Distinct "
        "from a page nobody has read, and from a page the parser found nothing on",
    )
    added.add_argument("--pdf", help="override the file the document record names")
    added.add_argument("--root", default="documents", type=Path)

    state = inner.add_parser("status", help="which pages have been read")
    state.add_argument("document_id")
    state.add_argument("--root", default="documents", type=Path)


def _dispatch(args) -> int:
    return {"init": _init, "show": _show, "add": _add, "status": _status}[args.action](args)


def _record_path(args) -> Path:
    return args.root / args.document_id / "document.json"


def _pdf_for(args, meta: dict) -> str:
    """The file to read, and a refusal if it is not the one the record names.

    The extraction cites printed page numbers, and a page number is a claim
    about one printing. Reading a different file under the same document id
    would file its pages under a document it is not, so the fingerprint is
    checked rather than trusted -- a reprint has the same title and different
    tables.
    """
    named = args.pdf or str(Path(meta["source_file"]["path_when_read"]).expanduser())
    if not Path(named).is_file():
        raise SystemExit(
            f"{named} is not here. The document is not committed, so this needs the file "
            f"itself; pass --pdf if it has moved."
        )
    if documents.sha256(named) != meta["source_file"]["sha256"]:
        raise SystemExit(
            f"{named} is not the file {args.document_id} was initialised from. Its pages "
            "may be numbered differently, so rows read from it would cite pages of a "
            "document this record is not about."
        )
    return named


def _init(args) -> int:
    path = _record_path(args)
    if path.exists():
        raise SystemExit(f"{path} already exists")
    pages = documents.page_count(args.pdf)
    documents.save(
        path,
        {
            "document_id": args.document_id,
            "title": args.title,
            "publisher": args.publisher,
            "copyright": args.copyright,
            "language": args.language,
            "printing": args.printing,
            "rights": "Copyright in this document is the publisher's. It is not "
            "distributed here and no licence over it is granted or claimed. What is "
            "held in this directory is the factual content of its tables, cited by "
            "printed page.",
            "source_file": {
                "name": Path(args.pdf).name,
                "path_when_read": documents.where_it_was(args.pdf),
                "sha256": documents.sha256(args.pdf),
                "pages": pages,
                "page_offset": args.page_offset,
            },
            "first_read": dt.date.today().isoformat(),
            "pages": documents.ledger(pages),
        },
    )
    print(f"{path}: {pages} pages, all unread")
    return 0


def _in_file(meta: dict, printed: int) -> int:
    """The file's page holding the page printed with this number.

    Every command here takes and records the printed number, because that is
    what a citation is for: a reader checking a claim has the document, not this
    file. The two happened to coincide in the first document read, which is
    exactly how an offset goes unnoticed until it puts every citation of the
    second one two pages out.
    """
    offset = meta["source_file"].get("page_offset", 0)
    inside = printed + offset
    if not 1 <= inside <= meta["source_file"]["pages"]:
        raise SystemExit(
            f"page {printed} is not in {meta['document_id']}: its printed pages run to "
            f"{meta['source_file']['pages'] - offset}."
        )
    return inside


def _carried(table: str, pdf: str, page: int) -> list[str] | dict | None:
    """Whatever the table above this page left open on it.

    The columns of a parameter address map, the effect type an effect list is
    still printing the parameters of: different things, and each is the
    difference between reading the rows on a table's later pages and losing
    them. Costs a read of each page walked back over.
    """
    return TABLES[table]["carry"](lambda at: documents.page_text(pdf, at), page)


def _heading(table: str, language: str) -> str:
    headings = TABLES[table]["headings"]
    if language not in headings:
        raise SystemExit(
            f"the heading a {language} edition prints over the {table} is not written "
            "down here. It is what a citation names, so it is read off a copy of that "
            "edition rather than translated out of another one."
        )
    return headings[language]


def _read(args, meta: dict) -> documents.Reading:
    pdf = _pdf_for(args, meta)
    inside = _in_file(meta, args.page)
    text = documents.page_text(pdf, inside)
    return TABLES[args.table]["read"](text, args.page, _carried(args.table, pdf, inside))


def _show(args) -> int:
    path = _record_path(args)
    meta = documents.load(path)
    pdf = _pdf_for(args, meta)
    inside = _in_file(meta, args.page)
    text = documents.page_text(pdf, inside)
    print(text)
    print(f"--- {args.table}, printed page {args.page} (file page {inside}) ---")
    reading = TABLES[args.table]["read"](text, args.page, _carried(args.table, pdf, inside))
    for row in reading.rows:
        marked = " (qualified)" if "needs_review" in row else ""
        print(f"  {TABLES[args.table]['line'](row)}{marked}")
    for missed in reading.not_extracted:
        print(f"  ? {missed}")
    print(f"  {len(reading.rows)} rows, {len(reading.not_extracted)} unread")
    return 0


def _forget(path: Path, args) -> int:
    """Drop what the parser filed against a page a reader says holds no table.

    A page of prose set in two columns puts an unrelated line beside another and
    reads as a table header, and the parser then files a note saying the table
    under it could not be read. Left in place beside a `no tables` verdict the
    two contradict each other, and the note is the one that gets counted: it
    names a table nobody claims is there.
    """
    table_path = path.parent / f"{args.table}.json"
    if not table_path.exists():
        return 0
    table = documents.load(table_path)
    kept = {
        "rows": [row for row in table["rows"] if row["page"] != args.page],
        "not_extracted": [
            missed for missed in table["not_extracted"] if missed["page"] != args.page
        ],
    }
    dropped = sum(len(table[name]) - len(kept[name]) for name in kept)
    if dropped:
        documents.save(table_path, {k: kept.get(k, v) for k, v in table.items() if k != "record"})
    return dropped


def _add(args) -> int:
    path = _record_path(args)
    meta = documents.load(path)
    page = str(_in_file(meta, args.page))

    if args.no_tables:
        meta["pages"][page][args.table] = documents.NO_TABLES
        documents.save(path, {k: v for k, v in meta.items() if k != "record"})
        dropped = _forget(path, args)
        also = f", {dropped} parser notes dropped" if dropped else ""
        print(f"page {args.page}: no tables{also}")
        return 0

    reading = _read(args, meta)
    if not reading:
        raise SystemExit(
            f"the parser found nothing on page {page}. That is not the same as the page "
            "holding nothing: look at it with `document show`, and if it really holds no "
            "table this can hold, say so with --no-tables."
        )

    heading = _heading(args.table, meta["language"])
    table_path = path.parent / f"{args.table}.json"
    table = (
        documents.load(table_path)
        if table_path.exists()
        else {
            "document_id": args.document_id,
            "table": args.table,
            "printed_heading": heading,
            "rows": [],
            "not_extracted": [],
        }
    )
    table["rows"] = [row for row in table["rows"] if row["page"] != args.page] + reading.rows
    table["rows"].sort(key=TABLES[args.table]["order"])
    table["not_extracted"] = [
        missed for missed in table["not_extracted"] if missed["page"] != args.page
    ] + reading.not_extracted
    documents.save(table_path, {k: v for k, v in table.items() if k != "record"})

    meta["pages"][page][args.table] = documents.READ if reading.rows else documents.NOTHING_READABLE
    documents.save(path, {k: v for k, v in meta.items() if k != "record"})
    print(f"page {args.page}: {len(reading.rows)} rows, {len(reading.not_extracted)} unread")
    return 0


def _status(args) -> int:
    """How far each table has been read, separately.

    Separately because that is how they are read. One pass over a document takes
    one kind of table out of it, and a total over both would have the pages of
    the one that is finished count towards the one nobody has started.
    """
    meta = documents.load(_record_path(args))
    # Reported in printed numbers, which is what everything else here takes and
    # what a reader would go and look at. The ledger is keyed by the file's own
    # pages because every page of the file has one and the front matter has no
    # printed number at all.
    offset = meta["source_file"].get("page_offset", 0)
    print(f"{meta['document_id']}: {meta['source_file']['pages']} pages")
    for table in documents.TABLES:
        counts: dict[str, int] = {}
        for page in meta["pages"]:
            state = documents.state_of(meta, page, table)
            counts[state] = counts.get(state, 0) + 1
        read = sorted(
            int(page) - offset
            for page in meta["pages"]
            if documents.state_of(meta, page, table) == documents.READ
        )
        print(f"  {table}")
        for state in (
            documents.READ,
            documents.NOTHING_READABLE,
            documents.NO_TABLES,
            documents.UNREAD,
        ):
            if counts.get(state):
                print(f"    {state:<18} {counts[state]}")
        if read:
            print(f"    pages read: {_ranges(read)}")
    return 0


def _ranges(pages: list[int]) -> str:
    out: list[str] = []
    for page in pages:
        if out and page == int(out[-1].split("-")[-1]) + 1:
            first = out[-1].split("-")[0]
            out[-1] = f"{first}-{page}"
        else:
            out.append(str(page))
    return ", ".join(out)
