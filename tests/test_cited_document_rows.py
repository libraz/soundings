"""A claim citing a printed row has to cite one the document directory holds.

`soundings inferences stale` checks the other half of `rests_on`: every measurement
a claim leans on carries the value it was made on, and a record republished with
that figure moved makes the claim stale. Nothing checked the printed rows, and the
two are the same kind of citation -- one names a record, one names a page, and both
are there so a reader can go and look.

The failure this catches does not look like a failure. A row typed from what a page
was expected to say reads exactly like a row read off one, and the two manuals in
this directory print overlapping lists under different numbering, so a citation
aimed at the wrong one of them is a sentence about a page nobody read. What makes
it worth a test rather than care is that the mistake is invisible at the point it
is made and expensive later: a claim is retracted rather than corrected.

The check is deliberately loose about shape and strict about existence. Citations
are written for a reader, not for a parser, so this does not impose a format on
them -- it asks that the row's own distinguishing parts, the thing named and the
values printed against it, turn up together in one row of the file the citation
names.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLAIMS = ROOT / "inferences"


def cited() -> list[tuple[Path, str, str]]:
    """Every (claim, file, row) a claim's `rests_on.document_rows` names."""
    out = []
    for path in sorted(CLAIMS.glob("*/*.json")):
        if path.name == "index.json":
            continue
        claim = json.loads(path.read_text())
        for block in claim.get("rests_on", {}).get("document_rows", []):
            for row in block["rows"]:
                out.append((path, block["file"], row))
    return out


def rows_of(path: Path) -> list[dict]:
    document = json.loads(path.read_text())
    return [row for row in document.get("rows", []) if isinstance(row, dict)]


def names_one_row(row: str) -> bool:
    """Whether this citation points at one printed row rather than at a set of them.

    A column of a conversion table and a summary over every row sharing a value are
    citations of a different kind, and asking those to look like a row would be
    asking the writer to write for the test.
    """
    head = row.strip().casefold()
    if head.startswith(("column ", "the ", "every ", "all ")):
        return False
    return "parameter" in head


def normalised(text: str) -> str:
    """Enough of a printed cell to compare two spellings of it.

    A manual sets its ranges with an en dash and a citation is typed with a hyphen,
    with or without spaces, and `8k` and `8.0k` are the same frequency. None of
    those is a different claim about the page, and all of them would fail a
    comparison of the strings.
    """
    text = text.casefold()
    for dash in ("\u2013", "\u2014", "\u2212"):
        text = text.replace(dash, "-")
    text = text.replace(".0k", "k").replace(" ", "")
    return text.replace("-", "").replace("/", ",")


def parameter_of(row: str) -> str | None:
    """Which parameter of its table the citation names, however it writes it."""
    match = re.search(r"parameter[= ](\d+)", row, re.IGNORECASE)
    return match.group(1) if match else None


def table_of(row: str) -> str:
    """What the citation names the table by: a type number, an address, or a name."""
    head = row.split(",")[0].strip()
    return re.split(r"\s*parameter", head, flags=re.IGNORECASE)[0].strip().rstrip(",=")


def is_parameter(entry: dict, number: str | None) -> bool:
    """Whether a row is parameter `number` of its table, in either notation.

    One manual numbers a type's parameters from one and the other files them by the
    address byte they are written to, and parameter n lives at `40 03 (0x03 + n-1)`.
    The two directories are read by different passes and neither is the canonical
    one, so a citation is checked against whichever notation the file it names uses.
    """
    if number is None:
        return False
    if "parameter_number" in entry:
        return str(entry["parameter_number"]) == str(number)
    if "address_lsb" in entry:
        return entry["address_lsb"].upper() == f"{0x03 + int(number) - 1:02X}"
    return False


def names_the_same_table(named: str, entry: dict) -> bool:
    """Whether a row belongs to the table the citation named.

    Three notations are in use across the claims and all three are a reader's
    shorthand for the same thing -- the effect's printed name, its number in the
    list, and the two bytes its type is written with. A test that accepted only one
    of them would be choosing which claims are allowed to cite a page.
    """
    if not named:
        return False
    if named.startswith("type="):
        return entry.get("type") == named.removeprefix("type=")
    if re.fullmatch(r"[0-9A-Fa-f]{2} [0-9A-Fa-f]{2}", named):
        msb, lsb = named.split()
        return (entry.get("msb"), entry.get("lsb")) == (msb.upper(), lsb.upper())
    return named.casefold() in str(entry.get("effect", "")).casefold()


CITED = cited()


def test_there_are_citations_to_check() -> None:
    """A test that silently checks nothing is worse than no test.

    The claims do rest on printed rows, so an empty list here means the citations
    moved rather than that they all pass.
    """
    assert len(CITED) > 50


@pytest.mark.parametrize(
    ("claim", "file", "row"),
    CITED,
    ids=[f"{c.stem}:{r[:40]}" for c, _, r in CITED],
)
def test_a_cited_row_is_in_the_file_it_names(claim: Path, file: str, row: str) -> None:
    where = ROOT / file
    assert where.is_file(), (
        f"{claim.name} cites {file}, which is not in the documents directory. "
        "A claim's printed evidence is a file a reader can open."
    )

    if not names_one_row(row):
        pytest.skip(
            "not a citation of a single printed row. A claim may rest on a column of a "
            "conversion table or on every row that shares a value, and those are cited as "
            "what they are rather than enumerated -- the file check above is all this can "
            "ask of them without turning prose written for a reader into a format."
        )

    named, number = table_of(row), parameter_of(row)
    here = [
        entry for entry in rows_of(where)
        if names_the_same_table(named, entry) and is_parameter(entry, number)
    ]
    assert here, (
        f"{claim.name} cites a row {where.name} does not have:\n"
        f"    {row}\n"
        f"  No row of that file is parameter {number} of {named!r}. Either the row is in "
        "the other manual -- the two print overlapping lists under different numbering -- "
        "or it is a row somebody expected rather than read, in which case it is an "
        "absence and belongs in the queue and not in `rests_on`."
    )

    # What the citation says the data column reads. The fields between the
    # parameter and the data are the parameter's own name and gloss, which a reader
    # may shorten -- so the data is looked for as a run of trailing fields rather
    # than as a fixed one: a printed cell like `Off/LPF/HPF` is often cited with its
    # states spelled out, and that is several commas.
    fields = [part.strip() for part in row.split(",")]
    if len(fields) < 3:
        return
    printed = [normalised(str(entry.get("data", ""))) for entry in here]
    for take in range(1, len(fields) - 1):
        quoted = normalised(",".join(fields[-take:]))
        # One way round only. A citation may stop short of the whole cell, but it
        # may not carry a value the cell does not -- which is how a byte acquires a
        # state the unit was never asked for, and a range acquires an end no table
        # was cut from.
        if quoted and any(quoted in cell for cell in printed):
            return

    pytest.fail(
        f"{claim.name} quotes a data column the page does not print:\n"
        f"    cited: {row}\n"
        f"    {where.name} prints: {here[0].get('data')}\n"
        "  The values are the evidence, so a claim reasoning about how many states a "
        "byte has or where a printed range ends is reasoning about this cell. The name "
        "and its gloss may be shortened; this may not."
    )
