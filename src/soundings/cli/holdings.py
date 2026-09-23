"""The takes ledger: what `.cache/takes` holds, and who has read it.

`build` writes `.cache/takes-ledger.json` from the takes on disk and the unit's
published archive; it never opens the machine and never reads a take's sample
data, only its manifest or its WAV header. The other three actions ask of that
one file, so a query is as cheap as the archive is large rather than as the
disk is: missing the file, they say to build it and exit 2 rather than walk
`.cache/takes` again on its behalf.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import ledger


def register(sub) -> None:
    p = sub.add_parser(
        "takes",
        help="the ledger of what .cache/takes holds, which published records read "
        "each directory, and whether it has changed since -- with no machine attached",
    )
    p.set_defaults(needs_unit=False, func=_dispatch)
    inner = p.add_subparsers(dest="action", required=True)

    built = inner.add_parser(
        "build",
        help="write .cache/takes-ledger.json from the takes on disk and the unit's "
        "published records",
    )
    built.add_argument("--unit", default="roland-sc8850-01", help="a directory under data/units")
    built.add_argument("--root", default=".", type=Path, help="repository root, for testing")

    row = inner.add_parser("row", help="one type and address's entries from the ledger")
    row.add_argument("--type", required=True, help="the effect type, as two hex bytes, 'MM LL'")
    row.add_argument("--slot", required=True, help="the address, as three hex bytes")
    row.add_argument("--root", default=".", type=Path, help="repository root, for testing")

    unread = inner.add_parser(
        "unread", help="directories no published record's consumed_by reaches, largest first"
    )
    unread.add_argument("--root", default=".", type=Path, help="repository root, for testing")
    unread.add_argument("--limit", type=int, default=40, help="how many directories to print")

    rewritten = inner.add_parser(
        "rewritten",
        help="directories whose newest take is younger than a record that read them",
    )
    rewritten.add_argument("--root", default=".", type=Path, help="repository root, for testing")


def _dispatch(args: argparse.Namespace) -> int:
    return {
        "build": _build,
        "row": _row,
        "unread": _unread,
        "rewritten": _rewritten,
    }[args.action](args)


def _build(args: argparse.Namespace) -> int:
    found = ledger.build(args.unit, root=args.root)
    out = Path(args.root) / ledger.LEDGER_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(found, indent=2) + "\n")
    unbound = found["unbound_records"]
    bad = sorted({u["reason"] for u in unbound} - set(ledger.REASON_CODES))
    print(
        f"{len(found['directories'])} directories, {len(unbound)} published records cite "
        ".cache and never reach one"
    )
    print(f"wrote {out}")
    if bad:
        print(f"reason code(s) outside the vocabulary: {', '.join(bad)}")
        return 1
    return 0


def _need_ledger(root: Path) -> dict | None:
    found = ledger.load(root)
    if found is None:
        print(f"no {ledger.LEDGER_PATH} here -- run `soundings takes build` first")
    return found


def _row(args: argparse.Namespace) -> int:
    found = _need_ledger(args.root)
    if found is None:
        return 2
    matches = ledger.row(found, args.type, args.slot)
    if not matches:
        print(f"no directory in the ledger is {args.type} at {args.slot}")
        return 0
    print(json.dumps(matches, indent=2))
    return 0


def _unread(args: argparse.Namespace) -> int:
    found = _need_ledger(args.root)
    if found is None:
        return 2
    rows = ledger.unread(found)
    for relpath, entry in rows[: args.limit]:
        print(f"{entry.get('bytes', 0) / 1e9:8.3f} GB  {relpath}")
    if not rows:
        print("every directory is read by something")
    return 0


def _rewritten(args: argparse.Namespace) -> int:
    found = _need_ledger(args.root)
    if found is None:
        return 2
    rows = ledger.rewritten(found)
    for relpath, entry in rows:
        print(f"{relpath}: read by {', '.join(entry['rewritten_after'])}")
    if not rows:
        print("nothing on disk is younger than a record that read it")
    return 0
