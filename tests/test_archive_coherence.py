"""Where two published records answer the same question, they have to agree.

Every other gate here reads one record and asks whether it is well formed. A
record can be well formed and still be contradicted by the one next to it, and
that is the failure a consumer of the archive actually meets: two files, both
valid, saying different things about the same unit, with nothing in either
saying which to believe.

The rule this holds is the one an alias scan cannot break without something
being wrong somewhere. A scan watches a set of regions and sends a set of
stimuli, and reports where each stimulus's value came to rest. Widening the
regions cannot hide a store that a narrower run found: the narrower run's
regions are a subset, so every address it could reach is still being watched.
A whole-map run that attributes less than a bounded run of the same kind is
therefore a statement about the runs, not about the unit, and one of them is
wrong.

It is compared per stimulus rather than by counting. Two runs rarely send the
same number of stimuli -- the address-kind pair here differs by almost four to
one -- so a count comparison reports a difference in what was asked as though it
were a difference in what was found. Only the stimuli both runs sent can be
compared at all, and over those the wider run must reach at least what the
narrower one did.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

UNITS = Path(__file__).resolve().parents[1] / "data" / "units"

KNOWN_TO_DISAGREE: dict[str, str] = {}
"""Records that break the rule, each with what is known about why.

An entry is not a pass. It says the disagreement was looked at and what was
found, so that the next reader starts from that rather than from the beginning,
and it comes out in the same change as the record that settles it.
"""


def _alias_records() -> list[tuple[str, dict]]:
    found = []
    for path in sorted(UNITS.rglob("*.json")):
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or data.get("regions_watched") is None:
            continue
        if not (data.get("attributed") or data.get("controls")):
            continue
        found.append((str(path.relative_to(UNITS)), data))
    return found


def _kind(data: dict) -> str:
    """A scan's kind, which the earliest runs left to their findings to say.

    Before `kind` was written at the top of a record it appeared on each
    attribution, so a run that attributed nothing had no kind at all. Reading
    it from the first finding recovers it for the runs that do have one rather
    than dropping them from the comparison, which would leave the rule with
    nothing to hold on exactly the records that predate it.
    """
    if data.get("kind"):
        return str(data["kind"])
    for row in data.get("attributed") or data.get("controls") or []:
        if isinstance(row, dict) and row.get("kind"):
            return str(row["kind"])
        if isinstance(row, dict) and str(row.get("control", "")).startswith("CC"):
            return "cc"
    return "?"


def _stimulus_name(raw: str) -> str:
    """One stimulus's name, spelled the way both records will agree on.

    `CC7 ch1` and `CC7` are one stimulus written two ways; the channel is
    already what the records are grouped by, so the suffix is dropped rather
    than allowed to split a stimulus in two and compare each half with nothing.

    Only that suffix. An address stimulus is named for the address it writes
    (`DT1 40 11 19`), so taking the first word instead would fold every one of
    them into a single name and report the difference between two address lists
    as a scan losing what it found.
    """
    name = raw.upper()
    head, _, tail = name.rpartition(" ")
    if head and tail.startswith("CH") and tail[2:].isdigit():
        return head
    return name


def _reached(data: dict) -> dict[str, set[str]]:
    """Where each stimulus's value came to rest, keyed by the stimulus."""
    out: dict[str, set[str]] = {}
    for row in data.get("attributed") or data.get("controls") or []:
        if not isinstance(row, dict):
            continue
        name = _stimulus_name(str(row.get("stimulus") or row.get("control") or ""))
        if name:
            out.setdefault(name, set()).update(row.get("stores_verbatim") or [])
    return out


def _sent(data: dict) -> set[str] | None:
    """The stimuli a run sent, or None where it did not record them.

    Needed separately from what it attributed, because those two sets differ by
    exactly the thing being measured. A stimulus absent from the findings was
    either sent and landed nothing or never sent at all, and a comparison that
    cannot tell those apart reports one run asking a different question as the
    other run losing an answer.

    None rather than an empty set: a run that recorded no stimulus list cannot
    be compared, and saying so is not the same as saying it sent nothing.
    """
    listed = data.get("stimuli_sent")
    if isinstance(listed, list):
        return {_stimulus_name(name) for name in listed if isinstance(name, str)}
    # A controller scan says what it sent as the range it swept rather than as a
    # list of names. Read as a range it is as good a record of the question as a
    # list is, and refusing to read it would put the one run worth comparing
    # beyond comparison for the sake of a spelling.
    swept = data.get("controllers_scanned")
    if isinstance(swept, str) and swept.count("-") == 1:
        low, _, high = swept.partition("-")
        if low.strip().isdigit() and high.strip().isdigit():
            return {f"CC{n}" for n in range(int(low), int(high) + 1)}
    return None


def _watched(data: dict) -> set[str] | None:
    """The addresses a scan watched, or None where it recorded only how many.

    A count was enough while every scan walked one map, and it stopped being
    enough the moment there were three: the widest count belongs to the map that
    asked two third bytes under each block, which misses thousands of addresses a
    later map holds, and the map built from single-address answers misses the
    ones that come back only inside a longer read. More regions is not more
    space, so containment has to be read from the addresses or not claimed.
    """
    listed = data.get("regions_watched_are")
    if not isinstance(listed, list):
        return None
    out = set()
    for region in listed:
        start = [int(b, 16) for b in region["address"].split()]
        for step in range(region["size"]):
            out.add("{:02X} {:02X} {:02X}".format(*start[:2], start[2] + step))
    return out


def _pairs() -> list[tuple[str, str]]:
    """Every narrower/wider pair of scans of one kind on one channel.

    Wider means it watched everything the other did and more, which is the only
    reading under which a lost address is about the runs rather than the unit.
    Where either record states only a count, the pair is left out: two scans of
    one kind whose watched space nobody can compare have nothing to say about
    each other, and pairing them by size would report the archive's own gap in
    what it recorded as the unit disagreeing with itself.
    """
    by_question: dict[tuple[str, object], list[tuple[str, dict]]] = {}
    for name, data in _alias_records():
        by_question.setdefault((_kind(data), data.get("channel")), []).append((name, data))
    pairs = []
    for scans in by_question.values():
        for narrow_name, narrow in scans:
            for wide_name, wide in scans:
                narrow_space, wide_space = _watched(narrow), _watched(wide)
                if narrow_space is None or wide_space is None:
                    continue
                if wide_space > narrow_space:
                    pairs.append((narrow_name, wide_name))
    return pairs


PAIRS = _pairs()


def test_the_archive_has_a_pair_to_compare() -> None:
    """A rule with nothing to hold passes for the wrong reason.

    The pairs are found by walking the archive, so a change to how a scan
    records itself can stop them being recognised without any test going red.
    """
    assert PAIRS, "no narrower/wider pair of alias scans was found to compare"


@pytest.mark.parametrize("narrow_name,wide_name", PAIRS, ids=lambda n: n)
def test_a_wider_scan_reaches_what_a_narrower_one_did(narrow_name: str, wide_name: str) -> None:
    narrow_data = json.loads((UNITS / narrow_name).read_text())
    wide_data = json.loads((UNITS / wide_name).read_text())
    narrow, wide = _reached(narrow_data), _reached(wide_data)
    narrow_sent, wide_sent = _sent(narrow_data), _sent(wide_data)

    if narrow_sent is None or wide_sent is None:
        unrecorded = narrow_name if narrow_sent is None else wide_name
        assert wide_name in KNOWN_TO_DISAGREE, (
            f"{unrecorded} does not record which stimuli it sent, so what it did not "
            f"attribute cannot be told from what it never asked, and it cannot be compared "
            f"with {narrow_name if unrecorded == wide_name else wide_name}."
        )
        return

    both_sent = narrow_sent & wide_sent
    lost = {
        stimulus: sorted(narrow[stimulus] - wide.get(stimulus, set()))
        for stimulus in both_sent & set(narrow)
    }
    lost = {stimulus: where for stimulus, where in lost.items() if where}
    if wide_name in KNOWN_TO_DISAGREE:
        assert lost, (
            f"{wide_name} is listed as disagreeing with a narrower scan and no longer does. "
            f"Delete its entry from KNOWN_TO_DISAGREE in the same change as whatever settled it."
        )
        return
    assert not lost, (
        f"{wide_name} watched more regions than {narrow_name} and reached less, over "
        f"{len(both_sent)} stimuli both of them sent.\n"
        f"  addresses it lost: {lost}\n"
        "A wider run's regions are a superset, so this is about the runs and not the unit."
    )
