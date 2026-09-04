"""Find what each reset restores, by breaking the state first and seeing what comes back.

Reading the machine after a reset says what it holds; it does not say the reset
put it there. Every byte that was already at its default reads the same whether
the reset touched it or ignored it, and a reset that does nothing at all scores
identically to one that restores everything. So the state is deliberately broken
first, and what the reset is credited with is only what it changed back.

**The mark has to be a value the address accepts.** A byte written outside its
range keeps what it had, so it was never broken, and a reset gets credit for it
either way. Only bytes read back as holding the mark are counted, and the count
is reported next to how many were attempted.

**A reset is measured against the power-on state, which no reset can be assumed
to reproduce.** That is the comparison worth making and it is also the one that
cannot be made twice from the same starting point: the second reset in a run
begins where the first left the unit, not where the mains switch did. The
sequence is therefore recorded with the results, and a reset examined on its own
terms needs its own power cycle.

**A byte that differs from power-on afterwards is reported whether or not this
run touched it.** Restoring the marks is the question; what else moved is the
part that a marked-byte-only comparison would never show.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import roland
from .midi import MidiLink

Address = tuple[int, int, int]

# Two marks so that a byte already holding one is still broken by the other. A
# byte the reset leaves alone then reads as the mark rather than as its default,
# which is the whole signal.
MARKS = (0x2A, 0x55)

METHOD = (
    "Each reset was preceded by writing a mark into every address the write probe found "
    "accepts any value, so that a byte the reset leaves alone reads as the mark rather than "
    "as its default. Only bytes read back as holding the mark are counted."
)

WHY_PRECEDED = (
    "So the three are comparable with each other rather than each being read against wherever "
    "the previous one left the unit. This same probe measured that reset as reproducing the "
    "power-on capture byte for byte, which is what makes it usable as a starting line."
)

WHY_NO_VERDICT = (
    "Neither is a verdict about the reset. A byte the snapshot did not reach, and a byte with "
    "no power-on value to be held against, were both broken by the mark and left without an "
    "answer, so they are named rather than dropped: every marked byte is in exactly one list "
    "here, and the lists add up to what was marked."
)

WHY_SKIPPED = (
    "A mark written to an address that holds nothing lands in whatever store that address is a "
    "window onto, so a block of them says what the reset did to that store, under names that "
    "keep none of it. Skipping them is not a gap in the coverage: the store itself is marked "
    "and read under its own name. Anything listed here was left unmarked and this run says "
    "nothing about it."
)


@dataclass
class Reset:
    label: str
    message: list[int]
    note: str = ""

    @staticmethod
    def universal(label: str, sub_id: int, note: str = "") -> Reset:
        return Reset(label, [0xF0, 0x7E, 0x7F, 0x09, sub_id, 0xF7], note)

    @staticmethod
    def roland(
        label: str, address: Address, value: int = 0x00, *, device_id=0x10, note=""
    ) -> Reset:
        return Reset(label, roland.dt1(address, [value], device_id=device_id), note)


def catalogue(device_id: int) -> list[Reset]:
    return [
        Reset.universal("GM1 System On", 0x01, "MIDI universal non-realtime, sub id 09 01"),
        Reset.universal("GM2 System On", 0x03, "MIDI universal non-realtime, sub id 09 03"),
        Reset.roland(
            "GS Reset",
            (0x40, 0x00, 0x7F),
            device_id=device_id,
            note="the Roland address the manual gives as the GS reset",
        ),
    ]


def named(label: str, device_id: int) -> Reset:
    """One reset out of the catalogue, by the label it is published under.

    A reset is used two ways: as a subject, and as the starting line a subject is
    measured from. The second wants it by name, and wants the same object the
    first will report, so that the state a run began in is the one the archive
    says it began in.
    """
    for reset in catalogue(device_id):
        if reset.label == label:
            return reset
    known = ", ".join(r.label for r in catalogue(device_id))
    raise KeyError(f"no reset called {label!r}; the catalogue holds {known}")


def mode_set(device_id: int, mode: int = 0x00) -> Reset:
    """System Mode Set, which reinitialises rather than resetting parameters.

    Kept out of the default catalogue and named separately, because it changes
    what the parts are rather than what they hold.
    """
    return Reset.roland(
        f"System Mode Set {mode + 1}",
        (0x00, 0x00, 0x7F),
        mode,
        device_id=device_id,
        note="reinitialises the unit; not a parameter reset",
    )


@dataclass
class ResetResult:
    label: str
    message: str
    note: str

    marked: dict[str, int] = field(default_factory=dict)
    """Address -> the mark it was verified to be holding before the reset."""

    refused_the_mark: list[str] = field(default_factory=list)
    restored: list[str] = field(default_factory=list)
    left_marked: dict[str, int] = field(default_factory=dict)
    changed_to_something_else: dict[str, list[int]] = field(default_factory=dict)
    """Address -> [mark, what it holds now] where that is neither the mark nor the default."""

    not_read_afterwards: list[str] = field(default_factory=list)
    """Marked, then absent from the snapshot taken after the reset."""

    missing_from_the_baseline: list[str] = field(default_factory=list)
    """Marked, but with no power-on value to be compared against."""

    differs_from_power_on: dict[str, list[int]] = field(default_factory=dict)
    """Every readable byte unequal to the power-on capture, marked or not."""

    regions_unread: int = 0

    def to_json(self) -> dict:
        return {
            "reset": self.label,
            "message": self.message,
            "note": self.note,
            "bytes_marked": len(self.marked),
            "marked": sorted(self.marked),
            "bytes_that_refused_the_mark": self.refused_the_mark,
            "not_read_afterwards": sorted(self.not_read_afterwards),
            "missing_from_the_baseline": sorted(self.missing_from_the_baseline),
            "why_those_two": WHY_NO_VERDICT,
            "restored_to_the_power_on_value": sorted(self.restored),
            "left_holding_the_mark": {a: f"{v:02X}" for a, v in sorted(self.left_marked.items())},
            "changed_to_neither": {
                a: [f"{v:02X}" for v in vs]
                for a, vs in sorted(self.changed_to_something_else.items())
            },
            "differs_from_power_on_afterwards": {
                a: [f"{v:02X}" for v in vs] for a, vs in sorted(self.differs_from_power_on.items())
            },
            "regions_unread": self.regions_unread,
        }


class Prober:
    def __init__(
        self,
        link: MidiLink,
        *,
        baseline: dict[Address, int],
        device_id: int = roland.DEFAULT_DEVICE_ID,
        settle: float = 0.6,
    ):
        self.link = link
        self.baseline = baseline
        self.device_id = device_id
        self.settle = settle

    def read_byte(self, address: Address) -> int | None:
        while self.link.receive(timeout=0.05):
            pass
        request = roland.rq1(address, 1, device_id=self.device_id)
        reply = roland.parse_dt1(self.link.exchange(request, timeout=0.6))
        if reply is None or reply.address != address or reply.size != 1:
            return None
        return reply.data[0]

    def mark(self, addresses: list[Address]) -> tuple[dict[Address, int], list[Address]]:
        """Write a mark to each address and keep only the ones that took it."""
        marked: dict[Address, int] = {}
        refused: list[Address] = []
        for address in addresses:
            current = self.read_byte(address)
            value = MARKS[0] if current != MARKS[0] else MARKS[1]
            self.link.send(roland.dt1(address, [value], device_id=self.device_id))
            time.sleep(0.02)
            if self.read_byte(address) == value:
                marked[address] = value
            else:
                refused.append(address)
        return marked, refused

    def apply(self, reset: Reset) -> None:
        self.link.send(reset.message)
        time.sleep(self.settle)


def compare(
    result: ResetResult,
    marked: dict[Address, int],
    after: dict[Address, int],
    baseline: dict[Address, int],
) -> None:
    """Sort every marked byte into exactly one outcome.

    Including the two that are not verdicts. A byte the snapshot did not reach,
    or one with no power-on value to be held against, was still broken by the
    mark, and dropping it makes the buckets add up to less than what was marked
    with nothing saying which byte went missing or why.
    """
    for address, mark in marked.items():
        name = f"{address[0]:02X} {address[1]:02X} {address[2]:02X}"
        now = after.get(address)
        default = baseline.get(address)
        if now is None:
            result.not_read_afterwards.append(name)
            continue
        if default is None:
            result.missing_from_the_baseline.append(name)
            continue
        if now == default:
            result.restored.append(name)
        elif now == mark:
            result.left_marked[name] = mark
        else:
            result.changed_to_something_else[name] = [mark, now]

    for address, was in baseline.items():
        now = after.get(address)
        if now is not None and now != was:
            result.differs_from_power_on[f"{address[0]:02X} {address[1]:02X} {address[2]:02X}"] = [
                was,
                now,
            ]


WHY_GAP = (
    "What was marked, less what got a verdict. A run made before the two outcomes that are "
    "not verdicts were named dropped those bytes silently, and kept only a count of what it "
    "marked, so the shortfall can be stated and the addresses cannot be recovered. Where this "
    "is zero every marked byte was accounted for."
)


def read_gap_again(payload: dict) -> dict[str, int]:
    """Say, per reset, how many marked bytes the record gives no outcome for.

    Read from the record's own counts, since a run that dropped a byte did not
    write down which one. Nothing here identifies an address; it says only that
    the arithmetic does not close, which is what the counts alone can support.
    """
    gaps = {}
    for result in payload.get("resets", []):
        accounted = (
            len(result.get("restored_to_the_power_on_value", []))
            + len(result.get("left_holding_the_mark", {}))
            + len(result.get("changed_to_neither", {}))
            + len(result.get("not_read_afterwards", []))
            + len(result.get("missing_from_the_baseline", []))
        )
        gaps[result["reset"]] = result["bytes_marked"] - accounted
    payload["marked_without_an_outcome"] = {"by_reset": gaps, "why": WHY_GAP}
    return gaps


def summarise(results: list[ResetResult]) -> str:
    lines = []
    for r in results:
        lines.append(
            f"{r.label}: {len(r.restored)} of {len(r.marked)} marked bytes came back, "
            f"{len(r.left_marked)} kept the mark, {len(r.changed_to_something_else)} went elsewhere"
        )
        if r.differs_from_power_on:
            lines.append(
                f"  {len(r.differs_from_power_on)} readable bytes differ from power-on afterwards"
            )
        else:
            lines.append("  the whole readable map matches the power-on capture")
        if r.refused_the_mark:
            lines.append(f"  {len(r.refused_the_mark)} addresses would not take a mark")
        if r.regions_unread:
            lines.append(f"  !! {r.regions_unread} region reads failed")
    return "\n".join(lines)


POWER_ON_NOTE = (
    "Read twice over. Nothing that writes was sent -- an Identity Request and RQ1 reads only "
    "-- so this is the state the unit powers up in rather than the state a run left behind. "
    "Addresses where the two reads disagreed are listed rather than resolved."
)

WHY_READ_TWICE = (
    "A byte read once has nothing to check it against. This is the baseline every reset is "
    "measured against, so a byte recorded here wrongly is credited to, or withheld from, "
    "every reset afterwards -- and the capture cannot be taken again without another power "
    "cycle. Reading the whole space a second time separates what the unit holds from what "
    "one read got wrong, at the cost of doubling a run that is only reads."
)

CAPTURED_IS_ASSERTED = (
    "That the unit had just been powered on is asserted by whoever ran this, not shown by the "
    "run. Nothing here can tell a unit fresh from the mains switch from one an earlier run "
    "wrote to, because both answer a read the same way. What the run can say is that it wrote "
    "nothing itself."
)


def agreed_bytes(
    first: dict[Address, int], second: dict[Address, int]
) -> tuple[dict[Address, int], list[Address], list[Address]]:
    """Split two reads of the same space into what they agree on and what they do not.

    A byte the two reads disagree about is left out of the baseline rather than
    resolved to either value. Deciding between them would need a third read, and
    a wrong decision here is invisible afterwards: the byte simply reads as
    having been changed by whichever reset is measured next.
    """
    agreed: dict[Address, int] = {}
    disagreed: list[Address] = []
    for address, value in first.items():
        if address not in second:
            continue
        if second[address] == value:
            agreed[address] = value
        else:
            disagreed.append(address)
    once = [a for a in first if a not in second] + [a for a in second if a not in first]
    return agreed, sorted(disagreed), sorted(once)


def power_on_record(
    first: dict[Address, int],
    second: dict[Address, int],
    *,
    unit_id: str,
    identity_reply: str,
    captured: str,
    regions_read: int,
    regions_unread: int,
) -> dict:
    """The capture a power-on run leaves, with what it rests on stated in it.

    Assembled here rather than where the run is driven, so a capture cannot be
    written without the two things a later reader has no way to recover: that
    the power cycle is an assertion rather than a measurement, and that a byte
    the two reads disagreed about was dropped rather than decided.
    """
    agreed, disagreed, once = agreed_bytes(first, second)
    return {
        "unit_id": unit_id,
        "captured": captured,
        "captured_is_asserted_not_measured": CAPTURED_IS_ASSERTED,
        "identity_reply": identity_reply,
        "note": POWER_ON_NOTE,
        "why_read_twice": WHY_READ_TWICE,
        "regions_read": regions_read,
        "regions_unread": regions_unread,
        "read_disagreed_at": [_address_text(a) for a in disagreed],
        "read_only_once": [_address_text(a) for a in once],
        "values": {_address_text(a): f"{v:02X}" for a, v in sorted(agreed.items())},
    }


def _address_text(address: Address) -> str:
    return " ".join(f"{b:02X}" for b in address)


__all__ = [
    "MARKS",
    "Prober",
    "Reset",
    "ResetResult",
    "agreed_bytes",
    "catalogue",
    "compare",
    "mode_set",
    "power_on_record",
    "summarise",
]
