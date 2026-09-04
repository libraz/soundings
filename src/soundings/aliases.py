"""Find where a message is stored, by watching the address space change.

The method is a difference: read a set of regions, send something, read them
again, and see which bytes moved. What makes it a measurement rather than a
coincidence is everything around that.

**A control run first, with nothing sent.** Two snapshots taken back to back
must be identical. Anything that moves on its own -- a running LFO, a counter, a
level meter -- would otherwise be attributed to whichever message happened to be
in flight, and the attribution would look exactly like a real one. Addresses
that fail the control are excluded by name rather than by hope.

**Primed, moved, and moved back.** The stimulus is sent three times: value A,
then B, then A again, with the baseline snapshot taken after the first A. A byte
is attributed only if it moved when B arrived and moved back when A returned.
Priming is what makes the first send informative: comparing against whatever the
unit happened to be holding drops every byte that already sat at A, and it drops
it silently, as an absence. The return leg costs no extra reads and asks for
more than a change -- a byte that merely drifted does not drift back on cue.

**What was sent is compared with what is stored.** An address that ends up
holding exactly what was transmitted is a different finding from one holding
something derived from it, and both readings are kept so the difference survives.

**A stimulus is any list of messages parameterised by one value.** A control
change is one. So are an NRPN or RPN triple, a bank select followed by a program
change, and a SysEx write. That is deliberate: the question of whether two
different messages reach one storage location is answered by running both
through the same procedure and comparing the addresses, not by trusting that
they were documented as the same parameter.

**A whole kind falling silent is not a finding about the unit.** The control
proves the snapshots and the diff work; it does not prove that an NRPN was
received, because it is not an NRPN. When no stimulus of a kind is attributed,
"the unit stores none of these" and "these never arrived" are the same
observation, and the caller is told so rather than shown a table of zeroes.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from . import roland
from .midi import MidiLink

METHOD = (
    "Each stimulus was sent at its low value, snapshotted, sent at its high value, snapshotted, "
    "and sent at its low value again. A byte is listed only if it moved both times, to a "
    "different value each time."
)

NOTE = (
    "A byte listed here followed the stimulus out and back. That says where the value is kept, "
    "not that anything uses it."
)

WHY_CONTROL = (
    "Sent before the first stimulus and after the last. Without it a scan that finds nothing "
    "cannot be told from a scan that cannot find anything; sent only once, it says nothing "
    "about the rest of the run."
)

WHY_RESIDUE = (
    "Every stimulus ends on the value it started with, so anything listed here is state one "
    "message carried into the next."
)

WHY_RECOVERED = (
    "Each stimulus that lands nothing is retried with its two values swapped. Anything listed "
    "here is something a single pass would have reported as absent."
)

WHY_KIND_REACHED = (
    "The control is a control change, so it cannot show that a message of another kind arrived. "
    "Where this is false, every negative in the run is about the path, not about the unit."
)

WHY_LANDED_OUTSIDE = (
    "A run that writes to an address and sees only that address move is evidence nothing "
    "mirrors it, but only if a write that did land elsewhere would have been seen. So a "
    "stimulus whose value turned up under another top byte is the demonstration, made by the "
    "run on itself. Where this is empty the negatives are about a scan never shown able to "
    "report a landing outside the block it wrote to."
)

NOT_SCANNED = (
    "Controllers 120 to 127 are channel mode messages. Sending one resets the channel state "
    "every later attribution is measured against, so they need a scan of their own."
)

RPN_PARKED = "RPN and NRPN were set to 7F 7F before the scan."

Address = tuple[int, int, int]


@dataclass
class Change:
    address: str
    """The address of the byte itself, not of the region it was read in."""

    before: int
    after: int


@dataclass
class Stimulus:
    """Something that can be sent at a value, and asked about twice."""

    label: str
    kind: str
    build: Callable[[int], list[list[int]]]
    values: tuple[int, int] = (0x20, 0x60)
    """The two values used. A stimulus whose parameter has a narrow range says so
    here rather than being scanned with values outside it, which a clamping
    address answers identically for both and so reads as storing nothing."""


@dataclass
class Attribution:
    label: str
    kind: str
    values: tuple[int, int]

    readings: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    """Address -> [(value sent, value read)] for the move and the move back."""

    verbatim: list[str] = field(default_factory=list)
    """Addresses that held exactly what was sent, both on the way out and back."""

    on_second_attempt: bool = False
    """The first pass found nothing here and the retry did. Worth seeing: it means
    a single pass of this scan can miss something that is there."""

    @property
    def addresses(self) -> list[str]:
        return sorted(self.readings)

    def to_json(self) -> dict:
        return {
            "stimulus": self.label,
            "kind": self.kind,
            "primed_then_sent": [f"{self.values[0]:02X}", f"{self.values[1]:02X}"],
            "found_only_on_the_second_attempt": self.on_second_attempt,
            "stores_verbatim": sorted(self.verbatim),
            "bytes": {
                a: [[f"{s:02X}", f"{r:02X}"] for s, r in pairs]
                for a, pairs in sorted(self.readings.items())
            },
        }


class Snapshotter:
    """Reads a fixed set of regions, the same way every time."""

    def __init__(
        self,
        link: MidiLink,
        regions: list[tuple[Address, int]],
        *,
        device_id: int = roland.DEFAULT_DEVICE_ID,
        timeout: float = 0.4,
    ):
        self.link = link
        self.regions = regions
        self.device_id = device_id
        self.timeout = timeout
        self.reads = 0
        self.unread = 0

    def take(self) -> dict[Address, int]:
        """Read every region and flatten it to one byte per address.

        A region that fails to read is left out rather than recorded as zeros: an
        absent byte cannot differ from itself, which keeps a failed read from
        appearing as a change on the next comparison.
        """
        out: dict[Address, int] = {}
        for start, length in self.regions:
            self.reads += 1
            request = roland.rq1(start, length, device_id=self.device_id)
            raw = self.link.exchange(request, timeout=self.timeout)
            if roland.malformation(raw) is not None:
                while self.link.receive(timeout=0.3):
                    pass
                raw = self.link.exchange(request, timeout=self.timeout)
            reply = roland.parse_dt1(raw)
            if reply is None or reply.address != start:
                self.unread += 1
                continue
            for i, value in enumerate(reply.data):
                out[(start[0], start[1], start[2] + i)] = value
        return out


def differences(before: dict[Address, int], after: dict[Address, int]) -> list[Change]:
    """Bytes present in both snapshots that hold a different value in the second."""
    changed = []
    for address, old in before.items():
        new = after.get(address)
        if new is not None and new != old:
            changed.append(Change(f"{address[0]:02X} {address[1]:02X} {address[2]:02X}", old, new))
    return sorted(changed, key=lambda c: c.address)


def control_run(shot: Snapshotter, *, rounds: int = 3, settle: float = 0.15) -> set[str]:
    """Names the addresses that change with nothing sent to them.

    Run before anything is attributed. Its output is an exclusion list, and an
    empty one is the result worth having -- it is what makes a later difference
    mean the message caused it.
    """
    restless: set[str] = set()
    previous = shot.take()
    for _ in range(rounds):
        time.sleep(settle)
        current = shot.take()
        restless.update(c.address for c in differences(previous, current))
        previous = current
    return restless


class Scanner:
    def __init__(self, shot: Snapshotter, *, restless: set[str], settle: float = 0.1):
        self.shot = shot
        self.restless = restless
        self.settle = settle
        self.recovered: list[str] = []
        """Stimuli a single pass missed and the retry found."""

        self.residue: dict[str, list[str]] = {}
        """Stimulus -> addresses it did not put back when its value returned.

        Every stimulus ends on the value it started with, so the watched space
        should end where it began. Anything left over is state carried into the
        next stimulus, whose baseline then holds it. Reported rather than
        corrected: which addresses a message fails to release is a fact about
        the unit, and a scan that quietly repaired it would not have found the
        bank latch."""

    def _send(self, messages: list[list[int]]) -> None:
        for message in messages:
            self.shot.link.send(message)
        time.sleep(self.settle)

    def attribute(self, stimulus: Stimulus) -> Attribution | None:
        """Attempt the stimulus, and attempt it again with the values swapped if
        nothing followed.

        A negative is the claim that needs the strength, so it is the one that is
        retried; a positive is not re-run because there is nothing to strengthen.
        The retry swaps the two values rather than choosing new ones, which keeps
        it inside whatever range the pair was chosen for and closes a failure
        this method has: if the unit already holds the value the stimulus primes
        with, the prime moves nothing, and a byte that would have been seen is
        instead absent. Sending them the other way round has the same byte
        moving.

        The retry was added after one program change went unattributed in a run
        whose own control passed, and which four later tests could neither
        reproduce nor explain. Message loss, storage latency and failed reads
        were each measured and none of them accounts for it. What is recorded is
        therefore that a single pass can miss, not why.
        """
        hit = self._attempt(stimulus, stimulus.values)
        if hit is not None:
            return hit
        hit = self._attempt(stimulus, (stimulus.values[1], stimulus.values[0]))
        if hit is not None:
            hit.on_second_attempt = True
            self.recovered.append(stimulus.label)
        return hit

    def _attempt(self, stimulus: Stimulus, values: tuple[int, int]) -> Attribution | None:
        """Prime, move, move back, and keep the bytes that did all three."""
        low, high = values
        self._send(stimulus.build(low))
        baseline = self.shot.take()

        self._send(stimulus.build(high))
        moved = self.shot.take()

        self._send(stimulus.build(low))
        returned = self.shot.take()

        kept = self.residue.setdefault(stimulus.label, [])
        for change in differences(baseline, returned):
            if change.address not in self.restless and change.address not in kept:
                kept.append(change.address)
        if not kept:
            del self.residue[stimulus.label]

        result = Attribution(label=stimulus.label, kind=stimulus.kind, values=values)
        out = {c.address: c.after for c in differences(baseline, moved)}
        back = {c.address: c.after for c in differences(moved, returned)}
        for address, went in out.items():
            if address in self.restless or address not in back:
                continue
            came = back[address]
            if came != went:
                result.readings[address] = [(high, went), (low, came)]
                if (went, came) == (high, low):
                    result.verbatim.append(address)
        return result if result.readings else None


def control_change(channel: int, controller: int, value: int) -> list[int]:
    return [0xB0 | (channel & 0x0F), controller & 0x7F, value & 0x7F]


def cc(channel: int, controller: int, *, values: tuple[int, int] = (0x20, 0x60)) -> Stimulus:
    return Stimulus(
        label=f"CC{controller}",
        kind="cc",
        build=lambda v: [control_change(channel, controller, v)],
        values=values,
    )


def nrpn(channel: int, msb: int, lsb: int, name: str = "") -> Stimulus:
    """Select an NRPN and set its data entry MSB, as one indivisible stimulus.

    The selector is re-sent for every value rather than once for the pair. A
    selector left standing from an earlier stimulus is the classic way for a data
    entry to land somewhere nobody named, and re-selecting costs two messages.
    """
    label = f"NRPN {msb:02X} {lsb:02X}" + (f" ({name})" if name else "")
    return Stimulus(
        label=label,
        kind="nrpn",
        build=lambda v: [
            control_change(channel, 99, msb),
            control_change(channel, 98, lsb),
            control_change(channel, 6, v),
        ],
    )


def rpn(channel: int, msb: int, lsb: int, name: str = "", values=(0x20, 0x60)) -> Stimulus:
    label = f"RPN {msb:02X} {lsb:02X}" + (f" ({name})" if name else "")
    return Stimulus(
        label=label,
        kind="rpn",
        build=lambda v: [
            control_change(channel, 101, msb),
            control_change(channel, 100, lsb),
            control_change(channel, 6, v),
        ],
        values=values,
    )


def program_change(channel: int, values: tuple[int, int] = (0x00, 0x30)) -> Stimulus:
    return Stimulus(
        label="program change",
        kind="channel",
        build=lambda v: [[0xC0 | (channel & 0x0F), v & 0x7F]],
        values=values,
    )


def bank_then_program(
    channel: int, controller: int, *, program: int = 0, values: tuple[int, int] = (0x00, 0x03)
) -> Stimulus:
    """A bank select followed by the program change that commits it.

    Bank select alone is attributed to nothing, which is what a latch looks like:
    the number is held somewhere no read reaches until a program change consumes
    it. Pairing them separates "not stored" from "not stored yet".

    **Both halves of the latch are driven every time, not just the one under
    test.** The pair is accepted or rejected whole: a bank the program does not
    exist in discards the program change entirely, and neither the bank byte nor
    the program byte moves. So a value left in the other half by an earlier
    stimulus does not merely add noise -- it silently switches off every program
    change for the rest of the run, and nothing in the address space says so,
    because the latch is not in the address space. That is what happened before
    this was written: a run set CC0 to 3, and from then on program change and
    map select both read as storing nothing, while the run's own control kept
    passing.
    """
    return Stimulus(
        label=f"CC{controller} then program change",
        kind="channel",
        build=lambda v: [
            control_change(channel, 0, v if controller == 0 else 0),
            control_change(channel, 32, v if controller == 32 else 0),
            [0xC0 | (channel & 0x0F), program & 0x7F],
        ],
        values=values,
    )


def pitch_bend(channel: int, values: tuple[int, int] = (0x20, 0x60)) -> Stimulus:
    return Stimulus(
        label="pitch bend",
        kind="channel",
        build=lambda v: [[0xE0 | (channel & 0x0F), 0x00, v & 0x7F]],
        values=values,
    )


def channel_pressure(channel: int, values: tuple[int, int] = (0x20, 0x60)) -> Stimulus:
    return Stimulus(
        label="channel pressure",
        kind="channel",
        build=lambda v: [[0xD0 | (channel & 0x0F), v & 0x7F]],
        values=values,
    )


def address_write(
    address: Address, *, device_id: int = roland.DEFAULT_DEVICE_ID, values=(0x20, 0x60)
) -> Stimulus:
    """Write one address by SysEx, to find every other address that follows it."""
    return Stimulus(
        label=f"DT1 {address[0]:02X} {address[1]:02X} {address[2]:02X}",
        kind="address",
        build=lambda v: [roland.dt1(address, [v], device_id=device_id)],
        values=values,
    )


# The GS part parameters reachable as an NRPN. Names are the specification's;
# whether this unit puts each one where the specification says is the question,
# so nothing here is used to label an address that was found.
GS_NRPN = (
    (0x01, 0x08, "vibrato rate"),
    (0x01, 0x09, "vibrato depth"),
    (0x01, 0x0A, "vibrato delay"),
    (0x01, 0x20, "TVF cutoff"),
    (0x01, 0x21, "TVF resonance"),
    (0x01, 0x63, "envelope attack"),
    (0x01, 0x64, "envelope decay"),
    (0x01, 0x66, "envelope release"),
)

# Addressed per drum note, so the note number is the low byte and these only
# mean anything on a part in drum mode.
GS_DRUM_NRPN = (
    (0x18, "drum pitch coarse"),
    (0x1A, "drum level"),
    (0x1C, "drum panpot"),
    (0x1D, "drum reverb send"),
    (0x1E, "drum chorus send"),
    (0x1F, "drum delay send"),
)

GS_RPN = (
    (0x00, 0x00, "pitch bend sensitivity", (0x02, 0x0C)),
    (0x00, 0x01, "master fine tune", (0x20, 0x60)),
    (0x00, 0x02, "master coarse tune", (0x3C, 0x44)),
    (0x00, 0x05, "modulation depth range", (0x01, 0x08)),
)


def universal(
    label: str,
    head: tuple[int, ...],
    *,
    realtime: bool = True,
    values: tuple[int, int] = (0x20, 0x60),
) -> Stimulus:
    """A universal system exclusive whose last byte before the terminator varies.

    Addressed to 7F, every device, rather than to this unit's identifier. These
    are defined for any instrument that answers them at all, and a unit ignoring
    its own broadcast form would itself be the finding.
    """
    kind = 0x7F if realtime else 0x7E
    return Stimulus(
        label=label,
        kind="universal",
        build=lambda v: [[0xF0, kind, 0x7F, *head, v & 0x7F, 0xF7]],
        values=values,
    )


def universal_stimuli(channel: int, note: int) -> list[Stimulus]:
    """The GM and GM2 universal messages, none of which this project had sent.

    Names are the specification's. Where each one lands, and whether it lands at
    all, is what the scan is for -- a device is free to answer any of them by
    doing nothing, and several are defined only for a device that claims GM2.

    The key-based control is the one that addresses a single drum note, so it is
    given the note the caller names rather than a fixed one, and it is the only
    member whose target changes with the note.
    """
    return [
        # Sub id 04, device control. The byte varied is the MSB of each pair,
        # since the LSB alone is finer than several of these are documented to
        # resolve, and a stimulus that moves nothing readable is a null about
        # the resolution wearing the parameter's name.
        universal("master volume", (0x04, 0x01, 0x00), values=(0x20, 0x7F)),
        universal("master balance", (0x04, 0x02, 0x00)),
        universal("master fine tuning", (0x04, 0x03, 0x00)),
        universal("master coarse tuning", (0x04, 0x04, 0x00), values=(0x3C, 0x44)),
        # Sub id 04 05, global parameter control: the slot path comes first and
        # the value last, so only the tail varies as it does for the rest.
        universal(
            "global parameter control, reverb type",
            (0x04, 0x05, 0x01, 0x01, 0x01, 0x01, 0x00),
            values=(0x00, 0x04),
        ),
        universal(
            "global parameter control, chorus type",
            (0x04, 0x05, 0x01, 0x01, 0x01, 0x02, 0x00),
            values=(0x00, 0x04),
        ),
        # Sub id 0A 01, key-based instrument control: the only universal message
        # shaped like the per-note planes this unit holds, and so the only one
        # that could land in a block nothing else has reached.
        *(
            universal(
                f"key-based control, {name} on note {note}",
                (0x0A, 0x01, channel & 0x0F, note & 0x7F, controller),
            )
            for controller, name in (
                (0x07, "level"),
                (0x0A, "panpot"),
                (0x5B, "reverb send"),
                (0x5D, "chorus send"),
            )
        ),
    ]


def landed_outside_its_own_block(found: list[Attribution]) -> dict[str, list[str]]:
    """For each address write, the addresses it reached under some other top byte.

    Only address writes have a block of their own to be outside of, so a scan of
    any other kind produces nothing here and can say nothing about mirroring.
    """
    outside = {}
    for hit in found:
        parts = hit.label.split()
        if len(parts) != 4 or parts[0] != "DT1":
            continue
        elsewhere = [a for a in hit.addresses if a[:2] != parts[1]]
        if elsewhere:
            outside[hit.label] = elsewhere
    return outside


def read_outside_again(payload: dict) -> dict[str, list[str]]:
    """Put the derived control into a record written before the scan computed one.

    Read from the record's own attributions through the same rule the scan uses,
    so an earlier run carries the control rather than being quietly without one.
    Where it comes back empty that is the finding: the run never showed it could
    report a write landing outside the block it was addressed to.
    """
    found = [
        Attribution(
            label=hit["stimulus"],
            kind=hit["kind"],
            values=(0, 0),
            readings=dict.fromkeys(hit["bytes"], []),
        )
        for hit in payload.get("attributed", [])
    ]
    outside = landed_outside_its_own_block(found)
    payload["landed_outside_its_own_block"] = {
        "by_stimulus": outside,
        "why": WHY_LANDED_OUTSIDE,
    }
    return outside


def summarise(found: list[Attribution], restless: set[str], unread: int) -> str:
    lines = [f"{len(found)} stimuli were stored somewhere this could see"]
    for a in found:
        where = ", ".join(
            f"{addr}={'/'.join(f'{r:02X}' for _, r in pairs)}"
            for addr, pairs in sorted(a.readings.items())
        )
        mark = "  (verbatim)" if a.verbatim else ""
        mark += "  [only on the second attempt]" if a.on_second_attempt else ""
        lines.append(f"  {a.label:34} -> {where}{mark}")
    if restless:
        lines.append(
            f"  {len(restless)} addresses moved with nothing sent and were excluded: "
            + ", ".join(sorted(restless)[:8])
        )
    else:
        lines.append("  nothing moved with nothing sent, so a difference means the message did it")
    if unread:
        lines.append(f"  !! {unread} region reads failed and were left out of the comparison")
    return "\n".join(lines)


__all__ = [
    "GS_DRUM_NRPN",
    "GS_NRPN",
    "GS_RPN",
    "Attribution",
    "Change",
    "Scanner",
    "Snapshotter",
    "Stimulus",
    "address_write",
    "bank_then_program",
    "cc",
    "channel_pressure",
    "control_change",
    "control_run",
    "differences",
    "nrpn",
    "pitch_bend",
    "program_change",
    "rpn",
    "summarise",
]
