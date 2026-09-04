"""What a second MIDI input is, measured from the other side of a cable move.

This unit has two MIDI inputs and one output, and a host with a single MIDI out
can be cabled to one input at a time. That is not an inconvenience to work
around; it is the shape of the measurement. Everything else in this harness rests
on a request going out and a reply coming back, and an input that answers no
request cannot be measured that way at all.

So a run is split in two with the cable moved between the halves, and the unit's
own memory is what carries the result across. The first phase sends and reads
nothing. The second reads, and sends nothing before it has read what it came
for. What makes the pair one measurement rather than two runs is that every
stimulus in the first phase carries a value used nowhere else in it: a byte found
holding that value in the second phase names the message that put it there,
without anything having watched it at the time.

**The state left behind is the result, so the first phase deliberately does not
put the unit back.** Every other command that writes here restores what it wrote,
because a run that leaves state contaminates the next one. Here the state is the
finding, and restoring it would be erasing the answer.

**The null is a landing where the archive already has it.** The question is
whether an input reaches a part set of its own. This unit's map holds two blocks
shaped identically down to the region -- 40 and 50, 85 regions and 1250 bytes
each, with the same bytes accepting a write in both -- and the port A records put
a control change on channel 1 at 40 11 19. A run through the other input that
puts it there too has found two inputs addressing one set of parts. A run that
puts it at 50 11 19 has found what the second set is for.

**Which physical socket the cable was in is asserted, not measured.** Nothing in
the data can tell one input from the other, and a later reader has no way to
recover it. The record carries the claim as a claim, the same way a power-on
capture carries the assertion that it followed a power cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import roland
from .midi import MidiLink

Address = tuple[int, int, int]

METHOD = (
    "Sent through one MIDI input with nothing read back, then read through the other after the "
    "cable was moved. Every stimulus carried a value used by no other stimulus in the run, so a "
    "byte that differs from the baseline and holds one of those values names the message that "
    "wrote it."
)

INPUT_IS_ASSERTED = (
    "Which of the unit's two MIDI inputs the cable was in is stated here and was not measured. "
    "The host has one MIDI output and both inputs are reached over the same cable, so nothing in "
    "the data distinguishes them and a later reader cannot recover which it was."
)

WHY_VALUES_UNIQUE = (
    "Two stimuli sent at one value would be indistinguishable afterwards, and the run would "
    "credit whichever of them the table happened to keep. The values are checked for uniqueness "
    "before anything is sent rather than chosen carefully and trusted."
)

WHY_NOT_RESTORED = (
    "The first phase leaves everything it wrote in place, because what it wrote is the only "
    "thing that survives the cable move. A restore would put the unit back and take the "
    "measurement with it. The second phase resets the unit once it has read what it came for."
)

WHY_SYSEX_PROBE = (
    "Asked before any stimulus and with no state depending on it, so a silence here is about the "
    "input rather than about anything the run went on to do. An input that answers neither a "
    "universal identity request nor a Roland data request cannot be measured by any other "
    "command in this harness, all of which read what they write."
)

WHY_UNEXPLAINED = (
    "Bytes that differ from the baseline while holding no value this run sent. A message does "
    "not only store what it carries -- a program change loads a whole part -- so these are "
    "expected rather than a fault, and they are listed because a run that reported only its own "
    "values could not tell a knock-on effect from a byte it never noticed moving."
)

WHY_BASELINE = (
    "The comparison is against a state captured earlier rather than against a snapshot taken at "
    "the head of this run, because the input under test cannot be read from. Anything that moved "
    "between the capture and the first stimulus is therefore credited to the run, so the "
    "baseline is only worth what the claim that nothing else was sent is worth."
)

WHY_LANDED_NOWHERE = (
    "A stimulus whose value turned up nowhere. On an input that stores nothing this is every one "
    "of them and says so; on an input that stores most of them it is about the parameter, since "
    "the run's other landings show the messages arrived. A region that did not answer is not in "
    "the comparison at all, so the unread count beside this is what bounds every null in it."
)

NOTHING_READ = (
    "A comparison against a baseline the snapshot shares no address with. Every stimulus would "
    "be reported as having landed nowhere and no byte as having moved, which is the same shape "
    "as an input that stores nothing and is indistinguishable from it in the record."
)

# The per-channel stimulus, and the value the first channel is sent at. One
# controller across every channel is what maps an input onto a set of parts:
# the value encodes which channel carried it, so the answer survives without
# anything having to be watched.
PER_CHANNEL_CONTROLLER = 7
PER_CHANNEL_FIRST_VALUE = 0x10

# Further controllers on the first channel, each with its own value. These exist
# to make the per-channel result a claim about the part rather than about one
# parameter: a block that takes the level and nothing else would be a different
# finding from one that takes all of them.
EXTRA_CONTROLLERS = ((10, 0x21), (11, 0x22), (91, 0x23), (93, 0x24), (71, 0x25), (74, 0x26))

PROGRAM_VALUE = 0x27

# One address in each of the two candidate part blocks. Chosen because the write
# probe found both accept a value and give it back, and no message in this
# unit's archive reaches either -- so a value found there was written by the
# SysEx that carried it and not by a control change that happens to land nearby.
WRITE_TARGETS: tuple[tuple[Address, int], ...] = (
    ((0x40, 0x11, 0x1A), 0x28),
    ((0x50, 0x11, 0x1A), 0x29),
)


@dataclass
class Sent:
    """One stimulus, its value, and the bytes that carried it."""

    label: str
    kind: str
    value: int
    messages: list[list[int]] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "stimulus": self.label,
            "kind": self.kind,
            "value": f"{self.value:02X}",
            "messages": [" ".join(f"{b:02X}" for b in m) for m in self.messages],
        }


def catalogue(
    *, channels: int = 16, device_id: int = roland.DEFAULT_DEVICE_ID, first_channel: int = 0
) -> list[Sent]:
    """Everything the sending phase transmits, in the order it goes out.

    The per-channel sweep comes first so that the extras on the first channel
    overwrite it there rather than the other way round; a stimulus whose value
    was replaced by a later one would be reported as having landed nowhere.
    """
    out = [
        Sent(
            label=f"CC{PER_CHANNEL_CONTROLLER} on channel {ch + 1}",
            kind="cc",
            value=PER_CHANNEL_FIRST_VALUE + ch,
            messages=[[0xB0 | (ch & 0x0F), PER_CHANNEL_CONTROLLER, PER_CHANNEL_FIRST_VALUE + ch]],
        )
        for ch in range(first_channel, first_channel + channels)
    ]
    ch = first_channel
    out += [
        Sent(
            label=f"CC{number} on channel {ch + 1}",
            kind="cc",
            value=value,
            messages=[[0xB0 | (ch & 0x0F), number, value]],
        )
        for number, value in EXTRA_CONTROLLERS
    ]
    # Both halves of the bank latch are driven, because the unit accepts or
    # rejects a bank and program together: a bank left standing from something
    # earlier discards the program change whole, and the run would report a
    # message that was never acted on as one the input does not store.
    out.append(
        Sent(
            label=f"program change on channel {ch + 1}",
            kind="channel",
            value=PROGRAM_VALUE,
            messages=[
                [0xB0 | (ch & 0x0F), 0, 0],
                [0xB0 | (ch & 0x0F), 32, 0],
                [0xC0 | (ch & 0x0F), PROGRAM_VALUE],
            ],
        )
    )
    out += [
        Sent(
            label=f"DT1 {address[0]:02X} {address[1]:02X} {address[2]:02X}",
            kind="address",
            value=value,
            messages=[roland.dt1(address, [value], device_id=device_id)],
        )
        for address, value in WRITE_TARGETS
    ]
    return out


def repeated_values(sent: list[Sent]) -> dict[int, list[str]]:
    """Values carried by more than one stimulus, which would be uncreditable."""
    seen: dict[int, list[str]] = {}
    for s in sent:
        seen.setdefault(s.value, []).append(s.label)
    return {value: labels for value, labels in seen.items() if len(labels) > 1}


def answers_sysex(link: MidiLink, *, device_id: int, probe: str = "40 01 30") -> dict:
    """Whether this input answers a request at all, asked two ways.

    Both are asked because they are answered by different parts of a unit: the
    identity request is a universal message every instrument is meant to answer,
    and the data request is Roland's own and carries this unit's device number.
    A silence on one and a reply on the other would be a finding about which
    kind of message the input receives; a silence on both is what makes the rest
    of this file necessary.
    """
    identity = link.exchange(roland.IDENTITY_REQUEST, timeout=1.0)
    reply = roland.parse_dt1(link.exchange(roland.rq1(probe, 1, device_id=device_id), timeout=1.0))
    return {
        "identity_request": " ".join(f"{b:02X}" for b in identity) if identity else "no reply",
        "data_request": f"{probe} answered" if reply is not None else f"{probe}: no reply",
        "answered": bool(identity) or reply is not None,
        "why": WHY_SYSEX_PROBE,
    }


def send(link: MidiLink, sent: list[Sent], *, settle: float = 0.05) -> None:
    """Transmit the catalogue, and read nothing back."""
    import time

    for stimulus in sent:
        for message in stimulus.messages:
            link.send(message)
        time.sleep(settle)


class NothingWasRead(RuntimeError):
    """The snapshot has no address the baseline also holds, so nothing can be compared.

    Raised rather than returned, because the empty answer is not distinguishable
    from a real one. A run whose reads all failed reports every stimulus as
    landing nowhere and every byte as unmoved -- which is a coherent, plausible
    result meaning the opposite of what happened. It has been produced once here,
    by a snapshot taken over a link whose replies had fallen a message behind:
    each read returned the previous one's answer, every region was dropped for
    the address not matching, and the comparison that followed announced no
    differences at all.
    """


def locate(
    sent: list[Sent], baseline: dict[str, int], after: dict[str, int]
) -> tuple[dict[str, list[str]], dict[str, list[int]], list[str]]:
    """Sort every byte that moved into the stimulus that wrote it, or into neither.

    Returns the addresses credited to each stimulus, the moved bytes holding no
    value the run sent, and the addresses read now that the baseline has nothing
    to compare against. The third is separate from the second because a byte with
    no baseline has not been shown to have moved, and counting it as unexplained
    would report a gap in an earlier capture as a change made by this run.
    """
    if not set(after) & set(baseline):
        raise NothingWasRead(NOTHING_READ)
    by_value = {s.value: s.label for s in sent}
    landed: dict[str, list[str]] = {s.label: [] for s in sent}
    unexplained: dict[str, list[int]] = {}
    unknown: list[str] = []
    for address in sorted(after):
        now = after[address]
        was = baseline.get(address)
        if was is None:
            unknown.append(address)
            continue
        if was == now:
            continue
        label = by_value.get(now)
        if label is None:
            unexplained[address] = [was, now]
        else:
            landed[label].append(address)
    return landed, unexplained, unknown


def blocks_reached(landed: dict[str, list[str]]) -> dict[str, int]:
    """How many credited bytes fell under each top byte.

    The verdict of the whole run in one line: the block a set of channel
    messages reaches is what an input addresses.
    """
    counted: dict[str, int] = {}
    for addresses in landed.values():
        for address in addresses:
            counted[address[:2]] = counted.get(address[:2], 0) + 1
    return dict(sorted(counted.items()))


def summarise(landed: dict[str, list[str]], unexplained: dict, unknown: list[str]) -> str:
    lines = []
    for label, addresses in landed.items():
        where = ", ".join(addresses) if addresses else "nowhere"
        lines.append(f"  {label:30} -> {where}")
    reached = blocks_reached(landed)
    lines.append(
        "  blocks reached: "
        + (", ".join(f"{block} ({n} bytes)" for block, n in reached.items()) or "none")
    )
    empty = [label for label, addresses in landed.items() if not addresses]
    if empty:
        lines.append(f"  {len(empty)} stimuli landed nowhere: {', '.join(empty)}")
    lines.append(f"  {len(unexplained)} bytes moved holding no value this run sent")
    if unknown:
        lines.append(f"  {len(unknown)} bytes read now have nothing in the baseline to compare")
    return "\n".join(lines)


__all__ = [
    "INPUT_IS_ASSERTED",
    "METHOD",
    "NOTHING_READ",
    "WHY_BASELINE",
    "WHY_LANDED_NOWHERE",
    "WHY_NOT_RESTORED",
    "WHY_SYSEX_PROBE",
    "WHY_UNEXPLAINED",
    "WHY_VALUES_UNIQUE",
    "NothingWasRead",
    "Sent",
    "answers_sysex",
    "blocks_reached",
    "catalogue",
    "locate",
    "repeated_values",
    "send",
    "summarise",
]
