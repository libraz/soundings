"""Roland exclusive messages: DT1 (write) and RQ1 (read).

The address and size are three seven-bit bytes each, and the checksum covers
both plus the data. Nothing here knows what any address means -- that is the
measurement's job, not the transport's.

**The model id is part of the address and not a property of the unit.** One
machine answers under more than one of them, each opening a separate space in
which the same three bytes mean something else, so a run has to say which one it
addressed and a record has to carry it. Defaulting it here and leaving it
unsayable would put every measurement in one space without recording the choice,
and a later reader could not tell a space that answered nothing from one that was
never asked.
"""

from __future__ import annotations

from dataclasses import dataclass

ROLAND_ID = 0x41
GS_MODEL_ID = 0x42
CMD_RQ1 = 0x11
CMD_DT1 = 0x12

DEFAULT_DEVICE_ID = 0x10


def checksum(payload: list[int]) -> int:
    return (128 - sum(payload) % 128) % 128


def _three(value: int) -> list[int]:
    if not 0 <= value < (1 << 21):
        raise ValueError(f"value {value} does not fit in three seven-bit bytes")
    return [(value >> 14) & 0x7F, (value >> 7) & 0x7F, value & 0x7F]


def address_bytes(address: int | tuple[int, int, int] | str) -> list[int]:
    """Accept 0x400130, (0x40, 0x01, 0x30) or '40 01 30'."""
    if isinstance(address, str):
        parts = [int(p, 16) for p in address.replace(",", " ").split()]
        if len(parts) != 3:
            raise ValueError(f"expected three address bytes, got {address!r}")
        return parts
    if isinstance(address, tuple):
        if len(address) != 3:
            raise ValueError(f"expected three address bytes, got {address!r}")
        return list(address)
    return _three(address)


def rq1(
    address: int | tuple[int, int, int] | str,
    size: int,
    *,
    device_id: int = DEFAULT_DEVICE_ID,
    model_id: int = GS_MODEL_ID,
) -> list[int]:
    payload = address_bytes(address) + _three(size)
    return [0xF0, ROLAND_ID, device_id, model_id, CMD_RQ1, *payload, checksum(payload), 0xF7]


def dt1(
    address: int | tuple[int, int, int] | str,
    data: list[int],
    *,
    device_id: int = DEFAULT_DEVICE_ID,
    model_id: int = GS_MODEL_ID,
) -> list[int]:
    payload = address_bytes(address) + list(data)
    return [0xF0, ROLAND_ID, device_id, model_id, CMD_DT1, *payload, checksum(payload), 0xF7]


@dataclass(frozen=True)
class Dt1Reply:
    address: tuple[int, int, int]
    data: list[int]
    checksum_ok: bool
    model_id: int
    """Which model id the reply came under, reported rather than checked here.

    Whether it is the one that was asked for is the caller's question: a reader
    aimed at one space wants a reply from another refused, while a run
    establishing which spaces a unit answers in wants to see it. Parsing it away
    would leave the second unable to ask.
    """

    @property
    def size(self) -> int:
        return len(self.data)


def malformation(raw: list[int]) -> str | None:
    """Say why these bytes are not one well formed SysEx message, or None if they are.

    Checking the interior matters as much as checking the frame. A message that
    begins with F0 and ends with F7 can still hold a second F0, or bytes with the
    high bit set, which no legal SysEx contains. One arrived on this path 113
    bytes long: an unterminated header for the requested address, thirty one
    three byte groups led by EF, then the correct reply, all inside one frame.
    Parsed without this check it reads as a 103 byte answer -- a plausible
    number, wrong, and indistinguishable from data.

    An empty buffer is not a malformation. Nothing arriving is an ordinary
    outcome of probing an address that does not exist.
    """
    if not raw:
        return None
    if raw[0] != 0xF0:
        return f"does not start with F0 (starts {raw[0]:02X})"
    if raw[-1] != 0xF7:
        return f"does not end with F7 (ends {raw[-1]:02X})"
    inner = raw[1:-1]
    extra = inner.count(0xF0)
    if extra:
        return f"holds {extra} further F0 inside one frame"
    high = sorted({b for b in inner if b > 0x7F})
    if high:
        return "carries bytes with the high bit set: " + " ".join(f"{b:02X}" for b in high)
    return None


def parse_dt1(raw: list[int]) -> Dt1Reply | None:
    """Parse a DT1 reply, or return None if this is not a well-formed one.

    The checksum is reported rather than enforced. A bad checksum means the path
    corrupted the message, which is a finding about the measurement setup and
    must not be silently discarded. A malformed frame is different and is
    refused outright: its length is not a length, so there is no reply to report.
    """
    if len(raw) < 11 or malformation(raw) is not None:
        return None
    if raw[1] != ROLAND_ID or raw[4] != CMD_DT1:
        return None
    body = raw[5:-2]
    if len(body) < 4:
        return None
    return Dt1Reply(
        address=(body[0], body[1], body[2]),
        data=body[3:],
        checksum_ok=raw[-2] == checksum(body),
        model_id=raw[3],
    )


IDENTITY_REQUEST = [0xF0, 0x7E, 0x7F, 0x06, 0x01, 0xF7]
GM_SYSTEM_ON = [0xF0, 0x7E, 0x7F, 0x09, 0x01, 0xF7]


def gs_reset(*, device_id: int = DEFAULT_DEVICE_ID) -> list[int]:
    return dt1("40 00 7F", [0x00], device_id=device_id)
