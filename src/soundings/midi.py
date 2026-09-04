"""MIDI transport for probing a hardware tone generator.

Wraps python-rtmidi so the rest of the harness works in terms of complete SysEx
messages and round trips rather than byte streams. Active Sensing (0xFE) is
dropped on receive: an idle module emits it continuously and it would otherwise
appear in every reply.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import rtmidi

ACTIVE_SENSING = 0xFE
SYSEX_START = 0xF0
SYSEX_END = 0xF7


class MidiError(RuntimeError):
    pass


@dataclass(frozen=True)
class Ports:
    """The names of an input and an output port, as the platform reports them."""

    input_name: str
    output_name: str


def list_ports() -> tuple[list[str], list[str]]:
    """Return (input port names, output port names)."""
    mi, mo = rtmidi.MidiIn(), rtmidi.MidiOut()
    try:
        return list(mi.get_ports()), list(mo.get_ports())
    finally:
        mi.delete()
        mo.delete()


def _resolve(names: list[str], needle: str | None, kind: str) -> int:
    if not names:
        raise MidiError(f"no MIDI {kind} ports on this system")
    if needle is None:
        if len(names) > 1:
            raise MidiError(f"several MIDI {kind} ports; name one of: " + ", ".join(names))
        return 0
    hits = [i for i, n in enumerate(names) if needle.lower() in n.lower()]
    if not hits:
        raise MidiError(f"no MIDI {kind} port matching {needle!r}; have: " + ", ".join(names))
    if len(hits) > 1:
        raise MidiError(
            f"{needle!r} matches several MIDI {kind} ports: " + ", ".join(names[i] for i in hits)
        )
    return hits[0]


class MidiLink:
    """A bidirectional MIDI connection to one device.

    The receive side is required, not optional: every probe in this harness is a
    round trip, and a link that can only transmit cannot measure anything.
    """

    def __init__(self, port: str | None = None, *, timeout: float = 0.5):
        self.timeout = timeout
        self._in = rtmidi.MidiIn()
        self._out = rtmidi.MidiOut()
        # Anything that goes wrong from here on has to give the two objects back.
        # Each of them is a platform MIDI client and the platform allows a finite
        # number of them; a construction that raises without releasing its pair
        # leaks two, and enough of those and the next MidiIn() fails inside the
        # library, where it is a C++ error that ends the process rather than an
        # exception anything here can catch. The symptom is a run that aborts on
        # startup with nothing wrong with the device.
        try:
            ins, outs = list(self._in.get_ports()), list(self._out.get_ports())
            i_idx = _resolve(ins, port, "input")
            o_idx = _resolve(outs, port, "output")
            self.ports = Ports(ins[i_idx], outs[o_idx])
            self._in.open_port(i_idx)
            self._out.open_port(o_idx)
            # rtmidi filters SysEx by default.
            self._in.ignore_types(sysex=False, timing=True, active_sense=True)
        except BaseException:
            self._in.delete()
            self._out.delete()
            raise

    def close(self) -> None:
        self._in.close_port()
        self._out.close_port()
        self._in.delete()
        self._out.delete()

    def __enter__(self) -> MidiLink:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def drain(self) -> None:
        """Discard anything already waiting on the input."""
        while self._in.get_message() is not None:
            pass

    def send(self, message: bytes | list[int]) -> None:
        self._out.send_message(list(message))

    def receive(self, timeout: float | None = None) -> list[int]:
        """Collect bytes until a complete SysEx has arrived or the timeout expires.

        Returns the raw bytes received with Active Sensing removed. A caller that
        wants a SysEx should check the framing itself; this deliberately does not
        raise on a partial message, because a partial message is a measurement
        result (it is how a dropped byte presents).
        """
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        got: list[int] = []
        while time.monotonic() < deadline:
            msg = self._in.get_message()
            if msg is None:
                time.sleep(0.001)
                continue
            data = [b for b in msg[0] if b != ACTIVE_SENSING]
            got.extend(data)
            if got and got[-1] == SYSEX_END:
                break
        return got

    def exchange(self, message: bytes | list[int], timeout: float | None = None) -> list[int]:
        """Send one message and return whatever came back before the timeout."""
        self.drain()
        self.send(message)
        return self.receive(timeout)
