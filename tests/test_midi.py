"""What the link does when it cannot be opened.

A MidiIn and a MidiOut are each a platform MIDI client, and the platform allows
a finite number of them. A construction that fails after making the pair and
before anything can close it leaks two, and the leak is invisible until the
count runs out -- at which point the next one fails inside the library, as a C++
error that ends the process rather than an exception this harness can catch. So
the release is what is tested, not the raising.

No hardware. The double stands in for the library, which is what makes the
assertion about the code rather than about this machine's port list.
"""

from __future__ import annotations

import pytest

from soundings import midi
from soundings.midi import MidiError, MidiLink


class FakePort:
    def __init__(self, names: list[str]):
        self.names = names
        self.deleted = False
        self.opened: int | None = None

    def get_ports(self) -> list[str]:
        return list(self.names)

    def open_port(self, index: int) -> None:
        self.opened = index

    def ignore_types(self, **_) -> None:
        pass

    def delete(self) -> None:
        self.deleted = True


class FakeRtMidi:
    def __init__(self, inputs: list[str], outputs: list[str]):
        self.made: list[FakePort] = []
        self._inputs = inputs
        self._outputs = outputs

    def MidiIn(self):  # noqa: N802 - the library's own name
        return self._make(self._inputs)

    def MidiOut(self):  # noqa: N802 - the library's own name
        return self._make(self._outputs)

    def _make(self, names: list[str]) -> FakePort:
        port = FakePort(names)
        self.made.append(port)
        return port


@pytest.fixture
def library(monkeypatch):
    def install(inputs, outputs):
        fake = FakeRtMidi(inputs, outputs)
        monkeypatch.setattr(midi, "rtmidi", fake)
        return fake

    return install


def test_a_link_that_cannot_find_its_port_gives_both_clients_back(library):
    """The failure is a name that matches nothing, which is the common one: a
    device unplugged, or a substring that stopped matching after a rename."""
    fake = library(["Scarlett 4i4 4th Gen"], ["Scarlett 4i4 4th Gen"])

    with pytest.raises(MidiError):
        MidiLink("nothing called this")

    assert len(fake.made) == 2
    assert all(port.deleted for port in fake.made)


def test_a_link_that_opens_keeps_its_clients(library):
    """The release must not be able to fire on the path that worked, or every
    run would be talking through a closed port."""
    fake = library(["Scarlett 4i4 4th Gen"], ["Scarlett 4i4 4th Gen"])

    link = MidiLink("Scarlett")

    assert not any(port.deleted for port in fake.made)
    assert link.ports.input_name == "Scarlett 4i4 4th Gen"


def test_an_ambiguous_name_is_refused_rather_than_guessed(library):
    """Two ports matching one substring is the case where picking either would
    put a whole run on the wrong device without saying so."""
    fake = library(["SC-8850 PART A", "SC-8850 PART B"], ["SC-8850 PART A"])

    with pytest.raises(MidiError):
        MidiLink("SC-8850")

    assert all(port.deleted for port in fake.made)
