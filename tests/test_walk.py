"""§C-2: `walk` writes a state and steps an address through it, recording as it goes.

Run end to end on fakes -- no MIDI port, no audio interface -- the same way
`test_takes_manifest.py` runs `cmd_contrast`. What is checked here is the shape
the design fixes: the order the state is written in, the names takes are kept
under, the manifest's top-level keys, and that a write the unit did not take
stops the run before anything is recorded.
"""

from __future__ import annotations

import json

import pytest

from soundings import roland, takes
from soundings.cli import build_parser, session
from soundings.cli.walk import cmd_walk

from .fakes import FakePorts, fake_capture_record


class UnitLink:
    """A unit that holds whatever was last written to any address and answers reads of it.

    The same shape as `test_takes_manifest.py`'s fixture: every DT1 write updates
    its own state and every RQ1 read answers out of it, which is enough to carry
    `cmd_walk` past every preparation and read-back it makes without touching
    hardware.
    """

    ports = FakePorts()

    def __init__(self) -> None:
        self.state: dict[str, list[int]] = {}
        self.sent: list[list[int]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def send(self, message) -> None:
        self.sent.append(list(message))
        parsed = roland.parse_dt1(message)
        if parsed is not None:
            self.state[" ".join(f"{b:02X}" for b in parsed.address)] = list(parsed.data)

    def receive(self, timeout: float = 0.0) -> list[int]:
        return []

    def exchange(self, message, timeout: float = 0.0) -> list[int]:
        asked = " ".join(f"{b:02X}" for b in message[5:8])
        got = self.state.get(asked)
        return roland.dt1(asked, got, device_id=0x10) if got is not None else []


class RefusingLink(UnitLink):
    """Like `UnitLink`, but one write is silently dropped rather than taken.

    So a run against it writes a byte, reads back whatever the address held
    before, and finds the two disagree -- the one failure `cmd_walk` has to stop
    on rather than record through.
    """

    def __init__(self, refuse_address: str, refuse_value: int) -> None:
        super().__init__()
        self.refuse_address = refuse_address
        self.refuse_value = refuse_value

    def send(self, message) -> None:
        parsed = roland.parse_dt1(message)
        if (
            parsed is not None
            and " ".join(f"{b:02X}" for b in parsed.address) == self.refuse_address
            and list(parsed.data) == [self.refuse_value]
        ):
            self.sent.append(list(message))
            return
        super().send(message)


class _PassingReport:
    passed = True

    def __str__(self) -> str:
        return "report: pass"


@pytest.fixture
def wired_unit(monkeypatch):
    """No MIDI port, no audio device: everything `cmd_walk` touches is faked."""
    monkeypatch.setattr("time.sleep", lambda _s: None)
    monkeypatch.setattr(
        session, "midi_selftest", lambda link, *, repeats, device_id: _PassingReport()
    )
    from soundings import capture

    monkeypatch.setattr(capture, "record", fake_capture_record)


def _args(save, **overrides):
    argv = [
        "walk",
        "--type",
        "01 50",
        "--slot",
        "40 03 04",
        "--settings",
        "0,32",
        "--stimulus",
        "struck",
        "--takes",
        "1",
        "--bypass",
        "1",
        "--silences",
        "1",
        "--save",
        str(save),
    ]
    for flag, value in overrides.items():
        argv += [f"--{flag.replace('_', '-')}", str(value)]
    return build_parser().parse_args(argv)


def _dt1_addresses(sent) -> list[str]:
    """The address of every well-formed DT1 write, in the order it was sent."""
    addresses = []
    for message in sent:
        parsed = roland.parse_dt1(message)
        if parsed is not None:
            addresses.append(" ".join(f"{b:02X}" for b in parsed.address))
    return addresses


def test_the_state_is_written_reset_type_routing_prepare_then_settings(
    wired_unit, tmp_path, monkeypatch
):
    link = UnitLink()
    monkeypatch.setattr(session, "MidiLink", lambda port: link)
    args = _args(
        tmp_path / "takes",
        prepare="40 41 22=01",
        bypass=0,
        silences=0,
    )

    assert cmd_walk(args) == 0

    addresses = _dt1_addresses(link.sent)
    assert addresses[:5] == [
        "40 00 7F",  # GS Reset
        "40 03 00",  # type
        "40 42 22",  # routing
        "40 41 22",  # --prepare
        "40 03 04",  # the first setting
    ]
    # And a second write to the swept address for the second setting.
    assert addresses.count("40 03 04") == 2


def test_setting_names_are_v_padded_bytes_then_out_then_silence(wired_unit, tmp_path, monkeypatch):
    monkeypatch.setattr(session, "MidiLink", lambda port: UnitLink())
    save = tmp_path / "takes"
    args = _args(save)

    assert cmd_walk(args) == 0

    manifest = json.loads((save / "takes-manifest.json").read_text())
    assert [t["setting"] for t in manifest["takes"]] == ["v000", "v032", "out", "silence"]


def test_manifest_top_level_carries_unit_state_plus_address_settings_stimuli_label(
    wired_unit, tmp_path, monkeypatch
):
    monkeypatch.setattr(session, "MidiLink", lambda port: UnitLink())
    save = tmp_path / "takes"
    args = _args(save)

    assert cmd_walk(args) == 0

    manifest = takes.method_of(save)
    assert manifest["type"] == "01 50"
    assert manifest["prepared"] == [
        {"address": "40 03 00", "bytes": "01 50"},
        {"address": "40 42 22", "bytes": "01"},
    ]
    assert manifest["settle_s"] == args.settle
    assert manifest["address"] == "40 03 04"
    assert manifest["settings"] == [0, 32]
    assert isinstance(manifest["stimuli"], list) and manifest["stimuli"][0]["name"] == "struck"
    assert manifest["label"]


def test_a_readback_mismatch_stops_without_recording(wired_unit, tmp_path, monkeypatch):
    """The first setting's write is refused, so nothing is ever recorded."""
    link = RefusingLink("40 03 04", 0x00)
    monkeypatch.setattr(session, "MidiLink", lambda port: link)
    save = tmp_path / "takes"
    args = _args(save)

    assert cmd_walk(args) == 1
    assert not (save / "takes-manifest.json").exists()
    assert not list(save.glob("*.wav"))


def test_an_interrupted_walk_leaves_a_manifest_naming_what_it_reached(
    wired_unit, tmp_path, monkeypatch
):
    """The second setting's write is refused, so the manifest still names the first."""
    link = RefusingLink("40 03 04", 0x20)
    monkeypatch.setattr(session, "MidiLink", lambda port: link)
    save = tmp_path / "takes"
    args = _args(save, bypass=0, silences=0)

    assert cmd_walk(args) == 1

    manifest = json.loads((save / "takes-manifest.json").read_text())
    assert [t["setting"] for t in manifest["takes"]] == ["v000"]
