"""§B: the state a preparing run left the unit in, read once and used twice.

`unit_state` is tested directly against the shape the design fixes, and then
`cmd_contrast` and `cmd_transfer` are run end to end on fakes -- no MIDI port, no
audio interface -- to show the manifest and the published JSON come out of the
same call rather than two expressions that can drift apart.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from soundings import roland, takes
from soundings.cli import build_parser, session

from .fakes import FakePorts, fake_capture_record

# unit_state, called directly


def test_unit_state_matches_the_designed_shape():
    prepare = [("40 03 00", (0x01, 0x50)), ("40 42 22", (0x01,))]

    assert takes.unit_state(prepare, as_held=["00", "7F"], settle_s=0.4) == {
        "type": "01 50",
        "prepared": [
            {"address": "40 03 00", "bytes": "01 50"},
            {"address": "40 42 22", "bytes": "01"},
        ],
        "values_as_held": ["00", "7F"],
        "settle_s": 0.4,
    }


def test_unit_state_type_is_null_without_a_40_03_00_write():
    state = takes.unit_state([("40 41 22", (0x01,))], settle_s=0.4)

    assert state["type"] is None
    assert state["prepared"] == [{"address": "40 41 22", "bytes": "01"}]


def test_unit_state_omits_values_as_held_when_not_given():
    state = takes.unit_state([], settle_s=0.4)

    assert "values_as_held" not in state
    assert state == {"type": None, "prepared": [], "settle_s": 0.4}


def test_unit_state_carries_values_as_held_when_given_even_if_empty_list():
    """`as_held` only when given, and an explicit empty list is still given."""
    state = takes.unit_state([], as_held=[], settle_s=0.4)

    assert state["values_as_held"] == []


# The write side, run through fakes


class UnitLink:
    """A unit that holds whatever was last written to any address and answers reads of it.

    Enough to carry `cmd_contrast` and `cmd_transfer` past their preparation and
    readback checks without touching hardware: every DT1 write updates its own
    state and every RQ1 read answers out of it.
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


class _PassingReport:
    passed = True

    def __str__(self) -> str:
        return "report: pass"


@pytest.fixture
def wired_unit(monkeypatch):
    """No MIDI port, no audio device: every command that touches the unit is faked."""
    monkeypatch.setattr("time.sleep", lambda _s: None)
    monkeypatch.setattr(session, "MidiLink", lambda port: UnitLink())
    monkeypatch.setattr(
        session, "midi_selftest", lambda link, *, repeats, device_id: _PassingReport()
    )
    from soundings import capture

    monkeypatch.setattr(capture, "record", fake_capture_record)


def test_contrast_writes_the_same_prepared_to_the_manifest_and_the_result(wired_unit, tmp_path):
    from soundings.cli import sound

    save = tmp_path / "takes"
    out = tmp_path / "result.json"
    args = build_parser().parse_args(
        [
            "contrast",
            "--address",
            "40 03 04",
            "--prepare",
            "40 03 00=01 50",
            "--values",
            "0,127",
            "--takes",
            "1",
            "--save",
            str(save),
            "--out",
            str(out),
        ]
    )

    assert sound.cmd_contrast(args) == 0

    published = json.loads(out.read_text())
    manifest = takes.method_of(save)

    assert published["prepared"] == [{"address": "40 03 00", "bytes": "01 50"}]
    assert manifest["prepared"] == published["prepared"]
    assert manifest["type"] == "01 50"
    assert manifest["settle_s"] == args.settle


def test_transfer_writes_the_same_prepared_to_the_manifest_and_the_result(
    wired_unit, tmp_path, monkeypatch
):
    from soundings import probe
    from soundings.cli import inject

    def fake_play_and_record(sweep, *, device, output_channels, input_channels):
        return np.tile(sweep.played[:, None], (1, len(input_channels)))

    monkeypatch.setattr(probe, "play_and_record", fake_play_and_record)

    save = tmp_path / "takes"
    out = tmp_path / "result.json"
    args = build_parser().parse_args(
        [
            "transfer",
            "--prepare",
            "40 03 00=01 61",
            "--seconds",
            "0.1",
            "--pad",
            "0.02",
            "--rate",
            "8000",
            "--low",
            "200",
            "--high",
            "3000",
            "--takes",
            "1",
            "--save",
            str(save),
            "--out",
            str(out),
        ]
    )

    assert inject.cmd_transfer(args) == 0

    published = json.loads(out.read_text())
    manifest = takes.method_of(save)

    assert published["prepared"] == [{"address": "40 03 00", "bytes": "01 61"}]
    assert manifest["prepared"] == published["prepared"]
    assert manifest["type"] == "01 61"
    assert manifest["settle_s"] == args.settle
