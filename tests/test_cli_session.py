"""The gate every command that touches the unit now goes through.

It used to be written out at the head of each command, so a command could be
added without it and nothing would say so. Single-sourced it is one thing to get
right and one thing to break, which is what these hold still.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from soundings.cli import main, report, session


class FakeReport:
    def __init__(self, passed: bool):
        self.passed = passed

    def __str__(self) -> str:
        return f"report: {'pass' if self.passed else 'FAIL'}"


class FakePorts:
    output_name = "Fake MIDI Out"


class FakeLink:
    ports = FakePorts()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False


@pytest.fixture
def wired(monkeypatch):
    """Stand in for the port and the selftest; returns a switch for the verdict."""
    link = FakeLink()
    link.closed = False
    state = {"passed": True, "repeats": None}

    def selftest(_link, *, repeats, device_id):
        state["repeats"] = repeats
        return FakeReport(state["passed"])

    monkeypatch.setattr(session, "MidiLink", lambda port: link)
    monkeypatch.setattr(session, "midi_selftest", selftest)
    return link, state


def args(**kw) -> argparse.Namespace:
    return argparse.Namespace(**{"port": None, "device_id": 0x10, "verify_reads": 20, **kw})


def test_a_passing_selftest_hands_over_the_link(wired):
    link, _ = wired
    with session.verified_link(args(), refusing="sweeping") as given:
        assert given is link


def test_a_failing_selftest_refuses_and_names_what_did_not_happen(wired):
    _, state = wired
    state["passed"] = False
    with pytest.raises(session.Refused) as stopped:
        with session.verified_link(args(), refusing="sweeping"):
            raise AssertionError("the body must not run")
    assert str(stopped.value) == "\nSelftest failed. Not sweeping."


def test_the_port_is_closed_even_when_the_gate_refuses(wired):
    link, state = wired
    state["passed"] = False
    with pytest.raises(session.Refused):
        with session.verified_link(args(), refusing="writing"):
            pass
    assert link.closed


def test_the_selftest_is_run_for_as_many_reads_as_asked(wired):
    _, state = wired
    with session.verified_link(args(verify_reads=30), refusing="sweeping"):
        pass
    assert state["repeats"] == 30


def test_the_port_is_named_only_when_the_command_asks(wired, capsys):
    with session.verified_link(args(), refusing="asking"):
        pass
    assert "Fake MIDI Out" not in capsys.readouterr().out

    with session.verified_link(args(), refusing="sweeping", show_port=True):
        pass
    assert "MIDI: Fake MIDI Out" in capsys.readouterr().out


def test_an_announcement_is_printed_before_the_report(wired, capsys):
    with session.verified_link(args(), refusing="writing", announce="Verifying the path"):
        pass
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["Verifying the path", "report: pass"]


def test_main_turns_a_refusal_into_a_message_and_a_failing_exit(capsys, monkeypatch):
    """The refusal has to reach the reader; an exit code alone says nothing."""

    def refuse(_args):
        raise session.Refused("\nSelftest failed. Not scanning.")

    monkeypatch.setattr("soundings.cli.wire.cmd_identity", refuse)
    assert main(["identity"]) == 1
    assert "Selftest failed. Not scanning." in capsys.readouterr().out


def test_a_run_without_out_writes_nothing(tmp_path, capsys):
    report.write_json(None, {"anything": 1})
    assert not list(tmp_path.iterdir())
    assert capsys.readouterr().out == ""


def test_a_result_is_written_indented_with_a_trailing_newline(tmp_path, capsys):
    """The archive is read as text and diffed; the shape of the file is part of it."""
    path = tmp_path / "unit" / "result.json"
    report.write_json(str(path), {"method": "how", "value": 1})
    raw = path.read_text()
    assert raw.endswith("\n")
    assert raw.splitlines()[1].startswith('  "')
    assert json.loads(raw) == {"method": "how", "value": 1}
    assert f"wrote {path}" in capsys.readouterr().out


def test_the_directory_is_made_for_a_unit_that_has_none_yet(tmp_path):
    path = tmp_path / "units" / "new-unit-01" / "sweep.json"
    report.write_json(str(path), {"regions": []})
    assert Path(path).exists()
