"""One command drives the unit at a time, and the harness is what enforces it.

The protocol has said hardware stages are serial since before there was a second
sweep to prove it. What it could not do is stop one: two runs shared a unit for
twenty-four minutes and the records of both became worthless, with nothing
visible from the second run's side.

No hardware: what is under test is the refusal.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from soundings import hardware
from soundings.cli import build_parser


@pytest.fixture(autouse=True)
def lock_in_tmp(tmp_path, monkeypatch):
    """Never the real path. A test that took the machine-wide lock would refuse a
    sweep running in another window, which is the opposite of the point."""
    monkeypatch.setattr(hardware, "LOCK", tmp_path / "unit.lock")


def test_nobody_holds_a_lock_that_was_never_taken() -> None:
    assert hardware.holder() is None


def test_the_lock_is_taken_for_the_run_and_dropped_after_it() -> None:
    with hardware.held("contrast"):
        found = hardware.holder()
        assert found is not None
        assert found["pid"] == os.getpid()
        assert found["command"] == "contrast"
    assert hardware.holder() is None
    assert not hardware.LOCK.exists()


def test_a_second_command_is_refused_while_the_first_holds_it() -> None:
    with hardware.held("sweep"), pytest.raises(hardware.Busy) as refused:
        with hardware.held("contrast"):
            pytest.fail("the second command should not have started")

    said = str(refused.value)
    # The refusal has to name the holder, or whoever reads it cannot act on it.
    assert str(os.getpid()) in said and "sweep" in said


def test_a_lock_left_by_a_process_that_died_is_not_a_holder() -> None:
    """The ordinary way a long unattended run ends is a kill, so a stale file has
    to be takeable. A stop that left every later run refused would be worse than
    the collision this guards against."""
    hardware.LOCK.write_text(json.dumps({"pid": 2**22, "command": "sweep", "since": 0.0}))

    assert hardware.holder() is None
    with hardware.held("contrast"):
        assert hardware.holder()["command"] == "contrast"


def test_a_run_drops_only_its_own_lock() -> None:
    """A run whose lock was broken as stale while it was somehow still alive must
    not delete the file belonging to whoever took it next."""
    with hardware.held("sweep"):
        hardware.LOCK.write_text(json.dumps({"pid": 2**22 + 1, "command": "other", "since": 0.0}))

    assert json.loads(hardware.LOCK.read_text())["command"] == "other"


def test_the_lock_is_not_a_read_followed_by_a_write() -> None:
    """Two commands started together both see no holder. Only one of them can
    create the file, which is what O_EXCL is for and what a check-then-write
    would get wrong exactly when two runs are launched from one script."""
    handle = os.open(hardware.LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(handle, "w") as f:
        f.write(json.dumps({"pid": os.getpid(), "command": "sweep", "since": 0.0}))

    with pytest.raises(hardware.Busy):
        with hardware.held("contrast"):
            pytest.fail("a live holder's file should not have been taken over")


def test_the_lock_file_is_finished_before_the_name_exists() -> None:
    """A file created empty and filled a moment later is readable and empty in
    between, and an empty file reads as nobody holding the unit. A second command
    arriving in that window would delete a live run's lock and sound alongside it,
    which no later check can catch. So the write happens before the name."""
    published = []
    linking = hardware.os.link

    def watched(source: str, target: str) -> None:
        published.append(
            (os.path.exists(target), json.loads(open(source).read())["command"]),
        )
        linking(source, target)

    with monkeypatched(hardware.os, "link", watched):
        with hardware.held("sweep"):
            assert hardware.holder()["command"] == "sweep"

    assert published == [(False, "sweep")]


def test_a_stale_file_is_taken_over_and_the_takeover_is_also_finished_first() -> None:
    """The way a long run ends is often a kill, so the file it leaves must not
    refuse the next run -- and the file that replaces it must arrive whole."""
    hardware.LOCK.write_text(json.dumps({"pid": 2**22 + 1, "command": "gone", "since": 0.0}))

    with hardware.held("contrast"):
        assert hardware.holder()["command"] == "contrast"
    assert hardware.holder() is None


@contextmanager
def monkeypatched(target: object, name: str, value: object) -> Iterator[None]:
    was = getattr(target, name)
    setattr(target, name, value)
    try:
        yield
    finally:
        setattr(target, name, was)


def _defaults(argv: list[str]) -> object:
    return build_parser().parse_args(argv)


def test_a_command_that_reads_takes_does_not_wait_for_the_unit() -> None:
    """Reading a block's records back while the next block is being captured is
    the ordinary way to work, so these must not queue."""
    assert _defaults(["verdict", "somewhere"]).needs_unit is False
    assert _defaults(["block", "plan.json", "records"]).needs_unit is False
    assert _defaults(["balance", "takes"]).needs_unit is False


def test_a_command_that_drives_the_unit_waits_its_turn() -> None:
    """The default is on, so a command added without a thought about this is safe
    rather than silently allowed to join a run in progress."""
    assert _defaults(["contrast", "--cc", "91"]).needs_unit is True
    assert _defaults(["selftest"]).needs_unit is True
