"""One unit and one interface, so one command drives them at a time.

The protocol says hardware stages are serial. It said so before the rule was
broken, which is why this exists: a written rule is followed by whoever read it,
and the driver scripts that run for hours unattended are written by whoever is in
a hurry.

**The second run is the one that cannot tell.** Its notes play, its takes record,
its records come out looking like every other record. The damage is in the *first*
run's takes, where the second run's notes land in the lead-in -- which is where
the noise floor is measured, and the noise floor is the yardstick every figure in
the archive is judged against. Measured once: two sweeps shared this unit for
twenty-four minutes and forty-four records had to be thrown away. The only reason
it was caught at all is that a take's lead-in guard names the case out loud.

The lock is machine-wide rather than per checkout, because a second checkout --
a git worktree, say -- reaches the same unit through the same cable.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

LOCK = Path(tempfile.gettempdir()) / "soundings-unit.lock"


class Busy(Exception):
    """Another command is driving the unit."""


def _alive(pid: int) -> bool:
    """Whether a process is still there, without touching it.

    Signal 0 asks the kernel to do the permission and existence checks and then
    deliver nothing, which is the question this needs. A process owned by someone
    else answers PermissionError, and that still means it exists.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def holder() -> dict | None:
    """Who holds the lock, or None if nobody does.

    A lock file left by a process that has since died is not a holder. That case
    is ordinary rather than exceptional: the way a long run ends is often a kill,
    and a stop that leaves the next run refused would be a worse failure than the
    one this guards against.
    """
    try:
        found = json.loads(LOCK.read_text())
    except (FileNotFoundError, ValueError):
        return None
    pid = found.get("pid")
    if not isinstance(pid, int) or not _alive(pid):
        return None
    return found


def _refusal(found: dict) -> str:
    ran_for = time.time() - float(found.get("since", 0.0))
    return (
        f"another command is driving the unit: pid {found.get('pid')} running "
        f"{found.get('command', 'something')}, started {ran_for / 60:.0f} minutes ago.\n"
        "One unit and one interface means two of these do not share; the second one's "
        "notes land in the first one's noise floor and both sets of numbers stop meaning "
        "anything, with nothing wrong on the second one's side to see.\n"
        "Stop it and wait for it to be gone before starting this."
    )


def _claim(mine: str) -> None:
    """Put a finished file at LOCK, or raise FileExistsError if it is taken.

    The contents are written before the name exists, rather than after. Creating
    the file and filling it a moment later leaves it readable and empty in
    between, and an empty file is what `holder` reads as nobody -- so a second
    command arriving inside that window deletes the lock of the run it is
    standing next to and drives the unit alongside it, which is the whole failure
    this module exists to stop. `os.link` refuses a name that is taken and
    publishes the file already complete, so there is no such window.
    """
    handle, staging = tempfile.mkstemp(dir=LOCK.parent, prefix=f"{LOCK.name}.")
    try:
        with os.fdopen(handle, "w") as f:
            f.write(mine)
        os.chmod(staging, 0o644)
        os.link(staging, LOCK)
    finally:
        os.unlink(staging)


@contextmanager
def held(command: str) -> Iterator[None]:
    """Hold the unit for the duration, refusing if anything else has it."""
    found = holder()
    if found is not None:
        raise Busy(_refusal(found))
    mine = json.dumps({"pid": os.getpid(), "command": command, "since": time.time()})
    try:
        _claim(mine)
    except FileExistsError:
        # Either a live holder that appeared between the check and here, or the
        # stale file `holder` just declined to believe in. Re-ask, and take it
        # over only when the answer is still nobody.
        found = holder()
        if found is not None:
            raise Busy(_refusal(found)) from None
        LOCK.unlink(missing_ok=True)
        _claim(mine)
    try:
        yield
    finally:
        _release()


def _release() -> None:
    """Drop the lock, and only ever our own.

    Checked rather than assumed: a run whose lock was broken as stale while it was
    somehow still alive would otherwise delete the file belonging to whoever took
    it next, and the failure that follows is the one this module exists to stop.
    """
    found = None
    try:
        found = json.loads(LOCK.read_text())
    except (FileNotFoundError, ValueError):
        return
    if found.get("pid") == os.getpid():
        LOCK.unlink(missing_ok=True)
