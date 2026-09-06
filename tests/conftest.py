"""Settings every test in this directory runs under.

The one thing enforced here is that no test ever takes the machine-wide lock on
the unit. A test that calls the command line entry point takes it for real
otherwise, and that is wrong in both directions: a run in progress refuses the
test, and a test that got the lock first would refuse the run -- which is the
failure the lock exists to prevent, arriving from the harness that checks it.

Caught by a suite failing while a sweep was in flight, having been refused by the
sweep's own lock. The refusal was correct; the test asking for it was not.
"""

from __future__ import annotations

import pytest

from soundings import hardware


@pytest.fixture(autouse=True)
def lock_away_from_the_unit(tmp_path, monkeypatch):
    """Point the lock at this test's own directory, never at the shared one."""
    monkeypatch.setattr(hardware, "LOCK", tmp_path / "unit.lock")
