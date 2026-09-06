"""Settings every test in this directory runs under.

What is enforced here is that no test inherits process-wide state another test
set, in either of the two places the harness keeps some.

The first is the machine-wide lock on the unit. A test that calls the command
line entry point takes it for real otherwise, and that is wrong in both
directions: a run in progress refuses the test, and a test that got the lock
first would refuse the run -- which is the failure the lock exists to prevent,
arriving from the harness that checks it.

Caught by a suite failing while a sweep was in flight, having been refused by the
sweep's own lock. The refusal was correct; the test asking for it was not.

The second is the invocation a record takes its identity from. It is held for the
process so that a writer does not have to be handed it, which means a test that
runs the entry point leaves it set for whatever runs next: a record written by a
later test claimed an earlier one's stage and device ID, and looked entirely
plausible doing it.
"""

from __future__ import annotations

import pytest

from soundings import hardware, record


@pytest.fixture(autouse=True)
def lock_away_from_the_unit(tmp_path, monkeypatch):
    """Point the lock at this test's own directory, never at the shared one."""
    monkeypatch.setattr(hardware, "LOCK", tmp_path / "unit.lock")


@pytest.fixture(autouse=True)
def claim_no_invocation():
    """Start each test outside a run, and leave it outside one."""
    record.invoked(None)
    yield
    record.invoked(None)
