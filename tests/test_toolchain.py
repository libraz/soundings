"""One Python environment manager, and the file that would show a second one.

The project is rye's: `requirements.lock` and `requirements-dev.lock` are what
resolved it, and `[tool.rye] dev-dependencies` is where pytest and ruff come
from. Running `uv` against this directory resolves it again and writes `uv.lock`
-- a second lockfile, from a resolver that does not read `[tool.rye]`, so it
contains neither of the dev dependencies the suite needs. Two lockfiles for one
project means the environment a reader builds depends on which command they
happened to use, which is the one thing a lockfile exists to stop.

It arrived by habit rather than by decision: `uv run python ...` is a shorter way
to reach the venv than `rye run python ...`, and newer uv locks the project on
every `uv run`. So the detector is the file, not the intent.

Deliberately not in `.gitignore`. An ignored `uv.lock` is one that keeps being
written and nobody sees, which leaves the divergence in place and removes the
only signal that it happened.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RYE_LOCKS = ("requirements.lock", "requirements-dev.lock")


def _pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_no_second_lockfile_at_the_root() -> None:
    """`uv.lock` means a resolver that cannot see the dev dependencies has run."""
    stray = ROOT / "uv.lock"
    assert not stray.exists(), (
        "uv.lock is present. The project is locked by rye, in "
        f"{' and '.join(RYE_LOCKS)}; uv resolves it separately and does not read "
        "[tool.rye] dev-dependencies, so its lockfile omits pytest and ruff. Delete "
        "it, and reach the environment with `rye run ...` or .venv/bin/python rather "
        "than `uv run ...`, which re-locks the project every time it is called."
    )


def test_the_lockfiles_rye_writes_are_present() -> None:
    """Naming the absent file is only a check if the present ones are named too."""
    missing = [name for name in RYE_LOCKS if not (ROOT / name).is_file()]
    assert not missing, (
        f"lockfiles rye writes are missing: {missing}. Without them the previous check "
        "is satisfied by a project that has no lockfile at all, which is not the state "
        "it is asserting."
    )


def test_pyproject_hands_the_project_to_rye_and_to_nobody_else() -> None:
    """The manager is declared, so a migration shows up here rather than in a diff."""
    pyproject = _pyproject()
    assert pyproject.get("tool", {}).get("rye", {}).get("managed") is True, (
        "[tool.rye] managed is not true. The lockfiles and this suite's guard both "
        "assume rye resolves the project; if that changed, change them together."
    )
    intruders = sorted(key for key in ("uv",) if key in pyproject.get("tool", {}))
    assert not intruders, (
        f"pyproject.toml carries configuration for another manager: {intruders}. Two "
        "managers configured in one file is how a project ends up with two lockfiles."
    )
    assert "dependency-groups" not in pyproject, (
        "pyproject.toml declares [dependency-groups], which rye does not install from. "
        "Dev dependencies live under [tool.rye] so that requirements-dev.lock covers "
        "them."
    )
