"""Flags several commands share, and the converters that read them.

Only the flags that are genuinely the same everywhere are here. `--settle` and
`--values` mean something different in each command that takes them and carry
their own defaults and help there; collapsing those would replace a true
description with a vague one, which costs more than the repetition saves.
"""

from __future__ import annotations

import argparse

SAVE_HELP = (
    "directory to keep every take in, as float WAV with a manifest. Device time is the "
    "scarce thing here and a verdict thrown away with its audio has to be re-recorded to "
    "be asked anything else; kept takes can be measured again with the machine unplugged"
)

LEAD_IN_HELP = (
    "dBFS the lead-in must stay under; above this something was sounding before the note and "
    "the noise floor, which is the yardstick for everything else, is wrong. Absolute rather "
    "than relative to the take, or a quiet stimulus reads the same as a contaminated one"
)


def number(text: str) -> int:
    """An int in any base the prefix names, so 0x40 and 64 are both accepted."""
    return int(text, 0)


def pair(text: str) -> tuple[int, ...]:
    """'0,127' -- the two values a stimulus is sent at."""
    return tuple(int(v, 0) for v in text.split(","))


def bank_program(spec: str) -> tuple[int, int]:
    """'80' or '8:80' -- a bare number is the GM bank, which is bank 0."""
    bank, _, program = spec.rpartition(":")
    return (int(bank or 0, 0), int(program, 0))


def add_out(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", help="write the result as JSON")


def add_audio(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--audio", help="substring of the audio input device name")


def add_channel(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--channel", type=int, default=0, help="zero based MIDI channel")


def add_verify_reads(
    parser: argparse.ArgumentParser, *, default: int = 20, help: str | None = None
) -> None:
    parser.add_argument("--verify-reads", type=int, default=default, help=help)


def add_save(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--save", help=SAVE_HELP)


def add_max_lead_in(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-lead-in", type=float, default=-60.0, help=LEAD_IN_HELP)
