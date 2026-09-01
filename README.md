# soundings

[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data: CC0](https://img.shields.io/badge/data-CC0--1.0-blue.svg)](data/LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)](#requirements)

**What hardware MIDI tone generators actually answer, measured from the machines themselves.**

Taking soundings means dropping a line to chart water you have no map of. This
project does that to the control planes of hardware tone generators: it writes to
an address, reads it back, and records what came out. The result is a machine-readable
archive of measured behaviour across GM, GM2, GS and XG — value ranges, clamping,
power-on defaults, which parameters alias which, and the algorithms behind the
effects.

## Why

The specifications exist on paper. What a given machine *does* is a separate
question, and it is the one that matters when you are implementing playback: an
address the manual documents may be write-only, a value it gives a range for may
clamp somewhere else, and a parameter reachable three different ways may or may
not land in the same place. None of that is written down anywhere machine-readable.

Software that plays these files is largely bad at it not because the samples are
different but because the control plane is ignored. This archive is the reference
that would let an implementation stop guessing.

## What is here, and what it claims

**Every published row is a value a machine returned.** Nothing is transcribed from
a manual, and nothing is inferred from other people's firmware analysis. Where a
hypothesis came from published literature or a patent, that is cited; where it
came from anywhere else, it is used to decide what to measure and never to fill
in an answer.

**Claims are about a unit, not a product line.** One machine of one model at one
firmware revision is one data point, and the archive says so. A second unit that
answers differently is not a contradiction — it is the second data point, and both
are kept.

```
data/
  units/<manufacturer>-<model>-<n>/
    meta.json          the individual: identity, firmware, measurement chain
    measurements.json  what it answered
tools/                 (nothing yet; the harness is the Python package)
src/soundings/         the measurement harness
docs/                  how to read the data, and how to reproduce it
```

## Status

Early. The measurement chain is built and verified against one unit; the address
space sweep has not been run yet.

| unit | control plane | effects |
|---|---|---|
| Roland SC-8850 | chain verified, sweep pending | not started |

## Requirements

- Python 3.11+, and [rye](https://rye.astral.sh/) to manage it
- A MIDI interface with **both** directions — every probe here is a round trip
- The hardware, and an audio interface for the effect measurements

```sh
rye sync
rye run soundings devices
```

## Run the selftest before trusting anything

Both ways this measurement can break are silent.

A MIDI path that drops bytes yields a *missing row*, not an error, so the address
space comes out with holes that read as absences. A capture path that drops
samples yields a recording whose levels are all correct and whose timeline is
compressed, so every decay time comes out short by the same factor and nothing
inside the data disagrees with anything else.

Neither is detectable afterwards from the measurements. They have to be excluded
first, against a stimulus whose answer is already known.

```sh
rye run soundings selftest --audio "<your interface>"
```

It verifies reply checksums, reads one address a hundred times and requires every
answer identical, then plays a note once a second and requires the recording to
agree that they were one second apart.

```sh
rye run soundings identity
rye run soundings read "40 01 30" 16
```

## Reproducing a measurement

Each unit's `meta.json` records the full chain — wiring, levels, knob positions,
capture method — because a measurement that cannot be repeated is an anecdote.
Settings are chosen for reproducibility where there is a choice: a volume control
is recorded at its hard stop, because that is the only position anyone else can
match exactly without markings.

## Licence

The harness is MIT. **The data is CC0** — these are measurements of physical
facts, and asserting a licence over them would imply an ownership that does not
exist. A citation is welcome and is not a condition.

This project is not affiliated with, endorsed by, or connected to any instrument
manufacturer. Model names identify the equipment measured.
