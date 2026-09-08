# soundings

[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/libraz/soundings/blob/main/LICENSE)
[![Data](https://img.shields.io/badge/data-CC0--1.0-blue)](https://github.com/libraz/soundings/blob/main/data/LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)](https://github.com/libraz/soundings)

`soundings` measures the MIDI control plane and audio behaviour of hardware MIDI tone generators. It includes a Python command-line harness and a versioned archive of results from individual units.

The project records observed behaviour rather than transcribing specifications: readable addresses, accepted values, reset and power-on state, message aliases, available tones and effects, and audio measurements. Results are scoped to a specific unit and measurement setup.

## Scope and status

The archive currently contains measurements for one Roland SC-8850 unit, under `data/units/roland-sc8850-01/`: its address map and the bounds of each region, what each address accepts and whether it holds a value of its own, the power-on state, what each reset restores, where each message family is stored, the tone and insertion effect catalogues, and audio measurements — repeatability, audible differences at single addresses and folded across whole blocks, the parameters of individual insertion effect types, and effect decay and modulation.

Measurements are not claims about every unit of a model or firmware revision. `meta.json` identifies the unit, records the firmware information available from it, and describes the MIDI and audio paths used for the measurement.

## Install

Requirements:

- Python 3.11 or later, managed with [Rye](https://rye.astral.sh/)
- A bidirectional MIDI interface for MIDI measurements
- The target unit; audio measurements also require an audio interface

```sh
rye sync
rye run soundings devices
```

List the available commands with:

```sh
rye run soundings --help
```

## Before measuring a unit

Run the self-test before collecting data:

```sh
rye run soundings selftest --audio "<audio interface>"
```

It checks SysEx reply checksums, repeated reads of a known address, and the audio capture timeline. This does not establish that a target unit is safe to probe. Read the command help and existing measurements before using a command against unfamiliar hardware; some requests can produce no reply or leave a unit requiring a power cycle.

Use `--port` to select the MIDI interface and `--device-id` to select a SysEx device ID. Both are global options and precede the subcommand. Commands that write result data accept `--out` where applicable.

## Commands

| Command | Purpose |
|---|---|
| `devices` / `selftest` | List MIDI and audio devices; verify the measurement paths. |
| `identity` / `read "40 01 30" 16` | Send an Identity Request; read an address with Roland RQ1. |
| `sweep` / `boundary` / `offsets` | Map the address space, test whether a region ends where the map says, and find addresses only a single-byte read reaches. |
| `write-probe` / `hold-probe` / `window-probe` | Measure what an address accepts, whether it holds a value of its own, and whether it shows another address's. |
| `power-on` / `reset-probe` | Capture the state the unit powers up in; measure what each reset restores. |
| `tone-map` / `efx-map` | Identify available tones or insertion effects. |
| `alias-scan` | Locate storage affected by MIDI messages. |
| `port-send` / `port-read` | Send into a MIDI input that answers nothing, and read what it left in the unit. |
| `repeat` / `contrast` / `verdict` | Establish repeatability, measure audible differences, and judge takes recorded earlier. |
| `transfer` / `motion` / `decay` / `vibrato` | Analyse an audio path, a time-varying effect, an effect's decay, or a take's pitch modulation. |
| `balance` | Measure what a parameter did to the level difference between the channels. |
| `plan` / `block` / `efx-params` | Choose the values a block is asked at, then fold saved records into one verdict per address or per effect parameter. |
| `efx-motion` / `efx-sort` | Track what an effect does over time, and sort the types by whether they stand still. |
| `index` / `complete` | List a unit's records; count a unit against the bar for a finished one. |

Commands that read saved records — `motion`, `decay`, `verdict`, `balance`, `vibrato`, `plan`, `block`, `efx-params`, `efx-motion`, `efx-sort`, `complete` and `index` — need no unit attached, so they run while another measurement holds the hardware.

## Data format

Each unit directory holds one subdirectory per measurement stage, named after the command that wrote the records in it. A record is named after what it is about — the address, block, effect type or controller the run asked at — and a run covering the whole address map rather than one part of it is `whole-map.json`.

```text
data/units/<manufacturer>-<model>-<n>/
  meta.json                 unit identity and measurement chain
  measurements.json         documented observations and spot checks
  index.json                generated listing of everything below

  sweep/                    readable address regions
  boundary/                 whether a region ends where the map says it does
  offsets/                  addresses only a single-byte read reaches
  write-probe/              accepted writes and read-back results
  hold-probe/               whether neighbouring addresses hold their own values
  window-probe/             addresses that show another address's value
  power-on/                 captured state after power-on
  reset-probe/              state changes caused by resets
  tone-map/                 available tones
  efx-map/                  available insertion effects
  alias-scan/               message-to-address mappings
  port-send/  port-read/    what a MIDI input that answers nothing leaves behind

  repeat/                   how well the unit repeats a take, which every
                            audio comparison below is measured against
  contrast/                 audio comparison at one address or controller
  block/                    those comparisons folded into one verdict per address
  efx-params/               one insertion effect type's parameters
  balance/                  what a parameter did to the level between channels
  transfer/                 what the path did to a swept sine
  motion/  decay/  vibrato/ an effect's modulation, its decay, a take's pitch
  efx-motion/  efx-sort/    which effect types stand still, and which do not
```

`soundings index <unit directory>` regenerates `index.json`, which names every record with the stage that wrote it and what the record says it is about.

Result files preserve the method, settings, and limits of each measurement. A claim derived from documentation or a patent is recorded as context for a measurement, not as a measured result.

## Reproducibility

`meta.json` records the MIDI and audio interfaces, cabling, capture method, levels, and relevant unit settings. Add equivalent information when contributing data so another person can repeat the measurement or assess differences between units.

## License

The measurement harness is [MIT](LICENSE). Data under `data/` is [CC0 1.0](data/LICENSE). Citation is appreciated but not required.

This project is not affiliated with, endorsed by, or connected to any instrument manufacturer. Product names identify measured equipment.
