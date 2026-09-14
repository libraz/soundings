# soundings

[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/libraz/soundings/blob/main/LICENSE)
[![Data](https://img.shields.io/badge/data-CC0--1.0-blue)](https://github.com/libraz/soundings/blob/main/data/LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)](https://github.com/libraz/soundings)
[![Docs](https://img.shields.io/badge/docs-soundings.libraz.net-2563eb)](https://soundings.libraz.net)

`soundings` measures the MIDI control plane and audio behaviour of hardware MIDI tone generators. It includes a Python command-line harness and a versioned archive of results from individual units.

The project records observed behaviour rather than transcribing specifications: readable addresses, accepted values, reset and power-on state, message aliases, available tones and effects, and audio measurements. Results are scoped to a specific unit and measurement setup.

The archive is published at **[soundings.libraz.net](https://soundings.libraz.net)** — address lookup, the address space drawn as one grid at two scales, the tone and insertion-effect catalogues, what a published document states shown beside the measurement it is about, and a browser emulator of the unit's control surface driven by the measurements alone. The site is built by [`soundings-homepage`](https://github.com/libraz/soundings-homepage), whose sync step reads a checkout of this repository and converts it into what the site serves. Nothing is asserted there that a record here does not establish, and an absence is rendered as *not measured* rather than as a zero or a blank.

## Scope and status

The archive currently contains measurements for one Roland SC-8850 unit, under `data/units/roland-sc8850-01/`: its address map and the bounds of each region, which offsets each block answers at, what each address accepts and whether it holds a value of its own, the power-on state, what each reset restores, where each message family is stored, the tone and insertion effect catalogues, and audio measurements — repeatability, audible differences at single addresses and folded across whole blocks, the parameters of individual insertion effect types, effect decay and modulation, and the quantity-by-setting sweeps: what a rate slot modulated at, what a parameter did to the level of each third octave, where a delay slot put the copy it returns, and what a parameter did to the balance between the two channels.

An address space repeats itself, so coverage is counted over shapes rather than addresses. Two blocks are the same shape when the same set of offsets answered in them; a stage measures one representative of each shape in full, and the fold is checked against a second block of that shape rather than assumed. `soundings complete <unit directory>` counts a unit against the bar for a finished one and says what is left.

Measurements are not claims about every unit of a model or firmware revision. `meta.json` identifies the unit, records the firmware information available from it, and describes the MIDI and audio paths used for the measurement.

Beside the archive, and deliberately outside `data/`, `documents/` holds the factual tables read out of published documentation — currently the SC-8850 and SC-88Pro owner's manuals — so that what a manufacturer stated can be held against what a unit was measured to do. Nothing in there is evidence about a unit: a row saying an address accepts `00 – 01` is evidence that a page said so.

## What is read out of the measurements

A record under `data/` says what the unit answered. It does not name an algorithm, fit a topology, estimate a coefficient, or sketch an implementation. The reading — which structure the answers are the response of, which table a byte indexes, what the era makes plausible, and what anybody outside this project has reported — lives in `inferences/`, and derivation happens in that one directory and nowhere else. References run one way: an inference cites a record, and no record cites an inference. Delete the directory and the archive is unchanged.

An archive carrying one interpretation is worth less than the measurements inside it, because every later reader has to work out what to subtract before they can use it. The test for a borderline sentence is whether a reader who disagrees with our reading could still use it. If it has to be believed rather than checked, it belongs in `inferences/`.

A claim closes on four gates, and the size of the residual is not one of them beyond a rejection bound: the residual against the span the effect itself commands, the properties the unit's own readings have, the strongest competing candidate in `candidates/` leaning where the winner does not, and no setting at which the model breaks down. Between the rejection bound and the noise floor the residual is reported and never optimised — driving it down is chasing the room and the converters. `not-chased.json` lists what a model is not asked to explain, each with the measurement that bounded it.

Two readings that both reproduce and that no measurement here separates are recorded as equivalent rather than left open. A claim standing on a prior or a community report alone is `standing_untested` and names the measurement that could have contradicted it. What is worth measuring next is derived rather than kept: the queue is the observable that would separate a standing reading from the claim, with the margin over what this rig resolves there, and an alternative nothing can separate says so and stays open with the reason.

```sh
rye run soundings inferences closed   # claims with nothing standing against them a measurement could settle
rye run soundings inferences open     # readings still standing, and what would separate them
rye run soundings inferences stale    # claims resting on figures that have moved since they were made
```

What has been read on the SC-8850 so far is its insertion-effect parameter map, class by class rather than type by type: the modulation-rate, delay-time and filter-frequency bytes as indexes into tables the manual prints, the effect-level byte as a stored seven-bit multiplier with a measured quantum, the tone-gain byte as decibels one per step, the output-pan and effect-balance bytes as pairs of ramps, and one equaliser type as a chain of four filter sections and an output gain.

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

Use `--port` to select the MIDI interface, `--device-id` to select a SysEx device ID, and `--model-id` to select which Roland model ID the three address bytes are read under — a unit answering under more than one opens a separate address space in each, and only the stages that need nothing of a particular family accept a value other than GS. All three are global options and precede the subcommand. Commands that write result data accept `--out` where applicable.

## Documentation

- [Measurement protocol](docs/en/measurement-protocol.md) — the order a unit is measured in, stage by stage, and what every record has to carry.
- [Adding a unit](docs/en/adding-a-unit.md) — what has to hold before the first stage runs on a machine that is not in the archive yet.
- [Completing a unit](docs/en/completing-a-unit.md) — the bar a unit is counted against, and what completion does not require.

These pages are the source; [soundings.libraz.net/docs/](https://soundings.libraz.net/docs/) renders them alongside a glossary and how to read a record, in English and Japanese.

## Commands

| Command | Purpose |
|---|---|
| `devices` / `selftest` | List MIDI and audio devices; verify the measurement paths. |
| `identity` / `read "40 01 30" 16` | Send an Identity Request; read an address with Roland RQ1. |
| `sweep` / `boundary` / `offsets` | Map the address space, test whether a region ends where the map says, and find addresses only a single-byte read reaches. |
| `watch-set` | Assemble the addresses a later stage watches out of what this unit has already answered, since a sweep bounds which blocks exist and not which addresses do. |
| `write-probe` / `hold-probe` / `window-probe` | Measure what an address accepts, whether it holds a value of its own, and whether it shows another address's. |
| `power-on` / `reset-probe` | Capture the state the unit powers up in; measure what each reset restores. |
| `tone-map` / `efx-map` | Identify available tones or insertion effects. |
| `alias-scan` | Locate storage affected by MIDI messages. |
| `port-send` / `port-read` | Send into a MIDI input that answers nothing, and read what it left in the unit. |
| `repeat` / `contrast` / `verdict` | Establish repeatability, measure audible differences, and judge takes recorded earlier. |
| `transfer` / `motion` / `decay` / `phase` / `vibrato` | Analyse an audio path, a time-varying effect, an effect's decay, the angle it turns each band by, or a take's pitch modulation. |
| `balance` / `balance-bands` / `arrival` | Measure what a parameter did to the level difference between the channels, to their separation band by band, and to what a take holds early against what it holds late. |
| `plan` / `block` / `efx-params` | Choose the values a block is asked at, then fold saved records into one verdict per address or per effect parameter. |
| `efx-rate` / `efx-bands` / `efx-time` | Read one insertion effect slot setting by setting, from takes already saved: what it modulated at, what it did to the level of each third octave, and where it put the copy it returns. |
| `efx-motion` / `efx-sort` | Track what an effect does over time, and sort the types by whether they stand still. |
| `index` / `complete` | List a unit's records; count a unit against the bar for a finished one. |
| `document` | Read a published document's tables into `documents/`, a page at a time — `init`, `show`, `add`, `status`. |
| `inferences` | Query what was read out of the measurements — `open`, `closed`, `stale`, `index`. |

Commands that read saved records, a document or an inference rather than the hardware — `watch-set`, `motion`, `decay`, `phase`, `verdict`, `balance`, `balance-bands`, `arrival`, `vibrato`, `plan`, `block`, `efx-rate`, `efx-bands`, `efx-time`, `efx-params`, `efx-motion`, `efx-sort`, `complete`, `index`, `document` and `inferences` — need no unit attached, so they run while another measurement holds the hardware.

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
  watch-set/                the addresses a later stage watches, assembled
                            from this unit's own sweep and offsets records
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
  balance-bands/            what it did to their separation band by band
  arrival/                  what it put in the window the direct sound arrives
                            in, against the window the effect's return does
  transfer/                 what the path did to a swept sine
  motion/  decay/  phase/   an effect's modulation, its decay, the angle it
  vibrato/                  turns each band by, a take's pitch
  efx-rate/                 what a rate slot modulated at, setting by setting
  efx-bands/                what a slot did to each third octave, setting by
                            setting
  efx-time/                 where a delay slot put the copy it returns
  efx-motion/  efx-sort/    which effect types stand still, and which do not
```

`soundings index <unit directory>` regenerates `index.json`, which names every record with the stage that wrote it and what the record says it is about.

Result files preserve the method, settings, and limits of each measurement. A claim derived from documentation or a patent is recorded as context for a measurement, not as a measured result.

What was read out of a published document is kept in its own tree, one directory per document, and never merged into a unit's records:

```text
documents/<manufacturer>-<model>-<document>-<language>/
  document.json             the document, the SHA-256 of the file that was
                            read, and which of its pages have been read
  address-map.json          the parameter address map, row by row
  block-map.json            which block begins at which address
  by-hand.json              rows the parser refused, read in by hand
  qualifications.json       a note attached to one address, saying the cell
                            is not the whole of what the page states there
  statements.json           a note printed under a table, the rows it was
                            read as reaching, and what is open
```

Only tables are copied. Prose printed around them is restated in this project's own words or named as not restated, and the documents themselves are not committed.

What was read out of both trees is kept in a third, with a schema and an index of its own, so that a measurement and a reading of one are distinguishable without reading any prose:

```text
inferences/
  not-chased.json           what a model is not asked to explain, each with
                            the measurement that bounded it
  candidates/<class>.json   a parameter class, what the era implies for it,
                            and every candidate reading of its byte
  models/<name>.json        one candidate, in the form the renderer reads
  schema/inference.json     the shape a claim has to have
  <unit>/index.json         one entry per claim, with the records it rests on
  <unit>/<claim>.json       the claim, what it rests on, what it predicted
                            before the run, the alternatives still standing,
                            and what it does not close
```

## Reproducibility

`meta.json` records the MIDI and audio interfaces, cabling, capture method, levels, and relevant unit settings. Add equivalent information when contributing data so another person can repeat the measurement or assess differences between units.

## License

The measurement harness is [MIT](LICENSE). Data under `data/` is [CC0 1.0](data/LICENSE), and so is everything under `inferences/` — [its own file](inferences/LICENSE) says the same thing, because the reason for the separation is not a legal one: a consumer harvesting the measurements must not pick up an interpretation along with them. Citation is appreciated but not required.

`documents/` is neither, and sits outside `data/` for that reason: `data/LICENSE` dedicates the measurements in that directory to the public domain, which is a dedication nobody here can make over somebody else's tables. The documents themselves are their publishers' and are not distributed here — see [documents/LICENSE](documents/LICENSE).

This project is not affiliated with, endorsed by, or connected to any instrument manufacturer. Product names identify measured equipment.
