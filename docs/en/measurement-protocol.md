# Measurement protocol

The order in which a unit is measured. Each stage is aimed by the files an
earlier stage wrote, so the order is part of the method rather than a
convenience: a stage run out of turn measures something the archive cannot
describe.

A claim that compares two units rests on both having been measured this way. A
unit measured differently can still be described on its own, but it cannot join
a comparison.

## Before any stage

Every command verifies the MIDI path before it writes anything. Run the
self-test on its own first, so a failure is read as a failure of the path rather
than of the stage that happened to be running:

```sh
rye run soundings selftest --audio "<audio interface>"
```

A self-test that passes says the harness can read the unit. It does not say the
unit is safe to probe. Two constraints hold for every stage:

- **Keep bulk requests at or below 64 bytes.** A larger request has been
  measured to stop a unit responding entirely, recoverable only by a power
  cycle.
- **Ask a known-good address periodically.** A unit that has stopped answering
  reads as a space of absent addresses, so a probe that treats silence as
  absence finishes successfully having measured nothing.

## Stages

### 1. Identity

```sh
rye run soundings identity
```

Records what the unit answers to an Identity Request. The reply goes into
`meta.json` verbatim, together with the rear-panel rating plate, the measurement
chain, and the unit's settings. Everything measured afterwards is scoped to that
record.

### 2. Address map

```sh
rye run soundings sweep --out data/units/<unit-id>/sweep/whole-map.json
```

Asks the unit which addresses answer. The map is what the unit answered, not
what a document lists, and it is what the following stages are aimed at. A
region absent from the map is absent from everything measured afterwards.

### 3. Power-on state

```sh
rye run soundings power-on --map data/units/<unit-id>/sweep/whole-map.json \
  --unit-id <unit-id> --out data/units/<unit-id>/power-on/whole-map.json
```

Reads the space twice over and writes nothing. What this produces is the state
the unit powers up in.

**This is the last stage that may be run before a stage that writes.** Stages 1
to 3 send reads only, so the capture is still a power-on capture when it is
taken after the sweep. From stage 4 onwards the unit is written to, and the
power-on state is then unavailable until the next power cycle. A map costs a
sweep to rebuild; this capture cannot be rebuilt at all.

The run cannot tell a unit fresh from the mains switch from one an earlier run
wrote to, because both answer a read the same way. What it records is therefore
the operator's statement of what was done beforehand, marked as a claim, beside
the one thing the run can say: that it wrote nothing itself.

Addresses where the two reads disagreed are listed rather than resolved.
Deciding between them would need a third read, and a wrong decision is invisible
afterwards -- the byte simply reads as having been changed by whichever reset is
measured against this capture next. Every later reset measurement is compared
against this file.

### 4. Windows

```sh
rye run soundings window-probe --stores <addr> <addr> <candidate>...
```

Establishes which blocks hold values of their own and which show another
block's. Runs before any stage interprets what an address holds, because a
window cannot be told from storage by writing to it and reading it back: the
write points the window and the read follows it, so the address answers with
exactly what it was given.

A block found to be a window is named in the records that describe it, and is
skipped by the stages that would otherwise measure the same store many times
over under names that keep none of it.

### 5. Accepted values

```sh
rye run soundings write-probe --map data/units/<unit-id>/sweep/whole-map.json \
  --resume --out data/units/<unit-id>/write-probe/whole-map.json
```

Writes each byte on its own and reads it back, recording what each address
accepts, clamps or refuses, and restoring it afterwards. Regions are written as
they finish, so an interrupted run resumes from the last whole region.

An address that answers no single-byte read is never written to, since there
would be nothing to put back. Those are recorded as skipped: which of them are
undefined, rather than reachable only as part of a larger block, is not settled
by this stage.

### 6. Independent storage

```sh
rye run soundings hold-probe --map data/units/<unit-id>/sweep/whole-map.json \
  --out data/units/<unit-id>/hold-probe/whole-map.json
```

Gives neighbouring addresses different values before reading any of them back,
which separates a run of addresses backed by one cell from a run of distinct
ones. The order is the method: written apart from read, an address that only
shows the last value written near it answers with that one value while its
neighbours answer with their own.

This does not find a block that mirrors another block, which is what stage 4
asks.

### 7. Aliases

```sh
rye run soundings alias-scan --map data/units/<unit-id>/sweep/whole-map.json \
  --kind cc --out data/units/<unit-id>/alias-scan/cc-ch1.json
```

Sends each message of a family and diffs the address space around it, which
locates the storage a message reaches. Run once per family: `cc`, `nrpn`,
`rpn`, `channel`, `drum-nrpn`, and `address` for writes by SysEx.

**Watch the whole map**, which is what `--prefix` does by default. A scan
restricted to one block bounds every negative finding it produces to that block,
and the record of "this message is stored nowhere" then means "nowhere in the
part of the space that was looked at".

`--kind address` writes to addresses an earlier scan attributed, which asks
whether a location has a third way in. It takes them from that unit's own
records:

```sh
rye run soundings alias-scan --map data/units/<unit-id>/sweep/whole-map.json \
  --kind address --addresses-from data/units/<unit-id>/alias-scan/cc-ch1.json \
  --out data/units/<unit-id>/alias-scan/sysex-ch1.json
```

### 8. Resets

```sh
rye run soundings reset-probe --baseline data/units/<unit-id>/power-on/whole-map.json \
  --map data/units/<unit-id>/sweep/whole-map.json \
  --write-probe data/units/<unit-id>/write-probe/whole-map.json \
  --out data/units/<unit-id>/reset-probe/whole-map.json
```

Breaks the state before each reset and reads the space afterwards, which
measures what each reset restores rather than what it is documented to restore.
Marks are placed only in addresses the write probe found take any value: an
address that clamps may keep what it had, and a byte that was never broken says
nothing about the reset.

Every reset is preceded by the same one, so the results are comparable with each
other rather than each being read against wherever the previous reset left the
unit.

### 9. Tones and effects

```sh
rye run soundings tone-map --out data/units/<unit-id>/tone-map/map-select-0.json
rye run soundings efx-map --out data/units/<unit-id>/efx-map/types.json
```

Asks for each tone and each insertion effect and reads back whether it was
taken. A map that samples before sweeping carries the sampling as a caveat;
`--exhaustive` asks every bank for all 128 programs and is the only form with
nothing to caveat.

### 10. Repeatability, before any audio comparison

```sh
rye run soundings repeat --audio "<audio interface>" \
  --out data/units/<unit-id>/repeat/program-0.json
```

Plays one stimulus twice and measures how closely the two takes agree. That
figure is the floor of every later difference measurement: below it, "no
difference" and "no resolution to see one" are the same reading.

The floor is a property of the unit and the chain together, so it is measured
per unit and re-measured whenever the chain changes.

### 11. Audible differences

```sh
rye run soundings contrast --cc 91 --audio "<audio interface>" \
  --out data/units/<unit-id>/contrast/cc-91.json
```

Compares two settings of one parameter against the repeatability floor, which
separates a parameter the unit stores from one it is heard through. A parameter
can be stored and not audible; the address-space stages cannot tell those apart
and this one can.

`transfer`, `motion` and `decay` measure an analogue path, a time-varying effect
and an effect's tail. `motion` and `decay` read takes and need no unit attached.

### 12. Whole blocks

```sh
rye run soundings plan data/units/<unit-id>/write-probe/whole-map.json "40 11" \
  --out plan-40-11.json
rye run soundings block plan-40-11.json <plain records> --gesture <gesture records> \
  --balance <balance record> --out data/units/<unit-id>/block/40-11.json
```

Asks every address in one block whether changing it changes the sound, rather
than the addresses a manual made interesting. Two passes: a plain note first, and
then the addresses it could not hear asked again with messages moved after the
setting, since a parameter that only decides whether a message is received has
nothing to receive under a bare note.

The pair each address is asked at comes from the write probe rather than from a
document, and the block is counted against the plan rather than against the
records that happen to be present -- a directory holding forty-five records of a
forty-seven address block reads exactly like a complete answer.

**Ask the block on the channel the block itself says it listens to.** That byte
is in the block and the unit will tell you. A stimulus on any other channel asks
a part the address does not address, and every address in the block answers
inaudible.

**A pair chosen from the accepted range alone cannot ask a channel byte.** The
write probe reports what the address takes; it does not know which channel the
stimulus is on. Where both values of the pair move the part away from the note
being played, both settings are silent and the run refuses to measure -- which
reads as a failed capture rather than as a badly chosen pair. Such an address is
asked at the part's own channel against off.

## Reading an audible verdict

A verdict can be reached three ways: the shape changed, the level changed, or one
setting's takes agreed with each other far worse than the other's. The first two
measure the sound. The third measures a spread, and a spread can be produced by
things that are not the parameter.

**A verdict resting on agreement alone requires the steadier setting to be steady
in absolute terms.** Against the plain note's own figure for the same address, not
against the other setting. A gate leaves the setting it blocks repeating as a
plain note does; two draws from a free-running phase can sit any distance apart
and mean nothing.

**A stimulus has to establish its own repeatability before its verdicts count.**
Stage 10 measures the floor for one stimulus. A stimulus that sounds more than one
voice has a relative phase the trigger cannot fix, and where the trigger scatter
is wider than a period of the note being played, the phase is free from take to
take and the level moves with it. Measured on one unit: a plain note agreed within
56 to 62 dB while a two-note stimulus landed between 1 and 66 dB, with 84 per cent
of its figures worse than 10 dB. Nothing it reported survived being asked twice.
Doubling the takes did not help.

**A verdict witnessed by exactly one stimulus is asked again before it is
published.** Every such verdict in one three-block sweep failed to reproduce,
including the ones whose first pass had looked soundest. Re-asking is cheap
against the cost of publishing a result that is not there.

**Three blocks holding the same parameters are three draws.** A verdict that
appears in one block and not the others is either a difference between parts,
which parameters do not usually have, or a measurement that did not reproduce.
Deciding between those needs a re-ask, not an argument.

## What every record carries

A stage that finds nothing is only worth reading if the run could have found
something. Each record therefore states the controls the run carried:

- **A positive control**, sent before the first stimulus and after the last. A
  control sent once says nothing about the rest of the run.
- **Whether the family under test arrived at all.** The positive control is a
  control change and cannot show that a message of another kind got through.
  Where this is absent, every negative in the run is about the path rather than
  about the unit.
- **Whether a landing outside the block written to would have been seen.** A run
  that writes to an address and sees only that address move is evidence nothing
  mirrors it, but only if a write that did land elsewhere would have been
  reported.
- **What was restored**, and what was skipped. What a run stopped covering is
  the caveat on everything it says.

## Running a stage that takes hours

The stages that sweep a whole block run for hours and are driven from
`procedures/`, which is tracked. A driver specific to one unit is a working note
and stays out; one that is part of the protocol belongs here, because a second
unit measured by a script nobody else has is not measured the same way.

```sh
./procedures/part-block.sh "40 11" 0        # block, and the channel it listens on
./procedures/stop-sweep.sh                  # stop it, and prove it stopped
```

**One address at a time as a fresh process, which is not an implementation
detail.** The two failures a long unattended run hits both cost the whole
address, and one of them ends the process outright, so asking again as a new
process is the only recovery available for it.

**Each stage is aimed by the previous stage's own record.** What the gesture pass
asks is what `soundings block` named as still open, read from the file rather than
written by hand: a list written twice is a judgement made twice, and the two
drift. A stage refuses to start when the record it should have been aimed by is
missing, because an empty list and a crashed read send the same thing to the next
pass -- nothing -- and the run then reports a pass with nothing to do.

## Measurement chain

The chain is part of the protocol. Interfaces, gain, cabling and capture method
stay identical across units, or a level comparison between two units cannot be
told from a difference between two chains.

Where a chain has to change, the difference is recorded in `meta.json` under
`measurement_chain`, and level comparisons involving that unit are marked as not
established.

Hardware stages are serial. One audio interface and one unit mean a background
run sounds during a foreground capture, and what survives is a set of numbers
whose reproducibility has quietly collapsed.

That rule was written before it was broken, so writing it down is not enough and
the harness enforces it: a command that drives the unit takes an exclusive lock
and a second one refuses to start. The refusal is the point. From inside the
second run nothing looks wrong -- its notes play, its takes record, its records
come out ordinary -- and the only trace is in the first run's takes, where the
other run's notes land in the lead-in that the noise floor is measured in.

**Stopping a run means stopping every process of it, and proving it.** The
drivers nest, each spawning one stage at a time, so killing the top of the tree
leaves a parent that starts the next stage as soon as the child being killed
dies. Kill parents before children, repeat until a scan comes back empty, and
treat a stop that cannot prove it stopped as a stop that did not happen. A
survivor is not idle; it is still playing notes into the unit.

**Two failures on a long unattended run are the machine refusing a resource for a
moment rather than anything about the address**: a capture that comes back short
of what was asked for, and the MIDI layer declining to create a client at all.
Both cost the whole address, and the second ends the process outright, so both
are retried by asking again as a fresh process. How many retries were needed is
recorded, because a rate that climbs is a fact about the chain.
