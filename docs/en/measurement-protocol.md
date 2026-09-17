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

### Which space the run is addressed to

Three address bytes name nothing on their own. What they name is decided by the
model ID they are sent under, and **a unit may answer under more than one**: one
in this archive has a display block reachable only outside the space the rest of
it lives in. Each model ID is a separate address space, and the same three bytes
in two of them are two different parameters.

```sh
rye run soundings --model-id 0x45 read "10 00 00"
```

Before the subcommand, like `--port` and `--device-id`, because it says what the
run is addressed to rather than what one stage does. It defaults to GS.

**Only the stages that need nothing of a family accept another value** --
`sweep`, `boundary`, `offsets`, `power-on`, `write-probe`, `window-probe`,
`hold-probe`, and the two commands that just look. Everything else refuses it
and says so, because what it sends is one family's own: a reset message, a fixed
effect address, a bank-and-program convention. Sent into another space those
write to whatever those bytes happen to mean there, and the run would read the
answer back as a measurement.

Every record carries the model ID it asked under, beside the device ID. A record
written before the field existed does not, which is not an unknown: nothing here
could send any model ID but GS until the flag existed.

The path is proved in the space the probe address is known to answer in, not in
the one being explored. A space that answers nothing is a finding for the stage
to report, not a reason for its self-test to refuse to start it -- so `selftest`
takes `--probe` for proving a second space deliberately.

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
what a document lists, and it is what the following stages are aimed at.

The sweep asks a named set of third bytes under each block, so it finds where a
block begins only if a block begins where it looked. **What it produces is a
list of blocks, not a list of addresses**, and reading it as the latter puts a
hole in every stage aimed at it. Stage 3 is what turns it into the second.

### 3. Offsets, and the shapes they fall into

```sh
rye run soundings offsets "40 10 00:128" ... --canary <addr> \
  --resume --out data/units/<unit-id>/offsets/whole-map.json
```

Asks every offset of every block the sweep found, one single-byte read each.
Reading forward from a region's end stops at the first address that answers
nothing, so a run of live addresses past a silent gap is invisible to it; only
asking every offset finds those.

Blocks whose answering offsets are the same set are the same **shape**, and an
address space repeats: on the first unit measured this way, 461 blocks held
twelve shapes and one of them appeared 204 times. Every stage after this one is
aimed at one representative per shape, and reports its coverage as a fraction
over shapes. Measuring all of them measures the same thing over again and
produces a figure that says nothing about how much is known.

The fold is checked rather than assumed: beside the representative, one further
block of the shape is asked and the agreement recorded. Where the two disagree
the shape was drawn wrongly and is split.

**Sends reads only, so it belongs before the power-on capture.** Ordering it
after a stage that writes would cost that capture, which cannot be rebuilt.

### 4. What each later stage watches

```sh
rye run soundings watch-set data/units/<unit-id> \
  --out data/units/<unit-id>/watch-set/reachable.json
rye run soundings watch-set data/units/<unit-id> --keep-windows \
  --out data/units/<unit-id>/watch-set/with-windows.json
rye run soundings watch-set data/units/<unit-id> --one-at-a-time \
  --out data/units/<unit-id>/watch-set/one-at-a-time.json
```

Asks the records rather than the unit. Every stage from here on takes a `--map`,
and that map is what bounds every negative finding the stage goes on to make --
so which map it is given is part of what its records mean.

**It is the union of both reads, and it has to be: neither contains the other.**
A region read returns one reply for a run of addresses and reaches addresses that
answer nothing when asked on their own. A single-byte read asks one address and
reaches addresses that begin where no region does, which is what stage 3 was
written to find. Giving a stage either half alone leaves a message able to land
where nothing is looking, and its record then says the message is stored
nowhere. On the first unit measured this way the sweep's map missed 3650
addresses that answer a read, and five stages were aimed at it before anyone
noticed.

Blocks measured to be a window are left out, since a value landing in one lands
in the store it points at and that store is already watched. `--keep-windows`
puts them back, for a capture of what each address held rather than of where a
message goes. `--one-at-a-time` holds only what a single read answers, each its
own region: a reply to a single read is answered for the address it was asked
about or not at all, so no byte in it can land on the address below its own.

A block this unit's records name as answering a region read with its first byte
and zeros after it is put in one address at a time, however the rest of the map is
asked, and the map says which blocks those were. The blocks are read from the
records and held nowhere in the code: which ones answer that way is one unit's
answer, and a list of them beside the builder would arrive at the next unit as an
assumption about it.

Nothing here is measured, and the file says so. It is rebuilt from the unit's own
records whenever those change, and it names them -- so a stage that has just
learnt something about how a block answers rebuilds the map before it runs again.
The command takes no unit and holds none, so it can be rebuilt while the hardware
is busy with something else.

### 5. Power-on state

```sh
rye run soundings power-on --map data/units/<unit-id>/watch-set/with-windows.json \
  --unit-id <unit-id> --out data/units/<unit-id>/power-on/regions.json
rye run soundings power-on --map data/units/<unit-id>/watch-set/one-at-a-time.json \
  --unit-id <unit-id> --out data/units/<unit-id>/power-on/one-at-a-time.json
```

Reads the space twice over and writes nothing. What this produces is the state
the unit powers up in.

**Taken both ways, after one power cycle, because the two reach different
addresses.** A region read reaches the ones that answer no single read; a single
read reaches the ones a short reply would otherwise misplace. One unit answers a
thirty-two byte request with thirty bytes, reproducibly and with a checksum that
verifies -- and the two addresses it leaves out are not the two that go silent
when the same block is asked an offset at a time. Such a reply cannot be laid
down over the addresses asked for, so it is refused and the region is recorded as
answered short. The baseline is the union of the two captures, and `soundings
complete` counts it that way.

**A reply of the length that was asked for can still hold one byte of data.** Two
blocks of that same unit answer a region read with a reply as long as the request
and a checksum that verifies, carrying the value of the address it starts at and
00 in every byte after it -- while all 128 offsets of each block, asked one at a
time, answer with their own offset. Nothing refuses a reply like that, since the
length is what makes a short one refusable, so every byte of it is published as a
value; and where the region's first byte is 00 anyway there is nothing to see.
What makes it visible is the comparison and not the run: hold every value in a
region capture against a single-byte read of the same address. On this unit that
is 37240 addresses, 220 of them disagree, and all 220 are in those two blocks.

**Do that comparison before a later stage rests on either capture.** A reset probe
took its baseline from the region read and chose each mark against a single read,
which wrote the baseline's own value into 220 of the 224 addresses it marked
there. All three resets came back restoring every one of them, which is the only
answer that comparison could return.

**This is the last stage that may be run before a stage that writes.** Stages 1
to 4 send reads or nothing at all, so the capture is still a power-on capture
when it is taken after them. From stage 6 onwards the unit is written to, and the
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

### 6. Windows

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

### 7. Accepted values

```sh
rye run soundings write-probe --map data/units/<unit-id>/watch-set/reachable.json \
  --resume --out data/units/<unit-id>/write-probe/whole-map.json
```

Writes each byte on its own and reads it back, recording what each address
accepts, clamps or refuses, and restoring it afterwards. Regions are written as
they finish, so an interrupted run resumes from the last whole region.

An address that answers no single-byte read is never written to, since there
would be nothing to put back. Those are recorded as skipped: which of them are
undefined, rather than reachable only as part of a larger block, is not settled
by this stage.

### 8. Independent storage

```sh
rye run soundings hold-probe --map data/units/<unit-id>/watch-set/reachable.json \
  --out data/units/<unit-id>/hold-probe/whole-map.json
```

Gives neighbouring addresses different values before reading any of them back,
which separates a run of addresses backed by one cell from a run of distinct
ones. The order is the method: written apart from read, an address that only
shows the last value written near it answers with that one value while its
neighbours answer with their own.

This does not find a block that mirrors another block, which is what stage 5
asks.

### 9. Aliases

```sh
rye run soundings alias-scan --map data/units/<unit-id>/watch-set/reachable.json \
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
rye run soundings alias-scan --map data/units/<unit-id>/watch-set/reachable.json \
  --kind address --addresses-from data/units/<unit-id>/alias-scan/cc-ch1.json \
  --out data/units/<unit-id>/alias-scan/sysex-ch1.json
```

### 10. Resets

```sh
rye run soundings reset-probe --baseline data/units/<unit-id>/power-on/one-at-a-time.json \
  --map data/units/<unit-id>/watch-set/reachable.json \
  --write-probe data/units/<unit-id>/write-probe/whole-map.json \
  --out data/units/<unit-id>/reset-probe/whole-map.json
```

Breaks the state before each reset and reads the space afterwards, which
measures what each reset restores rather than what it is documented to restore.
A mark is a value the address does not already hold, and it is taken from what
the write probe measured that address to accept. Read as "any address that takes
any value" this left out every byte with a range -- which is every byte a
document gives a function to, since the ones that accept anything are mostly the
ones nobody defined. A byte measured to take one value has no mark at all and is
left out: writing what it holds breaks nothing, and a reset leaving it alone
would then read as a reset restoring it.

Every reset is preceded by the same one, so the results are comparable with each
other rather than each being read against wherever the previous reset left the
unit.

### 11. Tones and effects

```sh
rye run soundings tone-map --out data/units/<unit-id>/tone-map/map-select-0.json
rye run soundings efx-map --out data/units/<unit-id>/efx-map/types.json
```

Asks for each tone and each insertion effect and reads back whether it was
taken. A map that samples before sweeping carries the sampling as a caveat;
`--exhaustive` asks every bank for all 128 programs and is the only form with
nothing to caveat.

### 12. Repeatability, before any audio comparison

```sh
rye run soundings repeat --audio "<audio interface>" \
  --out data/units/<unit-id>/repeat/program-0.json
```

Plays one stimulus twice and measures how closely the two takes agree. That
figure is the floor of every later difference measurement: below it, "no
difference" and "no resolution to see one" are the same reading.

The floor is a property of the unit and the chain together, so it is measured
per unit and re-measured whenever the chain changes.

### 13. Audible differences

```sh
rye run soundings contrast --cc 91 --audio "<audio interface>" \
  --out data/units/<unit-id>/contrast/cc-91.json
```

Compares two settings of one parameter against the repeatability floor, which
separates a parameter the unit stores from one it is heard through. A parameter
can be stored and not audible; the address-space stages cannot tell those apart
and this one can.

`transfer`, `motion`, `decay`, `phase` and `vibrato` measure an analogue path, a
time-varying effect, an effect's tail, the angle an effect turns each band by,
and whether a take's pitch was modulated at all. `balance`, `balance-bands` and
`arrival` read the stereo pair instead of one channel of it: what a parameter did
to the level between the two, to their separation band by band, and to what a
take holds early against what it holds late. All but `transfer` read takes and
need no unit attached.

**`motion` reads a modulated delay and only that.** Its positive control injects a
swing in milliseconds, so on a type whose modulator moves a level or a pan instead
it cannot pass and never could -- and the null it returns is correct and says
nothing about the type. `efx-sway` is the reader for those: it tracks the take's
own level and its own difference between channels, inside one take for the reason
`vibrato` is read that way, and reports the rate, the depth and one averaged cycle
of whatever was modulating. Which printed shape that cycle answers to is a
comparison against `documents/` and is not made in the record.

A comparison made in one channel cannot see a parameter that moved the other, and
one made on broadband level cannot see a parameter that changed one channel's
shape while leaving its level alone. Which of the three was asked is therefore
part of what a null from this stage means.

**A reading that weighs every frame the same hands itself to the wrong ones.** A
pitch estimator loses the note now and then -- a comb's notch takes the partial it
was following, or it lands an octave out -- and it comes back a few frames later.
Those frames sit hundreds of cents from their neighbours on a track that is
otherwise steady to a cent, and two steps of `vibrato` used to weigh them in full:
a cubic trend fitted by least squares over every frame, and a depth read by
projecting every frame onto a sinusoid. Neither has any notion of an outlier, so a
run of lost frames outvoted the modulation. The trend is now fitted on the frames
inside the track's own middle, and the depth is the swing between the track's
fifth and ninety-fifth percentile.

What it was worth: four takes of one part-level vibrato setting read 112.72,
147.97, 102.83 and 233.31 cents under the old reading and 80.89, 80.14, 80.84 and
80.91 under this one, and the shallowest depth the injected control reaches on
those records went from 50 cents to 3.

**A control that passes is not a track that is sound.** The stage injects a known
modulation into one take and reports which depths came back, so what it
establishes is that the *search* can find a periodic component in that series --
not that the series is the note's pitch, and not that the reading taken off it is
robust. It recovered three cents on a run whose every published figure was wrong.
The take it is injected into is also the one setting where nothing is sweeping,
which is the least like the rows it bounds.

So the check that catches this is not the control. It is to look at the series
itself over a take the swept parameter was held at nought in, **with a statistic
one frame cannot decide**: an end-to-end span over sixteen hundred frames says
whatever the worst of them says, and reading those same takes end to end is what
produced a published claim that their tracks were unusable when they were flat to
eight tenths of a cent.

`phase` carries a control the others do not need. It measures between two takes of
one note, which works only where the two takes are the same waveform, so every
band is published with the coherence of that band beside it: a phase measured
where the takes carry different signals is the angle between two unrelated things
and is as steady and as plausible as a real one. An unknown delay between two
takes is a straight line in frequency and cannot be separated from a delay the
effect put there, so the line is removed, what was removed is published in
samples, and the figure with it still in is published beside the figure without.

### 14. Whole blocks

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

### 15. An effect type's own parameters

```sh
rye run soundings efx-sort <contrast records> \
  --types-from data/units/<unit-id>/efx-map/types.json \
  --tracked data/units/<unit-id>/efx-motion/<record>.json \
  --out data/units/<unit-id>/efx-sort/<name>.json
rye run soundings efx-params <contrast records> --type 01 00 \
  --types-from data/units/<unit-id>/efx-map/types.json \
  --control <the routed-against-bypassed record from the same run> \
  --slots "40 03 03" "40 03 04" ... \
  --out data/units/<unit-id>/efx-params/01-00.json
```

Asks of one insertion effect type's parameters what stage 14 asks of a block.
**The gate is asked at the parameter, not at the type.** A gate is worth what it
turns away, and one asked of a whole type can pass every type a unit has, leaving
the number of parameters still to sweep exactly where it started.

**The route is established before any of it**: whether a signal presented to the
unit's analogue input reaches its effects, measured with a control proving the
raised state does something. Where it does not, sweeping a known signal through
an effect is unavailable on that unit, and the stage is bounded by what the
unit's own voices can support.

**The slots are given rather than derived.** Which address is which slot is a
fact about the unit. Left to the sorted file names, an address the run could not
measure renumbers every parameter after it, pairs each with the default of the
slot before, and the count then reports the type as having one parameter fewer
than it has.

**A pair that left one setting silent compared sound with silence.** Such an
address is withdrawn by name, and the record that replaced it travels with the
row rather than with whoever remembers the directory. The prepared state the run
was taken in is recorded with it too, since the verdict holds in that state and
not generally.

Two routes answer whether a type stands still: `efx-motion` reads the takes,
`efx-sort` reads what each type did to the unit's own repeatability. They rest on
different properties, so a type they disagree about is reported as a disagreement
rather than reconciled. Both take the type map, so a type with no record reads as
unsurveyed rather than as absent.

A third route answers more than that, from one take per type rather than a pair:

```sh
rye run soundings efx-partials <one directory of held-tone takes> \
  --carrier-hz 440 --hold 8.0 --control-at 0.45 \
  --out data/units/<unit-id>/efx-partials/<name>.json
```

Each order of the held tone is demodulated by its own frequency, which leaves that
partial's phase and its level over the take, and a projection is walked across a
grid of rates. **Nothing is predicted and no byte's value enters it**, which is
what lets the result be held against a claim about what a rate byte means.

**The three mechanisms are separated by which quantity moves.** A swept delay
turns each partial's phase in proportion to that partial's own frequency; an
all-pass section turns them all by the same angle; a level modulation turns none
of them. Reading either quantity alone returns a number for all three.

**A level swing is a comb only if the partials disagree.** How well a forward comb
fit explains its series is a gate against the wrong rate and none at all against
the wrong mechanism -- a plain level modulation is fitted as a comb explaining more
of its own series than a real comb does. A comb sweeps a notch, so a partial near
it swings far more than one on a peak; one envelope over the voice cannot do that.

**A peak is placed against the bypassed take, rate by rate, not against the middle
of its own grid.** A held tone is not steady, and a voice drifting on its own puts
a peak many times its own grid median without anything in the path.

**What the comb returns is an equivalent, not a length.** It is the delay a two
path comb would need to move its notches that far; a phaser and a wah move notches
with all-pass sections and have no delay line to be long, and a level series cannot
tell those apart.

### 16. What a parameter's byte stands for

Only for a parameter an audible verdict admitted, and only when a named
downstream question needs the curve. Screening ends; sweeping does not.

```sh
rye run soundings efx-rate <takes> --type 01 20 --slot "40 03 04" \
  --setting '<pattern with a value group>' --held "40 03 07=00" \
  --out data/units/<unit-id>/efx-rate/01-20-04-<what-was-asked>.json
rye run soundings efx-bands <takes> --type 01 00 --slot "40 03 07" \
  --setting '...' --reference '...' --control '...' --silence '...' \
  --out data/units/<unit-id>/efx-bands/01-00-07-<what-was-asked>.json
rye run soundings efx-time <takes> --type 01 40 --slot "40 03 03" \
  --setting '...' --control '...' \
  --out data/units/<unit-id>/efx-time/01-40-03-<what-was-asked>.json
rye run soundings efx-sway <takes> --type 01 26 --slot "40 03 03" \
  --setting '...' --still '...' --held "40 03 05=7F" \
  --out data/units/<unit-id>/efx-sway/01-26-03-<what-was-asked>.json
```

`--still` names the take with no modulation in it, which the control is injected
into. It is named rather than guessed: a take already carrying a modulation ends
up with two, the search finds the unit's own, and the control reads as having
failed on material it can read perfectly well.

Each reads a directory of takes a `--save` run left, so none of them needs the
unit attached and all of them run while the hardware is busy with something else.
`balance`, `balance-bands` and `arrival` read the same directories for what the
byte did to the stereo pair.

**A sweep says what changed with the byte and cannot say what was already
there.** Each reader therefore takes the state it reports against from the same
session, and as takes rather than as fields: the type loaded with nothing written
(`--untouched`), the setting the run held flat (`--reference`), the part routed
past the effect (`--control`), and the chain with nothing played (`--silence`).
Without the last, a setting that turns the output far enough down publishes the
floor's own shape as a profile.

**What else was held is part of the reading.** A type with two modulators returns
whichever dominates, and a band profile is the whole chain's, so a byte read with
another of the type's stages moved is a reading of something else and nothing in
the numbers says so. `--held` is what makes it nameable.

**A record has to be rebuildable from the invocation it carries, and one stage's
were not.** Their stored commands were missing the `--held` flags, so a replay
came back with the held block empty — a record saying nothing else in the chain
was moved, which is the one thing that decides what a band profile is a profile
of. Nothing failed and the bands were identical. Before republishing a stage from
its own stored command, replay one record and diff every key against what was
committed, both ways round: a key that vanished is as much a change as one that
moved.

**How a setting is read back out of a take's name belongs in the invocation.** A
run names its takes however its own question needed, so the pattern lands in the
record where a reader can check it rather than trust it.

**The resolution costs something in both directions.** A band wider than the
deviation in it reports that deviation shallower than it was, and a deviation
still growing where the band set ends is reported as though it had stopped there.
A finer set is not automatically better: on one unit, re-reading the same takes
four times finer scattered a peak instead of sharpening it, and the flat repeats'
own spread worsened with it. Read at the coarser of what the run resolved and
what the question needs, and say in the record which that was.

**A delay is read against its own resolution.** No peak is taken below the
transform's resolution, since a delay shorter than that comes back as its third
rahmonic, which stands as high as a delay and is not one. The search runs past
the printed end of the range, so a byte that runs further than the page says
shows as a reading rather than as the search's own edge.

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
Stage 11 measures the floor for one stimulus. A stimulus that sounds more than one
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

## Reading a record is not one of the stages

No stage above says what a reading means. Naming a structure, fitting a topology,
deciding which of several readings the evidence leaves standing — all of it
happens in `inferences/`, outside `data/`, and in that one directory. References
run one way: an inference cites a record, and no record cites an inference.

It bears on the protocol in one place, and that place is what to run next.

```sh
rye run soundings inferences open
```

**What is worth measuring next is derived rather than kept in a list.** Each
entry is the observable that would separate a reading still standing from the
claim, with the margin over what this rig resolves there. **An entry whose margin
is below one must not be queued**: a finer band grid was such an entry on one
unit, and the run would have come back with a worse number than the one it was
meant to improve. An alternative nothing here can separate says so and stays open
with the reason, rather than becoming a run that could not have answered it.

**A record is admitted or refused before any reading sees it, and the rule is the
record's own.** A rate reading fewer than half of the take's partials found, or
taken at the slowest rate the take could carry, is left out by name; so is a slot
on a type the page prints more than one rate for, because a take returns whichever
modulator dominates it. Each refusal becomes a queue entry rather than a silent
drop.

## Running a stage that takes hours

The stages that sweep a whole block run for hours and are driven from
`procedures/`, which is tracked. A driver specific to one unit is a working note
and stays out; one that is part of the protocol belongs here, because a second
unit measured by a script nobody else has is not measured the same way.

```sh
./procedures/part-block.sh "40 11" 0        # block, and the channel it listens on
./procedures/part-takes.sh <plan> plain     # one pass over a planned block
./procedures/stop-sweep.sh                  # stop it, and prove it stopped
./procedures/publish-efx-bands.py --takes <dir> --out data/units/<unit-id>/efx-bands
```

**A publisher enumerates runs; it does not read them.** The ones in
`procedures/` go through the same stage the command line exposes rather than
carrying a second implementation of the reading, so a record can be rebuilt from
the audio after the reader improves, and a fix lands in one place. A slot a run
swept and the publisher does not name produces no record and is listed at the
end — a publisher that quietly covers the runs it recognises leaves the rest
looking like runs that were never made.

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
