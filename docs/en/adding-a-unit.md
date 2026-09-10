# Adding a unit

What to establish about a unit before it is measured. The measurement protocol
describes the stages; this describes the state the unit and its record have to
be in before stage 1 runs.

## Identifier and directory

A unit is identified as `<manufacturer>-<model>-<n>`, lowercase, and everything
measured from it lives in `data/units/<unit-id>/`. The trailing number
distinguishes two units of the same model, which is the case the archive is
built for: a measurement describes the unit it was taken from, not the model.

Inside that directory there is one subdirectory per stage, named after the
command that wrote the records in it — `sweep/`, `write-probe/`, `alias-scan/`,
`contrast/`, `block/`, and the rest of the commands `soundings --help` lists. A
record is filed under the command that produced it and named after what it is
about: the address, block, effect type or controller the run asked at, with the
hex cased as the records write it. A run covering the whole of the address map
rather than one part of it is `whole-map.json`.

One directory under the stages holds no measurement: `watch-set/` is assembled
from the unit's own sweep and offsets records and says which addresses a later
stage should watch. It is named after the command that writes it like the rest,
and each file in it says in its own words that it is not a measurement and which
records it was built from. Rebuild it whenever those records change; a stage
aimed at a stale one is bounded by a space the unit is no longer known by.

Two files sit above the stages, because they are the unit rather than a
measurement of it: `meta.json`, which is its identity, and `measurements.json`,
which is the behaviours established about it and is kept by hand. A third,
`index.json`, lists everything in the directory and is generated —
`soundings index data/units/<unit-id> --out data/units/<unit-id>/index.json`
rewrites it, and a test fails while it disagrees with what is there.

Nothing is named after what it found. A file called after a verdict has to be
believed before it can be opened, it spells the finding differently on the next
unit, and it stops being true if the record is ever reissued.

## Before power-on

**Read the rear-panel rating plate and record it verbatim.** Model suffixes and
serial numbers are recorded as they read, not interpreted. A suffix that
indicates a regional or supply variant goes into `specifications_claimed` as the
plate spells it.

**Check the backup battery.** A unit old enough to be interesting is old enough
for its internal battery to have failed. The problem is not that settings are
lost. The power-on capture is the baseline every reset measurement is compared
against and the ground for any comparison with another unit, so a unit whose
power-on state differs because its battery is dead records a fault of that unit
as a property of its model. Once taken as a baseline, the two are no longer
distinguishable.

Replace the battery, if it needs replacing, before the first power-on capture.

## Both directions of the MIDI path

A self-test that passes is not evidence that the unit is receiving. One unit in
this archive does not route its DIN MIDI IN to the tone generator in three of
its four rear-selector positions, while MIDI OUT emits Active Sensing in every
position — which presents as a healthy path with a silent transmit side. The
selector is read only at power-on.

Confirm both directions explicitly, and record the selector position used in
`meta.json`.

## First runs

The first stages fix what everything else is measured against, and two of them
cannot be repeated later:

1. **The power-on capture**, which is unavailable once any stage has written to
   the unit.
2. **`repeat`**, which establishes the audio floor. A comparison between two
   units is bounded by the worse of their two floors, so the figure has to exist
   before any difference between them is claimed.

## Scope of what is recorded

A measurement describes the unit it was taken from. Findings are written as
being about that unit, or about the units measured, and never about a model, a
firmware revision or a family that was not measured.

`meta.json` carries the identity reply, what the unit reports of its firmware,
the rating plate, the measurement chain and the relevant unit settings. Where
the unit reports nothing usable — an all-zero software revision, for instance —
the record says the firmware is unresolved rather than leaving it absent.
