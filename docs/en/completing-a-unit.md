# Completing a unit

What has to be true before a unit's directory is finished. The protocol
describes the stages; this describes when there are no more of them to run.

A unit is complete when **every stage is accounted for**: either it was run and
its record meets the bar below, or the unit's `meta.json` says the stage does not
apply and why. Completion is per unit. Nothing waits on any other unit, and a
finished directory is usable on its own.

The distinction matters because a machine outside the family a stage was written
for has no answer to give it. "Not run" and "does not apply" look identical in a
directory listing and mean opposite things to a reader.

## The bar, stage by stage

| Stage | Complete when |
|---|---|
| Identity | `meta.json` carries the identity reply, the rear-panel plate verbatim, the measurement chain, the selector position, and a statement about firmware — including that it is unresolved, where the unit reports nothing usable. |
| Address map | The sweep covered the whole top-byte range and reports itself trustworthy and complete. |
| Power-on state | Captured before any stage that writes, with the disagreement list and the unread count present. |
| Windows | Every block the address map shows answering identically to another block has a window verdict, measured at more than one offset. A block left untested is a block whose every later record may be about somewhere else. |
| Accepted values | The write probe covered every region in the map, every region restored, and the bytes it skipped are listed. |
| Independent storage | The hold probe covered every region in the map. |
| Aliases | One scan per message family the unit answers to, each watching the whole map, each carrying its controls. Parts that were not scanned are named in the record. |
| Resets | Every reset the unit accepts, marked from a whole-map write probe and compared against the power-on capture. |
| Tones and effects | A tone map for every map-select the unit accepts, not only the first; the effect type map asked for every type. |
| Repeatability | A floor measured for this unit on this chain, and re-measured after any change to the chain. |
| Audible differences | Every parameter reachable by a message has an audible verdict, or falls under a stated exclusion. |
| Effect response | Every parameter found audible has the measurements the identification work needs (below). A parameter found inaudible needs none, and the null verdict is the record. |

## The audible verdict gates the expensive work

Sweeping an effect parameter across its range, capturing a response at each
setting, is the costliest measurement here and the one with no natural end. It
is bounded by making the cheap measurement first: **a parameter is swept only
after it has been found audible.**

A parameter the unit stores but is not heard through does not get a response
measurement. That it is stored and inaudible is itself the finding, and it is
already recorded.

For a parameter that is audible, the bar is the material an algorithm can be
identified from rather than a summary of it:

- the deconvolved response, kept as numbers beside the stimulus that produced it
- the harmonic orders separated from the linear response, not folded into it
- the parameter's own settings sampled across its range, so that the mapping
  from a seven-bit value to a physical quantity — a delay in milliseconds, a
  decay per octave band, a modulation rate — is a measured curve rather than two
  endpoints
- the quantisation visible in that curve, since a step is a fact about the
  machine and an interpolation over it is not

## What completion does not require

- **Every byte explained.** An address that answers, accepts a range and is
  reached by no message is completely measured. What it is for is not this
  archive's question.
- **Every part or channel scanned.** Scanning one of each kind and naming what
  was left is complete; scanning all sixteen is not more complete, it is more
  expensive.
- **A model of anything.** No algorithm is named, no topology fitted, no
  coefficient estimated. A unit is complete when it can be derived from, not
  when it has been.
- **Agreement with any other unit.** A result that differs from another unit's
  is a result.

## Stages that do not apply

Recorded in `meta.json`, naming the stage and the reason, in the same terms the
protocol names it. A stage is skipped because the unit has nothing to answer it
with — not because it was inconvenient, and not because the answer was expected
to be uninteresting.
