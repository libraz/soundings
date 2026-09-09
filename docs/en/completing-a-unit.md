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
| Address map | The sweep covered the whole top-byte range and reports itself trustworthy and complete. It bounds which blocks exist, and nothing more: the offsets stage is what makes it a list of addresses. |
| Offsets and shapes | Every offset of every block the sweep found has been asked, and the blocks are grouped into shapes by the set that answered. Each shape has a representative measured in full and one further block of the shape asked against it, with the agreement recorded. A shape whose two members disagree is split rather than folded. |
| Power-on state | Captured before any stage that writes, with the disagreement list and the unread count present. |
| Windows | Every block the address map shows answering identically to another block has a window verdict, measured at more than one offset. A block left untested is a block whose every later record may be about somewhere else. |
| Accepted values | The write probe covered every offset of every shape's representative, every region restored, and the bytes it skipped are listed. |
| Independent storage | The hold probe covered every offset of every shape's representative. |
| Aliases | One scan per message family the unit answers to, each watching the whole map, each carrying its controls. Parts that were not scanned are named in the record. |
| Resets | Every reset the unit accepts, marked from a whole-map write probe and compared against the power-on capture. |
| Tones and effects | A tone map for every map-select the unit accepts, not only the first; the effect type map asked for every type. |
| Repeatability | A floor measured for this unit on this chain, and re-measured after any change to the chain. |
| Audible differences | Every parameter reachable by a message has an audible verdict, or falls under a stated exclusion. |
| Whole blocks | One block of each kind swept in full and counted against its plan, with nothing left unasked and the addresses no pair can be built for named. A verdict that appeared in one block and not its peers was re-asked before it was published. |
| Effect response | The route is established first: whether a signal presented to the unit's analogue input reaches its effects, measured with a control proving the raised state does something. Where it does not, sweeping a known signal through an effect is unavailable on that unit and the stage is what its own voices can support. Then every effect parameter carries an audible verdict, taken at the parameter rather than at the effect type. The curves the identification work needs (below) are not required for completion. |

## Coverage is a fraction over shapes, not over addresses

An address space repeats itself, and by a large factor: on the first unit
measured this way, 461 blocks answered and held twelve shapes between them, one
of which appeared 204 times. A figure counted over addresses is then mostly a
count of how many times the same thing was measured, and it moves when the unit
has more parts rather than when more is known about it.

Every figure above is therefore a fraction over shapes. The address count is
still published, because a reader wants to know how much space is involved, but
it is not what any bar is set against.

The same applies to what a published document names. A document gives a function
to one row and the unit repeats that row across every part; counting the
expansion makes documented and undocumented space look alike. What is reported
is how many of a shape's offsets carry a function some document states, against
how many answered.

## The audible verdict gates the expensive work

Sweeping an effect parameter across its range, capturing a response at each
setting, is the costliest measurement here and the one with no natural end. Two
things bound it.

**The gate is asked at the parameter.** A gate is worth what it turns away, and
one asked of a whole effect type can pass every type a unit has, leaving the
number of parameters to sweep exactly where it started. So the audible verdict
that admits a sweep is the one taken at the parameter. Screening every parameter
of every type is a stage of its own, and its cost can be worked out before it is
begun: a count of parameters times the cost of one cheap comparison.

A parameter the unit stores but is not heard through does not get a response
measurement. That it is stored and inaudible is itself the finding.

**The sweep is asked for, not scheduled.** Screening ends; sweeping does not. A
unit is complete when what is audible in it has been mapped, with the bounds and
controls that make each null readable. The curves come afterwards, one parameter
at a time, when a named downstream question needs that parameter — the same rule
that decides what enters scope at all. A complete unit is one those questions can
be put to, not one that has already answered them.

For a parameter whose curve is asked for, the bar is the material an algorithm
can be identified from rather than a summary of it:

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
- **Every audible parameter swept.** The map of what is audible is the
  deliverable; the curves are drawn from it on demand. A unit that has to answer
  every question before it is finished is a unit that is never finished, and the
  questions are not all known yet.
- **A model of anything.** No algorithm is named, no topology fitted, no
  coefficient estimated. A unit is complete when it can be derived from, not
  when it has been.
- **Agreement with any other unit.** A result that differs from another unit's
  is a result.

## Stages that do not apply

Recorded in `meta.json` under `stages_that_do_not_apply`, keyed by the stage name
in the table above and carrying the reason. A stage is skipped because the unit
has nothing to answer it with — not because it was inconvenient, and not because
the answer was expected to be uninteresting.

## Asking where a unit stands

`soundings complete <unit directory>` counts a directory against the table and
prints what is left. The checks are structural — a file is present, a flag is
set, one count agrees with another — so a stage it reports as met was counted
rather than read. Where the bar turns on something only a reader can settle,
such as whether a control was sufficient or a null carried its bound, it says
it cannot decide instead of passing it.

What remains is given as counts of addresses and parameters, never as hours. A
rate belongs to a chain and a stimulus; one measured on one unit is not a fact
about the next, and the operator is the one who knows which applies.
