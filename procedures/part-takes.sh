#!/bin/bash
# Ask every address in one part block whether changing it changes the sound, at
# the pair `soundings plan` derived from the write probe.
#
# Operational driver, not a measurement: the measurement is `contrast`, and each
# address's own record and takes-manifest.json say how it was asked. Run from the
# repo root. Arguments: <plan json> <stage> [address list] [takes] [channel]
#
#   rye run soundings plan data/units/roland-sc8850-01/write-probe-wholemap.json \
#     "40 11" --out .cache/plan-40-11.json
#   ./backup/part-takes.sh .cache/plan-40-11.json plain
#   ./backup/part-takes.sh .cache/plan-40-11.json gesture .cache/deaf.txt
#   ./backup/part-takes.sh .cache/plan-40-11.json polyphony .cache/still-open.txt
#
# The standard procedure runs two of these: plain, then gesture on what the plain
# note could not hear. A gesture asks in a heavily prepared state the verdict then
# only holds in, so it is worth its device time on an address the plain note could
# not hear and nothing on one it could. Measured on this unit: 40 seconds an
# address plain, 2 minutes 39 seconds for the gesture's three stimuli.
#
# The polyphony stage is here and part-block.sh does not run it. Its two stimuli
# were built to ask what one note cannot -- whether a part sounds two voices at
# once -- and then measured to produce no yardstick that repeats: 84 per cent of
# their agreement figures worse than 10 dB where a plain note sits at 56 to 62,
# and nothing they reported surviving a second asking. The stage is kept because a
# stimulus that can ask the question is worth building; it is not kept because
# these two can.
#
# The address list for a rescue stage is one address per line, in the same form
# the plan writes them ("40 11 03"); with none given the whole plan is asked.
# Which addresses belong in it is a reading of the earlier stage's records, not a
# judgement made here.
#
# The part block and the channel have to agree: a stimulus on any other channel
# asks a part the address does not address, and answers inaudible for every one of
# them. The channel each block listens on is not assumed here -- it is the byte at
# `40 1n 02`, which power-on-state.json records as 00 for 40 11, 01 for 40 12 and
# 09 for 40 10. Pass it as the fifth argument, zero based, matching that byte.
#
# Re-running skips an address whose takes are already on disk, so a run stopped
# part way carries on rather than starting again. The take and record directories
# carry the block, so two parts swept in turn cannot land in one directory and
# read afterwards as one block's answer.

set -u

PLAN="${1:?a plan json, from 'soundings plan'}"
STAGE="${2:?stage: plain, gesture or polyphony}"
ONLY="${3:-}"
TAKES="${4:-4}"
CHANNEL="${5:-0}"

case "$STAGE" in
  plain)     STIMULUS="struck" ;;
  gesture)   STIMULUS="gesture" ;;
  polyphony) STIMULUS="polyphony" ;;
  *) echo "stage must be plain, gesture or polyphony, not $STAGE" >&2; exit 2 ;;
esac

BLOCK=$(python3 -c "
import json, sys
print(json.load(open(sys.argv[1]))['block'].replace(' ', '-'))
" "$PLAN") || exit 2

ROOT=".cache/takes/part-$BLOCK-$STAGE"
OUT=".cache/part-$BLOCK-$STAGE"

mkdir -p "$ROOT" "$OUT"

# Tab separated, and read with IFS set to a tab: an address is three hex bytes
# with spaces in it, so splitting a row on whitespace tears it into three fields
# and every run dies at argument parsing.
addresses=$(python3 -c "
import json, sys
plan = json.load(open(sys.argv[1]))
only = None
if len(sys.argv) > 2 and sys.argv[2]:
    only = {line.strip() for line in open(sys.argv[2]) if line.strip()}
for ask in plan['ask']:
    if only is not None and ask['address'] not in only:
        continue
    low, high = ask['values']
    print('\t'.join([ask['address'].replace(' ', '-'), ask['address'], str(low), str(high)]))
" "$PLAN" "$ONLY")

ATTEMPTS=3
REST=8
# Two failures on this chain are the machine refusing a resource for a moment
# rather than anything about the address, and both cost the whole address: a
# capture that came back short of what was asked for, which the run rightly
# refuses to measure through, and CoreMIDI declining to create a client at all,
# which aborts the process outright. Neither is retried inside the run -- the
# second cannot be, since an uncaught C++ exception ends it -- so the retry is
# here, where asking again is a fresh process. Measured over one unattended
# sweep: 6 of 25 addresses lost this way, and every one of them had already
# produced a usable verdict under at least one of its stimuli.
#
# How many were needed is reported rather than swallowed. A rate that climbs is
# a fact about the chain, and a pass that quietly retried its way to a full set
# would hide it.
total=0
failed=()
retried=()
while IFS=$'\t' read -r id address low high; do
  [ -z "${id:-}" ] && continue
  total=$((total + 1))
  if [ -f "$ROOT/$id/takes-manifest.json" ]; then
    echo "=== $id already captured, skipping ==="
    continue
  fi
  echo "=== $id ($total) $low against $high on channel $CHANNEL ==="
  for attempt in $(seq 1 "$ATTEMPTS"); do
    rye run soundings contrast \
      --address "$address" --values "$low,$high" \
      --stimulus "$STIMULUS" \
      --channel "$CHANNEL" \
      --takes "$TAKES" \
      --audio Scarlett \
      --save "$ROOT/$id" \
      --out "$OUT/$id.json"
    status=$?
    [ $status -eq 0 ] && break
    [ "$attempt" -ge "$ATTEMPTS" ] && break
    echo "=== $id attempt $attempt exit $status, resting ${REST}s and asking again ==="
    retried+=("$id")
    # Only ever an incomplete capture: a directory with a manifest is a finished
    # one, and this loop is not reached for those at all.
    if [ -d "$ROOT/$id" ] && [ ! -f "$ROOT/$id/takes-manifest.json" ]; then
      rm -rf "${ROOT:?}/${id:?}"
    fi
    sleep "$REST"
  done
  echo "=== $id exit $status ==="
  [ $status -ne 0 ] && failed+=("$id")
done <<< "$addresses"

echo "=== done: $total addresses asked, ${#failed[@]} failed: ${failed[*]:-none} ==="
echo "=== retried: ${#retried[@]} attempts over ${retried[*]:-no} addresses ==="
