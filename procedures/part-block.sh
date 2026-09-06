#!/bin/bash
# Sweep one whole part block, from the write probe to a finished block record.
#
# Operational driver, not a measurement. It chains what part-takes.sh does one
# stage at a time, and the list each rescue stage is given is read back out of the
# previous stage's own records rather than written by hand: `soundings block`
# already names what is left to try, and that name is `still_open`. A hand list
# would be a judgement made twice, and the two would drift.
#
# Run from the repo root. Arguments: <block> <channel> [takes]
#
#   ./backup/part-block.sh "40 12" 1
#
# The channel is zero based and has to be the byte the block itself listens on --
# `40 1n 02`, which power-on-state.json records. It is not derived here, because a
# unit whose part-to-channel map has been moved would then be swept against an
# assumption instead of against itself.
#
# Every stage skips an address already on disk, so the whole script is safe to run
# again after a stop: it walks back to where it got to and carries on.
#
# Two stages, not three. A third pass with two-voice stimuli was built and then
# withdrawn: neither of them produces a yardstick that repeats, and nothing either
# reported survived being asked a second time. See the measurement protocol.
#
# One address at a time as a fresh process, which is not incidental. The two
# failures a long unattended run hits -- a capture that comes back short, and the
# MIDI layer refusing to create a client -- cost the whole address, and the second
# ends the process outright. Asking again as a new process is the only recovery
# available for it, so the retry lives here rather than inside the run.

set -u

BLOCK="${1:?a block, e.g. '40 12'}"
CHANNEL="${2:?the zero based channel that block listens on}"
TAKES="${3:-4}"

TAG="${BLOCK// /-}"
PROBE="data/units/roland-sc8850-01/write-probe-wholemap.json"
PLAN=".cache/plan-$TAG.json"
HERE="$(dirname "$0")"

mkdir -p .cache

say() { echo; echo "########## $* ##########"; echo; }

# Addresses still worth device time, from a block record's own accounting.
still_open() {
  python3 -c "
import json, sys
print('\n'.join(json.load(open(sys.argv[1]))['still_open']))
" "$1"
}

# A balance record for one stage's takes, if that stage captured any. Absent is a
# valid outcome -- a rescue stage asked nothing when the pass before it heard
# everything -- and is not an error to be reported as one.
#
# The command's own output goes to stderr, and the path is the only thing this
# writes to stdout. A caller reads it through $(...), which captures the whole
# function's stdout rather than its last line: with the report left on stdout the
# caller got forty-five lines of balance findings as the file name, and the block
# read died of a name too long. That failure then read as a stage with nothing to
# do, because the list it should have produced came back empty.
balance_for() {
  local takes=".cache/takes/part-$TAG-$1" out=".cache/balance-$TAG-$1.json"
  if [ -d "$takes" ] && [ -n "$(ls -A "$takes" 2>/dev/null)" ]; then
    rye run soundings balance "$takes" --out "$out" >&2 || true
  fi
  [ -f "$out" ] && echo "$out"
}

# The addresses a stage leaves for the next one, refusing to carry on without
# them. An empty list and a crashed read look identical downstream -- both send
# nothing to the next pass -- so the block record has to exist before its
# `still_open` can be read as a measurement rather than as a missing file.
carry_forward() {
  local record="$1" list="$2"
  if [ ! -f "$record" ]; then
    echo "!! $record was not written, so what is left to try is unknown." >&2
    echo "!! Refusing to run the next pass on an empty list, which would read as a" >&2
    echo "!! pass with nothing to do. Fix the read and run this script again." >&2
    exit 1
  fi
  still_open "$record" > "$list"
}

say "$BLOCK: planning from the write probe"
rye run soundings plan "$PROBE" "$BLOCK" --out "$PLAN" || exit 1

say "$BLOCK: plain pass"
"$HERE/part-takes.sh" "$PLAN" plain "" "$TAKES" "$CHANNEL"

say "$BLOCK: reading the plain pass"
BAL_PLAIN=$(balance_for plain)
rye run soundings block "$PLAN" ".cache/part-$TAG-plain" \
  ${BAL_PLAIN:+--balance "$BAL_PLAIN"} \
  --out ".cache/block-$TAG-plain.json"
carry_forward ".cache/block-$TAG-plain.json" ".cache/still-open-$TAG-1.txt"
echo "$(wc -l < ".cache/still-open-$TAG-1.txt") addresses go to the gesture pass"

say "$BLOCK: gesture pass"
"$HERE/part-takes.sh" "$PLAN" gesture ".cache/still-open-$TAG-1.txt" "$TAKES" "$CHANNEL"

say "$BLOCK: reading both one-note passes"
BAL_GESTURE=$(balance_for gesture)
rye run soundings block "$PLAN" ".cache/part-$TAG-plain" \
  --gesture ".cache/part-$TAG-gesture" \
  ${BAL_PLAIN:+--balance "$BAL_PLAIN"} \
  ${BAL_GESTURE:+--balance "$BAL_GESTURE"} \
  --out ".cache/block-$TAG-final.json"

say "$BLOCK: done"
