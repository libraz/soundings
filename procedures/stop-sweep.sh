#!/bin/bash
# Stop every process of a part sweep, and prove nothing is left driving the unit.
#
# Killing the top of the tree is not enough, and neither is one pass. The drivers
# nest three deep and each spawns one stage at a time, so a parent left alive
# starts the next stage the moment the child being killed dies -- `pkill -P` on
# the top process reaches one level and leaves the rest running. What survives is
# not idle: it keeps playing notes into the same unit through the same interface.
#
# That happened here. A survivor from one sweep ran alongside a second sweep for
# twenty-four minutes, and the two were measuring the unit at once. It was caught
# only because a take's lead-in guard names the case -- "something was sounding
# before the note: the tail of the take before it, or another process driving the
# same unit" -- and forty-four records had to be thrown away.
#
# So: kill parents before children, repeat until a scan comes back empty, and exit
# non-zero if it never does. A stop that cannot prove it stopped is not a stop.

set -u

PATTERN="queue-parts.sh|part-block.sh|part-takes.sh|soundings contrast"

for round in 1 2 3 4 5 6; do
  # Outermost first, so a parent cannot outlive the child it would replace.
  pkill -f queue-parts.sh 2>/dev/null
  pkill -f part-block.sh 2>/dev/null
  pkill -f part-takes.sh 2>/dev/null
  pkill -f "soundings contrast" 2>/dev/null
  sleep 2
  if ! pgrep -f "$PATTERN" >/dev/null 2>&1; then
    echo "stopped after $round round(s); nothing is driving the unit"
    exit 0
  fi
  echo "round $round: still up, going again"
  # Escalate once the polite signal has had two rounds to work.
  [ "$round" -ge 2 ] && pkill -9 -f "$PATTERN" 2>/dev/null
done

echo "!! a sweep is still running after 6 rounds. Do not start another one:" >&2
pgrep -fl "$PATTERN" >&2
exit 1
