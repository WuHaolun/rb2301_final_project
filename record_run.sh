#!/usr/bin/env bash
# Record a rosbag of a navigation run: every topic (including hidden/action
# topics) except /particle_cloud and /evaluation.
#
# /evaluation is DWB's debug output (debug_trajectory_details: True). It is
# ~1 MB per message at 20 Hz, ~97% of the bag, and subscribing to it makes
# controller_server serialise all of that inside its control loop.
#
# Usage:
#   ./record_run.sh            # bag named run_YYYYmmdd_HHMMSS
#   ./record_run.sh astar_t1   # bag named astar_t1_YYYYmmdd_HHMMSS
#
# Start it after planning.launch.py is up, then start the mission.
# Ctrl+C to stop; the bag is finalised on exit.

set -eo pipefail

BAG_DIR="./rb2301_bags"
NAME="${1:-run}"
OUT="$BAG_DIR/${NAME}_$(date +%Y%m%d_%H%M%S)"

# Source ROS if this shell hasn't already (whichever distro is installed).
if [ -z "${ROS_DISTRO:-}" ]; then
    for d in foxy humble jazzy; do
        if [ -f "/opt/ros/$d/setup.bash" ]; then
            source "/opt/ros/$d/setup.bash"
            break
        fi
    done
fi

mkdir -p "$BAG_DIR"
echo "Recording to: $OUT"
echo "ROS distro:   ${ROS_DISTRO:-unknown}"
echo "Excluding:    /particle_cloud /evaluation"
echo "Press Ctrl+C to stop."

# Flag names differ across distros (Foxy/Humble: -x/--exclude,
# Jazzy: --exclude-regex; --all-topics/--exclude-topics are Jazzy-only),
# so pick them from what this ros2 bag actually supports.
EXCLUDE_RE='^/(particle_cloud|evaluation)$'
HELP="$(ros2 bag record -h 2>/dev/null || true)"
ARGS=(-a -o "$OUT")

if grep -q -- '--exclude-regex' <<<"$HELP"; then
    ARGS+=(--exclude-regex "$EXCLUDE_RE")
elif grep -qE -- '(^|[ [])-x[ ,]' <<<"$HELP"; then
    ARGS+=(-x "$EXCLUDE_RE")
else
    echo "WARNING: this ros2 bag cannot exclude topics; recording everything."
fi

# --include-hidden-topics (action topics etc.) may be missing on old distros.
if grep -q -- '--include-hidden-topics' <<<"$HELP"; then
    ARGS+=(--include-hidden-topics)
else
    echo "Note: no --include-hidden-topics here; hidden topics skipped."
fi

echo "Command:      ros2 bag record ${ARGS[*]}"

ros2 bag record "${ARGS[@]}" || true

echo
ros2 bag info "$OUT" 2>/dev/null || echo "No bag written at $OUT"
