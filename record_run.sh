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

# Source ROS if this shell hasn't already.
if [ -z "${ROS_DISTRO:-}" ]; then
    source /opt/ros/jazzy/setup.bash
fi

mkdir -p "$BAG_DIR"
echo "Recording to: $OUT"
echo "Excluding:    /particle_cloud /evaluation"
echo "Press Ctrl+C to stop."

ros2 bag record \
    --all-topics \
    --include-hidden-topics \
    --exclude-topics /particle_cloud /evaluation \
    -o "$OUT" || true

echo
ros2 bag info "$OUT" 2>/dev/null || echo "No bag written at $OUT"
