#!/usr/bin/env bash
set -e
workspace_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$workspace_dir/install/setup.bash"
rviz2 -d "$(ros2 pkg prefix rb2301_fp)/share/rb2301_fp/rviz/obstacle_course.rviz" "$@"
