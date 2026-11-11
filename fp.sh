#!/bin/bash
PYTHONPATH="$PWD/.venv/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"

source install/setup.bash
ros2 run rb2301_fp obstacle_course "$@"
