# Plan: detect "robot has passed the gate" and switch to final-zone behaviour

Status: plan only, nothing implemented. Written as a brief a subagent can follow.

## Context

The global planner (`global_planner.py` + `global_planner_core.py`) works through the zig-zag and the gate. After the gate two things go wrong:

- The swinging obstacles keep entering and leaving the occupancy grid, so the A* path changes on every replan.
- DWA should run with a different parameter set in the last section, but nothing tells the controller it is there.

Both need one signal: **the robot has passed through the gate**. The gate is the only wall on the course the robot drives *through*, and the planner already holds a multi-scan occupancy grid, so the test is done there: "just behind the robot there is a wall across the travel direction, with a gate-sized gap that the robot's track goes through".

The earlier single-scan `gate_detector` in `obstacle_course.py` was too fragile on the real robot. This replaces it for single-point-goal mode; do not reuse or modify that old code.

## Where things live (read these first)

| File | Relevant parts |
|---|---|
| `src/rb2301_fp/rb2301_fp/global_planner_core.py` | parameter block at the top, `OccupancyGrid` (`log_odds`, `occupied()`, `world_to_cell`, `cell_to_world`), `GlobalPlanner`, `PathFollower.select_goal` (returns the final goal when the path is empty) |
| `src/rb2301_fp/rb2301_fp/global_planner.py` | `GlobalPlannerNode`: `set_pose`, `map_timer_callback` (5 Hz), `plan_timer_callback` (1 Hz), `publish_path`, latched QoS pattern |
| `src/rb2301_fp/rb2301_fp/obstacle_course.py` | `init_global_path`, `path_callback`, `dwa_goal`, `publish_rviz_visualization`, `_sub_goal_2` (final travel direction: `'+x'` simulation, `'-y'` real) |
| `src/rb2301_fp/rb2301_fp/dynamic_window_approach.py` | `Config` (separate simulation / real branches) |
| `src/rb2301_fp/test/test_global_planner_core.py`, `test_path_following.py` | test patterns to follow |

Grid arrays are indexed `[row, col] = [iy, ix]`. Unknown cells have `log_odds == 0`; observed-free cells have `log_odds < 0`.

## Part 1 — `GateDetector` in `global_planner_core.py` (no ROS imports)

Add parameters to the block at the top of the file:

| Name | Default | Meaning |
|---|---|---|
| `FINAL_DIRECTION_SIM` / `FINAL_DIRECTION_REAL` | `'+x'` / `'-y'` | travel direction through the gate, one of `+x -x +y -y` |
| `GATE_MIN_WIDTH` / `GATE_MAX_WIDTH` | 0.45 / 0.95 m | accepted gap width (real gate is 0.6–0.8 m; margin for 5 cm cells and smear) |
| `GATE_MIN_FLANK` | 0.30 m | wall must continue at least this far on each side of the gap |
| `GATE_BACK_MIN` / `GATE_BACK_MAX` | 0.15 / 0.60 m | how far behind the robot to look for the wall line |
| `GATE_BAND_CELLS` | 3 | adjacent grid lines OR-ed together, for wall thickness and misalignment |
| `GATE_LATERAL_REACH` | 1.0 m | how far sideways to search for the gap edges |
| `GATE_FREE_FRACTION` | 0.8 | share of gap cells that must be observed free, not unknown |
| `GATE_CONFIRM` | 3 | consecutive positive checks before latching |
| `GATE_ARM_LATERAL` | 0.8 m | sideways displacement from the first pose before detection is armed; 0 disables arming |
| `GATE_MOTION_WINDOW` | 0.4 m | recent travel used to check the robot is moving along the final direction |

`class GateDetector`, constructed with the direction and the values above. One method, `update(grid, position) -> bool`, called with the `OccupancyGrid` and the robot `[x, y]`:

1. If already latched, return True.
2. Remember the first position. Keep a short trail of positions (enough to cover `GATE_MOTION_WINDOW`).
3. **Armed?** Displacement from the first position, measured perpendicular to the final direction, is at least `GATE_ARM_LATERAL`. If not, reset the confirm counter and return False.
4. **Moving the right way?** Over the last `GATE_MOTION_WINDOW` of trail, the displacement along the final direction is positive and at least 70% of the trail displacement. If not, reset and return False.
5. Because the direction is axis-aligned, a "line across the travel direction" is a grid column (for `±x`) or row (for `±y`). For each line between `GATE_BACK_MIN` and `GATE_BACK_MAX` behind the robot:
   - Take `occupied` OR-ed over `GATE_BAND_CELLS` adjacent lines, and `log_odds < 0` OR-ed over the same lines as "observed free".
   - The cell on the robot's own track must not be occupied.
   - Walk sideways from the track in both directions to the first occupied cell, within `GATE_LATERAL_REACH`. Both must exist.
   - Gap width = free cells between them × resolution, within `[GATE_MIN_WIDTH, GATE_MAX_WIDTH]`.
   - At least `GATE_FREE_FRACTION` of the gap cells are observed free.
   - On each side, the occupied run continuing outward from the gap edge is at least `GATE_MIN_FLANK` long, tolerating single-cell holes.
6. If any line passes, increment the confirm counter, else reset it. Latch at `GATE_CONFIRM`; store `gate_center` (world `[x, y]` of the gap middle on that line) and `gate_width`.

Why the flank test matters — it is what rejects the look-alikes:

- Corridor side walls run along the travel direction, so on a cross line they are only one wall-thickness wide.
- A static obstacle next to a side wall gives at most one long flank.
- The wall behind the robot after a corner has no free cell on the robot's track.
- An unobserved stretch of wall is unknown, not observed free.

Keep it to plain numpy slicing on a small window around the robot; it must cost well under a millisecond on the laptop.

## Part 2 — planner node (`global_planner.py`)

- New ROS parameters: `gate_detection` (default true) and `final_zone_mode` (`direct` default, or `keep`).
- Build the `GateDetector` together with the `GlobalPlanner` at the first pose, using `FINAL_DIRECTION_SIM` or `_REAL` according to `is_simulation`.
- Call `update()` at the end of `map_timer_callback` (5 Hz), including on ticks where the scan was skipped for yaw rate.
- On the tick it latches: log the gate centre and width once, and publish the centre once on a reliable, transient-local topic `gate_passed` (`geometry_msgs/PointStamped`, frame id = `visualization_frame`). The message existing means "passed"; the point is the gate centre for display.
- After latching:
  - `final_zone_mode:=direct`: `plan_timer_callback` skips planning and publishes an empty path. `PathFollower.select_goal` already falls back to the final goal on an empty path, so the controller needs no change for this.
  - `final_zone_mode:=keep`: planning continues unchanged; only the controller's DWA parameters change.
- With `gate_detection:=false` the node behaves exactly as today.
- Add `gate_passed=<bool>` to the existing `Performance:` log line.

## Part 3 — controller (`obstacle_course.py`) and DWA config

- In `dynamic_window_approach.Config`, add `self.final_zone = {}` in both the simulation and the real branch: a dict of `attribute name -> value` overrides, with a comment listing likely candidates (`max_speed`, `speed_cost_gain`, `obstacle_cost_gain`, `obstacle_turn_cost_gain`, `predict_time`, `to_goal_node_cost_gain`, `to_goal_node_dist_cost_gain`). Leave both dicts empty so behaviour is unchanged until they are tuned.
- Add `Config.apply_final_zone()`: set each override with `setattr`, raising `AttributeError` for a name that is not already an attribute. Validate the names once in `Config.__init__` as well, so a typo fails at start-up, not mid-run. (This avoids the existing `set_dwa_param` problem, which writes `to_goal_cost_gain`, a name DWA never reads. Do not use `set_dwa_param` here and do not change it.)
- In `init_global_path`, subscribe to `gate_passed` with the same latched QoS as `global_goal`. On the first message, only when `single_point_goal and use_global_path`: call `self.dwa_config.apply_final_zone()`, store the gate centre, set `self.stage = 'final_zone'`, log one line. Ignore repeats.
- In `publish_rviz_visualization`, add a `gate` marker at the stored centre (empty list, hence deleted, before the gate is passed), so it shows through both the `ros` and `ssh` transports.
- Do not touch the staged heading mode (`single_point_goal:=false`), `gate_detector`, `gate_stage` or `final_stage`.

## Constraints

- Runs on the robot's ROS 2 Foxy / Python 3.8: no newer syntax or APIs; numpy only, no scipy. The new code must pass the same Python 3.8 parse check the other modules have in the tests.
- `global_planner_core.py` stays free of ROS imports.
- No commits and no branch switching unless told to.

## Tests

Core, in `test_global_planner_core.py`, building grids by writing `log_odds` directly:

| Case | Expected |
|---|---|
| Cross wall with a 0.6, 0.7 and 0.8 m gap, robot driven through it | latches after `GATE_CONFIRM` updates; `gate_center` within one cell |
| Same, robot still in front of the wall | no detection |
| Corridor with side walls only | no detection |
| Solid wall behind the robot (after a corner) | no detection |
| 7 cm obstacle plus side wall forming a 0.6 m gap | no detection |
| 0.4 m box plus side wall forming a 0.6 m gap | no detection |
| Gap cells unknown instead of observed free | no detection |
| 1.2 m opening flanked by walls | no detection |
| Wall two cells thick, wall offset by one cell, single-cell holes in a flank | still detected |
| All four directions `+x -x +y -y` | detected |
| Not armed (sideways displacement below `GATE_ARM_LATERAL`) | no detection |
| Robot crossing the gap against the final direction | no detection |
| After latching, any later input | stays True |

Node and controller, following `test_path_following.py`:

- Planner publishes `gate_passed` once and then empty paths in `direct` mode; keeps planning in `keep` mode; `gate_detection:=false` is unchanged.
- Controller applies `final_zone` overrides exactly once on `gate_passed`; ignores it when `use_global_path` is false; an unknown override name raises at construction.
- All existing tests still pass.

## Verification in simulation

1. `colcon build --symlink-install --packages-select rb2301_fp`, then `python3 -m pytest src/rb2301_fp/test -q`.
2. `./gz_fp.sh`, `./fp.sh`, `./gp.sh`, `bash rviz_fp.sh`, over at least five randomised launches with obstacles enabled. For each run record the robot pose when `gate_passed` fires and compare it with the gate position printed by the launch.
3. Check there is no trigger in the first leg or at the static obstacles, and that the path stops changing after the gate in `direct` mode.
4. Repeat one run with `./gz_fp.sh disable_obstacles:=true`.

Report what was actually run. If Gazebo cannot be run, say so; unit tests alone do not show the detector works on real scans.

## Known risks and things to confirm

- **Final direction on the real robot.** `'-y'` is taken from `_sub_goal_2`; confirm it matches the real course before relying on it.
- **Swinging obstacle near the gate line in simulation.** In the world file `obstacle_1` has the same x (4.2) as the gate walls, so it may cross the gap and briefly change the measured width. The confirm count and latch should cope, but this needs to be seen in a run.
- **`direct` mode assumes open space after the gate.** If a static wall lies between the gate and the goal, DWA can get stuck again; use `final_zone_mode:=keep` in that case.
- **Real-robot timing.** The obstacles are released when the robot is at the gate, and this signal fires 0.15–0.6 m after it. If parameters must change before the gate, the same test can be run on lines ahead of the robot; that is not part of this plan.
- **Not verified on the Jetson or the real course.** Thresholds are first guesses for a 5 cm grid.

## Left out on purpose

- Slower hit evidence for cells beyond the gate, so swinging obstacles never reach the occupied threshold.
- Path hysteresis (keep the old path unless it is blocked or clearly longer).
- Choosing the actual final-zone DWA values: `Config.final_zone` is left empty for tuning.
