# Single-arm path-planning figure

This experiment evaluates the multiview path optimizer with one MoveIt robot.
It never creates a motion-plan request and never calls a controller. The only
robot operation is collision-aware IK through the standard
`moveit_msgs/srv/GetPositionIK` service, using `ik_link_name` and
`pose_stamped`.

## Run

Edit `config.yaml` for the active robot, source its MoveIt workspace, start
`move_group`, joint-state publication, and RViz, then run:

```bash
source /opt/ros/humble/setup.bash
source ~/your_moveit_ws/install/setup.bash
python3 scripts/path_planning_single/single_path_planner.py
```

The supplied YAML contains FR3-style defaults:

```yaml
planning_group: fr3_arm
ik_link_name: fr3_hand_tcp
```

For a Kinova configuration, use the exact names from its SRDF, for example:

```bash
python3 scripts/path_planning_single/single_path_planner.py \
  --planning-group manipulator \
  --ik-link-name end_effector_link
```

The actual Kinova group and link names vary by MoveIt configuration. Confirm
them in the loaded SRDF. Use `--no-use-sim-time` when the MoveIt graph uses wall
time. The script waits indefinitely for RViz by default; use
`--hold-seconds 20` for a finite run.

`camera_to_ik_link_xyz` and `camera_to_ik_link_rpy_deg` define
$^{C}T_{L}$, the fixed transform from the desired camera optical frame to the
MoveIt IK link. Identity means the IK link itself is treated as a camera frame
whose +Z axis looks at the scene center. Set this calibration correctly before
interpreting IK reachability.

## RViz displays

Set RViz's fixed frame to the configured `world_frame`, then add these displays:

| RViz display | Topic | Meaning |
|---|---|---|
| MarkerArray | `/single_path_planning/hemisphere` | Transparent sampling shell |
| MarkerArray | `/single_path_planning/candidates` | Red reachable dots and black X rejected samples |
| MarkerArray | `/single_path_planning/path_before_2opt` | Greedy path, neon magenta |
| MarkerArray | `/single_path_planning/path_after_2opt` | Final path, neon green `(25,255,0)` |
| MarkerArray | `/single_path_planning/path_comparison` | Both paths overlaid |
| MotionPlanning | `/single_path_planning/display_before_2opt` | Robot animation before 2-opt |
| MotionPlanning | `/single_path_planning/display_after_2opt` | Robot animation after 2-opt |

The marker topics are republished once per second and also use transient-local
durability. For a clean paper figure, show the hemisphere, candidates, and
comparison topics. For an unambiguous route check, enable the before-only and
after-only displays one at a time.

The generated `single_path_planning_result.json` records accepted/rejected
counts, both source orders, objective values, motion distance, and overlap
statistics for use in the paper.
