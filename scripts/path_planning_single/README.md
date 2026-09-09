# Single-arm trajectory comparison

This experiment compares annotation data collected from three traversal orders
over the same sampled and IK-reachable camera poses:

1. `random`: a reproducible random permutation.
2. `spiral`: the original golden-angle hemisphere sample order.
3. `hamilton_2opt`: an overlap-constrained nearest-neighbor Hamiltonian seed
   refined by 2-opt.

The planner itself never creates a motion-plan request and never calls a
controller. Its only robot operation is collision-aware IK through the standard
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
`--hold-seconds 20` for a finite run. Override `--random-seed 23` to create a
different reproducible random baseline.

Select one trajectory for an individual experiment:

```bash
python3 scripts/path_planning_single/single_path_planner.py \
  --trajectory-mode random
python3 scripts/path_planning_single/single_path_planner.py \
  --trajectory-mode spiral
python3 scripts/path_planning_single/single_path_planner.py \
  --trajectory-mode hamilton_2opt
```

Set `trajectory_mode: all` in `config.yaml`, or pass `--trajectory-mode all`,
to publish all three routes for the RViz comparison. A single-mode run publishes
only the common hemisphere/candidates and that route's path and robot animation.
The JSON still contains all computed results, with `trajectory_mode` and
`selected_trajectories` identifying the active experiment.

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
| MarkerArray | `/single_path_planning/path_random` | Random order, orange |
| MarkerArray | `/single_path_planning/path_spiral` | Normal spiral order, purple |
| MarkerArray | `/single_path_planning/path_hamilton_2opt` | Hamiltonian 2-opt route, neon green `(25,255,0)` |
| MarkerArray | `/single_path_planning/path_comparison` | All three routes overlaid |
| MotionPlanning | `/single_path_planning/display_random` | Robot animation in random order |
| MotionPlanning | `/single_path_planning/display_spiral` | Robot animation in spiral order |
| MotionPlanning | `/single_path_planning/display_hamilton_2opt` | Robot animation in optimized order |

The marker topics are republished once per second and also use transient-local
durability. For a clean paper figure, show the hemisphere, candidates, and
comparison topics. For an unambiguous route check, enable the individual path
or robot-animation displays one at a time.

The generated `single_path_planning_result.json` records camera poses, IK joint
solutions, the three source orders, objective and motion-distance metrics,
neighbor-overlap statistics, and the count of edges below the configured
overlap threshold. Random and spiral are intentionally retained as baselines
even if they violate that threshold; the Hamiltonian 2-opt route is constrained
by it. Use the recorded `sample_index` values to associate images and annotation
metrics with the same physical viewpoints in every trial.
