# Single-arm trajectory comparison

This experiment compares annotation data collected from three traversal orders
over the same sampled and IK-reachable camera poses:

1. `random`: a reproducible random permutation.
2. `spiral`: the original layer-major latitude/azimuth sample order.
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

The hemisphere uses `latitude_layers: L` elevations and
`azimuth_samples: K` uniformly spaced azimuths per elevation, producing
`L*K` candidate camera poses. Override these values with
`--latitude-layers L` and `--azimuth-samples K`.

With elevation measured upward from the world horizontal plane, the sampler
uses

```text
theta_l = theta_min + l (theta_max - theta_min)/(L - 1)
phi_k   = phi_offset + 2 pi k/K
p_lk    = p_c + rho [cos(theta_l) cos(phi_k),
                     cos(theta_l) sin(phi_k),
                     sin(theta_l)]^T.
```

The camera rotation is `[x y z]`, where `z` is the normalized direction from
`p_lk` to `p_c`, `x` is the normalized `z × z_world`, and `y = z × x`.

Path initialization is controlled by `path_start_mode`:

- `initial_pose` starts at the accepted viewpoint nearest the measured initial
  robot state and prevents 2-opt from replacing that first viewpoint.
- `all_accepted` tries every IK-accepted viewpoint as the greedy start and
  selects the lowest-cost complete path. This is the default and preserves the
  previous behavior.

Override it with `--path-start-mode initial_pose` or
`--path-start-mode all_accepted`.

If nearest-neighbor reaches a dead end for every permitted start, the planner
falls back to a deterministic, connectivity-pruned Hamiltonian search. Thus a
greedy failure no longer rejects an overlap-feasible route. A failure after the
fallback means either that no Hamiltonian path satisfies the overlap threshold
or that the bounded search limit was reached.

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

## Kinova scan execution

`single_view_scan.py` extends the no-motion planner with Kinova motion and
capture. Its defaults are in the `single_view_scan` section of `config.yaml`:

```yaml
planning_group: manipulator
ik_link_name: end_effector_link
end_effector_name: end_effector_link
ready_state: Ready
base_frame: base_link
output_suffix: hamilton_2opt
trajectory_mode: hamilton_2opt
```

Confirm the camera frame, segmentation camera frame, and RGB-D topics against
the active Kinova/Isaac configuration, then run:

```bash
python3 scripts/path_planning_single/single_view_scan.py \
  --output-suffix trial_01 \
  --trajectory-mode hamilton_2opt
```

Available motion orders are `random`, `spiral`, and `hamilton_2opt`. For a
reproducible random trial, also pass `--random-seed N`. The first physical
command moves the `manipulator` group to SRDF state `Ready`. Planning then uses
that joint state and the live `camera_frame -> end_effector_link` TF calibration.
Each retained camera pose is converted into an end-effector target and sent
with `move_cartesian` for `end_effector_link`. A failed Cartesian motion is
recorded and skipped.

`camera_frame` controls motion conversion and the saved `T_base_cam_i.npy`
transform. `segmentation_camera_frame` is independent and is passed only to
Isaac's segmentation service. The default Kinova simulation configuration uses
`camera_color_frame` for motion/TF and
`handeye_camera_color_optical_frame` for segmentation.

Index `0` is always captured at the initial `Ready` pose before any Cartesian
scan motion. Hemisphere sampling indices begin at `1`. The initial index is
also prepended to every recorded route and to the RViz route markers; route
optimization still operates only on the IK-accepted hemisphere samples.

For view index `i`, successful captures are:

```text
scan_output/trial_01/
├── save_images/color_i.png
├── save_images/depth_i.png
├── save_segment/segment_i/<camera-frame>/capture_000000/
│   ├── manifest.json
│   └── <visible-object>.png
├── save_TF/T_base_cam_i.npy
└── manifest.json
```

The NumPy matrix is the measured $T_{base\_link}^{camera}$ TF after motion and
settling. Pass `--no-save-segmentation` when the Isaac segmentation service is
not being used. The run-specific `manifest.json` records rejected
IK samples, every route, motion failures, and all saved paths.

At startup, existing `save_images`, `save_segment`, `save_TF`, and `manifest.json`
artifacts inside the selected run directory are removed before new data is
written. Other run directories and `collection_log` are preserved.
