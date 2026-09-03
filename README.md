# Dual-Arm Vision and Grasp Pipeline

This repository contains an experimental perception-to-grasp workflow for a
dual Franka FR3 system. It can spawn random YCB objects in Isaac Sim, scan a
scene with both hand-eye cameras, reconstruct RGB-D point clouds, segment
objects with Qwen3-VL and SAM, sample collision-aware grasps, visualize them,
and execute a selected grasp through `RobotAPI`.

## Pipeline

1. Spawn random objects and start the Isaac Sim ROS bridge.
2. Start MoveIt and `robot_interfaces` with simulated time enabled.
3. Move both hand-eye cameras around the scene and capture synchronized RGB-D
   images plus world-frame camera transforms.
4. Reconstruct the scene or run the vision notebook to create segmented object
   point clouds.
5. Sample and rank parallel-jaw grasp candidates.
6. Inspect the selected grasp, dry-run its command file, and explicitly enable
   execution when it is safe.

## Repository layout

| Path | Purpose |
| --- | --- |
| `scripts/run_random_objects.sh` | Spawn five random objects from the configured YCB pool. |
| `scripts/multi_view_scan.py` | Reachability-filtered, overlap-aware dual-arm scan and RGB-D capture. |
| `scripts/scan_trajectory.py` | Pure spiral generation, projection overlap, open-TSP, and constrained 2-opt. |
| `scripts/moveit_ik.py` | Read-only collision-aware MoveIt multi-tip IK client. |
| `config/multi_view_scan.yaml` | Complete configurable parameter set for multiview scanning. |
| `scripts/scene_reconstruction.py` | Merge captured RGB-D views into a world-frame scene cloud. |
| `scripts/vision_pipeline.ipynb` | Qwen3-VL detection, SAM segmentation, point-cloud generation, and grasp workflow. |
| `scripts/grasp_sampling.py` | Principal-curvature grasp sampling, collision checks, scoring, and command export. |
| `scripts/execute_grasp.py` | Validate, print, and optionally execute a grasp command. |
| `scripts/o3d_process.py` | Shared Open3D and point-cloud utilities. |
| `scripts/aux_math.py` | Shared pose, transform, and hemisphere geometry utilities. |
| `TASK.md` | Detailed multiview trajectory-planning contract and acceptance criteria. |
| `docs/multiview_scan_algorithm.md` | Publication-oriented derivation of the multiview planning algorithm. |
| `docs/multiview_scan_method_short.md` | Condensed single-section version for a paper methods section. |

## Requirements

- Ubuntu with ROS 2 Humble and MoveIt 2
- Isaac Sim at `~/isaacsim`
- Simulation repository at `~/Projects/dual_manipulation_isaac_sim`
- Built ROS workspace at `~/Projects/dual_arm_ws`
- The `robot_interfaces` services and Python `RobotAPI`
- Python packages used by the offline pipeline:
  - NumPy and SciPy
  - OpenCV and PyYAML
  - Open3D
  - Pillow, PyTorch, Transformers, and Jupyter for the vision notebook
- A CUDA-capable GPU is strongly recommended for Qwen3-VL and SAM

The notebook loads these Hugging Face models:

- `Qwen/Qwen3-VL-4B-Instruct`
- `facebook/sam-vit-base`

## Environment setup

Run ROS commands from the repository root after sourcing both workspaces:

```bash
cd ~/Projects/vla_dual_arm
source /opt/ros/humble/setup.bash
source ~/Projects/dual_arm_ws/install/setup.bash
```

The scan imports helper modules directly from `scripts/`. Run the scripts from
the repository root as shown below.

## 1. Spawn random objects in Isaac Sim

```bash
./scripts/run_random_objects.sh
```

The launcher asks `run_simulation_obj.py` to choose five unique objects, place
them within `x=[-0.3, 0.3]` and `y=[-0.3, 0.0]`, and use MIT joint control.
The duplicated `Strawberry` entry from the original list is included only once.

Use a fixed seed for a repeatable scene:

```bash
./scripts/run_random_objects.sh --seed 42
```

Additional arguments are forwarded to `run_simulation_obj.py`, for example:

```bash
./scripts/run_random_objects.sh --headless --seed 42
```

## 2. Start MoveIt and the robot services

Start the existing MoveIt/controller launch with `use_sim_time:=true`. Start
the robot interface explicitly with the same setting:

```bash
ros2 launch robot_interfaces robot_interfaces.launch.py use_sim_time:=true
```

Verify the clock before scanning:

```bash
ros2 topic hz /clock
ros2 param get /move_group use_sim_time
ros2 param get /robot_motion_server use_sim_time
```

Both parameters must be `True`, and `/clock` must be advancing. A message such
as the following means that a MoveIt or robot-interface node still uses wall
time while Isaac publishes simulation timestamps:

```text
Requested time 1788044963..., but latest received state has time 4179...
```

Restart the offending launch with `use_sim_time:=true`; changing only the scan
client cannot correct a wall-time MoveIt server.

## 3. Capture a multi-view scan

The scanner loads every default parameter from
`config/multi_view_scan.yaml`:

```bash
python3 scripts/multi_view_scan.py
```

Use another experiment configuration with `--config`. Any explicitly supplied
command-line option overrides the value in that YAML file:

```bash
python3 scripts/multi_view_scan.py \
  --config config/multi_view_scan.yaml \
  --view-count 32 \
  --minimum-neighbor-overlap 0.45
```

The scan defaults to simulated time and verifies that `/clock` advances before
moving. It then:

- removes previous `steps`, `grasp_commands`, `mask_segment`, and `pointclouds`
  directories, then creates a clean `steps` directory;
- checks both synchronized RGB-D streams and required TF frames;
- attempts to move `dual_arm` to the SRDF `ready` state;
- generates mirrored, equal-area golden-angle spiral camera pairs;
- sends each pair to collision-aware multi-tip `/compute_ik` without moving the
  robot and removes any unreachable pair;
- forms a pose- or joint-distance nearest-neighbor open-TSP path and improves
  it with 2-opt;
- permits a temporal edge only when projected surface overlap meets
  `--minimum-neighbor-overlap` for both cameras;
- publishes the complete IK waypoint sequence on `/display_planned_path` and
  the optimized camera routes on `/scan/left_camera_path` and
  `/scan/right_camera_path` before the first scan motion;
- publishes camera and TCP target frames for RViz preview;
- sends both arm targets together with `move_l_dual`;
- continues to the next viewpoint when a motion call raises `RobotAPIError`;
- saves images under `scan_output/steps/step_001`, `step_002`, and so on;
- saves IK rejections, before/after path metrics, optimized spiral order,
  successful capture records, and relative image paths in
  `scan_output/manifest.json`;
- attempts to return to `ready` after the scan.

Useful options include:

```bash
python3 scripts/multi_view_scan.py \
  --output-dir scan_output \
  --center 0.40 0.0 0.0 \
  --radius 0.4 \
  --view-count 30 \
  --azimuth-bounds 10 135 \
  --elevation-bounds 30 75 \
  --scan-volume-radius 0.15 \
  --projection-samples 2048 \
  --minimum-neighbor-overlap 0.35 \
  --overlap-weight 0.5 \
  --distance-metric pose \
  --pose-translation-weight 1.0 \
  --pose-rotation-weight 0.10 \
  --two-opt-passes 30 \
  --ik-timeout 0.25 \
  --trajectory-point-time 0.25 \
  --trajectory-preview-time 5 \
  --camera-timeout 10 \
  --motion-timeout 120
```

The overlap model projects a deterministic spherical surface proxy through the
intrinsics in `--camera-yaml`. If planning cannot connect all reachable poses
at the requested threshold, the scan stops before executing a view. Reduce
`--minimum-neighbor-overlap`, increase `--view-count`, or narrow the angular
bounds. `--overlap-weight` trades a shorter path under the selected motion
metric against greater neighbor overlap while the threshold remains a hard
constraint.

The default YAML uses `distance_metric: pose`. For two paired viewpoints, pose
distance combines the root-sum-square translation of both TCPs with their
shortest orientation changes. `pose_translation_weight` scales translation,
while `pose_rotation_weight` converts radians to equivalent translation cost.
Use `--distance-metric joint` to restore Euclidean distance between the
multi-tip IK joint solutions.

To inspect the complete route in RViz before execution, use the MoveIt Motion
Planning display subscribed to `/display_planned_path` and add two `Path`
displays for `/scan/left_camera_path` and `/scan/right_camera_path`. The
publishers use transient-local durability; configure the RViz Path displays to
use `Transient Local` durability if RViz connects after publication. The robot
animation connects collision-checked IK waypoints for preview only; the
interpolated full route is not itself a prevalidated MoveIt trajectory. Each
actual `move_l_dual` request still performs its normal runtime planning and
validation. Set `--trajectory-preview-time 0` to publish without waiting.

Run the offline planner tests without commanding the robot:

```bash
source /opt/ros/humble/setup.bash
source ~/Projects/dual_arm_ws/install/setup.bash
python3 -m unittest discover -s tests -v
```

Use `--no-use-sim-time` only when the entire ROS graph is running on wall time,
such as a real-robot deployment.

### Camera transport

`RobotAPI` supports raw `sensor_msgs/Image` RGB streams and compressed
`sensor_msgs/CompressedImage` streams. Compressed JPEG/PNG and Isaac Sim H.264
payloads are decoded automatically when the RGB topic ends in `/compressed`.
Depth remains a raw `sensor_msgs/Image` stream.

## 4. Reconstruct the full scene

Validate the manifest, images, and calibration without loading Open3D:

```bash
python3 scripts/scene_reconstruction.py \
  --scan-dir scan_output \
  --validate-only
```

Build and display the world-frame point cloud:

```bash
python3 scripts/scene_reconstruction.py \
  --scan-dir scan_output \
  --visualize
```

By default this writes:

- `scan_output/scene_reconstruction.ply`
- `scan_output/scene_reconstruction.json`

Camera intrinsics default to:

```text
~/Projects/dual_manipulation_isaac_sim/env/urdf/camera.yaml
```

Only manifest records with `status: captured` are reconstructed. Actual TF
camera poses are preferred; use `--pose-source desired` to reconstruct from
planned camera poses instead.

## 5. Segment objects and create object clouds

Open the notebook:

```bash
jupyter lab scripts/vision_pipeline.ipynb
```

The segmentation section reads color images, depth images, and camera poses
from `scan_output/manifest.json`. Qwen3-VL identifies objects and bounding
boxes, SAM produces masks, and the masked RGB-D views are transformed into the
manifest world frame.

The notebook writes a point-cloud index similar to:

```text
scan_output/
├── manifest.json
├── steps/
│   ├── step_001/
│   ├── step_002/
│   └── ...
├── mask_segment/
├── pointclouds/
│   ├── index.json
│   ├── scene.json
│   └── <object>.json
└── grasp_commands/
    └── <object>.json
```

Review the notebook's `SCAN_DIR`, `CAMERA_YAML`, user prompt, selected object,
gripper geometry, and `EXECUTE_GRASP` values before running all cells.
`EXECUTE_GRASP` defaults to `False`.

## 6. Generate and visualize a grasp

The notebook includes grasp sampling and visualization. The same operation can
be run from the command line after object clouds have been saved:

```bash
python3 scripts/grasp_sampling.py \
  scan_output/pointclouds/apple.json \
  --scene-cloud scan_output/pointclouds/scene.json \
  --object-name Apple \
  --output scan_output/grasp_commands/apple.json \
  --samples 5000 \
  --top 10
```

The sampler estimates normals and local principal-curvature directions. It
aligns gripper `+Z` opposite the sampled outward normal, tests both local
principal directions for the closing axis, rejects finger/palm and pregrasp
collisions, and globally ranks the remaining candidates.

Gripper-frame convention:

- `+X`: parallel-jaw closing axis
- `+Y`: finger/hand depth direction
- `+Z`: palm toward fingertips
- origin: contact plane, with `grasp_depth` from the origin to the fingertips

## 7. Validate and execute a grasp

Always perform a dry run first:

```bash
python3 scripts/execute_grasp.py scan_output/grasp_commands/apple.json
```

The dry run validates the score, opening, frames, action order, and pose values,
then prints every command without moving the robot.

Execute only after checking the visualized grasp and planning scene:

```bash
python3 scripts/execute_grasp.py \
  scan_output/grasp_commands/apple.json \
  --minimum-score 0.2 \
  --execute
```

The sequence is:

1. Open the selected gripper.
2. Move to the pregrasp pose.
3. Move to the grasp pose.
4. Close the gripper.

Execution aborts immediately if any command raises an error.

## Troubleshooting

### No synchronized camera images

Confirm that RGB and depth topics exist and publish continuously:

```bash
ros2 topic list -t | grep handeye
ros2 topic hz /left_handeye/color/image_raw/compressed
ros2 topic hz /left_handeye/aligned_depth_to_color/image_raw
```

The RGB and depth timestamps must be close enough for the approximate
synchronizer. Increasing `--camera-timeout` does not fix a missing topic or a
message-type mismatch.

### Current robot state timeout

Compare all clock settings:

```bash
ros2 param get /move_group use_sim_time
ros2 param get /robot_motion_server use_sim_time
ros2 topic echo /clock --once
```

All simulation nodes must use the same `/clock`. Restart nodes that report
`False`; changing the parameter after MoveIt has initialized may leave stale
state monitors, so a clean restart is preferred.

### Open3D import failure

Install Open3D in the same Python environment used to run the offline scripts.
ROS sourcing does not install the `open3d` Python package.

## Safety

Grasp execution and multi-view scanning can command both arms. Confirm the
planning scene, target TF frames, collision objects, speed limits, and emergency
stop before using `--execute` or running the scanner on hardware. Keep the
default dry-run workflow when reviewing newly generated grasp commands.
