#!/usr/bin/env python3
"""Move both hand-eye cameras over a hemisphere and capture RGB-D images.

Each scan step contains a mirrored pair of poses. The pose with world-frame
``y > 0`` is assigned to the left arm and the pose with ``y < 0`` is assigned
to the right arm. Both targets are sent in one coordinated ``move_l_dual``
call.

Desired poses describe camera optical frames, not TCPs. At startup, TF is used
to obtain each rigid camera-to-TCP transform and convert camera poses to TCP
targets.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from multi_view_scan.aux_math import (  # noqa: E402
    matrix_from_pose,
    matrix_to_pose,
    transform_from_euler,
)
from multi_view_scan.moveit_ik import MoveItIKClient  # noqa: E402
from robot_api import RobotAPI, RobotAPIError  # noqa: E402
from multi_view_scan.scan_trajectory import (  # noqa: E402
    CameraIntrinsics,
    ReachableViewpoint,
    SpiralViewpoint,
    combined_edge_costs,
    dual_pose_distance,
    joint_distance_matrix,
    measure_path,
    nearest_neighbor_open_path,
    pairwise_overlap_matrix,
    pose_distance_matrix,
    sample_scan_volume,
    spiral_hemisphere_pairs,
    two_opt_open_path,
)


DEFAULT_CAMERA_YAML = Path(
    "/home/hier-tony/Projects/dual_manipulation_isaac_sim/env/urdf/camera.yaml"
)
DEFAULT_SCAN_CONFIG = (
    Path(__file__).resolve().parents[2] / "config" / "multi_view_scan.yaml"
)


def transform_record(transform: np.ndarray) -> dict[str, list[float]]:
    """Return JSON-friendly XYZ and XYZW values for a transform."""
    position, quaternion = matrix_to_pose(transform)
    return {
        "position_xyz": position.tolist(),
        "quaternion_xyzw": quaternion.tolist(),
    }


def actual_camera_transform(
    robot: RobotAPI, world_frame: str, camera_frame: str, timeout: float
) -> np.ndarray:
    """Read an actual camera pose from TF as a homogeneous transform."""
    position, quaternion = robot.get_transform_pos_quat(
        world_frame, camera_frame, timeout=timeout
    )
    return matrix_from_pose(position, quaternion)


def reset_generated_directories(output_dir: Path) -> Path:
    """Remove generated scan products safely and recreate the steps directory."""
    directory_names = ("steps", "grasp_commands", "mask_segment", "pointclouds")
    targets = [output_dir / name for name in directory_names]

    # Validate every path before deleting any previous scan data.
    for target in targets:
        if target.is_symlink():
            raise RuntimeError(f"refusing to clean symlinked directory: {target}")
        if target.exists() and not target.is_dir():
            raise RuntimeError(f"cleanup path exists but is not a directory: {target}")

    for target in targets:
        if target.exists():
            shutil.rmtree(target)
            print(f"Removed previous generated data: {target}")

    steps_dir = output_dir / "steps"
    steps_dir.mkdir(parents=True)
    return steps_dir


def load_camera_intrinsics(path: Path) -> CameraIntrinsics:
    """Load the pinhole intrinsics used by projected-overlap constraints."""
    calibration_path = path.expanduser().resolve()
    if not calibration_path.is_file():
        raise FileNotFoundError(f"camera calibration does not exist: {calibration_path}")
    data = yaml.safe_load(calibration_path.read_text(encoding="utf-8"))
    try:
        intrinsics = data["intrinsics"]
        width, height = intrinsics["resolution"]
        return CameraIntrinsics(
            width=int(width),
            height=int(height),
            fx=float(intrinsics["fx"]),
            fy=float(intrinsics["fy"]),
            cx=float(intrinsics["cx"]),
            cy=float(intrinsics["cy"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid camera intrinsics in {calibration_path}: {error}") from error


def load_scan_config(
    path: Path, parser: argparse.ArgumentParser
) -> dict[str, object]:
    """Load and type-check argparse defaults from a multiview scan YAML file."""
    config_path = path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"multiview scan config does not exist: {config_path}")
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(
        document.get("multi_view_scan"), dict
    ):
        raise ValueError(
            f"{config_path} must contain a 'multi_view_scan' parameter mapping"
        )
    values = document["multi_view_scan"]
    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest not in {"help", "config"}
    }
    unknown = sorted(set(values) - set(actions))
    if unknown:
        raise ValueError(
            f"unknown multiview scan parameter(s) in {config_path}: "
            + ", ".join(unknown)
        )

    # Apply argparse's declared scalar type to YAML scalars and list elements.
    defaults: dict[str, object] = {}
    for name, value in values.items():
        action = actions[name]
        if isinstance(action, argparse.BooleanOptionalAction):
            if not isinstance(value, bool):
                raise ValueError(f"YAML parameter {name!r} must be true or false")
            defaults[name] = value
            continue
        if action.choices is not None and value not in action.choices:
            choices = ", ".join(str(choice) for choice in action.choices)
            raise ValueError(f"YAML parameter {name!r} must be one of: {choices}")
        converter = action.type
        if action.nargs is not None:
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"YAML parameter {name!r} must be a list")
            defaults[name] = [converter(item) if converter else item for item in value]
        else:
            defaults[name] = converter(value) if converter else value
    return defaults


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_SCAN_CONFIG,
        help="YAML parameter file; explicit command-line options override it",
    )
    parser.add_argument(
        "--center", type=float, nargs=3, default=(0.40, 0.0, 0.00),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.4,
        help="camera distance from the scan center in meters",
    )
    parser.add_argument(
        "--view-count",
        type=int,
        default=30,
        help="number of mirrored spiral pose pairs generated before IK filtering",
    )
    parser.add_argument(
        "--azimuth-bounds", type=float, nargs=2, default=(10.0, 135.0),
        metavar=("MIN", "MAX"),
        help="spiral azimuth bounds in degrees on the left half",
    )
    parser.add_argument(
        "--elevation-bounds", type=float, nargs=2, default=(30.0, 75.0),
        metavar=("MIN", "MAX"),
        help="spiral elevation bounds in degrees",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("scan_output"))
    parser.add_argument(
        "--camera-yaml",
        type=Path,
        default=DEFAULT_CAMERA_YAML,
        help="camera calibration used for projected-overlap scoring",
    )
    parser.add_argument("--world-frame", default="world")
    parser.add_argument("--left-tcp-frame", default="left_fr3_hand_tcp")
    parser.add_argument("--right-tcp-frame", default="right_fr3_hand_tcp")
    parser.add_argument("--left-camera-frame", default="left_handeye_camera_color_optical_frame")
    parser.add_argument("--right-camera-frame", default="right_handeye_camera_color_optical_frame")
    parser.add_argument("--left-target-frame-prefix", default="scan_left_camera_target")
    parser.add_argument("--right-target-frame-prefix", default="scan_right_camera_target")
    parser.add_argument("--left-rgb-topic", default="/left_handeye/color/image_raw/compressed")
    parser.add_argument(
        "--left-depth-topic", default="/left_handeye/aligned_depth_to_color/image_raw"
    )
    parser.add_argument("--right-rgb-topic", default="/right_handeye/color/image_raw/compressed")
    parser.add_argument(
        "--right-depth-topic", default="/right_handeye/aligned_depth_to_color/image_raw"
    )
    parser.add_argument("--planning-group", default="dual_arm")
    parser.add_argument(
        "--ik-service",
        default="/compute_ik",
        help="MoveIt GetPositionIK service used only for reachability filtering",
    )
    parser.add_argument(
        "--joint-state-topic",
        default="/joint_states",
        help="full robot state used as the multi-tip IK seed",
    )
    parser.add_argument(
        "--display-trajectory-topic",
        default="/display_planned_path",
        help="MoveIt DisplayTrajectory topic for the route after 2-opt",
    )
    parser.add_argument(
        "--initial-display-trajectory-topic",
        default="/scan/display_trajectory_before_2opt",
        help="MoveIt DisplayTrajectory topic for the greedy route before 2-opt",
    )
    parser.add_argument(
        "--left-camera-path-topic",
        default="/scan/left_camera_path",
        help="nav_msgs/Path topic for the optimized left-camera route",
    )
    parser.add_argument(
        "--right-camera-path-topic",
        default="/scan/right_camera_path",
        help="nav_msgs/Path topic for the optimized right-camera route",
    )
    parser.add_argument(
        "--trajectory-comparison-topic",
        default="/scan/trajectory_comparison",
        help="MarkerArray topic containing wide before/after camera routes",
    )
    parser.add_argument(
        "--before-trajectory-marker-topic",
        default="/scan/trajectory_before_2opt",
        help="MarkerArray topic containing only the route before 2-opt",
    )
    parser.add_argument(
        "--optimized-trajectory-marker-topic",
        default="/scan/trajectory_after_2opt",
        help="MarkerArray topic containing only the route after 2-opt",
    )
    parser.add_argument(
        "--trajectory-line-width",
        type=float,
        default=0.015,
        help="width in meters of RViz trajectory comparison lines",
    )
    parser.add_argument(
        "--before-trajectory-color-rgb",
        type=int,
        nargs=3,
        default=(255, 0, 255),
        metavar=("R", "G", "B"),
        help="RGB color for the greedy route before 2-opt",
    )
    parser.add_argument(
        "--optimized-trajectory-color-rgb",
        type=int,
        nargs=3,
        default=(25, 255, 0),
        metavar=("R", "G", "B"),
        help="RGB color for the optimized route after 2-opt",
    )
    parser.add_argument(
        "--ik-timeout",
        type=float,
        default=0.25,
        help="MoveIt solver timeout per candidate in seconds",
    )
    parser.add_argument(
        "--joint-state-timeout",
        type=float,
        default=2.0,
        help="wall-clock timeout for IK service/state messages in seconds",
    )
    parser.add_argument(
        "--scan-volume-radius",
        type=float,
        default=0.15,
        help="radius in meters of the spherical overlap surface proxy",
    )
    parser.add_argument(
        "--projection-samples",
        type=int,
        default=2048,
        help="deterministic surface samples used by overlap projection",
    )
    parser.add_argument(
        "--minimum-neighbor-overlap",
        type=float,
        default=0.35,
        help="hard minimum [0,1] overlap for every consecutive camera pair",
    )
    parser.add_argument(
        "--overlap-weight",
        type=float,
        default=0.5,
        help="penalty weight for low overlap in each TSP edge cost",
    )
    parser.add_argument(
        "--distance-metric",
        choices=("joint", "pose"),
        default="joint",
        help="minimize IK joint displacement or coordinated TCP-pose distance",
    )
    parser.add_argument(
        "--pose-translation-weight",
        type=float,
        default=1.0,
        help="weight for dual-TCP translation when distance-metric=pose",
    )
    parser.add_argument(
        "--pose-rotation-weight",
        type=float,
        default=0.1,
        help="meters-per-radian weight for dual-TCP rotation in pose distance",
    )
    parser.add_argument(
        "--two-opt-passes",
        type=int,
        default=30,
        help="maximum deterministic overlap-constrained 2-opt passes",
    )
    parser.add_argument("--motion-timeout", type=float, default=15.0)
    parser.add_argument("--camera-timeout", type=float, default=5.0)
    parser.add_argument("--tf-timeout", type=float, default=2.0)
    parser.add_argument("--clock-timeout", type=float, default=5.0)
    parser.add_argument(
        "--use-sim-time",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use and validate Isaac Sim /clock (default: enabled)",
    )
    parser.add_argument(
        "--target-preview-time",
        type=float,
        default=1.0,
        help="seconds to show each target TF in RViz before starting its motion",
    )
    parser.add_argument(
        "--trajectory-point-time",
        type=float,
        default=0.25,
        help="seconds between IK waypoints in the RViz robot animation",
    )
    parser.add_argument(
        "--trajectory-preview-time",
        type=float,
        default=5.0,
        help="seconds to show the complete RViz preview before the first motion",
    )
    parser.add_argument("--settle-time", type=float, default=3.0)
    return parser


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse the selected YAML file first, then apply explicit CLI overrides."""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_SCAN_CONFIG)
    selected, _ = config_parser.parse_known_args(arguments)
    parser = build_parser()
    parser.set_defaults(**load_scan_config(selected.config, parser))
    return parser.parse_args(arguments)


def validate_args(args: argparse.Namespace) -> None:
    if not all(math.isfinite(value) for value in args.center):
        raise ValueError("center coordinates must all be finite")
    if not math.isfinite(args.radius) or args.radius <= 0.0:
        raise ValueError("radius must be finite and greater than zero")
    if args.view_count <= 0:
        raise ValueError("view-count must be positive")
    if not 0.0 < args.azimuth_bounds[0] < args.azimuth_bounds[1] < 180.0:
        raise ValueError("azimuth-bounds must satisfy 0 < MIN < MAX < 180")
    if not 0.0 <= args.elevation_bounds[0] < args.elevation_bounds[1] < 90.0:
        raise ValueError("elevation-bounds must satisfy 0 <= MIN < MAX < 90")
    for name in (
        "motion_timeout",
        "camera_timeout",
        "tf_timeout",
        "clock_timeout",
        "ik_timeout",
        "joint_state_timeout",
        "scan_volume_radius",
        "trajectory_point_time",
        "trajectory_line_width",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be greater than zero")
    if args.projection_samples <= 0:
        raise ValueError("projection-samples must be positive")
    if not 0.0 <= args.minimum_neighbor_overlap <= 1.0:
        raise ValueError("minimum-neighbor-overlap must lie in [0, 1]")
    if not math.isfinite(args.overlap_weight) or args.overlap_weight < 0.0:
        raise ValueError("overlap-weight must be finite and nonnegative")
    pose_weights = (args.pose_translation_weight, args.pose_rotation_weight)
    if (
        not all(math.isfinite(weight) and weight >= 0.0 for weight in pose_weights)
        or sum(pose_weights) <= 0.0
    ):
        raise ValueError("pose-distance weights must be nonnegative and not both zero")
    if args.two_opt_passes < 0:
        raise ValueError("two-opt-passes must be nonnegative")
    for name in ("before_trajectory_color_rgb", "optimized_trajectory_color_rgb"):
        components = getattr(args, name)
        if len(components) != 3 or any(
            not 0 <= component <= 255 for component in components
        ):
            raise ValueError(
                f"{name.replace('_', '-')} must contain three values in [0, 255]"
            )
    if not math.isfinite(args.settle_time) or args.settle_time < 0.0:
        raise ValueError("settle-time must not be negative")
    if not math.isfinite(args.target_preview_time) or args.target_preview_time < 0.0:
        raise ValueError("target-preview-time must not be negative")
    if (
        not math.isfinite(args.trajectory_preview_time)
        or args.trajectory_preview_time < 0.0
    ):
        raise ValueError("trajectory-preview-time must not be negative")


def apply_camera_rolls(
    viewpoints: list[SpiralViewpoint],
    left_roll: np.ndarray,
    right_roll: np.ndarray,
) -> list[SpiralViewpoint]:
    """Apply fixed local optical-frame rolls to every spiral viewpoint."""
    return [
        SpiralViewpoint(
            source_index=viewpoint.source_index,
            azimuth_deg=viewpoint.azimuth_deg,
            elevation_deg=viewpoint.elevation_deg,
            left_camera_pose=viewpoint.left_camera_pose @ left_roll,
            right_camera_pose=viewpoint.right_camera_pose @ right_roll,
        )
        for viewpoint in viewpoints
    ]


def filter_reachable_viewpoints(
    ik_client: MoveItIKClient,
    viewpoints: list[SpiralViewpoint],
    left_camera_to_tcp: np.ndarray,
    right_camera_to_tcp: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[ReachableViewpoint], tuple[str, ...], np.ndarray, list[dict]]:
    """Discard pose pairs that fail collision-aware multi-tip MoveIt IK."""
    if not ik_client.wait_for_service(args.joint_state_timeout):
        raise TimeoutError(
            f"MoveIt IK service {args.ik_service!r} is unavailable after "
            f"{args.joint_state_timeout:.3f} seconds"
        )
    current_positions = ik_client.wait_for_joint_state(args.joint_state_timeout)
    solved: list[tuple[SpiralViewpoint, np.ndarray, np.ndarray, dict[str, float]]] = []
    rejected: list[dict] = []

    # Convert each camera target to TCP and test both tips in one IK request.
    for number, viewpoint in enumerate(viewpoints, start=1):
        left_tcp_pose = viewpoint.left_camera_pose @ left_camera_to_tcp
        right_tcp_pose = viewpoint.right_camera_pose @ right_camera_to_tcp
        left_position, left_quaternion = matrix_to_pose(left_tcp_pose)
        right_position, right_quaternion = matrix_to_pose(right_tcp_pose)
        result = ik_client.solve(
            {
                args.left_tcp_frame: RobotAPI.pose(left_position, left_quaternion),
                args.right_tcp_frame: RobotAPI.pose(right_position, right_quaternion),
            },
            planning_group=args.planning_group,
            frame_id=args.world_frame,
            timeout=args.ik_timeout,
            joint_state_timeout=args.joint_state_timeout,
            avoid_collisions=True,
        )
        if not result.success:
            rejected.append(
                {
                    "spiral_index": viewpoint.source_index + 1,
                    "azimuth_deg": viewpoint.azimuth_deg,
                    "elevation_deg": viewpoint.elevation_deg,
                    "error_code": result.error_code,
                    "reason": result.message,
                }
            )
            print(
                f"IK [{number}/{len(viewpoints)}] rejected spiral "
                f"{viewpoint.source_index + 1:03d}: {result.message}"
            )
            continue
        solved.append(
            (viewpoint, left_tcp_pose, right_tcp_pose, dict(result.joint_positions))
        )
        print(
            f"IK [{number}/{len(viewpoints)}] accepted spiral "
            f"{viewpoint.source_index + 1:03d}"
        )

    if not solved:
        raise RuntimeError("MoveIt IK rejected every spiral viewpoint")

    # Compare only actuated motion joints; gripper joints stay constant during scanning.
    common_names = set(solved[0][3])
    for _, _, _, positions in solved[1:]:
        common_names.intersection_update(positions)
    joint_names = tuple(
        sorted(name for name in common_names if "finger_joint" not in name)
    )
    if not joint_names:
        raise RuntimeError("MoveIt IK solutions contain no common motion joints")
    missing_current = [name for name in joint_names if name not in current_positions]
    if missing_current:
        raise RuntimeError(
            "current /joint_states is missing IK joints: " + ", ".join(missing_current)
        )

    reachable = [
        ReachableViewpoint(
            viewpoint=viewpoint,
            left_tcp_pose=left_tcp_pose,
            right_tcp_pose=right_tcp_pose,
            joint_values=np.asarray(
                [positions[name] for name in joint_names], dtype=np.float64
            ),
        )
        for viewpoint, left_tcp_pose, right_tcp_pose, positions in solved
    ]
    start_values = np.asarray(
        [current_positions[name] for name in joint_names], dtype=np.float64
    )
    return reachable, joint_names, start_values, rejected


def optimize_viewpoint_path(
    reachable: list[ReachableViewpoint],
    start_joint_values: np.ndarray,
    start_left_tcp_pose: np.ndarray,
    start_right_tcp_pose: np.ndarray,
    center: np.ndarray,
    intrinsics: CameraIntrinsics,
    args: argparse.Namespace,
) -> tuple[list[ReachableViewpoint], list[ReachableViewpoint], dict]:
    """Build a constrained open TSP path and refine it with deterministic 2-opt."""
    projection_points = sample_scan_volume(
        center, args.scan_volume_radius, args.projection_samples
    )
    overlaps = pairwise_overlap_matrix(reachable, projection_points, intrinsics)
    if args.distance_metric == "joint":
        motion_distances = joint_distance_matrix(reachable)
        start_distances = np.asarray(
            [
                np.linalg.norm(viewpoint.joint_values - start_joint_values)
                for viewpoint in reachable
            ],
            dtype=np.float64,
        )
    elif args.distance_metric == "pose":
        motion_distances = pose_distance_matrix(
            reachable,
            args.pose_translation_weight,
            args.pose_rotation_weight,
        )
        start_distances = np.asarray(
            [
                dual_pose_distance(
                    start_left_tcp_pose,
                    start_right_tcp_pose,
                    viewpoint.left_tcp_pose,
                    viewpoint.right_tcp_pose,
                    args.pose_translation_weight,
                    args.pose_rotation_weight,
                )
                for viewpoint in reachable
            ],
            dtype=np.float64,
        )
    else:
        raise ValueError("distance-metric must be 'joint' or 'pose'")
    edge_costs = combined_edge_costs(
        motion_distances, overlaps, args.overlap_weight
    )

    # Seed the open TSP greedily, then improve it without breaking overlap edges.
    try:
        initial_path = nearest_neighbor_open_path(
            edge_costs,
            overlaps,
            start_distances,
            args.minimum_neighbor_overlap,
        )
    except ValueError as error:
        raise RuntimeError(
            "cannot connect every reachable viewpoint with the requested "
            f"minimum overlap {args.minimum_neighbor_overlap:.3f}: {error}"
        ) from error
    optimized_path = two_opt_open_path(
        initial_path,
        edge_costs,
        overlaps,
        start_distances,
        args.minimum_neighbor_overlap,
        args.two_opt_passes,
    )
    initial_metrics = measure_path(
        initial_path, motion_distances, overlaps, edge_costs, start_distances
    )
    optimized_metrics = measure_path(
        optimized_path, motion_distances, overlaps, edge_costs, start_distances
    )
    neighbor_overlaps = [
        float(overlaps[first, second])
        for first, second in zip(optimized_path, optimized_path[1:])
    ]
    diagnostics = {
        "distance_metric": args.distance_metric,
        "pose_translation_weight": args.pose_translation_weight,
        "pose_rotation_weight": args.pose_rotation_weight,
        "initial_order": [
            reachable[index].viewpoint.source_index + 1 for index in initial_path
        ],
        "optimized_order": [
            reachable[index].viewpoint.source_index + 1 for index in optimized_path
        ],
        "initial_metrics": asdict(initial_metrics),
        "optimized_metrics": asdict(optimized_metrics),
        "optimized_neighbor_overlaps": neighbor_overlaps,
    }
    return (
        [reachable[index] for index in initial_path],
        [reachable[index] for index in optimized_path],
        diagnostics,
    )


def reachable_viewpoint_record(
    reachable: ReachableViewpoint, joint_names: tuple[str, ...]
) -> dict:
    """Return the IK and pose data needed to audit one retained candidate."""
    viewpoint = reachable.viewpoint
    return {
        "spiral_index": viewpoint.source_index + 1,
        "azimuth_deg": viewpoint.azimuth_deg,
        "elevation_deg": viewpoint.elevation_deg,
        "left_camera_pose": transform_record(viewpoint.left_camera_pose),
        "right_camera_pose": transform_record(viewpoint.right_camera_pose),
        "left_tcp_pose": transform_record(reachable.left_tcp_pose),
        "right_tcp_pose": transform_record(reachable.right_tcp_pose),
        "ik_joint_positions": {
            name: float(value)
            for name, value in zip(joint_names, reachable.joint_values)
        },
    }


def run_scan(args: argparse.Namespace) -> None:
    center = np.asarray(args.center, dtype=np.float64)
    left_camera_quarter_turn = transform_from_euler("z", -90.0, degrees=True)
    right_camera_quarter_turn = transform_from_euler("z", 90.0, degrees=True)
    intrinsics = load_camera_intrinsics(args.camera_yaml)

    # Generate camera-frame targets before starting ROS; IK will prune them later.
    spiral_viewpoints = spiral_hemisphere_pairs(
        center,
        args.radius,
        args.view_count,
        args.azimuth_bounds,
        args.elevation_bounds,
    )
    spiral_viewpoints = apply_camera_rolls(
        spiral_viewpoints, left_camera_quarter_turn, right_camera_quarter_turn
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    steps_dir = reset_generated_directories(output_dir)
    manifest_path = output_dir / "manifest.json"

    manifest = {
        "configuration_file": str(args.config.expanduser().resolve()),
        "world_frame": args.world_frame,
        "center_xyz": center.tolist(),
        "radius": args.radius,
        "left_assignment": "world y > 0",
        "right_assignment": "world y < 0",
        "left_camera_local_z_rotation_deg": -90.0,
        "right_camera_local_z_rotation_deg": 90.0,
        "use_sim_time": args.use_sim_time,
        "steps_directory": str(steps_dir.relative_to(output_dir)),
        "camera_calibration": str(args.camera_yaml.expanduser().resolve()),
        "camera_intrinsics": asdict(intrinsics),
        "planner": {
            "pipeline": [
                "golden_angle_spiral",
                "collision_aware_multi_tip_ik_filter",
                "nearest_neighbor_open_tsp",
                "overlap_constrained_2_opt",
            ],
            "requested_candidate_count": args.view_count,
            "azimuth_bounds_deg": list(args.azimuth_bounds),
            "elevation_bounds_deg": list(args.elevation_bounds),
            "scan_volume_radius_m": args.scan_volume_radius,
            "projection_samples": args.projection_samples,
            "minimum_neighbor_overlap": args.minimum_neighbor_overlap,
            "overlap_weight": args.overlap_weight,
            "distance_metric": args.distance_metric,
            "pose_translation_weight": args.pose_translation_weight,
            "pose_rotation_weight": args.pose_rotation_weight,
            "two_opt_passes": args.two_opt_passes,
            "ik_service": args.ik_service,
            "ik_timeout_s": args.ik_timeout,
            "trajectory_visualization": {
                "before_2opt_display_trajectory_topic": (
                    args.initial_display_trajectory_topic
                ),
                "after_2opt_display_trajectory_topic": args.display_trajectory_topic,
                "left_camera_path_topic": args.left_camera_path_topic,
                "right_camera_path_topic": args.right_camera_path_topic,
                "comparison_marker_topic": args.trajectory_comparison_topic,
                "before_2opt_marker_topic": args.before_trajectory_marker_topic,
                "after_2opt_marker_topic": args.optimized_trajectory_marker_topic,
                "line_width_m": args.trajectory_line_width,
                "before_2opt_color_rgb": list(args.before_trajectory_color_rgb),
                "after_2opt_color_rgb": list(args.optimized_trajectory_color_rgb),
                "point_interval_s": args.trajectory_point_time,
                "preview_time_s": args.trajectory_preview_time,
                "continuous_path_collision_checked": False,
            },
        },
        "captures": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # RobotAPI stores one synchronized RGB-D stream, so use one client for each
    # hand-eye camera. Only the left-camera client sends robot commands.
    with ExitStack() as stack:
        robot = stack.enter_context(
            RobotAPI(
                node_name="multi_view_scan_left",
                call_timeout=args.motion_timeout,
                use_sim_time=args.use_sim_time,
                enable_camera=True,
                rgb_topic=args.left_rgb_topic,
                depth_topic=args.left_depth_topic,
            )
        )
        right_camera = stack.enter_context(
            RobotAPI(
                node_name="multi_view_scan_right_camera",
                call_timeout=args.motion_timeout,
                use_sim_time=args.use_sim_time,
                enable_camera=True,
                rgb_topic=args.right_rgb_topic,
                depth_topic=args.right_depth_topic,
            )
        )
        ik_client = stack.enter_context(
            MoveItIKClient(
                service_name=args.ik_service,
                joint_state_topic=args.joint_state_topic,
                display_trajectory_topic=args.display_trajectory_topic,
                initial_display_trajectory_topic=(
                    args.initial_display_trajectory_topic
                ),
                left_camera_path_topic=args.left_camera_path_topic,
                right_camera_path_topic=args.right_camera_path_topic,
                trajectory_comparison_topic=args.trajectory_comparison_topic,
                before_trajectory_marker_topic=args.before_trajectory_marker_topic,
                optimized_trajectory_marker_topic=(
                    args.optimized_trajectory_marker_topic
                ),
                use_sim_time=args.use_sim_time,
            )
        )

        # Confirm both private nodes receive an advancing Isaac Sim clock.
        if args.use_sim_time:
            left_sim_time = robot.wait_for_sim_time(args.clock_timeout)
            right_sim_time = right_camera.wait_for_sim_time(args.clock_timeout)
            print(
                "Simulation clock is advancing: "
                f"left={left_sim_time:.3f}s, right={right_sim_time:.3f}s"
            )

        # Fail before moving if either camera stream or hand-eye TF is missing.
        robot.get_camera_images(wait_timeout=args.camera_timeout)
        right_camera.get_camera_images(wait_timeout=args.camera_timeout)
        left_camera_to_tcp = matrix_from_pose(
            *robot.get_transform_pos_quat(
                args.left_camera_frame, args.left_tcp_frame, timeout=args.tf_timeout
            )
        )
        right_camera_to_tcp = matrix_from_pose(
            *robot.get_transform_pos_quat(
                args.right_camera_frame, args.right_tcp_frame, timeout=args.tf_timeout
            )
        )

        manifest["left_camera_to_tcp"] = transform_record(left_camera_to_tcp)
        manifest["right_camera_to_tcp"] = transform_record(right_camera_to_tcp)

        print("Moving dual_arm to SRDF ready configuration")
        try:
            ready_result = robot.move_group_state("ready", planning_group=args.planning_group)
            print("Motion finished")
        except RobotAPIError as error:
            print(f"  ready motion failed; continuing with scan: {error}")
            manifest["ready_motion"] = {
                "status": "motion_failed",
                "motion_error": str(error),
            }
        else:
            print("  robot reached ready configuration")
            manifest["ready_motion"] = {
                "status": "completed",
                "motion_result": ready_result.message,
            }

        # Prune unreachable pose pairs, then optimize one coordinated dual-arm path.
        reachable, joint_names, start_joint_values, rejected = (
            filter_reachable_viewpoints(
                ik_client,
                spiral_viewpoints,
                left_camera_to_tcp,
                right_camera_to_tcp,
                args,
            )
        )
        current_left_tcp = actual_camera_transform(
            robot, args.world_frame, args.left_tcp_frame, args.tf_timeout
        )
        current_right_tcp = actual_camera_transform(
            robot, args.world_frame, args.right_tcp_frame, args.tf_timeout
        )
        initial_viewpoints, optimized_viewpoints, path_diagnostics = (
            optimize_viewpoint_path(
                reachable,
                start_joint_values,
                current_left_tcp,
                current_right_tcp,
                center,
                intrinsics,
                args,
            )
        )
        manifest["planner"].update(
            {
                "reachable_candidate_count": len(reachable),
                "rejected_candidate_count": len(rejected),
                "joint_names": list(joint_names),
                "start_joint_positions": {
                    name: float(value)
                    for name, value in zip(joint_names, start_joint_values)
                },
                "start_left_tcp_pose": transform_record(current_left_tcp),
                "start_right_tcp_pose": transform_record(current_right_tcp),
                "rejected_candidates": rejected,
                "reachable_candidates": [
                    reachable_viewpoint_record(viewpoint, joint_names)
                    for viewpoint in reachable
                ],
                **path_diagnostics,
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        metrics = path_diagnostics["optimized_metrics"]
        print(
            f"Trajectory planned: {len(reachable)}/{len(spiral_viewpoints)} "
            f"candidates reachable, metric={args.distance_metric}, "
            f"motion distance={metrics['motion_distance']:.3f}, "
            f"minimum overlap={metrics['minimum_neighbor_overlap']:.3f}"
        )

        center_pose = matrix_from_pose(center, [0, 0, 0, 1])
        robot.publish_static_transform(args.world_frame, "scan_center", center_pose)

        # Publish both complete routes before executing the optimized motion.
        current_left_camera = actual_camera_transform(
            robot, args.world_frame, args.left_camera_frame, args.tf_timeout
        )
        current_right_camera = actual_camera_transform(
            robot, args.world_frame, args.right_camera_frame, args.tf_timeout
        )
        before_left_preview_poses = [
            RobotAPI.pose(*matrix_to_pose(transform))
            for transform in [
                current_left_camera,
                *[
                    item.viewpoint.left_camera_pose
                    for item in initial_viewpoints
                ],
            ]
        ]
        before_right_preview_poses = [
            RobotAPI.pose(*matrix_to_pose(transform))
            for transform in [
                current_right_camera,
                *[
                    item.viewpoint.right_camera_pose
                    for item in initial_viewpoints
                ],
            ]
        ]
        optimized_left_preview_poses = [
            RobotAPI.pose(*matrix_to_pose(transform))
            for transform in [
                current_left_camera,
                *[
                    item.viewpoint.left_camera_pose
                    for item in optimized_viewpoints
                ],
            ]
        ]
        optimized_right_preview_poses = [
            RobotAPI.pose(*matrix_to_pose(transform))
            for transform in [
                current_right_camera,
                *[
                    item.viewpoint.right_camera_pose
                    for item in optimized_viewpoints
                ],
            ]
        ]
        ik_client.publish_trajectory_comparison(
            joint_names=joint_names,
            start_positions=start_joint_values,
            before_waypoint_positions=[
                item.joint_values for item in initial_viewpoints
            ],
            optimized_waypoint_positions=[
                item.joint_values for item in optimized_viewpoints
            ],
            frame_id=args.world_frame,
            before_left_camera_poses=before_left_preview_poses,
            before_right_camera_poses=before_right_preview_poses,
            optimized_left_camera_poses=optimized_left_preview_poses,
            optimized_right_camera_poses=optimized_right_preview_poses,
            point_interval=args.trajectory_point_time,
            line_width=args.trajectory_line_width,
            before_color_rgb=args.before_trajectory_color_rgb,
            optimized_color_rgb=args.optimized_trajectory_color_rgb,
        )
        print(
            "Published RViz trajectory comparison: "
            f"before={args.initial_display_trajectory_topic}, "
            f"after={args.display_trajectory_topic}, "
            f"wide lines={args.trajectory_comparison_topic}, "
            f"isolated before={args.before_trajectory_marker_topic}, "
            f"isolated after={args.optimized_trajectory_marker_topic}"
        )
        animation_duration = max(
            len(initial_viewpoints), len(optimized_viewpoints)
        ) * args.trajectory_point_time
        preview_wait = (
            max(args.trajectory_preview_time, animation_duration)
            if args.trajectory_preview_time
            else 0.0
        )
        manifest["planner"]["trajectory_visualization"].update(
            {
                "animation_duration_s": animation_duration,
                "effective_preview_wait_s": preview_wait,
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        if preview_wait:
            print(f"Waiting {preview_wait:.2f}s for the RViz preview before motion")
            time.sleep(preview_wait)

        # Execute the optimized path; failures remain recorded but do not stop the scan.
        breakpoint()
        for step_number, reachable_viewpoint in enumerate(
            optimized_viewpoints, start=1
        ):
            viewpoint = reachable_viewpoint.viewpoint
            azimuth = viewpoint.azimuth_deg
            elevation = viewpoint.elevation_deg
            left_camera_pose = viewpoint.left_camera_pose
            right_camera_pose = viewpoint.right_camera_pose
            left_tcp_pose = reachable_viewpoint.left_tcp_pose
            right_tcp_pose = reachable_viewpoint.right_tcp_pose

            left_tcp_position, left_tcp_quaternion = matrix_to_pose(left_tcp_pose)
            right_tcp_position, right_tcp_quaternion = matrix_to_pose(right_tcp_pose)

            left_target_frame = f"{args.left_target_frame_prefix}_{step_number:03d}"
            right_target_frame = f"{args.right_target_frame_prefix}_{step_number:03d}"
            robot.publish_dynamic_transform(
                "scan_center", left_target_frame, np.linalg.inv(center_pose) @ left_camera_pose
            )
            robot.publish_dynamic_transform(
                "scan_center",
                left_target_frame + "_tcp",
                np.linalg.inv(center_pose) @ left_tcp_pose,
            )
            robot.publish_dynamic_transform(
                "scan_center", right_target_frame, np.linalg.inv(center_pose) @ right_camera_pose
            )
            robot.publish_dynamic_transform(
                "scan_center",
                right_target_frame + "_tcp",
                np.linalg.inv(center_pose) @ right_tcp_pose,
            )
            print(
                f"Published target TFs: {left_target_frame}, {right_target_frame}"
            )
            if args.target_preview_time:
                time.sleep(args.target_preview_time)

            print(
                f"[{step_number}/{len(optimized_viewpoints)}] moving both arms: "
                f"spiral={viewpoint.source_index + 1:03d}, "
                f"azimuth=+/-{azimuth:g} deg, elevation={elevation:g} deg"
            )

            try:
                result = robot.move_l_dual(
                    RobotAPI.pose(left_tcp_position, left_tcp_quaternion),
                    RobotAPI.pose(right_tcp_position, right_tcp_quaternion),
                    planning_group=args.planning_group,
                    position_tolerance=0.015,
                    orientation_tolerance=0.1,
                    tracking_timeout=args.motion_timeout + 2.0,
                    timeout=args.motion_timeout,
                )
            except RobotAPIError as error:
                print(f"  motion failed; skipping capture and continuing: {error}")
                manifest["captures"].append(
                    {
                        "step": step_number,
                        "spiral_index": viewpoint.source_index + 1,
                        "azimuth_deg": azimuth,
                        "elevation_deg": elevation,
                        "status": "motion_failed",
                        "motion_error": str(error),
                        "left": {
                            "assignment": "y > 0",
                            "target_frame": left_target_frame,
                            "desired_camera_pose": transform_record(left_camera_pose),
                        },
                        "right": {
                            "assignment": "y < 0",
                            "target_frame": right_target_frame,
                            "desired_camera_pose": transform_record(right_camera_pose),
                        },
                    }
                )
                manifest_path.write_text(
                    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
                )
                continue

            if args.settle_time:
                time.sleep(args.settle_time)

            step_dir = steps_dir / f"step_{step_number:03d}"
            left_color, left_depth = robot.save_camera_images(
                str(step_dir / "left_color.png"),
                str(step_dir / "left_depth.png"),
                wait_timeout=args.camera_timeout,
            )
            right_color, right_depth = right_camera.save_camera_images(
                str(step_dir / "right_color.png"),
                str(step_dir / "right_depth.png"),
                wait_timeout=args.camera_timeout,
            )

            actual_left = actual_camera_transform(
                robot, args.world_frame, args.left_camera_frame, args.tf_timeout
            )
            actual_right = actual_camera_transform(
                robot, args.world_frame, args.right_camera_frame, args.tf_timeout
            )
            manifest["captures"].append(
                {
                    "step": step_number,
                    "spiral_index": viewpoint.source_index + 1,
                    "azimuth_deg": azimuth,
                    "elevation_deg": elevation,
                    "status": "captured",
                    "motion_result": result.message,
                    "left": {
                        "assignment": "y > 0",
                        "target_frame": left_target_frame,
                        "desired_camera_pose": transform_record(left_camera_pose),
                        "actual_camera_pose": transform_record(actual_left),
                        "color_image": str(left_color.relative_to(output_dir)),
                        "depth_image": str(left_depth.relative_to(output_dir)),
                    },
                    "right": {
                        "assignment": "y < 0",
                        "target_frame": right_target_frame,
                        "desired_camera_pose": transform_record(right_camera_pose),
                        "actual_camera_pose": transform_record(actual_right),
                        "color_image": str(right_color.relative_to(output_dir)),
                        "depth_image": str(right_depth.relative_to(output_dir)),
                    },
                }
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            print(f"  captured {left_color} and {right_color}")

        captured_pairs = sum(
            capture["status"] == "captured" for capture in manifest["captures"]
        )
        failed_pairs = len(manifest["captures"]) - captured_pairs
        print(
            f"Scan complete: {captured_pairs * 2} views saved under {output_dir}; "
            f"{failed_pairs} motion pair(s) skipped"
        )

        # Return to ready while the RobotAPI context is still alive.
        try:
            robot.move_group_state("ready", planning_group=args.planning_group)
            print("Robot returned to ready configuration")
        except RobotAPIError as error:
            print(f"final ready motion failed; scan data is preserved: {error}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    run_scan(args)


if __name__ == "__main__":
    main()
