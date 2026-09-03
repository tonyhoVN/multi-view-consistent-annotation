#!/usr/bin/env python3
"""Move both hand-eye cameras over a hemisphere and capture RGB-D images.

Each scan step contains a mirrored pair of poses. The pose with world-frame
``y > 0`` is assigned to the left arm and the pose with ``y < 0`` is assigned
to the right arm. Both targets are sent in one coordinated ``move_l_dual`` or 
``move_dual`` call.

Desired poses describe camera optical frames, not TCPs. At startup, TF is used
to obtain each rigid camera-to-TCP transform and convert camera poses to TCP
targets.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

from aux_math import (
    hemisphere_pairs,
    matrix_from_pose,
    matrix_to_pose,
    transform_from_euler,
)
from robot_api import RobotAPI, RobotAPIError


def parse_float_list(value: str) -> list[float]:
    """Parse a comma-separated CLI list of finite floating-point values."""
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exception:
        raise argparse.ArgumentTypeError(str(exception)) from exception
    if not values or not all(math.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of finite numbers")
    return values


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--center", type=float, nargs=3, default=(0.40, 0.0, 0.00),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--radius", type=float, default=0.4)
    parser.add_argument(
        "--azimuths", type=parse_float_list, default=parse_float_list("25,50,75"),
        help="positive angles in degrees, mirrored across world y=0",
    )
    parser.add_argument(
        "--elevations", type=parse_float_list, default=parse_float_list("45,60,75"),
        help="angles above the horizontal plane in degrees",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("scan_output"))
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
    parser.add_argument("--settle-time", type=float, default=1.0)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not all(math.isfinite(value) for value in args.center):
        raise ValueError("center coordinates must all be finite")
    if not math.isfinite(args.radius) or args.radius <= 0.0:
        raise ValueError("radius must be finite and greater than zero")
    if not all(0.0 < angle < 90.0 for angle in args.azimuths):
        raise ValueError("azimuths must be between 0 and 90 degrees")
    if not all(0.0 <= angle < 90.0 for angle in args.elevations):
        raise ValueError("elevations must be between 0 (inclusive) and 90 degrees")
    for name in ("motion_timeout", "camera_timeout", "tf_timeout", "clock_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be greater than zero")
    if not math.isfinite(args.settle_time) or args.settle_time < 0.0:
        raise ValueError("settle-time must not be negative")
    if not math.isfinite(args.target_preview_time) or args.target_preview_time < 0.0:
        raise ValueError("target-preview-time must not be negative")


def run_scan(args: argparse.Namespace) -> None:
    center = np.asarray(args.center, dtype=np.float64)
    left_camera_quarter_turn = transform_from_euler("z", -90.0, degrees=True)
    right_camera_quarter_turn = transform_from_euler("z", 90.0, degrees=True)
    azimuths = [i for i in range(10, 150, 25)]
    elevations = [45,60,75]
    # pairs = hemisphere_pairs(center, args.radius, args.azimuths, args.elevations)
    pairs = hemisphere_pairs(center, args.radius, azimuths, elevations)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    steps_dir = reset_generated_directories(output_dir)
    manifest_path = output_dir / "manifest.json"

    manifest = {
        "world_frame": args.world_frame,
        "center_xyz": center.tolist(),
        "radius": args.radius,
        "left_assignment": "world y > 0",
        "right_assignment": "world y < 0",
        "left_camera_local_z_rotation_deg": -90.0,
        "right_camera_local_z_rotation_deg": 90.0,
        "use_sim_time": args.use_sim_time,
        "steps_directory": str(steps_dir.relative_to(output_dir)),
        "captures": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

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
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

        center_pose = matrix_from_pose(center, [0, 0, 0, 1])
        robot.publish_static_transform(args.world_frame, "scan_center", center_pose)

        for step_number, (azimuth, elevation, left_camera_pose, right_camera_pose) in enumerate(
            pairs, start=1
        ):
            # Roll the optical frame while preserving its position and look direction.
            left_camera_pose = left_camera_pose @ left_camera_quarter_turn
            right_camera_pose = right_camera_pose @ right_camera_quarter_turn

            left_tcp_pose = left_camera_pose @ left_camera_to_tcp
            right_tcp_pose = right_camera_pose @ right_camera_to_tcp

            left_tcp_position, left_tcp_quaternion = matrix_to_pose(left_tcp_pose)
            right_tcp_position, right_tcp_quaternion = matrix_to_pose(right_tcp_pose)

            left_target_frame = f"{args.left_target_frame_prefix}_{step_number:03d}"
            right_target_frame = f"{args.right_target_frame_prefix}_{step_number:03d}"
            robot.publish_dynamic_transform(
                "scan_center", left_target_frame, np.linalg.inv(center_pose) @ left_camera_pose
            )
            # robot.publish_dynamic_transform(
            #     "scan_center", left_target_frame + "_tcp", np.linalg.inv(center_pose) @ left_tcp_pose
            # )
            robot.publish_dynamic_transform(
                "scan_center", right_target_frame, np.linalg.inv(center_pose) @ right_camera_pose
            )
            # robot.publish_dynamic_transform(
            #     "scan_center", right_target_frame + "_tcp", np.linalg.inv(center_pose) @ right_tcp_pose
            # )
            print(
                f"Published target TFs: {left_target_frame}, {right_target_frame}"
            )
            if args.target_preview_time:
                time.sleep(args.target_preview_time)

            print(
                f"[{step_number}/{len(pairs)}] moving both arms: "
                f"azimuth=+/-{azimuth:g} deg, elevation={elevation:g} deg"
            )

            breakpoint()

            try:
                # result = robot.move_dual(
                #     RobotAPI.pose(left_tcp_position, left_tcp_quaternion),
                #     RobotAPI.pose(right_tcp_position, right_tcp_quaternion),
                #     planning_group=args.planning_group,
                #     timeout=args.motion_timeout,
                # )
                result = robot.move_l_dual(
                    RobotAPI.pose(left_tcp_position, left_tcp_quaternion),
                    RobotAPI.pose(right_tcp_position, right_tcp_quaternion),
                    planning_group=args.planning_group,
                    position_tolerance=0.015,       # meters: 5 mm
                    orientation_tolerance=0.1,    # radians: 0.57 deg
                    tracking_timeout=args.motion_timeout + 2.0,  # seconds: allow extra time for dual-arm tracking
                    timeout=args.motion_timeout,
                )
            except RobotAPIError as error:
                print(f"  motion failed; skipping capture and continuing: {error}")
                manifest["captures"].append(
                    {
                        "step": step_number,
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
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
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
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"  captured {left_color} and {right_color}")

    captured_pairs = sum(
        capture["status"] == "captured" for capture in manifest["captures"]
    )
    failed_pairs = len(manifest["captures"]) - captured_pairs
    print(
        f"Scan complete: {captured_pairs * 2} views saved under {output_dir}; "
        f"{failed_pairs} motion pair(s) skipped"
    )

    # Comeback home 
    try:
        robot.move_group_state("ready", planning_group="dual_arm")
        print("Motion finished")
    except RobotAPIError as error:
        print(f"ready motion failed; continuing with scan: {error}")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    run_scan(args)


if __name__ == "__main__":
    main()
