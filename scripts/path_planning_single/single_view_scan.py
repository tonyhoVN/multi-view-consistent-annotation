#!/usr/bin/env python3
"""Plan and execute a Kinova single-camera multiview hemisphere scan."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Sequence

import numpy as np
import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from multi_view_scan.aux_math import (  # noqa: E402
    matrix_from_pose,
    matrix_to_pose,
)
from multi_view_scan.isaac_segmentation import (  # noqa: E402
    IsaacSegmentationClient,
    SegmentationServiceError,
)
from multi_view_scan.moveit_ik import build_display_trajectory  # noqa: E402
from multi_view_scan.scan_trajectory import (  # noqa: E402
    combined_edge_costs,
    measure_path,
    nearest_neighbor_open_path,
    sample_scan_volume,
    two_opt_open_path,
)
from path_planning_single.planning import (  # noqa: E402
    ReachableViewpoint,
    baseline_orders,
    joint_distances,
    overlap_violation_count,
    pairwise_overlaps,
    sample_hemisphere,
)
from path_planning_single.single_path_planner import (  # noqa: E402
    DEFAULT_CONFIG,
    StandardIKClient,
    build_visualizations,
    load_configuration,
    load_intrinsics,
    validate_configuration,
)
from robot_api import RobotAPI, RobotAPIError  # noqa: E402
from scan_layout import ScanRunLayout  # noqa: E402


TRAJECTORY_MODES = ("random", "spiral", "hamilton_2opt")


def build_parser() -> argparse.ArgumentParser:
    """Create runtime overrides for planning, Kinova motion, and capture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-suffix")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--trajectory-mode", choices=TRAJECTORY_MODES)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--planning-group")
    parser.add_argument("--ik-link-name")
    parser.add_argument("--end-effector-name")
    parser.add_argument("--ready-state")
    parser.add_argument("--world-frame")
    parser.add_argument("--base-frame")
    parser.add_argument("--camera-frame")
    parser.add_argument("--segmentation-camera-frame")
    parser.add_argument("--rgb-topic")
    parser.add_argument("--depth-topic")
    parser.add_argument("--latitude-layers", type=int)
    parser.add_argument("--azimuth-samples", type=int)
    parser.add_argument(
        "--path-start-mode", choices=("initial_pose", "all_accepted")
    )
    parser.add_argument("--motion-timeout", type=float)
    parser.add_argument("--camera-timeout", type=float)
    parser.add_argument("--tf-timeout", type=float)
    parser.add_argument("--segmentation-timeout", type=float)
    parser.add_argument("--settle-time", type=float)
    parser.add_argument("--target-preview-time", type=float)
    parser.add_argument("--trajectory-preview-time", type=float)
    parser.add_argument("--segmentation-service")
    parser.add_argument(
        "--save-segmentation",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--use-sim-time",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


def load_scan_configuration(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Merge the base planner YAML, scan YAML section, and explicit CLI values."""
    parser = build_parser()
    parsed = parser.parse_args(arguments)
    config_path = parsed.config.expanduser()
    planning = load_configuration(["--config", str(config_path)])
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    scan_values = document.get("single_view_scan") if isinstance(document, dict) else None
    if not isinstance(scan_values, dict):
        raise ValueError("config must contain a 'single_view_scan' mapping")

    values = vars(planning).copy()
    values.update(scan_values)
    for name, value in vars(parsed).items():
        if name != "config" and value is not None:
            values[name] = value
    values["config"] = parsed.config
    values["output_dir"] = Path(values["output_dir"])
    return argparse.Namespace(**values)


def validate_scan_configuration(config: argparse.Namespace) -> None:
    """Validate scanner-specific values after the shared planner parameters."""
    validate_configuration(config)
    if config.trajectory_mode not in TRAJECTORY_MODES:
        raise ValueError(
            f"trajectory_mode must be one of: {', '.join(TRAJECTORY_MODES)}"
        )
    if not isinstance(config.output_suffix, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", config.output_suffix
    ):
        raise ValueError("output_suffix must be a safe non-empty filename suffix")
    required = (
        config.end_effector_name,
        config.ready_state,
        config.base_frame,
        config.camera_frame,
        config.rgb_topic,
        config.depth_topic,
    )
    if not all(required):
        raise ValueError("Kinova link, state, frame, and camera names are required")
    positive = (
        config.motion_timeout,
        config.camera_timeout,
        config.tf_timeout,
        config.segmentation_timeout,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in positive):
        raise ValueError("scan timeouts must be finite and positive")
    nonnegative = (
        config.settle_time,
        config.target_preview_time,
        config.trajectory_preview_time,
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in nonnegative):
        raise ValueError("settle and preview times must be nonnegative")
    if config.save_segmentation and not (
        config.segmentation_service and config.segmentation_camera_frame
    ):
        raise ValueError(
            "segmentation_service and segmentation_camera_frame are required "
            "when segmentation is enabled"
        )


def output_directories(
    config: argparse.Namespace, manifest_path: Path | None = None
) -> dict[str, Path]:
    """Clear and recreate scan artifacts inside the prefix-scoped run folder."""
    layout = ScanRunLayout(config.output_dir, config.output_suffix)
    directories = {
        "images": layout.images,
        "segments": layout.segments,
        "transforms": layout.transforms,
    }

    # Validate every exact target before removing any data from a previous run.
    for directory in directories.values():
        if directory.is_symlink():
            raise RuntimeError(f"refusing to clear symlinked output: {directory}")
        if directory.exists() and not directory.is_dir():
            raise RuntimeError(f"scan output is not a directory: {directory}")
    if manifest_path is not None:
        if manifest_path.is_symlink():
            raise RuntimeError(
                f"refusing to clear symlinked manifest: {manifest_path}"
            )
        if manifest_path.exists() and not manifest_path.is_file():
            raise RuntimeError(f"scan manifest is not a file: {manifest_path}")
    for directory in directories.values():
        if directory.exists():
            shutil.rmtree(directory)
            print(f"Removed previous suffix output: {directory}")
        directory.mkdir(parents=True)
    return directories


def clear_previous_manifest(manifest_path: Path) -> None:
    """Remove the exact suffix manifest while refusing unsafe path types."""
    if manifest_path.is_symlink():
        raise RuntimeError(f"refusing to clear symlinked manifest: {manifest_path}")
    if manifest_path.exists() and not manifest_path.is_file():
        raise RuntimeError(f"scan manifest is not a file: {manifest_path}")
    if manifest_path.exists():
        manifest_path.unlink()
        print(f"Removed previous suffix manifest: {manifest_path}")


def build_scan_plan(
    config: argparse.Namespace,
    client: StandardIKClient,
    camera_to_link: np.ndarray,
) -> dict[str, object]:
    """Sample poses, reject failed IK, and construct all three route orders."""
    center = np.asarray(config.center, dtype=np.float64)
    intrinsics = load_intrinsics(Path(config.camera_yaml))
    sampled = sample_hemisphere(
        center,
        config.radius,
        config.latitude_layers,
        config.azimuth_samples,
        config.elevation_bounds,
        config.azimuth_offset_deg,
    )
    current_state = client.wait_until_ready(config)
    raw_solutions = []
    rejected = []

    # Test every desired camera pose with collision-aware MoveIt IK.
    for number, viewpoint in enumerate(sampled, start=1):
        ik_link_pose = viewpoint.camera_pose @ camera_to_link
        success, positions, error_code = client.solve(ik_link_pose, config)
        if success:
            raw_solutions.append((viewpoint, ik_link_pose, positions))
            print(f"IK [{number}/{len(sampled)}] accepted sample {number}")
        else:
            rejected.append(viewpoint.source_index)
            print(
                f"IK [{number}/{len(sampled)}] rejected sample {number}: "
                f"MoveIt code {error_code}"
            )
    if len(raw_solutions) < 2:
        raise RuntimeError("fewer than two hemisphere samples passed MoveIt IK")

    # Retain only common arm joints and express every IK solution consistently.
    current_positions = dict(zip(current_state.name, current_state.position))
    excluded = tuple(str(value).lower() for value in config.excluded_joint_substrings)
    joint_names = [
        name
        for name in raw_solutions[0][2]
        if name in current_positions
        and not any(token in name.lower() for token in excluded)
        and all(name in solution for _, _, solution in raw_solutions)
    ]
    if not joint_names:
        raise RuntimeError("no common planning joints found in IK and JointState")
    reachable = [
        ReachableViewpoint(
            viewpoint,
            link_pose,
            np.asarray([positions[name] for name in joint_names]),
        )
        for viewpoint, link_pose, positions in raw_solutions
    ]
    start_values = np.asarray([current_positions[name] for name in joint_names])

    # Score common route geometry, then construct baseline and optimized orders.
    surface_points = sample_scan_volume(
        center, config.scan_volume_radius, config.projection_samples
    )
    overlaps = pairwise_overlaps(reachable, surface_points, intrinsics)
    distances = joint_distances(reachable)
    start_distances = np.asarray(
        [np.linalg.norm(item.joint_values - start_values) for item in reachable]
    )
    costs = combined_edge_costs(distances, overlaps, config.overlap_weight)
    random_order, spiral_order = baseline_orders(reachable, config.random_seed)
    greedy_order = nearest_neighbor_open_path(
        costs,
        overlaps,
        start_distances,
        config.minimum_neighbor_overlap,
        config.path_start_mode,
    )
    optimized_order = two_opt_open_path(
        greedy_order,
        costs,
        overlaps,
        start_distances,
        config.minimum_neighbor_overlap,
        config.two_opt_passes,
        lock_first=config.path_start_mode == "initial_pose",
    )
    orders = {
        "random": random_order,
        "spiral": spiral_order,
        "hamilton_2opt": optimized_order,
    }
    metrics = {
        mode: measure_path(order, distances, overlaps, costs, start_distances)
        for mode, order in orders.items()
    }
    return {
        "sampled": sampled,
        "reachable": reachable,
        "rejected": rejected,
        "joint_names": joint_names,
        "start_values": start_values,
        "orders": orders,
        "metrics": metrics,
        "overlaps": overlaps,
        "greedy_order": greedy_order,
        "camera_to_link": camera_to_link,
    }


def publish_selected_plan(
    config: argparse.Namespace,
    client: StandardIKClient,
    plan: dict[str, object],
    initial_camera_pose: np.ndarray,
) -> None:
    """Publish the sampling shell, spiral baseline, and selected trajectory."""
    sampled = plan["sampled"]
    reachable = plan["reachable"]
    orders = plan["orders"]
    stamp = client.node.get_clock().now().to_msg()
    markers = build_visualizations(
        config,
        sampled,
        reachable,
        set(plan["rejected"]),
        orders["random"],
        orders["spiral"],
        orders["hamilton_2opt"],
        stamp,
        initial_camera_pose,
    )
    selected_marker_key = {
        "random": "random_path",
        "spiral": "spiral_path",
        "hamilton_2opt": "optimized_path",
    }[config.trajectory_mode]
    selected_display_key = {
        "random": "random_display",
        "spiral": "spiral_display",
        "hamilton_2opt": "optimized_display",
    }[config.trajectory_mode]

    # Always expose the spiral baseline so it remains available for comparison.
    routes_to_publish = [("spiral_path", "spiral_display", "spiral")]
    if config.trajectory_mode != "spiral":
        routes_to_publish.append(
            (selected_marker_key, selected_display_key, config.trajectory_mode)
        )
    client.publishers["hemisphere"].publish(markers["hemisphere"])
    client.publishers["candidates"].publish(markers["candidates"])
    for marker_key, display_key, mode in routes_to_publish:
        display = build_display_trajectory(
            plan["joint_names"],
            plan["start_values"],
            [reachable[index].joint_values for index in orders[mode]],
            config.trajectory_point_time,
        )
        display.trajectory_start.joint_state.header.stamp = stamp
        display.trajectory[0].joint_trajectory.header.stamp = stamp
        client.publishers[marker_key].publish(markers[marker_key])
        client.publishers[display_key].publish(display)


def save_segmentation(
    client: IsaacSegmentationClient | None,
    segment_root: Path,
    camera_frame: str,
    timeout: float,
    output_root: Path,
) -> dict[str, str]:
    """Save one Isaac mask set, or explicitly record that it was disabled."""
    if client is None:
        return {"status": "disabled"}
    try:
        capture_directory = client.save(segment_root, camera_frame, timeout)
        relative = capture_directory.relative_to(output_root)
    except (SegmentationServiceError, ValueError, OSError) as error:
        return {"status": "failed", "error": str(error)}
    return {
        "status": "saved",
        "directory": str(relative),
        "manifest": str(relative / "manifest.json"),
    }


def capture_view(
    robot: RobotAPI,
    segmentation_client: IsaacSegmentationClient | None,
    config: argparse.Namespace,
    directories: dict[str, Path],
    output_root: Path,
    path_index: int,
    sample_index: int,
    *,
    initial_view: bool,
    motion_result: str | None = None,
) -> dict[str, object]:
    """Capture RGB-D, segmentation, and measured camera TF for one view."""
    if config.settle_time:
        time.sleep(config.settle_time)

    # Keep every modality under the same reserved or sampled integer index.
    color_path, depth_path = robot.save_camera_images(
        str(directories["images"] / f"color_{sample_index}.png"),
        str(directories["images"] / f"depth_{sample_index}.png"),
        wait_timeout=config.camera_timeout,
    )
    segment_root = directories["segments"] / f"segment_{sample_index}"
    segmentation = save_segmentation(
        segmentation_client,
        segment_root,
        config.segmentation_camera_frame,
        config.segmentation_timeout,
        output_root,
    )

    camera_position, camera_quaternion = robot.get_transform_pos_quat(
        config.base_frame,
        config.camera_frame,
        timeout=config.tf_timeout,
    )
    transform_path = directories["transforms"] / f"T_base_cam_{sample_index}.npy"
    np.save(transform_path, matrix_from_pose(camera_position, camera_quaternion))
    record: dict[str, object] = {
        "path_index": path_index,
        "sample_index": sample_index,
        "initial_view": initial_view,
        "status": "captured",
        "color_image": str(color_path.relative_to(output_root)),
        "depth_image": str(depth_path.relative_to(output_root)),
        "segmentation": segmentation,
        "camera_transform": str(transform_path.relative_to(output_root)),
    }
    if motion_result is not None:
        record["motion_result"] = motion_result
    return record


def run_scan(config: argparse.Namespace) -> None:
    """Execute the selected reachable route and save synchronized scan products."""
    validate_scan_configuration(config)
    layout = ScanRunLayout(config.output_dir, config.output_suffix)
    output_root = layout.root
    manifest_path = layout.manifest
    directories = output_directories(config, manifest_path)
    clear_previous_manifest(manifest_path)

    # Home first so IK seeding and route start cost use the Ready joint state.
    with RobotAPI(
        node_name="kinova_single_view_scan_ready",
        call_timeout=config.motion_timeout,
        use_sim_time=config.use_sim_time,
    ) as ready_robot:
        if config.use_sim_time:
            ready_robot.wait_for_sim_time(config.service_timeout)
        print(
            f"Moving {config.planning_group} to SRDF state {config.ready_state!r}"
        )
        ready_robot.move_group_state(
            config.ready_state,
            planning_group=config.planning_group,
            timeout=config.motion_timeout,
        )
        camera_to_link = matrix_from_pose(
            *ready_robot.get_transform_pos_quat(
                config.camera_frame,
                config.ik_link_name,
                timeout=config.tf_timeout,
            )
        )
        initial_camera_pose = matrix_from_pose(
            *ready_robot.get_transform_pos_quat(
                config.world_frame,
                config.camera_frame,
                timeout=config.tf_timeout,
            )
        )

    ik_client = StandardIKClient(config)
    try:
        plan = build_scan_plan(config, ik_client, camera_to_link)
        publish_selected_plan(config, ik_client, plan, initial_camera_pose)
        if config.trajectory_preview_time:
            print(f"Previewing selected route for {config.trajectory_preview_time:.1f}s")
            time.sleep(config.trajectory_preview_time)

        reachable = plan["reachable"]
        selected_order = plan["orders"][config.trajectory_mode]
        manifest = {
            "configuration": str(config.config.expanduser().resolve()),
            "output_suffix": config.output_suffix,
            "trajectory_mode": config.trajectory_mode,
            "path_start_mode": config.path_start_mode,
            "camera_frame": config.camera_frame,
            "segmentation_camera_frame": config.segmentation_camera_frame,
            "initial_view_index": 0,
            "sampling_index_base": 1,
            "latitude_layers": config.latitude_layers,
            "azimuth_samples": config.azimuth_samples,
            "sample_count": len(plan["sampled"]),
            "total_view_count": len(plan["sampled"]) + 1,
            "reachable_count": len(reachable),
            "rejected_sample_indices": [index + 1 for index in plan["rejected"]],
            "camera_to_ik_link": plan["camera_to_link"].tolist(),
            "directories": {
                name: str(path.relative_to(output_root))
                for name, path in directories.items()
            },
            "routes": {
                mode: {
                    "sample_indices": [
                        0,
                        *[
                            reachable[index].viewpoint.source_index + 1
                            for index in order
                        ],
                    ],
                    "metrics": asdict(plan["metrics"][mode]),
                    "overlap_violation_count": overlap_violation_count(
                        order,
                        plan["overlaps"],
                        config.minimum_neighbor_overlap,
                    ),
                }
                for mode, order in plan["orders"].items()
            },
            "captures": [],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        # RobotAPI owns camera synchronization and all Kinova motion services.
        with ExitStack() as stack:
            robot = stack.enter_context(
                RobotAPI(
                    node_name="kinova_single_view_scan",
                    call_timeout=config.motion_timeout,
                    use_sim_time=config.use_sim_time,
                    enable_camera=True,
                    rgb_topic=config.rgb_topic,
                    depth_topic=config.depth_topic,
                )
            )
            segmentation_client = None
            if config.save_segmentation:
                segmentation_client = stack.enter_context(
                    IsaacSegmentationClient(
                        config.segmentation_service,
                        use_sim_time=config.use_sim_time,
                    )
                )
                segmentation_client.wait_for_service(config.segmentation_timeout)

            # Check sensor and TF inputs before commanding the robot.
            if config.use_sim_time:
                robot.wait_for_sim_time(config.service_timeout)
            robot.get_camera_images(wait_timeout=config.camera_timeout)
            robot.get_transform_pos_quat(
                config.base_frame, config.camera_frame, timeout=config.tf_timeout
            )

            # Index zero always records the Ready pose before any scan motion.
            print(f"[0/{len(selected_order)}] capturing initial Ready view as index 0")
            manifest["captures"].append(
                capture_view(
                    robot,
                    segmentation_client,
                    config,
                    directories,
                    output_root,
                    0,
                    0,
                    initial_view=True,
                )
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )

            # Move through the chosen route; sampled hemisphere indices begin at one.
            for path_index, reachable_index in enumerate(selected_order, start=1):
                item = reachable[reachable_index]
                sample_index = item.viewpoint.source_index + 1
                target_frame = f"single_scan_camera_target_{sample_index}"
                robot.publish_dynamic_transform(
                    config.world_frame, target_frame, item.viewpoint.camera_pose
                )
                if config.target_preview_time:
                    time.sleep(config.target_preview_time)
                print(
                    f"[{path_index}/{len(selected_order)}] moving to sample "
                    f"{sample_index}/{len(plan['sampled'])}"
                )
                link_position, link_quaternion = matrix_to_pose(item.ik_link_pose)
                try:
                    motion = robot.move_cartesian(
                        {
                            config.end_effector_name: RobotAPI.pose(
                                link_position, link_quaternion
                            )
                        },
                        planning_group=config.planning_group,
                        relative=False,
                        timeout=config.motion_timeout,
                    )
                except RobotAPIError as error:
                    print(f"  motion failed; skipping sample: {error}")
                    manifest["captures"].append(
                        {
                            "path_index": path_index,
                            "sample_index": sample_index,
                            "initial_view": False,
                            "status": "motion_failed",
                            "error": str(error),
                        }
                    )
                    manifest_path.write_text(
                        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
                    )
                    continue

                manifest["captures"].append(
                    capture_view(
                        robot,
                        segmentation_client,
                        config,
                        directories,
                        output_root,
                        path_index,
                        sample_index,
                        initial_view=False,
                        motion_result=motion.message,
                    )
                )
                manifest_path.write_text(
                    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
                )

            # End in the same named configuration used at startup.
            try:
                robot.move_group_state(
                    config.ready_state,
                    planning_group=config.planning_group,
                    timeout=config.motion_timeout,
                )
            except RobotAPIError as error:
                print(f"Final Ready motion failed; scan data is preserved: {error}")
    finally:
        ik_client.close()


def main() -> None:
    run_scan(load_scan_configuration())


if __name__ == "__main__":
    main()
