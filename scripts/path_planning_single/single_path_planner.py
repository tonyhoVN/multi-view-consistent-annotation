#!/usr/bin/env python3
"""Plan and visualize a single-arm multiview route without executing motion."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np
import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from multi_view_scan.aux_math import matrix_to_pose, transform_from_euler  # noqa: E402
from multi_view_scan.moveit_ik import build_display_trajectory  # noqa: E402
from multi_view_scan.scan_trajectory import (  # noqa: E402
    CameraIntrinsics,
    combined_edge_costs,
    measure_path,
    nearest_neighbor_open_path,
    sample_scan_volume,
    two_opt_open_path,
)
from path_planning_single.planning import (  # noqa: E402
    ReachableViewpoint,
    baseline_orders,
    hemisphere_triangles,
    joint_distances,
    overlap_violation_count,
    pairwise_overlaps,
    sample_hemisphere,
)

from geometry_msgs.msg import Point, Pose, PoseStamped  # noqa: E402
from moveit_msgs.msg import DisplayTrajectory, MoveItErrorCodes  # noqa: E402
from moveit_msgs.srv import GetPositionIK  # noqa: E402
import rclpy  # noqa: E402
from rclpy.duration import Duration  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from rclpy.qos import (  # noqa: E402
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import JointState  # noqa: E402
from visualization_msgs.msg import Marker, MarkerArray  # noqa: E402


DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")
TRAJECTORY_MODES = ("random", "spiral", "hamilton_2opt", "all")


def load_configuration(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Load YAML parameters and apply the commonly needed robot CLI overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--planning-group")
    parser.add_argument("--ik-link-name")
    parser.add_argument("--world-frame")
    parser.add_argument("--compute-ik-service")
    parser.add_argument("--joint-state-topic")
    parser.add_argument("--latitude-layers", type=int)
    parser.add_argument("--azimuth-samples", type=int)
    parser.add_argument(
        "--path-start-mode", choices=("initial_pose", "all_accepted")
    )
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--trajectory-mode", choices=TRAJECTORY_MODES)
    parser.add_argument("--hold-seconds", type=float)
    parser.add_argument(
        "--use-sim-time", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--avoid-collisions", action=argparse.BooleanOptionalAction, default=None
    )
    parsed = parser.parse_args(arguments)
    document = yaml.safe_load(parsed.config.expanduser().read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(
        document.get("single_path_planning"), dict
    ):
        raise ValueError("config must contain a 'single_path_planning' mapping")
    values = dict(document["single_path_planning"])
    for name, value in vars(parsed).items():
        if name != "config" and value is not None:
            values[name] = value
    values["config"] = parsed.config
    return argparse.Namespace(**values)


def validate_configuration(config: argparse.Namespace) -> None:
    """Reject invalid geometry, optimization, and visualization parameters early."""
    required_names = (
        "planning_group",
        "ik_link_name",
        "world_frame",
        "compute_ik_service",
        "joint_state_topic",
    )
    if not all(getattr(config, name) for name in required_names):
        raise ValueError("robot group, link, frame, service, and joint topic are required")
    if config.latitude_layers < 1 or config.azimuth_samples < 3:
        raise ValueError(
            "latitude_layers must be positive and azimuth_samples at least 3"
        )
    if config.two_opt_passes < 0:
        raise ValueError("two_opt_passes cannot be negative")
    if isinstance(config.random_seed, bool) or not isinstance(config.random_seed, int):
        raise ValueError("random_seed must be an integer")
    if config.trajectory_mode not in TRAJECTORY_MODES:
        raise ValueError(
            f"trajectory_mode must be one of: {', '.join(TRAJECTORY_MODES)}"
        )
    if config.path_start_mode not in {"initial_pose", "all_accepted"}:
        raise ValueError("path_start_mode must be 'initial_pose' or 'all_accepted'")
    positive = (
        config.radius,
        config.service_timeout,
        config.joint_state_timeout,
        config.ik_timeout,
        config.scan_volume_radius,
        config.line_width,
        config.point_size,
        config.rejected_cross_size,
        config.rejected_cross_line_width,
        config.trajectory_point_time,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in positive):
        raise ValueError("radius, timeouts, and visualization sizes must be positive")
    if not 0.0 <= config.minimum_neighbor_overlap <= 1.0:
        raise ValueError("minimum_neighbor_overlap must lie in [0, 1]")
    if not 0.0 <= config.hemisphere_alpha <= 1.0:
        raise ValueError("hemisphere_alpha must lie in [0, 1]")
    for name in (
        "hemisphere_color_rgb",
        "accepted_color_rgb",
        "rejected_color_rgb",
        "random_color_rgb",
        "spiral_color_rgb",
        "optimized_color_rgb",
    ):
        color = getattr(config, name)
        if len(color) != 3 or any(not 0 <= int(value) <= 255 for value in color):
            raise ValueError(f"{name} must contain three values in [0, 255]")


def load_intrinsics(path: Path) -> CameraIntrinsics:
    """Load the camera intrinsics used by projected-overlap evaluation."""
    document = yaml.safe_load(path.expanduser().read_text(encoding="utf-8"))
    intrinsics = document["intrinsics"]
    width, height = intrinsics["resolution"]
    return CameraIntrinsics(
        int(width),
        int(height),
        float(intrinsics["fx"]),
        float(intrinsics["fy"]),
        float(intrinsics["cx"]),
        float(intrinsics["cy"]),
    )


class StandardIKClient:
    """Call standard single-tip MoveIt IK and publish transient RViz messages."""

    def __init__(self, config: argparse.Namespace) -> None:
        rclpy.init()
        self.node = rclpy.create_node(
            "single_path_planning",
            parameter_overrides=[Parameter("use_sim_time", value=config.use_sim_time)],
            automatically_declare_parameters_from_overrides=True,
        )
        self.client = self.node.create_client(GetPositionIK, config.compute_ik_service)
        self.joint_state: JointState | None = None
        self.node.create_subscription(
            JointState,
            config.joint_state_topic,
            self._joint_state_callback,
            qos_profile_sensor_data,
        )
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.publishers = {
            "hemisphere": self.node.create_publisher(
                MarkerArray, config.hemisphere_topic, qos
            ),
            "candidates": self.node.create_publisher(
                MarkerArray, config.candidate_topic, qos
            ),
            "random_path": self.node.create_publisher(
                MarkerArray, config.random_path_topic, qos
            ),
            "spiral_path": self.node.create_publisher(
                MarkerArray, config.spiral_path_topic, qos
            ),
            "optimized_path": self.node.create_publisher(
                MarkerArray, config.optimized_path_topic, qos
            ),
            "comparison": self.node.create_publisher(
                MarkerArray, config.comparison_topic, qos
            ),
            "random_display": self.node.create_publisher(
                DisplayTrajectory, config.random_display_topic, qos
            ),
            "spiral_display": self.node.create_publisher(
                DisplayTrajectory, config.spiral_display_topic, qos
            ),
            "optimized_display": self.node.create_publisher(
                DisplayTrajectory, config.optimized_display_topic, qos
            ),
        }

    def _joint_state_callback(self, message: JointState) -> None:
        if message.name and len(message.name) == len(message.position):
            self.joint_state = message

    def wait_until_ready(self, config: argparse.Namespace) -> JointState:
        """Wait for the standard IK service and one valid current joint state."""
        if not self.client.wait_for_service(timeout_sec=config.service_timeout):
            raise TimeoutError(f"IK service unavailable: {config.compute_ik_service}")
        deadline = time.monotonic() + config.joint_state_timeout
        while self.joint_state is None and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        if self.joint_state is None:
            raise TimeoutError("no valid current JointState received")
        return self.joint_state

    def solve(
        self, target_pose: np.ndarray, config: argparse.Namespace
    ) -> tuple[bool, dict[str, float], int]:
        """Solve one pose through GetPositionIK's standard singular target fields."""
        position_array, quaternion_array = matrix_to_pose(target_pose)
        position = tuple(map(float, position_array))
        quaternion = tuple(map(float, quaternion_array))
        target = PoseStamped()
        target.header.frame_id = config.world_frame
        target.header.stamp = self.node.get_clock().now().to_msg()
        target.pose = Pose()
        target.pose.position.x, target.pose.position.y, target.pose.position.z = position
        (
            target.pose.orientation.x,
            target.pose.orientation.y,
            target.pose.orientation.z,
            target.pose.orientation.w,
        ) = quaternion

        request = GetPositionIK.Request()
        request.ik_request.group_name = config.planning_group
        request.ik_request.ik_link_name = config.ik_link_name
        request.ik_request.pose_stamped = target
        request.ik_request.avoid_collisions = bool(config.avoid_collisions)
        request.ik_request.timeout = Duration(seconds=config.ik_timeout).to_msg()
        request.ik_request.robot_state.is_diff = True
        request.ik_request.robot_state.joint_state = self.joint_state
        request.ik_request.robot_state.joint_state.header.stamp = target.header.stamp

        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(
            self.node,
            future,
            timeout_sec=config.ik_timeout + config.service_timeout,
        )
        if not future.done() or future.exception() is not None:
            return False, {}, 0
        response = future.result()
        code = int(response.error_code.val)
        if code != MoveItErrorCodes.SUCCESS:
            return False, {}, code
        names = response.solution.joint_state.name
        positions = response.solution.joint_state.position
        if not names or len(names) != len(positions):
            return False, {}, code
        return True, dict(zip(names, map(float, positions))), code

    def close(self) -> None:
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def point_message(point: Sequence[float]) -> Point:
    return Point(x=float(point[0]), y=float(point[1]), z=float(point[2]))


def reset_marker(frame_id: str, stamp) -> Marker:
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.action = Marker.DELETEALL
    return marker


def route_marker(
    poses: Sequence[np.ndarray],
    frame_id: str,
    stamp,
    namespace: str,
    color_rgb: Sequence[int],
    width: float,
    alpha: float = 1.0,
) -> Marker:
    """Build one wide camera-center line strip in world coordinates."""
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = namespace
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = width
    marker.color.r, marker.color.g, marker.color.b = [
        int(value) / 255.0 for value in color_rgb
    ]
    marker.color.a = alpha
    marker.points = [point_message(pose[:3, 3]) for pose in poses]
    return marker


def build_visualizations(
    config: argparse.Namespace,
    sampled,
    reachable: Sequence[ReachableViewpoint],
    rejected_indices: set[int],
    random_order: Sequence[int],
    spiral_order: Sequence[int],
    optimized_order: Sequence[int],
    stamp,
    initial_camera_pose: np.ndarray | None = None,
) -> dict[str, MarkerArray]:
    """Build the hemisphere, candidate points, and three route marker arrays."""
    frame = config.world_frame
    shell = Marker()
    shell.header.frame_id = frame
    shell.header.stamp = stamp
    shell.ns = "transparent_sampling_hemisphere"
    shell.type = Marker.TRIANGLE_LIST
    shell.action = Marker.ADD
    shell.pose.orientation.w = 1.0
    shell.scale.x = shell.scale.y = shell.scale.z = 1.0
    shell.color.r, shell.color.g, shell.color.b = [
        int(value) / 255.0 for value in config.hemisphere_color_rgb
    ]
    shell.color.a = float(config.hemisphere_alpha)
    shell.points = [
        point_message(point)
        for point in hemisphere_triangles(
            config.center,
            config.radius,
            config.elevation_bounds,
            config.hemisphere_azimuth_segments,
            config.hemisphere_elevation_segments,
        )
    ]

    # Reachable samples are red dots; rejected samples are tangent-plane black Xs.
    accepted = Marker()
    accepted.header.frame_id = frame
    accepted.header.stamp = stamp
    accepted.ns = "reachable_camera_samples"
    accepted.id = 0
    accepted.type = Marker.SPHERE_LIST
    accepted.action = Marker.ADD
    accepted.pose.orientation.w = 1.0
    accepted.scale.x = accepted.scale.y = accepted.scale.z = config.point_size
    accepted.color.r, accepted.color.g, accepted.color.b = [
        int(value) / 255.0 for value in config.accepted_color_rgb
    ]
    accepted.color.a = 1.0
    accepted_indices = set(range(len(sampled))) - rejected_indices
    accepted.points = [
        point_message(sampled[index].camera_pose[:3, 3])
        for index in sorted(accepted_indices)
    ]

    rejected = Marker()
    rejected.header.frame_id = frame
    rejected.header.stamp = stamp
    rejected.ns = "ik_rejected_camera_samples"
    rejected.id = 1
    rejected.type = Marker.LINE_LIST
    rejected.action = Marker.ADD
    rejected.pose.orientation.w = 1.0
    rejected.scale.x = config.rejected_cross_line_width
    rejected.color.r, rejected.color.g, rejected.color.b = [
        int(value) / 255.0 for value in config.rejected_color_rgb
    ]
    rejected.color.a = 1.0

    # Each X uses the camera-frame X/Y axes, making it tangent to the shell.
    half_size = 0.5 * config.rejected_cross_size
    for index in sorted(rejected_indices):
        pose = sampled[index].camera_pose
        center_point = pose[:3, 3]
        first_diagonal = (pose[:3, 0] + pose[:3, 1]) / math.sqrt(2.0)
        second_diagonal = (pose[:3, 0] - pose[:3, 1]) / math.sqrt(2.0)
        rejected.points.extend(
            [
                point_message(center_point - half_size * first_diagonal),
                point_message(center_point + half_size * first_diagonal),
                point_message(center_point - half_size * second_diagonal),
                point_message(center_point + half_size * second_diagonal),
            ]
        )

    route_prefix = [] if initial_camera_pose is None else [initial_camera_pose]
    random_poses = route_prefix + [
        reachable[index].viewpoint.camera_pose for index in random_order
    ]
    spiral_poses = route_prefix + [
        reachable[index].viewpoint.camera_pose for index in spiral_order
    ]
    optimized_poses = route_prefix + [
        reachable[index].viewpoint.camera_pose for index in optimized_order
    ]
    random_route = route_marker(
        random_poses,
        frame,
        stamp,
        "random_sequence",
        config.random_color_rgb,
        config.line_width * 1.7,
        0.45,
    )
    spiral_route = route_marker(
        spiral_poses,
        frame,
        stamp,
        "normal_spiral",
        config.spiral_color_rgb,
        config.line_width * 1.35,
        0.65,
    )
    optimized_route = route_marker(
        optimized_poses,
        frame,
        stamp,
        "hamilton_2opt",
        config.optimized_color_rgb,
        config.line_width,
    )
    return {
        "hemisphere": MarkerArray(markers=[reset_marker(frame, stamp), shell]),
        "candidates": MarkerArray(markers=[reset_marker(frame, stamp), accepted, rejected]),
        "random_path": MarkerArray(markers=[reset_marker(frame, stamp), random_route]),
        "spiral_path": MarkerArray(markers=[reset_marker(frame, stamp), spiral_route]),
        "optimized_path": MarkerArray(
            markers=[reset_marker(frame, stamp), optimized_route]
        ),
        "comparison": MarkerArray(
            markers=[
                reset_marker(frame, stamp),
                random_route,
                spiral_route,
                optimized_route,
            ]
        ),
    }


def run(config: argparse.Namespace) -> None:
    """Run IK filtering and path optimization, publish results, and never move."""
    validate_configuration(config)
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
    camera_to_link = transform_from_euler(
        "xyz",
        config.camera_to_ik_link_rpy_deg,
        translation=config.camera_to_ik_link_xyz,
        degrees=True,
    )
    client = StandardIKClient(config)
    try:
        current_state = client.wait_until_ready(config)
        raw_solutions = []
        rejected = []

        # Filter all samples through collision-aware standard single-tip IK.
        for number, viewpoint in enumerate(sampled, start=1):
            ik_link_pose = viewpoint.camera_pose @ camera_to_link
            success, positions, error_code = client.solve(ik_link_pose, config)
            if success:
                raw_solutions.append((viewpoint, ik_link_pose, positions))
                print(f"IK [{number}/{len(sampled)}] accepted sample {number:03d}")
            else:
                rejected.append(viewpoint.source_index)
                print(
                    f"IK [{number}/{len(sampled)}] rejected sample {number:03d}: "
                    f"MoveIt code {error_code}"
                )
        if len(raw_solutions) < 2:
            raise RuntimeError("fewer than two hemisphere samples passed MoveIt IK")

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

        # Build all three routes over the exact same IK-reachable camera poses.
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
        random_metrics = measure_path(
            random_order, distances, overlaps, costs, start_distances
        )
        spiral_metrics = measure_path(
            spiral_order, distances, overlaps, costs, start_distances
        )
        greedy_metrics = measure_path(
            greedy_order, distances, overlaps, costs, start_distances
        )
        optimized_metrics = measure_path(
            optimized_order, distances, overlaps, costs, start_distances
        )

        stamp = client.node.get_clock().now().to_msg()
        markers = build_visualizations(
            config,
            sampled,
            reachable,
            set(rejected),
            random_order,
            spiral_order,
            optimized_order,
            stamp,
        )
        displays = {
            name: build_display_trajectory(
                joint_names,
                start_values,
                [reachable[index].joint_values for index in order],
                config.trajectory_point_time,
            )
            for name, order in (
                ("random_display", random_order),
                ("spiral_display", spiral_order),
                ("optimized_display", optimized_order),
            )
        }
        for display in displays.values():
            display.trajectory_start.joint_state.header.stamp = stamp
            display.trajectory[0].joint_trajectory.header.stamp = stamp

        # Serialize sample IDs and poses so downstream trials can replay each order.
        def source_order(order: Sequence[int]) -> list[int]:
            return [reachable[index].viewpoint.source_index + 1 for index in order]

        def trajectory_record(order, metrics) -> dict:
            return {
                "order": source_order(order),
                "metrics": asdict(metrics),
                "overlap_violation_count": overlap_violation_count(
                    order, overlaps, config.minimum_neighbor_overlap
                ),
            }

        reachable_records = []
        for item in reachable:
            position, quaternion = matrix_to_pose(item.viewpoint.camera_pose)
            reachable_records.append(
                {
                    "sample_index": item.viewpoint.source_index + 1,
                    "azimuth_deg": item.viewpoint.azimuth_deg,
                    "elevation_deg": item.viewpoint.elevation_deg,
                    "camera_position_xyz": position.tolist(),
                    "camera_quaternion_xyzw": quaternion.tolist(),
                    "joint_positions": dict(zip(joint_names, item.joint_values.tolist())),
                }
            )

        result = {
            "configuration": str(config.config.expanduser().resolve()),
            "motion_executed": False,
            "trajectory_mode": config.trajectory_mode,
            "selected_trajectories": (
                list(TRAJECTORY_MODES[:-1])
                if config.trajectory_mode == "all"
                else [config.trajectory_mode]
            ),
            "planning_group": config.planning_group,
            "ik_link_name": config.ik_link_name,
            "path_start_mode": config.path_start_mode,
            "latitude_layers": config.latitude_layers,
            "azimuth_samples": config.azimuth_samples,
            "sample_count": len(sampled),
            "reachable_count": len(reachable),
            "rejected_sample_indices": [index + 1 for index in rejected],
            "joint_names": joint_names,
            "minimum_neighbor_overlap": config.minimum_neighbor_overlap,
            "random_seed": config.random_seed,
            "reachable_viewpoints": reachable_records,
            "trajectories": {
                "random": trajectory_record(random_order, random_metrics),
                "spiral": trajectory_record(spiral_order, spiral_metrics),
                "hamilton_2opt": trajectory_record(
                    optimized_order, optimized_metrics
                ),
            },
            "optimizer": {
                "nearest_neighbor_seed_order": source_order(greedy_order),
                "nearest_neighbor_seed_metrics": asdict(greedy_metrics),
            },
        }
        result_path = Path(config.result_json).expanduser().resolve()
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

        # Select one experimental route or publish the complete paper comparison.
        route_keys = {
            "random": ("random_path", "random_display"),
            "spiral": ("spiral_path", "spiral_display"),
            "hamilton_2opt": ("optimized_path", "optimized_display"),
        }
        if config.trajectory_mode == "all":
            marker_names = [
                "hemisphere",
                "candidates",
                "random_path",
                "spiral_path",
                "optimized_path",
                "comparison",
            ]
            display_names = list(displays)
        else:
            path_name, display_name = route_keys[config.trajectory_mode]
            marker_names = ["hemisphere", "candidates", path_name]
            display_names = [display_name]

        # Publish once with transient durability, then refresh markers while held.
        for name in marker_names:
            client.publishers[name].publish(markers[name])
        for name in display_names:
            client.publishers[name].publish(displays[name])
        print(f"Trajectory mode: {config.trajectory_mode}")
        print(f"Planned {len(reachable)}/{len(sampled)} reachable views:")
        for label, order, metrics in (
            ("random", random_order, random_metrics),
            ("spiral", spiral_order, spiral_metrics),
            ("Hamiltonian + 2-opt", optimized_order, optimized_metrics),
        ):
            violations = overlap_violation_count(
                order, overlaps, config.minimum_neighbor_overlap
            )
            print(
                f"  {label}: objective={metrics.objective:.4f}, "
                f"motion={metrics.motion_distance:.4f}, "
                f"mean overlap={metrics.mean_neighbor_overlap:.3f}, "
                f"overlap violations={violations}"
            )
        print(f"Results: {result_path}")
        print("No motion was planned or executed. Press Ctrl-C to stop publishing.")
        deadline = (
            time.monotonic() + config.hold_seconds if config.hold_seconds > 0.0 else None
        )
        next_publish = time.monotonic() + 1.0
        while rclpy.ok() and (deadline is None or time.monotonic() < deadline):
            rclpy.spin_once(client.node, timeout_sec=0.1)
            if time.monotonic() >= next_publish:
                for name in marker_names:
                    client.publishers[name].publish(markers[name])
                next_publish = time.monotonic() + 1.0
    except KeyboardInterrupt:
        pass
    finally:
        client.close()


def main() -> None:
    run(load_configuration())


if __name__ == "__main__":
    main()
