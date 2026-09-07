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
    hemisphere_triangles,
    joint_distances,
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


def load_configuration(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Load YAML parameters and apply the commonly needed robot CLI overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--planning-group")
    parser.add_argument("--ik-link-name")
    parser.add_argument("--world-frame")
    parser.add_argument("--compute-ik-service")
    parser.add_argument("--joint-state-topic")
    parser.add_argument("--view-count", type=int)
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
    if config.view_count <= 1 or config.two_opt_passes < 0:
        raise ValueError("view_count must exceed one and two_opt_passes cannot be negative")
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
        "before_color_rgb",
        "after_color_rgb",
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
            "before_path": self.node.create_publisher(
                MarkerArray, config.before_path_topic, qos
            ),
            "after_path": self.node.create_publisher(
                MarkerArray, config.after_path_topic, qos
            ),
            "comparison": self.node.create_publisher(
                MarkerArray, config.comparison_topic, qos
            ),
            "before_display": self.node.create_publisher(
                DisplayTrajectory, config.before_display_topic, qos
            ),
            "after_display": self.node.create_publisher(
                DisplayTrajectory, config.after_display_topic, qos
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
    initial_order: Sequence[int],
    optimized_order: Sequence[int],
    stamp,
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

    initial_poses = [reachable[index].viewpoint.camera_pose for index in initial_order]
    optimized_poses = [reachable[index].viewpoint.camera_pose for index in optimized_order]
    before = route_marker(
        initial_poses,
        frame,
        stamp,
        "before_2opt",
        config.before_color_rgb,
        config.line_width * 1.5,
        0.60,
    )
    after = route_marker(
        optimized_poses,
        frame,
        stamp,
        "after_2opt",
        config.after_color_rgb,
        config.line_width,
    )
    return {
        "hemisphere": MarkerArray(markers=[reset_marker(frame, stamp), shell]),
        "candidates": MarkerArray(markers=[reset_marker(frame, stamp), accepted, rejected]),
        "before_path": MarkerArray(markers=[reset_marker(frame, stamp), before]),
        "after_path": MarkerArray(markers=[reset_marker(frame, stamp), after]),
        "comparison": MarkerArray(markers=[reset_marker(frame, stamp), before, after]),
    }


def run(config: argparse.Namespace) -> None:
    """Run IK filtering and path optimization, publish results, and never move."""
    validate_configuration(config)
    center = np.asarray(config.center, dtype=np.float64)
    intrinsics = load_intrinsics(Path(config.camera_yaml))
    sampled = sample_hemisphere(
        center,
        config.radius,
        config.view_count,
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

        # Construct the constrained greedy path and refine the complete route.
        surface_points = sample_scan_volume(
            center, config.scan_volume_radius, config.projection_samples
        )
        overlaps = pairwise_overlaps(reachable, surface_points, intrinsics)
        distances = joint_distances(reachable)
        start_distances = np.asarray(
            [np.linalg.norm(item.joint_values - start_values) for item in reachable]
        )
        costs = combined_edge_costs(distances, overlaps, config.overlap_weight)
        initial_order = nearest_neighbor_open_path(
            costs, overlaps, start_distances, config.minimum_neighbor_overlap
        )
        optimized_order = two_opt_open_path(
            initial_order,
            costs,
            overlaps,
            start_distances,
            config.minimum_neighbor_overlap,
            config.two_opt_passes,
        )
        initial_metrics = measure_path(
            initial_order, distances, overlaps, costs, start_distances
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
            initial_order,
            optimized_order,
            stamp,
        )
        before_display = build_display_trajectory(
            joint_names,
            start_values,
            [reachable[index].joint_values for index in initial_order],
            config.trajectory_point_time,
        )
        after_display = build_display_trajectory(
            joint_names,
            start_values,
            [reachable[index].joint_values for index in optimized_order],
            config.trajectory_point_time,
        )
        for display in (before_display, after_display):
            display.trajectory_start.joint_state.header.stamp = stamp
            display.trajectory[0].joint_trajectory.header.stamp = stamp

        result = {
            "configuration": str(config.config.expanduser().resolve()),
            "motion_executed": False,
            "planning_group": config.planning_group,
            "ik_link_name": config.ik_link_name,
            "sample_count": len(sampled),
            "reachable_count": len(reachable),
            "rejected_sample_indices": [index + 1 for index in rejected],
            "joint_names": joint_names,
            "initial_order": [reachable[index].viewpoint.source_index + 1 for index in initial_order],
            "optimized_order": [reachable[index].viewpoint.source_index + 1 for index in optimized_order],
            "initial_metrics": asdict(initial_metrics),
            "optimized_metrics": asdict(optimized_metrics),
        }
        result_path = Path(config.result_json).expanduser().resolve()
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

        # Publish once with transient durability, then republish markers while held.
        for name, message in markers.items():
            client.publishers[name].publish(message)
        client.publishers["before_display"].publish(before_display)
        client.publishers["after_display"].publish(after_display)
        print(
            f"Planned {len(reachable)}/{len(sampled)} reachable views; "
            f"objective {initial_metrics.objective:.4f} -> "
            f"{optimized_metrics.objective:.4f}"
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
                for name, message in markers.items():
                    client.publishers[name].publish(message)
                next_publish = time.monotonic() + 1.0
    except KeyboardInterrupt:
        pass
    finally:
        client.close()


def main() -> None:
    run(load_configuration())


if __name__ == "__main__":
    main()
