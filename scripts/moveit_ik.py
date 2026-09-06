"""Read-only MoveIt multi-tip IK client used to filter scan viewpoints."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Mapping, Optional, Sequence

from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, Pose, PoseStamped
from moveit_msgs.msg import DisplayTrajectory, MoveItErrorCodes, RobotTrajectory
from moveit_msgs.srv import GetPositionIK
from nav_msgs.msg import Path
import rclpy
from rclpy.context import Context
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray


@dataclass(frozen=True)
class IKResult:
    """One collision-aware MoveIt IK response."""

    success: bool
    joint_positions: Mapping[str, float]
    message: str
    error_code: int


def build_display_trajectory(
    joint_names: Sequence[str],
    start_positions: Sequence[float],
    waypoint_positions: Sequence[Sequence[float]],
    point_interval: float,
) -> DisplayTrajectory:
    """Build an RViz animation through current and optimized IK configurations."""
    names = list(joint_names)
    start = [float(value) for value in start_positions]
    waypoints = [[float(value) for value in point] for point in waypoint_positions]
    if not names or len(set(names)) != len(names):
        raise ValueError("display trajectory requires unique joint names")
    if len(start) != len(names) or any(len(point) != len(names) for point in waypoints):
        raise ValueError("every display trajectory point must match the joint names")
    if not all(math.isfinite(value) for point in [start, *waypoints] for value in point):
        raise ValueError("display trajectory positions must be finite")
    if not math.isfinite(point_interval) or point_interval <= 0.0:
        raise ValueError("display trajectory point interval must be finite and positive")

    display = DisplayTrajectory()
    display.trajectory_start.is_diff = True
    display.trajectory_start.joint_state.name = names
    display.trajectory_start.joint_state.position = start
    trajectory = RobotTrajectory()
    trajectory.joint_trajectory.joint_names = names

    # Include the current state so RViz animates the approach to the first view.
    for index, positions in enumerate([start, *waypoints]):
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(seconds=index * point_interval).to_msg()
        trajectory.joint_trajectory.points.append(point)
    display.trajectory.append(trajectory)
    return display


def build_camera_path(
    poses: Sequence[Pose], frame_id: str, stamp: Time
) -> Path:
    """Build a world-frame RViz path through a sequence of camera poses."""
    if not frame_id:
        raise ValueError("camera path frame must not be empty")
    if not all(isinstance(pose, Pose) for pose in poses):
        raise TypeError("every camera path item must be a geometry_msgs Pose")
    path = Path()
    path.header.frame_id = frame_id
    path.header.stamp = stamp
    for pose in poses:
        stamped = PoseStamped()
        stamped.header = path.header
        stamped.pose = pose
        path.poses.append(stamped)
    return path


def build_trajectory_markers(
    *,
    before_left_poses: Sequence[Pose],
    before_right_poses: Sequence[Pose],
    optimized_left_poses: Sequence[Pose],
    optimized_right_poses: Sequence[Pose],
    frame_id: str,
    stamp: Time,
    line_width: float,
    before_color_rgb: Sequence[int],
    optimized_color_rgb: Sequence[int],
) -> MarkerArray:
    """Build wide neon line strips comparing camera routes before and after 2-opt."""
    pose_groups = (
        (
            "trajectory_0_before_2opt_left",
            before_left_poses,
            before_color_rgb,
            1.6,
            0.55,
        ),
        (
            "trajectory_0_before_2opt_right",
            before_right_poses,
            before_color_rgb,
            1.6,
            0.55,
        ),
        (
            "trajectory_1_after_2opt_left",
            optimized_left_poses,
            optimized_color_rgb,
            1.0,
            1.0,
        ),
        (
            "trajectory_1_after_2opt_right",
            optimized_right_poses,
            optimized_color_rgb,
            1.0,
            1.0,
        ),
    )
    if not frame_id:
        raise ValueError("trajectory marker frame must not be empty")
    if not math.isfinite(line_width) or line_width <= 0.0:
        raise ValueError("trajectory marker line width must be finite and positive")
    for color in (before_color_rgb, optimized_color_rgb):
        if len(color) != 3 or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 255
            for value in color
        ):
            raise ValueError("trajectory marker colors must contain three integers in [0, 255]")
    if not all(
        isinstance(pose, Pose)
        for _, poses, _, _, _ in pose_groups
        for pose in poses
    ):
        raise TypeError("every trajectory marker item must be a geometry_msgs Pose")

    markers = MarkerArray()
    delete_all = Marker()
    delete_all.header.frame_id = frame_id
    delete_all.header.stamp = stamp
    delete_all.action = Marker.DELETEALL
    markers.markers.append(delete_all)

    # Draw the pre-2-opt path as a wider halo, leaving the optimized path visible
    # wherever both paths share the same coplanar edge.
    for marker_id, (namespace, poses, color, width_scale, alpha) in enumerate(
        pose_groups
    ):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(line_width) * width_scale
        marker.color.r = color[0] / 255.0
        marker.color.g = color[1] / 255.0
        marker.color.b = color[2] / 255.0
        marker.color.a = alpha
        marker.points = [
            Point(x=pose.position.x, y=pose.position.y, z=pose.position.z)
            for pose in poses
        ]
        markers.markers.append(marker)
    return markers


def split_trajectory_markers(
    comparison: MarkerArray,
) -> tuple[MarkerArray, MarkerArray]:
    """Split the validated comparison into independent before/after messages."""
    before = MarkerArray()
    optimized = MarkerArray()
    delete_markers = [
        marker for marker in comparison.markers if marker.action == Marker.DELETEALL
    ]
    route_markers = [
        marker for marker in comparison.markers if marker.action == Marker.ADD
    ]
    if len(delete_markers) != 1 or len(route_markers) != 4:
        raise ValueError("trajectory comparison must contain one reset and four routes")

    before.markers = [
        delete_markers[0],
        *[marker for marker in route_markers if "before_2opt" in marker.ns],
    ]
    optimized.markers = [
        delete_markers[0],
        *[marker for marker in route_markers if "after_2opt" in marker.ns],
    ]
    if len(before.markers) != 3 or len(optimized.markers) != 3:
        raise ValueError("trajectory comparison has invalid before/after namespaces")
    return before, optimized


class MoveItIKClient:
    """Own a private ROS node for collision-aware `/compute_ik` requests."""

    def __init__(
        self,
        *,
        node_name: str = "multi_view_scan_ik",
        service_name: str = "/compute_ik",
        joint_state_topic: str = "/joint_states",
        display_trajectory_topic: str = "/display_planned_path",
        initial_display_trajectory_topic: str = "/scan/display_trajectory_before_2opt",
        left_camera_path_topic: str = "/scan/left_camera_path",
        right_camera_path_topic: str = "/scan/right_camera_path",
        trajectory_comparison_topic: str = "/scan/trajectory_comparison",
        before_trajectory_marker_topic: str = "/scan/trajectory_before_2opt",
        optimized_trajectory_marker_topic: str = "/scan/trajectory_after_2opt",
        use_sim_time: bool = True,
    ) -> None:
        names = (
            node_name,
            service_name,
            joint_state_topic,
            display_trajectory_topic,
            initial_display_trajectory_topic,
            left_camera_path_topic,
            right_camera_path_topic,
            trajectory_comparison_topic,
            before_trajectory_marker_topic,
            optimized_trajectory_marker_topic,
        )
        if not all(names):
            raise ValueError("node, service, joint-state, and display names must not be empty")
        if not isinstance(use_sim_time, bool):
            raise TypeError("use_sim_time must be a bool")

        self._context = Context()
        rclpy.init(context=self._context)
        self._node = rclpy.create_node(
            node_name,
            context=self._context,
            parameter_overrides=[Parameter("use_sim_time", value=use_sim_time)],
            automatically_declare_parameters_from_overrides=True,
        )
        self._executor = MultiThreadedExecutor(num_threads=2, context=self._context)
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            name=f"{node_name}_executor",
            daemon=True,
        )
        self._closed = False
        self._joint_condition = threading.Condition()
        self._joint_state: Optional[JointState] = None
        self._joint_receipt_time = 0.0
        self._joint_subscription = self._node.create_subscription(
            JointState,
            joint_state_topic,
            self._joint_state_callback,
            qos_profile_sensor_data,
        )
        self._client = self._node.create_client(GetPositionIK, service_name)
        display_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._display_trajectory_publisher = self._node.create_publisher(
            DisplayTrajectory, display_trajectory_topic, display_qos
        )
        self._initial_display_trajectory_publisher = self._node.create_publisher(
            DisplayTrajectory, initial_display_trajectory_topic, display_qos
        )
        self._left_camera_path_publisher = self._node.create_publisher(
            Path, left_camera_path_topic, display_qos
        )
        self._right_camera_path_publisher = self._node.create_publisher(
            Path, right_camera_path_topic, display_qos
        )
        self._trajectory_comparison_publisher = self._node.create_publisher(
            MarkerArray, trajectory_comparison_topic, display_qos
        )
        self._before_trajectory_marker_publisher = self._node.create_publisher(
            MarkerArray, before_trajectory_marker_topic, display_qos
        )
        self._optimized_trajectory_marker_publisher = self._node.create_publisher(
            MarkerArray, optimized_trajectory_marker_topic, display_qos
        )
        self._spin_thread.start()

    def _joint_state_callback(self, message: JointState) -> None:
        """Retain the newest complete joint state and its wall-clock receipt time."""
        if len(message.name) != len(message.position) or not message.name:
            return
        with self._joint_condition:
            self._joint_state = message
            self._joint_receipt_time = time.monotonic()
            self._joint_condition.notify_all()

    def wait_for_joint_state(self, timeout: float) -> dict[str, float]:
        """Return a recently received full-state seed using a wall-clock timeout."""
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("joint-state timeout must be finite and positive")
        deadline = time.monotonic() + timeout
        with self._joint_condition:
            while (
                self._joint_state is None
                or time.monotonic() - self._joint_receipt_time > timeout
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        "no recent joint state received within "
                        f"{timeout:.3f} seconds"
                    )
                self._joint_condition.wait(remaining)
            message = self._joint_state
            return {
                name: float(position)
                for name, position in zip(message.name, message.position)
            }

    def wait_for_service(self, timeout: float) -> bool:
        """Wait for MoveIt's IK service using a finite wall-clock timeout."""
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("service timeout must be finite and positive")
        return self._client.wait_for_service(timeout_sec=timeout)

    def solve(
        self,
        targets: Mapping[str, Pose],
        *,
        planning_group: str,
        frame_id: str,
        timeout: float,
        joint_state_timeout: float,
        avoid_collisions: bool = True,
    ) -> IKResult:
        """Solve one multi-tip IK target without planning or executing motion."""
        if not targets or not planning_group or not frame_id:
            raise ValueError("targets, planning group, and frame ID must not be empty")
        if not all(isinstance(pose, Pose) for pose in targets.values()):
            raise TypeError("every IK target must be a geometry_msgs Pose")
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("IK timeout must be finite and positive")
        if not self.wait_for_service(joint_state_timeout):
            return IKResult(False, {}, "MoveIt /compute_ik service is unavailable", 0)

        # Seed all robot joints so joints outside the selected group retain state.
        seed_positions = self.wait_for_joint_state(joint_state_timeout)
        request = GetPositionIK.Request()
        request.ik_request.group_name = planning_group
        request.ik_request.avoid_collisions = bool(avoid_collisions)
        request.ik_request.timeout = Duration(seconds=timeout).to_msg()
        request.ik_request.robot_state.is_diff = True
        request.ik_request.robot_state.joint_state.name = list(seed_positions)
        request.ik_request.robot_state.joint_state.position = list(seed_positions.values())
        request.ik_request.robot_state.joint_state.header.stamp = (
            self._node.get_clock().now().to_msg()
        )

        # Multi-tip WbDOpt receives both synchronized TCP targets in one request.
        stamp = self._node.get_clock().now().to_msg()
        for tip_name, pose in targets.items():
            target = PoseStamped()
            target.header.frame_id = frame_id
            target.header.stamp = stamp
            target.pose = pose
            request.ik_request.ik_link_names.append(tip_name)
            request.ik_request.pose_stamped_vector.append(target)

        future = self._client.call_async(request)
        completed = threading.Event()
        future.add_done_callback(lambda _: completed.set())
        response_timeout = timeout + joint_state_timeout
        if not completed.wait(response_timeout):
            future.cancel()
            return IKResult(
                False,
                {},
                f"MoveIt IK response timed out after {response_timeout:.3f} seconds",
                0,
            )
        exception = future.exception()
        if exception is not None:
            return IKResult(False, {}, f"MoveIt IK call failed: {exception}", 0)

        response = future.result()
        error_code = int(response.error_code.val)
        if error_code != MoveItErrorCodes.SUCCESS:
            return IKResult(
                False,
                {},
                f"MoveIt returned IK error code {error_code}",
                error_code,
            )
        names = response.solution.joint_state.name
        positions = response.solution.joint_state.position
        if not names or len(names) != len(positions):
            return IKResult(False, {}, "MoveIt returned an invalid joint state", error_code)
        return IKResult(
            True,
            {name: float(position) for name, position in zip(names, positions)},
            "IK solution found",
            error_code,
        )

    def publish_trajectory_comparison(
        self,
        *,
        joint_names: Sequence[str],
        start_positions: Sequence[float],
        before_waypoint_positions: Sequence[Sequence[float]],
        optimized_waypoint_positions: Sequence[Sequence[float]],
        frame_id: str,
        before_left_camera_poses: Sequence[Pose],
        before_right_camera_poses: Sequence[Pose],
        optimized_left_camera_poses: Sequence[Pose],
        optimized_right_camera_poses: Sequence[Pose],
        point_interval: float,
        line_width: float,
        before_color_rgb: Sequence[int],
        optimized_color_rgb: Sequence[int],
    ) -> None:
        """Publish before/after robot animations and wide camera-route overlays."""
        before_display = build_display_trajectory(
            joint_names, start_positions, before_waypoint_positions, point_interval
        )
        optimized_display = build_display_trajectory(
            joint_names, start_positions, optimized_waypoint_positions, point_interval
        )
        stamp = self._node.get_clock().now().to_msg()
        for display in (before_display, optimized_display):
            display.trajectory_start.joint_state.header.stamp = stamp
            display.trajectory[0].joint_trajectory.header.stamp = stamp
        left_path = build_camera_path(optimized_left_camera_poses, frame_id, stamp)
        right_path = build_camera_path(optimized_right_camera_poses, frame_id, stamp)
        markers = build_trajectory_markers(
            before_left_poses=before_left_camera_poses,
            before_right_poses=before_right_camera_poses,
            optimized_left_poses=optimized_left_camera_poses,
            optimized_right_poses=optimized_right_camera_poses,
            frame_id=frame_id,
            stamp=stamp,
            line_width=line_width,
            before_color_rgb=before_color_rgb,
            optimized_color_rgb=optimized_color_rgb,
        )
        before_markers, optimized_markers = split_trajectory_markers(markers)

        # Publish isolated route messages as well as the combined comparison so
        # each result can be enabled and verified independently in RViz.
        self._initial_display_trajectory_publisher.publish(before_display)
        self._display_trajectory_publisher.publish(optimized_display)
        self._left_camera_path_publisher.publish(left_path)
        self._right_camera_path_publisher.publish(right_path)
        self._before_trajectory_marker_publisher.publish(before_markers)
        self._optimized_trajectory_marker_publisher.publish(optimized_markers)
        self._trajectory_comparison_publisher.publish(markers)

    def close(self) -> None:
        """Stop the executor and release private ROS resources."""
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(timeout_sec=2.0)
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
        self._executor.remove_node(self._node)
        self._node.destroy_node()
        if self._context.ok():
            self._context.shutdown()

    def __enter__(self) -> "MoveItIKClient":
        return self

    def __exit__(self, _exception_type, _exception, _traceback) -> None:
        self.close()
