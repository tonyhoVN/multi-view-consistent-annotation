"""Deterministic tests for multiview scan planning without robot motion."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import unittest

from builtin_interfaces.msg import Time
from geometry_msgs.msg import Pose
import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from multi_view_scan.aux_math import (  # noqa: E402
    look_at_camera_pose,
    transform_from_euler,
)
from multi_view_scan.moveit_ik import (  # noqa: E402
    IKResult,
    build_camera_path,
    build_display_trajectory,
    build_trajectory_markers,
    split_trajectory_markers,
)
from multi_view_scan.multi_view_scan import (  # noqa: E402
    DEFAULT_SCAN_CONFIG,
    build_parser,
    filter_reachable_viewpoints,
    load_scan_config,
    optimize_viewpoint_path,
    parse_args,
    reset_generated_directories,
)
from multi_view_scan.scan_trajectory import (  # noqa: E402
    CameraIntrinsics,
    ReachableViewpoint,
    SpiralViewpoint,
    baseline_path_orders,
    dual_pose_distance,
    overlap_violation_count,
    path_has_minimum_overlap,
    pose_distance_matrix,
    projected_overlap,
    sample_scan_volume,
    spiral_hemisphere_pairs,
    nearest_neighbor_open_path,
    open_path_cost,
    two_opt_open_path,
)


class SpiralAndProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.intrinsics = CameraIntrinsics(640, 480, 616.67, 616.67, 320.0, 240.0)

    def test_spiral_is_deterministic_mirrored_and_looks_at_center(self) -> None:
        center = np.array([0.4, 0.0, 0.1])
        first = spiral_hemisphere_pairs(center, 0.4, 12, (10.0, 135.0), (45.0, 75.0))
        second = spiral_hemisphere_pairs(center, 0.4, 12, (10.0, 135.0), (45.0, 75.0))

        self.assertEqual(len(first), 12)
        for first_view, second_view in zip(first, second):
            np.testing.assert_allclose(first_view.left_camera_pose, second_view.left_camera_pose)
            self.assertGreater(first_view.left_camera_pose[1, 3], 0.0)
            self.assertLess(first_view.right_camera_pose[1, 3], 0.0)
            self.assertAlmostEqual(
                first_view.left_camera_pose[1, 3],
                -first_view.right_camera_pose[1, 3],
            )
            expected_direction = center - first_view.left_camera_pose[:3, 3]
            expected_direction /= np.linalg.norm(expected_direction)
            np.testing.assert_allclose(
                first_view.left_camera_pose[:3, 2], expected_direction, atol=1e-12
            )

    def test_projected_overlap_is_symmetric_bounded_and_degrades(self) -> None:
        center = np.zeros(3)
        points = sample_scan_volume(center, 0.15, 4096)
        first = look_at_camera_pose([0.4, 0.0, 0.2], center)
        nearby = look_at_camera_pose([0.38, 0.12, 0.2], center)
        opposite = look_at_camera_pose([-0.4, 0.0, 0.2], center)

        identity_overlap = projected_overlap(first, first, points, self.intrinsics)
        nearby_overlap = projected_overlap(first, nearby, points, self.intrinsics)
        reverse_overlap = projected_overlap(nearby, first, points, self.intrinsics)
        opposite_overlap = projected_overlap(first, opposite, points, self.intrinsics)

        self.assertAlmostEqual(identity_overlap, 1.0)
        self.assertAlmostEqual(nearby_overlap, reverse_overlap)
        self.assertTrue(0.0 <= opposite_overlap < nearby_overlap <= 1.0)


class PathOptimizationTests(unittest.TestCase):
    def test_spiral_mode_selects_source_order_for_execution(self) -> None:
        pose = np.eye(4)
        reachable = [
            ReachableViewpoint(
                SpiralViewpoint(index, 0.0, 45.0, pose, pose),
                pose,
                pose,
                np.array([float(position)]),
            )
            for position, index in enumerate((3, 0, 2, 1))
        ]
        args = argparse.Namespace(
            scan_volume_radius=0.1,
            projection_samples=64,
            distance_metric="joint",
            pose_translation_weight=1.0,
            pose_rotation_weight=0.0,
            overlap_weight=0.5,
            minimum_neighbor_overlap=0.0,
            two_opt_passes=2,
            random_seed=9,
            trajectory_mode="spiral",
        )

        _, selected, diagnostics = optimize_viewpoint_path(
            reachable,
            np.zeros(1),
            pose,
            pose,
            np.zeros(3),
            CameraIntrinsics(640, 480, 600.0, 600.0, 320.0, 240.0),
            args,
        )

        self.assertEqual(
            [item.viewpoint.source_index for item in selected], [0, 1, 2, 3]
        )
        self.assertEqual(diagnostics["trajectory_mode"], "spiral")
        self.assertEqual(set(diagnostics["trajectories"]), {
            "random", "spiral", "hamilton_2opt"
        })

    def test_random_and_spiral_baselines_use_same_reachable_set(self) -> None:
        pose = np.eye(4)
        reachable = [
            ReachableViewpoint(
                SpiralViewpoint(index, 0.0, 45.0, pose, pose),
                pose,
                pose,
                np.zeros(2),
            )
            for index in (6, 1, 4, 2)
        ]

        random_first, spiral = baseline_path_orders(reachable, 19)
        random_second, _ = baseline_path_orders(reachable, 19)

        self.assertEqual(random_first, random_second)
        self.assertEqual(sorted(random_first), list(range(4)))
        self.assertEqual(
            [reachable[index].viewpoint.source_index for index in spiral],
            [1, 2, 4, 6],
        )

    def test_overlap_violations_are_counted_for_baselines(self) -> None:
        overlaps = np.array(
            [[1.0, 0.8, 0.2], [0.8, 1.0, 0.4], [0.2, 0.4, 1.0]]
        )

        self.assertEqual(overlap_violation_count([0, 1, 2], overlaps, 0.5), 1)

    def test_dual_pose_distance_combines_translation_and_rotation(self) -> None:
        identity = np.eye(4)
        translated_left = np.eye(4)
        translated_left[0, 3] = 1.0
        translated_right = np.eye(4)
        translated_right[1, 3] = 2.0
        rotated_left = transform_from_euler("z", 90.0, degrees=True)

        translation_distance = dual_pose_distance(
            identity,
            identity,
            translated_left,
            translated_right,
            1.0,
            0.1,
        )
        rotation_distance = dual_pose_distance(
            identity,
            identity,
            rotated_left,
            identity,
            1.0,
            0.1,
        )

        self.assertAlmostEqual(translation_distance, np.sqrt(5.0))
        self.assertAlmostEqual(rotation_distance, 0.1 * np.pi / 2.0)

    def test_pose_distance_matrix_is_symmetric(self) -> None:
        first_pose = np.eye(4)
        second_pose = np.eye(4)
        second_pose[0, 3] = 0.25
        spiral = SpiralViewpoint(0, 30.0, 45.0, first_pose, first_pose)
        first = ReachableViewpoint(spiral, first_pose, first_pose, np.zeros(2))
        second = ReachableViewpoint(spiral, second_pose, second_pose, np.ones(2))

        distances = pose_distance_matrix([first, second], 1.0, 0.1)

        np.testing.assert_allclose(distances, distances.T)
        np.testing.assert_allclose(np.diag(distances), 0.0)
        self.assertAlmostEqual(distances[0, 1], np.sqrt(2.0) * 0.25)

    def test_nearest_neighbor_uses_best_start_and_returns_permutation(self) -> None:
        costs = np.array(
            [
                [0.0, 1.0, 5.0],
                [1.0, 0.0, 1.0],
                [5.0, 1.0, 0.0],
            ]
        )
        overlaps = np.ones((3, 3))
        starts = np.array([100.0, 100.0, 0.0])

        path = nearest_neighbor_open_path(costs, overlaps, starts, 0.5)

        self.assertEqual(path, [2, 1, 0])
        self.assertEqual(sorted(path), [0, 1, 2])

    def test_nearest_neighbor_rejects_disconnected_overlap_graph(self) -> None:
        costs = np.ones((3, 3)) - np.eye(3)
        overlaps = np.eye(3)
        overlaps[0, 1] = overlaps[1, 0] = 0.8

        with self.assertRaisesRegex(ValueError, "no overlap-feasible"):
            nearest_neighbor_open_path(costs, overlaps, np.zeros(3), 0.5)

    def test_two_opt_improves_cost_without_breaking_overlap(self) -> None:
        costs = np.array(
            [
                [0.0, 10.0, 1.0, 0.0],
                [10.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 10.0],
                [0.0, 1.0, 10.0, 0.0],
            ]
        )
        overlaps = np.ones((4, 4))
        overlaps[0, 3] = overlaps[3, 0] = 0.1
        starts = np.array([0.0, 20.0, 20.0, 20.0])
        initial = [0, 1, 2, 3]

        optimized = two_opt_open_path(initial, costs, overlaps, starts, 0.5, 10)

        self.assertEqual(optimized, [0, 2, 1, 3])
        self.assertLess(
            open_path_cost(optimized, costs, starts),
            open_path_cost(initial, costs, starts),
        )
        self.assertTrue(path_has_minimum_overlap(optimized, overlaps, 0.5))


class _FakeIKClient:
    def __init__(self) -> None:
        self.call_count = 0

    def wait_for_joint_state(self, _timeout: float) -> dict[str, float]:
        return {"joint_a": 0.0, "joint_b": 0.0, "left_finger_joint": 0.04}

    def wait_for_service(self, _timeout: float) -> bool:
        return True

    def solve(self, _targets, **_kwargs) -> IKResult:
        self.call_count += 1
        if self.call_count == 2:
            return IKResult(False, {}, "synthetic collision", -31)
        return IKResult(
            True,
            {
                "joint_a": float(self.call_count),
                "joint_b": float(self.call_count + 1),
                "left_finger_joint": 0.04,
            },
            "IK solution found",
            1,
        )


class ScanIntegrationHelperTests(unittest.TestCase):
    def test_default_yaml_contains_every_scan_parameter(self) -> None:
        parser = build_parser()
        configured = load_scan_config(DEFAULT_SCAN_CONFIG, parser)
        expected = {
            action.dest
            for action in parser._actions
            if action.dest not in {"help", "config"}
        }

        self.assertEqual(set(configured), expected)
        defaults = parse_args([])
        self.assertEqual(defaults.distance_metric, "pose")
        self.assertEqual(defaults.trajectory_mode, "hamilton_2opt")
        self.assertEqual(defaults.random_seed, 7)
        self.assertEqual(defaults.before_trajectory_color_rgb, [255, 0, 255])
        self.assertEqual(defaults.optimized_trajectory_color_rgb, [25, 255, 0])

    def test_cli_values_override_selected_yaml_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "scan.yaml"
            config_path.write_text(
                "multi_view_scan:\n"
                "  view_count: 12\n"
                "  use_sim_time: false\n"
                "  output_dir: yaml_output\n",
                encoding="utf-8",
            )

            args = parse_args(
                [
                    "--config",
                    str(config_path),
                    "--view-count",
                    "20",
                    "--trajectory-mode",
                    "random",
                ]
            )

        self.assertEqual(args.config, config_path)
        self.assertEqual(args.view_count, 20)
        self.assertEqual(args.trajectory_mode, "random")
        self.assertFalse(args.use_sim_time)
        self.assertEqual(args.output_dir, Path("yaml_output"))

    def test_rviz_preview_contains_start_waypoints_and_camera_path(self) -> None:
        display = build_display_trajectory(
            ("joint_a", "joint_b"),
            (0.0, 0.5),
            ((1.0, 1.5), (2.0, 2.5)),
            0.25,
        )
        trajectory = display.trajectory[0].joint_trajectory

        self.assertTrue(display.trajectory_start.is_diff)
        self.assertEqual(trajectory.joint_names, ["joint_a", "joint_b"])
        self.assertEqual(len(trajectory.points), 3)
        self.assertEqual(list(trajectory.points[-1].positions), [2.0, 2.5])
        self.assertEqual(trajectory.points[-1].time_from_start.nanosec, 500_000_000)

        poses = [Pose(), Pose()]
        poses[1].position.x = 0.4
        camera_path = build_camera_path(poses, "world", Time(sec=12))
        self.assertEqual(camera_path.header.frame_id, "world")
        self.assertEqual(camera_path.header.stamp.sec, 12)
        self.assertEqual(len(camera_path.poses), 2)
        self.assertAlmostEqual(camera_path.poses[-1].pose.position.x, 0.4)

    def test_rviz_comparison_uses_wide_contrasting_neon_lines(self) -> None:
        before = [Pose(), Pose()]
        optimized = [Pose(), Pose(), Pose()]
        before[1].position.x = 0.3
        optimized[-1].position.y = 0.4

        marker_array = build_trajectory_markers(
            before_left_poses=before,
            before_right_poses=before,
            optimized_left_poses=optimized,
            optimized_right_poses=optimized,
            frame_id="world",
            stamp=Time(sec=12),
            line_width=0.015,
            before_color_rgb=(255, 0, 255),
            optimized_color_rgb=(25, 255, 0),
        )

        self.assertEqual(len(marker_array.markers), 5)
        self.assertEqual(
            marker_array.markers[0].action, marker_array.markers[0].DELETEALL
        )
        before_marker, _, optimized_marker, _ = marker_array.markers[1:]
        self.assertEqual(before_marker.ns, "trajectory_0_before_2opt_left")
        self.assertEqual(optimized_marker.ns, "trajectory_1_after_2opt_left")
        self.assertAlmostEqual(before_marker.scale.x, 0.024)
        self.assertAlmostEqual(optimized_marker.scale.x, 0.015)
        self.assertAlmostEqual(before_marker.color.a, 0.55)
        self.assertEqual(optimized_marker.color.a, 1.0)
        self.assertEqual(len(before_marker.points), 2)
        self.assertEqual(
            (before_marker.color.r, before_marker.color.g, before_marker.color.b),
            (1.0, 0.0, 1.0),
        )
        self.assertEqual(len(optimized_marker.points), 3)
        self.assertAlmostEqual(optimized_marker.color.r, 25.0 / 255.0)
        self.assertEqual(optimized_marker.color.g, 1.0)
        self.assertEqual(optimized_marker.color.b, 0.0)

        before_only, optimized_only = split_trajectory_markers(marker_array)
        self.assertEqual(len(before_only.markers), 3)
        self.assertEqual(len(optimized_only.markers), 3)
        self.assertTrue(
            all("before_2opt" in marker.ns for marker in before_only.markers[1:])
        )
        self.assertTrue(
            all("after_2opt" in marker.ns for marker in optimized_only.markers[1:])
        )

    def test_unreachable_candidate_is_removed_by_fake_ik(self) -> None:
        viewpoints = spiral_hemisphere_pairs(
            [0.4, 0.0, 0.0], 0.4, 3, (10.0, 135.0), (45.0, 75.0)
        )
        args = argparse.Namespace(
            joint_state_timeout=0.1,
            left_tcp_frame="left_tcp",
            right_tcp_frame="right_tcp",
            planning_group="dual_arm",
            world_frame="world",
            ik_timeout=0.1,
        )

        reachable, joint_names, start_values, rejected = filter_reachable_viewpoints(
            _FakeIKClient(),
            viewpoints,
            np.eye(4),
            np.eye(4),
            args,
        )

        self.assertEqual([item.viewpoint.source_index for item in reachable], [0, 2])
        self.assertEqual(joint_names, ("joint_a", "joint_b"))
        np.testing.assert_array_equal(start_values, [0.0, 0.0])
        self.assertEqual(rejected[0]["spiral_index"], 2)
        self.assertEqual(rejected[0]["reason"], "synthetic collision")

    def test_cleanup_recreates_only_empty_steps_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            for name in ("steps", "grasp_commands", "mask_segment", "pointclouds"):
                directory = output_dir / name
                directory.mkdir()
                (directory / "old-data.txt").write_text("old", encoding="utf-8")

            steps_dir = reset_generated_directories(output_dir)

            self.assertEqual(steps_dir, output_dir / "steps")
            self.assertEqual(list(steps_dir.iterdir()), [])
            for name in ("grasp_commands", "mask_segment", "pointclouds"):
                self.assertFalse((output_dir / name).exists())


if __name__ == "__main__":
    unittest.main()
