"""Offline tests for the single-arm hemisphere planner."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from builtin_interfaces.msg import Time


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from path_planning_single.planning import (  # noqa: E402
    CameraViewpoint,
    ReachableViewpoint,
    baseline_orders,
    hemisphere_triangles,
    joint_distances,
    overlap_violation_count,
    sample_hemisphere,
)
from path_planning_single.single_path_planner import (  # noqa: E402
    build_visualizations,
    load_configuration,
    validate_configuration,
)
from path_planning_single.single_view_scan import (  # noqa: E402
    clear_previous_manifest,
    load_scan_configuration,
    output_directories,
    save_segmentation,
    validate_scan_configuration,
)


class SinglePathPlanningTests(unittest.TestCase):
    def test_kinova_scan_configuration_and_suffix_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            config = load_scan_configuration(
                [
                    "--output-dir",
                    str(output_root),
                    "--output-suffix",
                    "spiral_1",
                ]
            )

            validate_scan_configuration(config)
            directories = output_directories(config)

            self.assertEqual(config.planning_group, "manipulator")
            self.assertEqual(config.ready_state, "Ready")
            self.assertEqual(config.camera_frame, "camera_color_frame")
            self.assertEqual(
                config.segmentation_camera_frame,
                "handeye_camera_color_optical_frame",
            )
            self.assertGreaterEqual(config.trajectory_preview_time, 0.0)
            self.assertEqual(
                directories["images"], output_root / "spiral_1" / "save_images"
            )
            self.assertEqual(
                directories["segments"], output_root / "spiral_1" / "save_segment"
            )
            self.assertEqual(
                directories["transforms"], output_root / "spiral_1" / "save_TF"
            )

            # Reusing the suffix removes stale products from all three folders.
            stale_file = directories["images"] / "color_9.png"
            stale_file.write_bytes(b"stale")
            recreated = output_directories(config)
            self.assertFalse((recreated["images"] / "color_9.png").exists())

            manifest = output_root / "spiral_1" / "manifest.json"
            manifest.write_text("stale", encoding="utf-8")
            clear_previous_manifest(manifest)
            self.assertFalse(manifest.exists())

    def test_disabled_segmentation_does_not_require_a_service(self) -> None:
        record = save_segmentation(
            None,
            Path("/tmp/segment_2"),
            "camera_frame",
            35.0,
            Path("/tmp"),
        )

        self.assertEqual(record, {"status": "disabled"})

    def test_trajectory_mode_can_be_selected_from_cli(self) -> None:
        config = load_configuration(["--trajectory-mode", "spiral"])

        validate_configuration(config)
        self.assertEqual(config.trajectory_mode, "spiral")

    def test_hemisphere_sampling_is_deterministic_and_looks_inward(self) -> None:
        center = np.array([0.4, 0.0, 0.2])
        first = sample_hemisphere(center, 0.5, 4, 5, (10.0, 80.0))
        second = sample_hemisphere(center, 0.5, 4, 5, (10.0, 80.0))

        self.assertEqual(len(first), 20)
        for left, right in zip(first, second):
            np.testing.assert_allclose(left.camera_pose, right.camera_pose)
            offset = left.camera_pose[:3, 3] - center
            self.assertAlmostEqual(np.linalg.norm(offset), 0.5)
            self.assertGreater(offset[2], 0.0)
            np.testing.assert_allclose(
                left.camera_pose[:3, 2], -offset / np.linalg.norm(offset)
            )

        # Every latitude contains K uniformly spaced azimuth samples.
        self.assertEqual([point.elevation_deg for point in first[:5]], [10.0] * 5)
        np.testing.assert_allclose(
            [point.azimuth_deg for point in first[:5]],
            [0.0, 72.0, 144.0, 216.0, 288.0],
        )

    def test_transparent_shell_geometry_has_two_triangles_per_cell(self) -> None:
        triangles = hemisphere_triangles(
            [0.0, 0.0, 0.0], 0.4, (0.0, 90.0), 12, 4
        )

        self.assertEqual(triangles.shape, (12 * 4 * 2 * 3, 3))
        np.testing.assert_allclose(np.linalg.norm(triangles, axis=1), 0.4)
        self.assertTrue(np.all(triangles[:, 2] >= -1e-12))

    def test_hemisphere_marker_has_valid_unit_scale(self) -> None:
        config = load_configuration([])
        sampled = sample_hemisphere(config.center, config.radius, 1, 3, (15.0, 80.0))
        reachable = [
            ReachableViewpoint(viewpoint, viewpoint.camera_pose, np.zeros(7))
            for viewpoint in sampled[:2]
        ]

        messages = build_visualizations(
            config, sampled, reachable, {2}, [1, 0], [0, 1], [1, 0], Time()
        )
        shell = messages["hemisphere"].markers[1]
        accepted = messages["candidates"].markers[1]
        rejected = messages["candidates"].markers[2]

        self.assertEqual((shell.scale.x, shell.scale.y, shell.scale.z), (1.0, 1.0, 1.0))
        self.assertGreater(shell.color.a, 0.0)
        self.assertLess(shell.color.a, 1.0)
        self.assertEqual(accepted.type, accepted.SPHERE_LIST)
        self.assertEqual((accepted.color.r, accepted.color.g, accepted.color.b), (1.0, 40.0 / 255.0, 40.0 / 255.0))
        self.assertEqual(len(accepted.points), 2)
        self.assertEqual(rejected.type, rejected.LINE_LIST)
        self.assertEqual((rejected.color.r, rejected.color.g, rejected.color.b), (0.0, 0.0, 0.0))
        self.assertEqual(len(rejected.points), 4)

        self.assertEqual(set(messages), {
            "hemisphere",
            "candidates",
            "random_path",
            "spiral_path",
            "optimized_path",
            "comparison",
        })
        self.assertEqual(len(messages["comparison"].markers), 4)

    def test_baselines_share_views_and_random_order_is_reproducible(self) -> None:
        pose = np.eye(4)
        source_indices = [8, 2, 5, 1]
        reachable = [
            ReachableViewpoint(
                CameraViewpoint(index, 0.0, 30.0, pose), pose, np.zeros(2)
            )
            for index in source_indices
        ]

        first_random, spiral = baseline_orders(reachable, random_seed=17)
        second_random, _ = baseline_orders(reachable, random_seed=17)

        self.assertEqual(first_random, second_random)
        self.assertEqual([source_indices[index] for index in spiral], [1, 2, 5, 8])
        self.assertEqual(sorted(first_random), list(range(len(reachable))))

    def test_overlap_violation_count_checks_consecutive_edges(self) -> None:
        overlaps = np.array(
            [
                [1.0, 0.7, 0.2],
                [0.7, 1.0, 0.4],
                [0.2, 0.4, 1.0],
            ]
        )

        self.assertEqual(overlap_violation_count([0, 1, 2], overlaps, 0.5), 1)
        self.assertEqual(overlap_violation_count([0], overlaps, 0.5), 0)

    def test_single_arm_joint_distance_is_symmetric(self) -> None:
        pose = np.eye(4)
        viewpoint = CameraViewpoint(0, 0.0, 30.0, pose)
        reachable = [
            ReachableViewpoint(viewpoint, pose, np.array([0.0, 1.0])),
            ReachableViewpoint(viewpoint, pose, np.array([3.0, 5.0])),
        ]

        distances = joint_distances(reachable)

        np.testing.assert_allclose(distances, distances.T)
        self.assertAlmostEqual(distances[0, 1], 5.0)


if __name__ == "__main__":
    unittest.main()
