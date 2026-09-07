"""Offline tests for the single-arm hemisphere planner."""

from pathlib import Path
import sys
import unittest

import numpy as np
from builtin_interfaces.msg import Time


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from path_planning_single.planning import (  # noqa: E402
    CameraViewpoint,
    ReachableViewpoint,
    hemisphere_triangles,
    joint_distances,
    sample_hemisphere,
)
from path_planning_single.single_path_planner import (  # noqa: E402
    build_visualizations,
    load_configuration,
)


class SinglePathPlanningTests(unittest.TestCase):
    def test_hemisphere_sampling_is_deterministic_and_looks_inward(self) -> None:
        center = np.array([0.4, 0.0, 0.2])
        first = sample_hemisphere(center, 0.5, 20, (10.0, 80.0))
        second = sample_hemisphere(center, 0.5, 20, (10.0, 80.0))

        self.assertEqual(len(first), 20)
        for left, right in zip(first, second):
            np.testing.assert_allclose(left.camera_pose, right.camera_pose)
            offset = left.camera_pose[:3, 3] - center
            self.assertAlmostEqual(np.linalg.norm(offset), 0.5)
            self.assertGreater(offset[2], 0.0)
            np.testing.assert_allclose(
                left.camera_pose[:3, 2], -offset / np.linalg.norm(offset)
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
        sampled = sample_hemisphere(config.center, config.radius, 3, (15.0, 80.0))
        reachable = [
            ReachableViewpoint(viewpoint, viewpoint.camera_pose, np.zeros(7))
            for viewpoint in sampled[:2]
        ]

        messages = build_visualizations(
            config, sampled, reachable, {2}, [0, 1], [1, 0], Time()
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
