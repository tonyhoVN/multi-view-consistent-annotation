#!/usr/bin/env python3
"""Sample and score parallel-jaw grasps from a segmented object point cloud.

The generated command sequence is descriptive and does not move the robot.
Candidate transforms use +X as the finger-closing axis and +Z from the palm
toward the fingertips, opposite the sampled outward normal. The motion from
pregrasp to grasp is along +Z.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import numpy.typing as npt
import open3d as o3d
from scipy.spatial import cKDTree

from aux_math import matrix_to_pose
from o3d_process import (
    estimate_outward_normals,
    estimate_principal_curvature_directions,
    point_array,
    read_point_cloud,
    visualize_grasp,
    voxel_downsample,
)


@dataclass(frozen=True)
class GripperGeometry:
    """Parallel-jaw geometry in meters, centered at the contact plane.

    +Z points from palm to fingertips. ``grasp_depth`` is the distance from the
    contact-plane origin to the fingertips along +Z. X aligned with parallel gripper movement.
    """

    min_opening: float = 0.010
    max_opening: float = 0.080
    contact_padding: float = 0.003
    finger_reach: float = 0.060 # Tip of finger
    hand_depth: float = 0.060 # Hand in Y direction
    outer_width: float = 0.105 # From left end to right end in X direction
    grasp_depth: float = 0.020 # From fringer tip to  
    pregrasp_distance: float = 0.100


@dataclass(frozen=True)
class SamplingCriteria:
    """Sampling and acceptance criteria."""

    samples: int = 5000
    voxel_size: float = 0.003
    normal_neighbors: int = 24
    antipodal_angle_deg: float = 35.0
    contact_radius: float = 0.010
    minimum_contact_neighbors: int = 5
    maximum_candidates: int = 50
    random_seed: int = 7


@dataclass(frozen=True)
class GraspCandidate:
    """One accepted grasp pose and its score components."""

    transform: np.ndarray
    pregrasp_transform: np.ndarray
    required_opening: float
    score: float
    contact_a: Optional[np.ndarray] = None
    contact_b: Optional[np.ndarray] = None
    local_score: float = 0.0
    approach_score: float = 0.0
    center_score: float = 0.0


def _validate_configuration(
    geometry: GripperGeometry, criteria: SamplingCriteria
) -> None:
    if not 0.0 <= geometry.min_opening < geometry.max_opening:
        raise ValueError("gripper opening limits are invalid")
    if geometry.contact_padding < 0.0:
        raise ValueError("contact padding must not be negative")
    if geometry.finger_reach <= 0.0 or geometry.pregrasp_distance <= 0.0:
        raise ValueError("finger reach and pregrasp distance must be positive")
    if not 0.0 <= geometry.grasp_depth <= geometry.finger_reach:
        raise ValueError("grasp depth must be between zero and finger reach")
    if geometry.hand_depth <= 0.0 or geometry.outer_width <= geometry.max_opening:
        raise ValueError("hand depth and outer width are inconsistent")
    if criteria.samples <= 0 or criteria.maximum_candidates <= 0:
        raise ValueError("sample and candidate counts must be positive")
    if criteria.voxel_size <= 0.0 or criteria.contact_radius <= 0.0:
        raise ValueError("voxel size and contact radius must be positive")
    if criteria.normal_neighbors < 3:
        raise ValueError("normal-neighbor count must be at least three")
    if criteria.minimum_contact_neighbors < 1:
        raise ValueError("minimum contact-neighbor count must be positive")
    if not 0.0 < criteria.antipodal_angle_deg < 90.0:
        raise ValueError("antipodal angle must be between zero and 90 degrees")


def _approach_axis(closing_axis: np.ndarray, preferred: np.ndarray) -> np.ndarray:
    """Project a preferred approach direction perpendicular to the closing axis."""
    approach = preferred - np.dot(preferred, closing_axis) * closing_axis
    norm = np.linalg.norm(approach)
    if norm < 1e-8:
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(fallback, closing_axis)) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0])
        approach = fallback - np.dot(fallback, closing_axis) * closing_axis
        norm = np.linalg.norm(approach)
    return approach / norm


def _grasp_transform(
    center: np.ndarray, closing_axis: np.ndarray, approach_axis: np.ndarray
) -> np.ndarray:
    """Build a right-handed gripper pose with X closing and Z approaching."""
    x_axis = closing_axis / np.linalg.norm(closing_axis)
    z_axis = approach_axis / np.linalg.norm(approach_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    transform[:3, 3] = center
    return transform


def _palm_clearance(
    points: np.ndarray,
    center: np.ndarray,
    rotation: np.ndarray,
    geometry: GripperGeometry,
) -> tuple[bool, float]:
    """Reject objects that extend into the nominal palm volume."""
    local = (points - center) @ rotation
    within_hand_cross_section = (
        (np.abs(local[:, 0]) <= geometry.outer_width / 2.0)
        & (np.abs(local[:, 1]) <= geometry.hand_depth / 2.0)
    )
    rear_extent = local[within_hand_cross_section, 2].min(initial=0.0)
    clearance = geometry.finger_reach + rear_extent
    return clearance >= 0.0, float(np.clip(clearance / geometry.finger_reach, 0.0, 1.0))


def _approach_is_clear(
    scene_points: Optional[np.ndarray],
    center: np.ndarray,
    rotation: np.ndarray,
    geometry: GripperGeometry,
) -> bool:
    """Check the pregrasp swept box when a scene cloud is supplied."""
    if scene_points is None:
        return True
    local = (scene_points - center) @ rotation
    corridor = (
        (np.abs(local[:, 0]) <= geometry.outer_width / 2.0)
        & (np.abs(local[:, 1]) <= geometry.hand_depth / 2.0)
        & (local[:, 2] < -geometry.finger_reach)
        & (local[:, 2] > -geometry.pregrasp_distance)
    )
    return not bool(np.any(corridor))


def sample_grasps_temp(
    object_cloud: o3d.geometry.PointCloud | npt.ArrayLike,
    *,
    scene_cloud: o3d.geometry.PointCloud | npt.ArrayLike | None = None,
    geometry: GripperGeometry = GripperGeometry(),
    criteria: SamplingCriteria = SamplingCriteria(),
    preferred_approach: npt.ArrayLike = (0.0, 0.0, -1.0),
) -> list[GraspCandidate]:
    """Return candidates from the previous random antipodal-pair sampler.

    This method is retained temporarily for comparison with :func:`sample_grasps`.
    """
    # Verify object point cloud
    _validate_configuration(geometry, criteria)
    points = voxel_downsample(point_array(object_cloud), criteria.voxel_size)
    if len(points) < 20:
        raise ValueError("too few points remain after voxel downsampling")

    # Verify scene point cloud
    scene_points = None if scene_cloud is None else point_array(scene_cloud)

    # Estimate normal
    normals = estimate_outward_normals(points, criteria.normal_neighbors)
    tree = cKDTree(points)
    centroid = points.mean(axis=0)
    object_radius = max(float(np.linalg.norm(points - centroid, axis=1).max()), 1e-6)


    preferred = np.asarray(preferred_approach, dtype=np.float64).reshape(3)
    preferred /= np.linalg.norm(preferred)
    antipodal_threshold = math.cos(math.radians(criteria.antipodal_angle_deg))
    max_contact_distance = geometry.max_opening - 2.0 * geometry.contact_padding
    min_contact_distance = max(0.0, geometry.min_opening - 2.0 * geometry.contact_padding)
    rng = np.random.default_rng(criteria.random_seed)
    candidates: list[GraspCandidate] = []

    for _ in range(criteria.samples):
        # Random select first point
        first = int(rng.integers(len(points)))
        neighbors = tree.query_ball_point(points[first], max_contact_distance)

        # If cannot find antipodal within the max_contact_distance
        if len(neighbors) < 2: 
            continue

        # Randomly select the antipodal point
        second = int(neighbors[int(rng.integers(len(neighbors)))])
        if first == second:
            continue

        # Check distance bwt 2 points within open range
        difference = points[second] - points[first]
        distance = float(np.linalg.norm(difference))
        if distance < min_contact_distance or distance > max_contact_distance:
            continue

        # 
        closing = difference / distance # Closing direction
        alignment_forward = min(
            -float(np.dot(normals[first], closing)),
            float(np.dot(normals[second], closing)),
        )
        alignment_reverse = min(
            float(np.dot(normals[first], closing)),
            -float(np.dot(normals[second], closing)),
        )
        antipodal_score = max(alignment_forward, alignment_reverse)
        if antipodal_score < antipodal_threshold:
            continue
        if alignment_reverse > alignment_forward:
            closing = -closing

        center = (points[first] + points[second]) / 2.0
        approach = _approach_axis(closing, preferred)
        transform = _grasp_transform(center, closing, approach)
        palm_is_clear, palm_score = _palm_clearance(
            points, center, transform[:3, :3], geometry
        )
        if not palm_is_clear or not _approach_is_clear(
            scene_points, center, transform[:3, :3], geometry
        ):
            continue

        support_a = len(tree.query_ball_point(points[first], criteria.contact_radius))
        support_b = len(tree.query_ball_point(points[second], criteria.contact_radius))
        support_score = min(
            1.0,
            min(support_a, support_b) / criteria.minimum_contact_neighbors,
        )
        if support_score < 1.0:
            continue

        # Center score: if grasping closed to center of object pointcloud
        required_opening = distance + 2.0 * geometry.contact_padding
        center_score = math.exp(-float(np.linalg.norm(center - centroid)) / object_radius)

        # Approach score: if object is closed to 
        approach_score = max(0.0, float(np.dot(approach, preferred)))
        width_score = 1.0 - (
            (required_opening - geometry.min_opening)
            / (geometry.max_opening - geometry.min_opening)
        )
        width_score = float(np.clip(width_score, 0.0, 1.0))
        score = (
            0.40 * antipodal_score
            # + 0.20 * center_score
            + 0.15 * approach_score
            + 0.10 * width_score
            + 0.10 * support_score
            + 0.05 * palm_score
        )

        pregrasp = transform.copy()
        pregrasp[:3, 3] = center - approach * geometry.pregrasp_distance
        candidates.append(
            GraspCandidate(
                transform=transform,
                pregrasp_transform=pregrasp,
                contact_a=points[first],
                contact_b=points[second],
                required_opening=required_opening,
                score=score,
                local_score=antipodal_score,
                approach_score=approach_score,
                center_score=center_score,
            )
        )

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    unique: list[GraspCandidate] = []
    for candidate in candidates:
        duplicate = any(
            np.linalg.norm(candidate.transform[:3, 3] - accepted.transform[:3, 3]) < 0.010
            and abs(
                np.dot(candidate.transform[:3, 0], accepted.transform[:3, 0])
            ) > 0.95
            for accepted in unique
        )
        if not duplicate:
            unique.append(candidate)
        if len(unique) >= criteria.maximum_candidates:
            break
    return unique


def _open_gripper_collides_temp(
    collision_tree: cKDTree,
    collision_points: np.ndarray,
    transform: np.ndarray,
    geometry: GripperGeometry,
) -> bool:
    """Test points against the two open fingers and the nominal palm volume."""
    palm_thickness = 0.012
    finger_base_z = geometry.grasp_depth - geometry.finger_reach
    fingertip_z = geometry.grasp_depth
    search_radius = float(
        np.linalg.norm(
            [
                geometry.outer_width / 2.0,
                geometry.hand_depth / 2.0,
                max(abs(finger_base_z - palm_thickness), abs(fingertip_z)),
            ]
        )
    )
    nearby = collision_tree.query_ball_point(transform[:3, 3], search_radius)
    if not nearby:
        return False

    # Express nearby scene and object points in the candidate gripper frame.
    local = (collision_points[nearby] - transform[:3, 3]) @ transform[:3, :3]
    abs_x = np.abs(local[:, 0])
    within_y = np.abs(local[:, 1]) <= geometry.hand_depth / 2.0
    within_finger_depth = (
        (local[:, 2] >= finger_base_z) & (local[:, 2] <= fingertip_z)
    )
    inside_open_fingers = (
        (abs_x >= geometry.max_opening / 2.0)
        & (abs_x <= geometry.outer_width / 2.0)
        & within_y
        & within_finger_depth
    )
    inside_palm = (
        (abs_x <= geometry.outer_width / 2.0)
        & within_y
        & (local[:, 2] >= finger_base_z - palm_thickness)
        & (local[:, 2] < finger_base_z)
    )
    return bool(np.any(inside_open_fingers | inside_palm))


def _open_gripper_collides(
    collision_tree: cKDTree,
    collision_points: np.ndarray,
    transform: np.ndarray,
    geometry: GripperGeometry,
) -> bool:
    """Test points against the two open fingers and the nominal palm volume."""
    finger_base_z = geometry.grasp_depth - geometry.finger_reach
    fingertip_z = geometry.grasp_depth
    search_radius = float(
        np.linalg.norm(
            [
                geometry.outer_width / 2.0,
                geometry.hand_depth / 2.0,
                geometry.finger_reach,
            ]
        )
    )
    nearby = collision_tree.query_ball_point(transform[:3, 3], search_radius)
    if not nearby:
        return False

    # Express nearby scene and object points in the candidate gripper frame.
    local = (collision_points[nearby] - transform[:3, 3]) @ transform[:3, :3]

    # Check if any points inside fingers Boundary box
    within_x = ((np.abs(local[:, 0]) >= geometry.max_opening / 2.0) & (np.abs(local[:, 0]) <= geometry.outer_width / 2.0))
    within_y = (np.abs(local[:, 1]) <= geometry.hand_depth / 2.0)
    within_z = ((local[:, 2] >= finger_base_z) & (local[:, 2] <= fingertip_z))
    inside_open_fingers = (within_x & within_y & within_z)

    # # Check if 2 finger tips collides with table
    # left_finger_local = np.array([-(geometry.max_opening/2), 0, fingertip_z])
    # right_finger_local = np.array([(geometry.max_opening/2), 0, fingertip_z])
    # left_finger_global = transform @ left_finger_local
    # right_finger_global = transform @ right_finger_local

    return bool(np.any(inside_open_fingers))


def _local_grasp_for_principal_direction(
    sample_index: int,
    principal_direction: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    object_tree: cKDTree,
    collision_tree: cKDTree,
    collision_points: np.ndarray,
    geometry: GripperGeometry,
    criteria: SamplingCriteria,
) -> Optional[GraspCandidate]:
    """Build and validate a grasp aligned with one principal-curvature axis."""
    sample_point = points[sample_index]
    z_axis = -normals[sample_index]
    z_axis /= np.linalg.norm(z_axis)
    x_axis = principal_direction - np.dot(principal_direction, z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-8:
        return None
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))

    finger_base_z = geometry.grasp_depth - geometry.finger_reach
    fingertip_z = geometry.grasp_depth
    neighborhood_radius = float(
        np.linalg.norm(
            [
                geometry.max_opening ,
                geometry.hand_depth ,
                max(abs(finger_base_z), abs(fingertip_z)),
            ]
        )
    )
    nearby = np.asarray(
        object_tree.query_ball_point(sample_point, neighborhood_radius), dtype=np.int64
    )
    if len(nearby) < criteria.minimum_contact_neighbors:
        return None

    # The sampled point is the X/Y center and Z=0 contact-plane origin.
    # local = (points[nearby] - sample_point) @ rotation
    local = (points - sample_point) @ rotation # take all points in the object cloud to check if any point collides with the gripper
    active = (
        (np.abs(local[:, 0]) <= geometry.max_opening / 2.0)
        & (np.abs(local[:, 1]) <= geometry.hand_depth / 2.0)
        & (local[:, 2] >= finger_base_z)
        & (local[:, 2] <= fingertip_z)
    )
    # active_indices = nearby[active]
    active_indices = np.where(active)[0]
    if len(active_indices) < criteria.minimum_contact_neighbors:
        return None

    # Infer the required opening from every object point inside the grasp box.
    active_x = local[active, 0]
    object_width = float(np.ptp(active_x))
    required_opening = object_width + 2.0 * geometry.contact_padding
    if not geometry.min_opening <= required_opening <= geometry.max_opening:
        return None

    center = sample_point
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = center
    if _open_gripper_collides(
        collision_tree, collision_points, transform, geometry
    ):
        return None

    # Average the requested X/Z normal-alignment sum over all enclosed points.
    local_normals = normals[active_indices]
    x_alignment = np.abs(local_normals @ rotation[:, 0])
    z_alignment = np.abs(local_normals @ rotation[:, 2])
    local_score = float(np.sum(0.7 * x_alignment + 0.3 * z_alignment) / len(active_indices))
    pregrasp = transform.copy()
    pregrasp[:3, 3] = center - rotation[:, 2] * geometry.pregrasp_distance
    return GraspCandidate(
        transform=transform,
        pregrasp_transform=pregrasp,
        required_opening=required_opening,
        score=local_score,
        local_score=local_score,
    )


def sample_grasps(
    object_cloud: o3d.geometry.PointCloud | npt.ArrayLike,
    *,
    scene_cloud: o3d.geometry.PointCloud | npt.ArrayLike | None = None,
    geometry: GripperGeometry = GripperGeometry(),
    criteria: SamplingCriteria = SamplingCriteria(),
    preferred_approach: npt.ArrayLike = (0.0, 0.0, -1.0),
) -> list[GraspCandidate]:
    """Sample principal-curvature grasps and return globally ranked candidates.

    For each sampled point, +Z opposes its outward normal and +X is tested along
    its two local principal-curvature directions. Only the locally better valid
    direction is retained before pregrasp validation and global ranking.
    """
    _validate_configuration(geometry, criteria)

    # Prepare object normals, principal-curvature frames, and spatial search trees.
    points = voxel_downsample(point_array(object_cloud), criteria.voxel_size)
    if len(points) < 20:
        raise ValueError("too few points remain after voxel downsampling")
    normals = estimate_outward_normals(points, criteria.normal_neighbors)
    principal_directions = estimate_principal_curvature_directions(
        points, normals, criteria.normal_neighbors
    )
    object_tree = cKDTree(points)
    scene_points = None if scene_cloud is None else point_array(scene_cloud)
    collision_points = (
        points if scene_points is None else np.vstack((scene_points, points))
    )
    collision_tree = cKDTree(collision_points)
    centroid = points.mean(axis=0)
    object_radius = max(
        float(np.linalg.norm(points - centroid, axis=1).max()), 1e-6
    )

    # Normalize the requested direction of motion from pregrasp to grasp.
    preferred = np.asarray(preferred_approach, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(preferred)) or np.linalg.norm(preferred) < 1e-8:
        raise ValueError("preferred_approach must be a finite nonzero vector")
    preferred /= np.linalg.norm(preferred)
    rng = np.random.default_rng(criteria.random_seed)
    local_best_candidates: list[GraspCandidate] = []

    # Keep the better of the two principal-curvature closing directions.
    for _ in range(criteria.samples):
        sample_index = int(rng.integers(len(points)))
        local_best: Optional[GraspCandidate] = None
        for principal_direction in principal_directions[sample_index]:
            candidate = _local_grasp_for_principal_direction(
                sample_index,
                principal_direction,
                points,
                normals,
                object_tree,
                collision_tree,
                collision_points,
                geometry,
                criteria,
            )
            if candidate is not None and (
                local_best is None or candidate.local_score > local_best.local_score
            ):
                local_best = candidate
        if local_best is not None:
            local_best_candidates.append(local_best)

    # Reject pregrasp collisions, then combine local, approach, and center scores.
    global_candidates: list[GraspCandidate] = []
    for candidate in local_best_candidates:
        if _open_gripper_collides(
            collision_tree, collision_points, candidate.pregrasp_transform, geometry
        ):
            continue
        motion_direction = candidate.transform[:3, 2]
        approach_score = max(0.0, float(np.dot(motion_direction, preferred)))
        center_distance = float(np.linalg.norm(candidate.transform[:3, 3] - centroid))
        center_score = math.exp(-center_distance / object_radius)
        normalized_local_score = candidate.local_score / 2.0
        global_score = (
            0.60 * normalized_local_score
            + 0.25 * approach_score
            + 0.15 * center_score
        )
        global_candidates.append(
            GraspCandidate(
                transform=candidate.transform,
                pregrasp_transform=candidate.pregrasp_transform,
                required_opening=candidate.required_opening,
                score=global_score,
                local_score=candidate.local_score,
                approach_score=approach_score,
                center_score=center_score,
            )
        )

    # Sort globally and remove nearly identical positions and orientations.
    global_candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    unique: list[GraspCandidate] = []
    for candidate in global_candidates:
        duplicate = any(
            np.linalg.norm(candidate.transform[:3, 3] - accepted.transform[:3, 3]) < 0.010
            and abs(float(np.dot(candidate.transform[:3, 0], accepted.transform[:3, 0]))) > 0.95
            and float(np.dot(candidate.transform[:3, 2], accepted.transform[:3, 2])) > 0.95
            for accepted in unique
        )
        if not duplicate:
            unique.append(candidate)
        if len(unique) >= criteria.maximum_candidates:
            break
    return unique


def select_arm(position: Sequence[float], center_tolerance: float = 1e-4) -> str:
    """Assign a grasp using the project's world-Y arm convention."""
    y_position = float(position[1])
    if y_position > center_tolerance:
        return "left"
    if y_position < -center_tolerance:
        return "right"
    return "left"


def pose_record(transform: np.ndarray) -> dict[str, list[float]]:
    """Convert a homogeneous transform to JSON-compatible pose values."""
    position, quaternion = matrix_to_pose(transform)
    return {
        "position_xyz": position.tolist(),
        "quaternion_xyzw": quaternion.tolist(),
    }


def build_gripper_commands(
    candidate: GraspCandidate,
    *,
    object_name: str = "object",
    arm: Optional[str] = None,
) -> dict[str, Any]:
    """Build an API-neutral, ordered grasp command sequence."""
    selected_arm = arm or select_arm(candidate.transform[:3, 3])
    if selected_arm not in {"left", "right"}:
        raise ValueError("arm must be 'left' or 'right'")
    end_effector = f"{selected_arm}_ee"
    # planning_group = f"{selected_arm}_arm"
    planning_group = "dual_arm"
    return {
        "object": object_name,
        "arm": selected_arm,
        "end_effector": end_effector,
        "planning_group": planning_group,
        "score": candidate.score,
        "required_opening_m": candidate.required_opening,
        # "criteria": {
        #     "local_normal_alignment": candidate.local_score,
        #     "approach": candidate.approach_score,
        #     "center": candidate.center_score,
        # },
        "commands": [
            {"action": "open_gripper", "end_effector": end_effector},
            {
                "action": "move_cartesian",
                "planning_group": planning_group,
                "pose": pose_record(candidate.pregrasp_transform),
            },
            {
                "action": "move_cartesian",
                "planning_group": planning_group,
                "pose": pose_record(candidate.transform),
            },
            {"action": "close_gripper", "end_effector": end_effector},
        ],
    }


def candidate_record(candidate: GraspCandidate) -> dict[str, Any]:
    """Return a complete JSON-compatible representation of a candidate."""
    record = asdict(candidate)
    record["transform"] = candidate.transform.tolist()
    record["pregrasp_transform"] = candidate.pregrasp_transform.tolist()
    record["contact_a"] = (
        None if candidate.contact_a is None else candidate.contact_a.tolist()
    )
    record["contact_b"] = (
        None if candidate.contact_b is None else candidate.contact_b.tolist()
    )
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("point_cloud", type=Path)
    parser.add_argument("--scene-cloud", type=Path)
    parser.add_argument("--object-name", default="object")
    parser.add_argument("--output", type=Path, default=Path("grasp_command.json"))
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-opening", type=float, default=0.080)
    parser.add_argument("--top", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    geometry = GripperGeometry(max_opening=args.max_opening)
    criteria = SamplingCriteria(
        samples=args.samples,
        random_seed=args.seed,
        maximum_candidates=max(args.top, 1),
    )
    object_cloud = read_point_cloud(args.point_cloud)
    scene_cloud = read_point_cloud(args.scene_cloud) if args.scene_cloud else None
    candidates = sample_grasps(
        object_cloud, scene_cloud=scene_cloud, geometry=geometry, criteria=criteria
    )
    if not candidates:
        raise RuntimeError("no grasp satisfied the configured criteria")

    output = {
        "best_command": build_gripper_commands(
            candidates[0], object_name=args.object_name
        ),
        "candidates": [candidate_record(candidate) for candidate in candidates[: args.top]],
        "geometry": asdict(geometry),
        "sampling": asdict(criteria),
    }
    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Generated {len(candidates)} valid grasp candidate(s)")
    print(f"Saved best gripper command: {output_path}")


if __name__ == "__main__":
    main()
