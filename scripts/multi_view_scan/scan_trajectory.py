"""Pure geometry and path optimization for coordinated multi-view scanning."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import numpy.typing as npt

from multi_view_scan.aux_math import look_at_camera_pose


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera calibration used for frustum-overlap projection."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        values = (self.fx, self.fy, self.cx, self.cy)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera width and height must be positive")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("camera intrinsics must be finite")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("camera focal lengths must be positive")


@dataclass(frozen=True)
class SpiralViewpoint:
    """One mirrored dual-camera target from the original spiral sequence."""

    source_index: int
    azimuth_deg: float
    elevation_deg: float
    left_camera_pose: np.ndarray
    right_camera_pose: np.ndarray


@dataclass(frozen=True)
class ReachableViewpoint:
    """A spiral target augmented with TCP poses and one MoveIt IK solution."""

    viewpoint: SpiralViewpoint
    left_tcp_pose: np.ndarray
    right_tcp_pose: np.ndarray
    joint_values: np.ndarray


@dataclass(frozen=True)
class PathMetrics:
    """Auditable quality metrics for an open scan path."""

    objective: float
    motion_distance: float
    mean_neighbor_overlap: float
    minimum_neighbor_overlap: float


def spiral_hemisphere_pairs(
    center: npt.ArrayLike,
    radius: float,
    count: int,
    azimuth_bounds_deg: Sequence[float],
    elevation_bounds_deg: Sequence[float],
) -> list[SpiralViewpoint]:
    """Generate deterministic equal-area golden-angle targets and mirror world Y."""
    center_array = np.asarray(center, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(center_array)):
        raise ValueError("scan center must be finite")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    if count <= 0:
        raise ValueError("viewpoint count must be positive")
    if len(azimuth_bounds_deg) != 2 or len(elevation_bounds_deg) != 2:
        raise ValueError("azimuth and elevation bounds require MIN and MAX")

    azimuth_min, azimuth_max = map(float, azimuth_bounds_deg)
    elevation_min, elevation_max = map(float, elevation_bounds_deg)
    if not 0.0 < azimuth_min < azimuth_max < 180.0:
        raise ValueError("azimuth bounds must satisfy 0 < MIN < MAX < 180")
    if not 0.0 <= elevation_min < elevation_max < 90.0:
        raise ValueError("elevation bounds must satisfy 0 <= MIN < MAX < 90")

    # Uniform spacing in sin(elevation) gives approximately equal surface area.
    sine_min = math.sin(math.radians(elevation_min))
    sine_max = math.sin(math.radians(elevation_max))
    golden_fraction = (math.sqrt(5.0) - 1.0) / 2.0
    viewpoints: list[SpiralViewpoint] = []
    for source_index in range(count):
        elevation_fraction = (source_index + 0.5) / count
        sine_elevation = sine_min + elevation_fraction * (sine_max - sine_min)
        elevation_deg = math.degrees(math.asin(sine_elevation))
        azimuth_fraction = (0.5 + source_index * golden_fraction) % 1.0
        azimuth_deg = azimuth_min + azimuth_fraction * (azimuth_max - azimuth_min)

        elevation = math.radians(elevation_deg)
        azimuth = math.radians(azimuth_deg)
        horizontal_radius = radius * math.cos(elevation)
        x = center_array[0] + horizontal_radius * math.cos(azimuth)
        y_offset = horizontal_radius * math.sin(azimuth)
        z = center_array[2] + radius * math.sin(elevation)
        left_position = np.array([x, center_array[1] + y_offset, z])
        right_position = np.array([x, center_array[1] - y_offset, z])
        if left_position[1] <= 0.0 or right_position[1] >= 0.0:
            raise ValueError(
                "spiral bounds must place left targets at world y > 0 and "
                "right targets at world y < 0"
            )
        viewpoints.append(
            SpiralViewpoint(
                source_index=source_index,
                azimuth_deg=azimuth_deg,
                elevation_deg=elevation_deg,
                left_camera_pose=look_at_camera_pose(left_position, center_array),
                right_camera_pose=look_at_camera_pose(right_position, center_array),
            )
        )
    return viewpoints


def sample_scan_volume(
    center: npt.ArrayLike, radius: float, sample_count: int
) -> np.ndarray:
    """Return deterministic near-uniform samples on a spherical scan boundary."""
    center_array = np.asarray(center, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(center_array)):
        raise ValueError("scan center must be finite")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("scan-volume radius must be finite and positive")
    if sample_count <= 0:
        raise ValueError("projection sample count must be positive")

    indices = np.arange(sample_count, dtype=np.float64)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    directions_z = 1.0 - 2.0 * (indices + 0.5) / sample_count
    radial_xy = np.sqrt(np.maximum(0.0, 1.0 - directions_z * directions_z))
    angles = indices * golden_angle
    directions = np.column_stack(
        (radial_xy * np.cos(angles), radial_xy * np.sin(angles), directions_z)
    )
    return center_array + directions * radius


def baseline_path_orders(
    viewpoints: Sequence[ReachableViewpoint], random_seed: int
) -> tuple[list[int], list[int]]:
    """Return random and source-spiral orders over the same reachable poses."""
    spiral_order = sorted(
        range(len(viewpoints)),
        key=lambda index: viewpoints[index].viewpoint.source_index,
    )
    random_order = list(spiral_order)
    np.random.default_rng(random_seed).shuffle(random_order)
    return random_order, spiral_order


def overlap_violation_count(
    order: Sequence[int], overlaps: npt.ArrayLike, minimum_overlap: float
) -> int:
    """Count neighboring route edges below a projected-overlap threshold."""
    overlap_matrix = np.asarray(overlaps, dtype=np.float64)
    return sum(
        float(overlap_matrix[first, second]) < minimum_overlap
        for first, second in zip(order, order[1:])
    )


def projection_visibility(
    camera_pose: npt.ArrayLike,
    world_points: npt.ArrayLike,
    intrinsics: CameraIntrinsics,
    near_depth: float = 1e-6,
) -> np.ndarray:
    """Mark world points whose pinhole projections lie inside one camera image."""
    pose = np.asarray(camera_pose, dtype=np.float64)
    points = np.asarray(world_points, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError("camera pose must be a 4-by-4 matrix")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("world points must have shape (N, 3)")
    if not math.isfinite(near_depth) or near_depth < 0.0:
        raise ValueError("near depth must be finite and nonnegative")

    # Pose columns are camera axes expressed in world coordinates.
    camera_points = (points - pose[:3, 3]) @ pose[:3, :3]
    depth = camera_points[:, 2]
    positive_depth = depth > near_depth
    safe_depth = np.where(positive_depth, depth, 1.0)
    image_x = intrinsics.fx * camera_points[:, 0] / safe_depth + intrinsics.cx
    image_y = intrinsics.fy * camera_points[:, 1] / safe_depth + intrinsics.cy
    return (
        positive_depth
        & (image_x >= 0.0)
        & (image_x < intrinsics.width)
        & (image_y >= 0.0)
        & (image_y < intrinsics.height)
    )


def projected_overlap(
    first_pose: npt.ArrayLike,
    second_pose: npt.ArrayLike,
    world_points: npt.ArrayLike,
    intrinsics: CameraIntrinsics,
) -> float:
    """Return Jaccard overlap of projected, front-facing surface samples."""
    first = np.asarray(first_pose, dtype=np.float64)
    second = np.asarray(second_pose, dtype=np.float64)
    points = np.asarray(world_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("world points must have shape (N, 3)")
    if len(points) == 0:
        return 0.0

    # The deterministic sphere is an object-surface proxy for occlusion overlap.
    center = np.mean(points, axis=0)
    normals = points - center
    normal_lengths = np.linalg.norm(normals, axis=1)
    valid_normals = normal_lengths > 1e-12
    normals[valid_normals] /= normal_lengths[valid_normals, None]
    first_front_facing = (
        np.einsum("ij,ij->i", normals, first[:3, 3] - points) > 0.0
    )
    second_front_facing = (
        np.einsum("ij,ij->i", normals, second[:3, 3] - points) > 0.0
    )
    first_visible = (
        projection_visibility(first, points, intrinsics)
        & valid_normals
        & first_front_facing
    )
    second_visible = (
        projection_visibility(second, points, intrinsics)
        & valid_normals
        & second_front_facing
    )
    union = np.count_nonzero(first_visible | second_visible)
    if union == 0:
        return 0.0
    intersection = np.count_nonzero(first_visible & second_visible)
    return float(intersection / union)


def pairwise_overlap_matrix(
    viewpoints: Sequence[ReachableViewpoint],
    world_points: npt.ArrayLike,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Build symmetric pair overlap using the weaker of the two camera arms."""
    count = len(viewpoints)
    overlap = np.eye(count, dtype=np.float64)
    for first in range(count):
        for second in range(first + 1, count):
            left_overlap = projected_overlap(
                viewpoints[first].viewpoint.left_camera_pose,
                viewpoints[second].viewpoint.left_camera_pose,
                world_points,
                intrinsics,
            )
            right_overlap = projected_overlap(
                viewpoints[first].viewpoint.right_camera_pose,
                viewpoints[second].viewpoint.right_camera_pose,
                world_points,
                intrinsics,
            )
            overlap[first, second] = overlap[second, first] = min(
                left_overlap, right_overlap
            )
    return overlap


def joint_distance_matrix(viewpoints: Sequence[ReachableViewpoint]) -> np.ndarray:
    """Build symmetric Euclidean distances between coordinated IK solutions."""
    if not viewpoints:
        return np.empty((0, 0), dtype=np.float64)
    joint_count = len(viewpoints[0].joint_values)
    values = np.vstack(
        [np.asarray(viewpoint.joint_values, dtype=np.float64) for viewpoint in viewpoints]
    )
    if values.shape[1] != joint_count or not np.all(np.isfinite(values)):
        raise ValueError("IK joint vectors must have one consistent finite shape")
    differences = values[:, None, :] - values[None, :, :]
    return np.linalg.norm(differences, axis=2)


def rotation_geodesic_distance(
    first_rotation: npt.ArrayLike, second_rotation: npt.ArrayLike
) -> float:
    """Return the shortest SO(3) rotation angle between two orientations."""
    first = np.asarray(first_rotation, dtype=np.float64)
    second = np.asarray(second_rotation, dtype=np.float64)
    if first.shape != (3, 3) or second.shape != (3, 3):
        raise ValueError("rotations must be 3-by-3 matrices")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("rotations must be finite")
    cosine = (np.trace(first.T @ second) - 1.0) / 2.0
    return float(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def dual_pose_distance(
    first_left: npt.ArrayLike,
    first_right: npt.ArrayLike,
    second_left: npt.ArrayLike,
    second_right: npt.ArrayLike,
    translation_weight: float,
    rotation_weight: float,
) -> float:
    """Measure coordinated TCP translation and orientation displacement."""
    transforms = [
        np.asarray(transform, dtype=np.float64)
        for transform in (first_left, first_right, second_left, second_right)
    ]
    if any(transform.shape != (4, 4) for transform in transforms):
        raise ValueError("TCP poses must be 4-by-4 matrices")
    if not all(np.all(np.isfinite(transform)) for transform in transforms):
        raise ValueError("TCP poses must be finite")
    if (
        not math.isfinite(translation_weight)
        or not math.isfinite(rotation_weight)
        or translation_weight < 0.0
        or rotation_weight < 0.0
        or translation_weight + rotation_weight <= 0.0
    ):
        raise ValueError("pose-distance weights must be finite, nonnegative, and nonzero")

    first_left, first_right, second_left, second_right = transforms
    left_translation = np.linalg.norm(first_left[:3, 3] - second_left[:3, 3])
    right_translation = np.linalg.norm(first_right[:3, 3] - second_right[:3, 3])
    translation_distance = math.hypot(left_translation, right_translation)
    left_rotation = rotation_geodesic_distance(
        first_left[:3, :3], second_left[:3, :3]
    )
    right_rotation = rotation_geodesic_distance(
        first_right[:3, :3], second_right[:3, :3]
    )
    rotation_distance = math.hypot(left_rotation, right_rotation)
    return float(
        translation_weight * translation_distance
        + rotation_weight * rotation_distance
    )


def pose_distance_matrix(
    viewpoints: Sequence[ReachableViewpoint],
    translation_weight: float,
    rotation_weight: float,
) -> np.ndarray:
    """Build pairwise coordinated TCP-pose distances for reachable views."""
    count = len(viewpoints)
    distances = np.zeros((count, count), dtype=np.float64)
    for first in range(count):
        for second in range(first + 1, count):
            distance = dual_pose_distance(
                viewpoints[first].left_tcp_pose,
                viewpoints[first].right_tcp_pose,
                viewpoints[second].left_tcp_pose,
                viewpoints[second].right_tcp_pose,
                translation_weight,
                rotation_weight,
            )
            distances[first, second] = distances[second, first] = distance
    return distances


def combined_edge_costs(
    motion_distances: npt.ArrayLike,
    overlaps: npt.ArrayLike,
    overlap_weight: float,
) -> np.ndarray:
    """Combine the selected motion metric with a low-overlap penalty."""
    distances = np.asarray(motion_distances, dtype=np.float64)
    overlap = np.asarray(overlaps, dtype=np.float64)
    if distances.shape != overlap.shape or distances.ndim != 2:
        raise ValueError("distance and overlap matrices must have the same 2D shape")
    if not math.isfinite(overlap_weight) or overlap_weight < 0.0:
        raise ValueError("overlap weight must be finite and nonnegative")
    if np.any((overlap < 0.0) | (overlap > 1.0)):
        raise ValueError("overlap values must lie in [0, 1]")
    costs = distances + overlap_weight * (1.0 - overlap)
    np.fill_diagonal(costs, 0.0)
    return costs


def path_has_minimum_overlap(
    path: Sequence[int], overlaps: npt.ArrayLike, minimum_overlap: float
) -> bool:
    """Return whether every temporal edge satisfies the overlap constraint."""
    matrix = np.asarray(overlaps, dtype=np.float64)
    return all(
        matrix[first, second] + 1e-12 >= minimum_overlap
        for first, second in zip(path, path[1:])
    )


def open_path_cost(
    path: Sequence[int], edge_costs: npt.ArrayLike, start_costs: npt.ArrayLike
) -> float:
    """Return start-to-first cost plus every edge in an open path."""
    if not path:
        return 0.0
    edges = np.asarray(edge_costs, dtype=np.float64)
    starts = np.asarray(start_costs, dtype=np.float64)
    return float(
        starts[path[0]]
        + sum(edges[first, second] for first, second in zip(path, path[1:]))
    )


def _backtracking_hamiltonian_path(
    costs: np.ndarray,
    overlap: np.ndarray,
    candidate_starts: Sequence[int],
    minimum_overlap: float,
    max_states: int = 1_000_000,
) -> list[int] | None:
    """Find a feasible path after greedy search fails, with bounded DFS."""
    count = len(costs)
    adjacency = [
        [
            neighbor
            for neighbor in range(count)
            if neighbor != vertex
            and overlap[vertex, neighbor] + 1e-12 >= minimum_overlap
        ]
        for vertex in range(count)
    ]
    explored_states = 0
    dead_states: set[tuple[int, frozenset[int]]] = set()

    def remaining_is_connected(current: int, unvisited: set[int]) -> bool:
        """Prune branches whose remaining induced graph is disconnected."""
        pending = [current]
        seen = {current}
        allowed = unvisited | {current}
        while pending:
            vertex = pending.pop()
            for neighbor in adjacency[vertex]:
                if neighbor in allowed and neighbor not in seen:
                    seen.add(neighbor)
                    pending.append(neighbor)
        return unvisited.issubset(seen)

    def search(path: list[int], unvisited: set[int]) -> list[int] | None:
        nonlocal explored_states
        explored_states += 1
        if explored_states > max_states:
            raise RuntimeError(
                "Hamiltonian fallback exceeded its search limit; lower the overlap "
                "threshold or increase sampling connectivity"
            )
        if not unvisited:
            return path.copy()
        current = path[-1]
        state = (current, frozenset(unvisited))
        if state in dead_states:
            return None
        if not remaining_is_connected(current, unvisited):
            dead_states.add(state)
            return None

        # Visit constrained vertices first, then prefer the lower motion cost.
        candidates = [vertex for vertex in adjacency[current] if vertex in unvisited]
        candidates.sort(
            key=lambda vertex: (
                sum(neighbor in unvisited for neighbor in adjacency[vertex]),
                costs[current, vertex],
                vertex,
            )
        )
        for vertex in candidates:
            path.append(vertex)
            unvisited.remove(vertex)
            result = search(path, unvisited)
            if result is not None:
                return result
            unvisited.add(vertex)
            path.pop()
        dead_states.add(state)
        return None

    for start in candidate_starts:
        result = search([start], set(range(count)) - {start})
        if result is not None:
            return result
    return None


def nearest_neighbor_open_path(
    edge_costs: npt.ArrayLike,
    overlaps: npt.ArrayLike,
    start_costs: npt.ArrayLike,
    minimum_overlap: float,
    start_mode: str = "all_accepted",
) -> list[int]:
    """Build a feasible greedy path from the initial-nearest or every start."""
    costs = np.asarray(edge_costs, dtype=np.float64)
    overlap = np.asarray(overlaps, dtype=np.float64)
    starts = np.asarray(start_costs, dtype=np.float64)
    count = len(starts)
    if costs.shape != (count, count) or overlap.shape != (count, count):
        raise ValueError("path matrices and start costs have inconsistent sizes")
    if not 0.0 <= minimum_overlap <= 1.0:
        raise ValueError("minimum overlap must lie in [0, 1]")
    if start_mode not in {"initial_pose", "all_accepted"}:
        raise ValueError("start_mode must be 'initial_pose' or 'all_accepted'")
    if count == 0:
        return []

    # Either anchor at the initial-nearest view or search every accepted start.
    ordered_starts = sorted(range(count), key=lambda index: (starts[index], index))
    candidate_starts = ordered_starts[:1] if start_mode == "initial_pose" else ordered_starts
    best_path: list[int] | None = None
    best_cost = math.inf
    for start in candidate_starts:
        path = [start]
        unvisited = set(range(count)) - {start}
        while unvisited:
            current = path[-1]
            feasible = [
                index
                for index in unvisited
                if overlap[current, index] + 1e-12 >= minimum_overlap
            ]
            if not feasible:
                break
            selected = min(feasible, key=lambda index: (costs[current, index], index))
            path.append(selected)
            unvisited.remove(selected)
        if unvisited:
            continue
        candidate_cost = open_path_cost(path, costs, starts)
        if candidate_cost < best_cost - 1e-12:
            best_path = path
            best_cost = candidate_cost

    if best_path is None:
        best_path = _backtracking_hamiltonian_path(
            costs,
            overlap,
            candidate_starts,
            minimum_overlap,
        )
    if best_path is None:
        raise ValueError("no overlap-feasible Hamiltonian path visits every viewpoint")
    return best_path


def two_opt_open_path(
    initial_path: Sequence[int],
    edge_costs: npt.ArrayLike,
    overlaps: npt.ArrayLike,
    start_costs: npt.ArrayLike,
    minimum_overlap: float,
    max_passes: int,
    lock_first: bool = False,
) -> list[int]:
    """Improve an open path with deterministic overlap-constrained 2-opt."""
    if max_passes < 0:
        raise ValueError("2-opt passes must be nonnegative")
    path = list(initial_path)
    if len(set(path)) != len(path):
        raise ValueError("initial path contains duplicate vertices")
    if not path_has_minimum_overlap(path, overlaps, minimum_overlap):
        raise ValueError("initial path violates the minimum overlap constraint")

    # Reversals may change the first vertex, so the start-state cost is included.
    for _ in range(max_passes):
        current_cost = open_path_cost(path, edge_costs, start_costs)
        best_path = path
        best_cost = current_cost
        first_reversal_index = 1 if lock_first else 0
        for first in range(first_reversal_index, len(path) - 1):
            for last in range(first + 1, len(path)):
                candidate = path[:first] + list(reversed(path[first : last + 1])) + path[last + 1 :]
                if not path_has_minimum_overlap(candidate, overlaps, minimum_overlap):
                    continue
                candidate_cost = open_path_cost(candidate, edge_costs, start_costs)
                if candidate_cost < best_cost - 1e-12:
                    best_path = candidate
                    best_cost = candidate_cost
        if best_path == path:
            break
        path = best_path
    return path


def measure_path(
    path: Sequence[int],
    motion_distances: npt.ArrayLike,
    overlaps: npt.ArrayLike,
    edge_costs: npt.ArrayLike,
    start_motion_distances: npt.ArrayLike,
) -> PathMetrics:
    """Calculate objective, motion distance, and temporal-overlap statistics."""
    if not path:
        return PathMetrics(0.0, 0.0, 1.0, 1.0)
    distances = np.asarray(motion_distances, dtype=np.float64)
    overlap = np.asarray(overlaps, dtype=np.float64)
    start_distances = np.asarray(start_motion_distances, dtype=np.float64)
    neighbor_overlaps = [
        float(overlap[first, second]) for first, second in zip(path, path[1:])
    ]
    motion_distance = float(
        start_distances[path[0]]
        + sum(distances[first, second] for first, second in zip(path, path[1:]))
    )
    return PathMetrics(
        objective=open_path_cost(path, edge_costs, start_distances),
        motion_distance=motion_distance,
        mean_neighbor_overlap=(
            float(np.mean(neighbor_overlaps)) if neighbor_overlaps else 1.0
        ),
        minimum_neighbor_overlap=(min(neighbor_overlaps) if neighbor_overlaps else 1.0),
    )
