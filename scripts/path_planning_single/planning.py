"""Pure geometry and optimization helpers for the single-arm experiment."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import numpy.typing as npt

from multi_view_scan.aux_math import look_at_camera_pose
from multi_view_scan.scan_trajectory import CameraIntrinsics, projected_overlap


@dataclass(frozen=True)
class CameraViewpoint:
    """One sampled camera target on the hemisphere."""

    source_index: int
    azimuth_deg: float
    elevation_deg: float
    camera_pose: np.ndarray


@dataclass(frozen=True)
class ReachableViewpoint:
    """A camera target with its IK-link pose and one MoveIt solution."""

    viewpoint: CameraViewpoint
    ik_link_pose: np.ndarray
    joint_values: np.ndarray


def sample_hemisphere(
    center: npt.ArrayLike,
    radius: float,
    latitude_layers: int,
    azimuth_samples: int,
    elevation_bounds_deg: Sequence[float],
    azimuth_offset_deg: float = 0.0,
) -> list[CameraViewpoint]:
    """Generate a deterministic latitude-azimuth grid on an upper hemisphere."""
    center_array = np.asarray(center, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(center_array)):
        raise ValueError("hemisphere center must be finite")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive")
    if latitude_layers < 1 or azimuth_samples < 3:
        raise ValueError(
            "latitude_layers must be positive and azimuth_samples at least 3"
        )
    if len(elevation_bounds_deg) != 2:
        raise ValueError("elevation bounds require MIN and MAX")
    elevation_min, elevation_max = map(float, elevation_bounds_deg)
    if not 0.0 <= elevation_min < elevation_max <= 90.0:
        raise ValueError("elevation bounds must satisfy 0 <= MIN < MAX <= 90")

    elevations = np.linspace(elevation_min, elevation_max, latitude_layers)
    azimuth_offset = math.radians(azimuth_offset_deg)
    viewpoints = []
    for layer_index, elevation_deg in enumerate(elevations):
        elevation = math.radians(float(elevation_deg))
        for azimuth_index in range(azimuth_samples):
            azimuth = azimuth_offset + 2.0 * math.pi * azimuth_index / azimuth_samples
            direction = np.array(
                [
                    math.cos(elevation) * math.cos(azimuth),
                    math.cos(elevation) * math.sin(azimuth),
                    math.sin(elevation),
                ]
            )
            position = center_array + radius * direction
            source_index = layer_index * azimuth_samples + azimuth_index
            viewpoints.append(
                CameraViewpoint(
                    source_index=source_index,
                    azimuth_deg=math.degrees(azimuth) % 360.0,
                    elevation_deg=float(elevation_deg),
                    camera_pose=look_at_camera_pose(position, center_array),
                )
            )
    return viewpoints


def hemisphere_triangles(
    center: npt.ArrayLike,
    radius: float,
    elevation_bounds_deg: Sequence[float],
    azimuth_segments: int,
    elevation_segments: int,
) -> np.ndarray:
    """Return triangle vertices for a translucent bounded hemisphere shell."""
    center_array = np.asarray(center, dtype=np.float64).reshape(3)
    if azimuth_segments < 3 or elevation_segments < 1:
        raise ValueError("hemisphere mesh resolution is too small")
    elevation_min, elevation_max = map(math.radians, elevation_bounds_deg)
    vertices = []

    # Tessellate each latitude-longitude cell into two consistently wound faces.
    for elevation_index in range(elevation_segments):
        low = elevation_min + (elevation_max - elevation_min) * (
            elevation_index / elevation_segments
        )
        high = elevation_min + (elevation_max - elevation_min) * (
            (elevation_index + 1) / elevation_segments
        )
        for azimuth_index in range(azimuth_segments):
            first = 2.0 * math.pi * azimuth_index / azimuth_segments
            second = 2.0 * math.pi * (azimuth_index + 1) / azimuth_segments

            def point(elevation: float, azimuth: float) -> np.ndarray:
                return center_array + radius * np.array(
                    [
                        math.cos(elevation) * math.cos(azimuth),
                        math.cos(elevation) * math.sin(azimuth),
                        math.sin(elevation),
                    ]
                )

            p00, p01 = point(low, first), point(low, second)
            p10, p11 = point(high, first), point(high, second)
            vertices.extend((p00, p01, p11, p00, p11, p10))
    return np.asarray(vertices, dtype=np.float64)


def pairwise_overlaps(
    viewpoints: Sequence[ReachableViewpoint],
    surface_points: npt.ArrayLike,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Compute the symmetric projected-overlap matrix for one camera."""
    count = len(viewpoints)
    overlaps = np.eye(count, dtype=np.float64)
    for first in range(count):
        for second in range(first + 1, count):
            value = projected_overlap(
                viewpoints[first].viewpoint.camera_pose,
                viewpoints[second].viewpoint.camera_pose,
                surface_points,
                intrinsics,
            )
            overlaps[first, second] = overlaps[second, first] = value
    return overlaps


def joint_distances(viewpoints: Sequence[ReachableViewpoint]) -> np.ndarray:
    """Return pairwise Euclidean distances between single-arm IK solutions."""
    if not viewpoints:
        return np.empty((0, 0), dtype=np.float64)
    values = np.vstack([viewpoint.joint_values for viewpoint in viewpoints])
    if not np.all(np.isfinite(values)):
        raise ValueError("IK joint values must be finite")
    return np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)


def baseline_orders(
    viewpoints: Sequence[ReachableViewpoint], random_seed: int
) -> tuple[list[int], list[int]]:
    """Return reproducible random and original spiral orders over reachable views."""
    spiral_order = sorted(
        range(len(viewpoints)),
        key=lambda index: viewpoints[index].viewpoint.source_index,
    )
    random_order = list(spiral_order)
    np.random.default_rng(random_seed).shuffle(random_order)
    return random_order, spiral_order


def overlap_violation_count(
    order: Sequence[int], overlaps: np.ndarray, minimum_overlap: float
) -> int:
    """Count consecutive camera pairs below the requested overlap threshold."""
    return sum(
        float(overlaps[first, second]) < minimum_overlap
        for first, second in zip(order, order[1:])
    )
