"""Pure geometry and optimization helpers for the single-arm experiment."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import numpy.typing as npt

from aux_math import look_at_camera_pose
from scan_trajectory import CameraIntrinsics, projected_overlap


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
    count: int,
    elevation_bounds_deg: Sequence[float],
    azimuth_offset_deg: float = 0.0,
) -> list[CameraViewpoint]:
    """Generate deterministic equal-area golden-angle poses on an upper hemisphere."""
    center_array = np.asarray(center, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(center_array)):
        raise ValueError("hemisphere center must be finite")
    if not math.isfinite(radius) or radius <= 0.0 or count <= 0:
        raise ValueError("radius and viewpoint count must be positive")
    if len(elevation_bounds_deg) != 2:
        raise ValueError("elevation bounds require MIN and MAX")
    elevation_min, elevation_max = map(float, elevation_bounds_deg)
    if not 0.0 <= elevation_min < elevation_max <= 90.0:
        raise ValueError("elevation bounds must satisfy 0 <= MIN < MAX <= 90")

    # Uniform spacing in sin(elevation) produces equal-area surface samples.
    sine_min = math.sin(math.radians(elevation_min))
    sine_max = math.sin(math.radians(elevation_max))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    viewpoints = []
    for index in range(count):
        fraction = (index + 0.5) / count
        elevation = math.asin(sine_min + fraction * (sine_max - sine_min))
        azimuth = math.radians(azimuth_offset_deg) + index * golden_angle
        direction = np.array(
            [
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ]
        )
        position = center_array + radius * direction
        viewpoints.append(
            CameraViewpoint(
                source_index=index,
                azimuth_deg=math.degrees(azimuth) % 360.0,
                elevation_deg=math.degrees(elevation),
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
