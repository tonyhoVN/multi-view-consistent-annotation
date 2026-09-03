from __future__ import annotations

import math
from typing import Iterable, Optional, Union

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R


def matrix_from_pose(trans, quat):
    """Convert position and quaternion to 4x4 transformation matrix."""
    trans = np.asarray(trans, dtype=np.float64).reshape(3)
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    rot = R.from_quat(quat)
    mat = np.eye(4)
    mat[:3, :3] = rot.as_matrix()
    mat[:3, 3] = trans
    return mat

def matrix_from_translate(trans):
    trans = np.asarray(trans, dtype=np.float64).reshape(3)
    T = np.eye(4)
    T[:3, 3] = trans
    return T

def matrix_to_pose(mat):
    """Convert 4x4 transformation matrix to (position, quaternion)."""
    pos = mat[:3, 3]
    rot = R.from_matrix(mat[:3, :3])
    quat = rot.as_quat()
    return pos, quat


def quaternion_from_matrix(
    rotation: npt.ArrayLike,
) -> tuple[float, float, float, float]:
    """Convert a 3-by-3 rotation matrix to a normalized XYZW quaternion."""
    quaternion = R.from_matrix(np.asarray(rotation, dtype=np.float64)).as_quat()
    return tuple(float(value) for value in quaternion)


def look_at_camera_pose(
    position: npt.ArrayLike, center: npt.ArrayLike
) -> np.ndarray:
    """Create an optical-frame pose whose +Z axis points at ``center``."""
    position = np.asarray(position, dtype=np.float64).reshape(3)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    optical_z = center - position
    distance = np.linalg.norm(optical_z)
    if distance < 1e-8:
        raise ValueError("camera position must differ from the look-at center")
    optical_z /= distance

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    optical_x = np.cross(optical_z, world_up)
    if np.linalg.norm(optical_x) < 1e-8:
        world_up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        optical_x = np.cross(optical_z, world_up)
    optical_x /= np.linalg.norm(optical_x)
    optical_y = np.cross(optical_z, optical_x)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack((optical_x, optical_y, optical_z))
    transform[:3, 3] = position
    return transform


def hemisphere_pairs(
    center: npt.ArrayLike,
    radius: float,
    azimuths_deg: Iterable[float],
    elevations_deg: Iterable[float],
) -> list[tuple[float, float, np.ndarray, np.ndarray]]:
    """Generate mirrored left/right camera poses on an upper-front hemisphere."""
    center = np.asarray(center, dtype=np.float64).reshape(3)
    pairs = []
    for elevation_deg in elevations_deg:
        elevation = math.radians(elevation_deg)
        horizontal_radius = radius * math.cos(elevation)
        height = radius * math.sin(elevation)
        for azimuth_deg in azimuths_deg:
            azimuth = math.radians(azimuth_deg)
            x = center[0] + horizontal_radius * math.cos(azimuth)
            y_offset = horizontal_radius * math.sin(azimuth)
            z = center[2] + height

            left_position = np.array([x, center[1] + y_offset, z])
            right_position = np.array([x, center[1] - y_offset, z])
            if left_position[1] <= 0.0 or right_position[1] >= 0.0:
                raise ValueError(
                    "scan center and angles must place left poses at world y > 0 "
                    "and right poses at world y < 0"
                )
            pairs.append(
                (
                    azimuth_deg,
                    elevation_deg,
                    look_at_camera_pose(left_position, center),
                    look_at_camera_pose(right_position, center),
                )
            )
    return pairs

def rot_matrix_from_euler(roll, pitch, yaw, degrees=False):
    """Create rotation matrix from Euler angles."""
    rotation = R.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
    return rotation.as_matrix()

def transform_from_euler(
    seq: str,
    angles: Union[float, npt.ArrayLike],
    translation: Optional[npt.ArrayLike] = None,
    degrees: bool = False
) -> np.ndarray:
    """
    Create a 4x4 transformation matrix from Euler angles and optional translation.

    Parameters:
        seq: Axis sequence for Euler angles (e.g., 'xyz', 'zyx').
        angles: Euler angles (list, tuple, or np.ndarray).
        translation: Optional translation (list, tuple, or np.ndarray of shape (3,)).
        degrees: Whether angles are in degrees.

    Returns:
        4x4 transformation matrix (np.ndarray)
    """
    rotation = R.from_euler(seq, angles, degrees=degrees)
    T = np.eye(4)
    T[:3, :3] = rotation.as_matrix()
    if translation is not None:
        T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return T

def transform_around_pivot(
    seq: str,
    angles: Union[float, npt.ArrayLike],
    radius: float,
    degrees: bool = False
) -> np.ndarray:
    rotation = R.from_euler(seq, angles, degrees=degrees)
    R_pivot_target = rotation.as_matrix()
    p_pivot_target = R_pivot_target @ np.array([0, 0, radius])
    return matrix_from_pose(p_pivot_target, rotation.as_quat())

def rot_around_pivot(
    seq: str,
    angles: Union[float, npt.ArrayLike],
    radius: float,
    degrees: bool = False
) -> np.ndarray:
    rotation = R.from_euler(seq, angles, degrees=degrees)
    R_A_B = rotation.as_matrix()
    p_A0_A = p_B0_B = np.array([0, 0, radius])
    p_AB_A = p_A0_A - R_A_B @ p_B0_B
    return matrix_from_pose(p_AB_A, rotation.as_quat())
