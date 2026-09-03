"""Shared Open3D loading, processing, and visualization helpers."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

DEPTH_SCALE = 1.0
DEPTH_TRUNC = 3.0


def point_array(point_cloud: Any, *, minimum_points: int = 20) -> np.ndarray:
    """Return the finite XYZ values from an Open3D cloud or array-like input."""
    if minimum_points < 1:
        raise ValueError("minimum_points must be at least one")
    source = point_cloud.points if isinstance(
        point_cloud, o3d.geometry.PointCloud
    ) else point_cloud
    points = np.asarray(source, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("point cloud must have shape (N, 3)")
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < minimum_points:
        raise ValueError(
            f"point cloud must contain at least {minimum_points} finite points"
        )
    return points


def point_cloud_from_array(points: Any) -> o3d.geometry.PointCloud:
    """Create an Open3D point cloud from finite array-like XYZ points."""
    xyz = point_array(points, minimum_points=1)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz)
    return cloud


def voxel_downsample(points: Any, voxel_size: float) -> np.ndarray:
    """Voxel-downsample XYZ data with Open3D and return its point array."""
    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("voxel_size must be finite and greater than zero")
    cloud = point_cloud_from_array(points)
    downsampled = cloud.voxel_down_sample(float(voxel_size))
    return np.asarray(downsampled.points, dtype=np.float64).copy()


def estimate_outward_normals(points: Any, neighbors: int) -> np.ndarray:
    """Estimate Open3D KNN normals and orient them away from the centroid."""
    xyz = point_array(points, minimum_points=3)
    if neighbors < 3:
        raise ValueError("neighbors must be at least three")
    cloud = point_cloud_from_array(xyz)
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(
            knn=min(int(neighbors), len(xyz))
        ),
        fast_normal_computation=False,
    )
    cloud.normalize_normals()
    normals = np.asarray(cloud.normals, dtype=np.float64).copy()
    outward = xyz - xyz.mean(axis=0)
    normals[np.einsum("ij,ij->i", normals, outward) < 0.0] *= -1.0
    return normals


def estimate_principal_curvature_directions(
    points: Any,
    normals: Any,
    neighbors: int,
) -> np.ndarray:
    """Estimate two orthogonal principal-curvature directions at every point.

    A quadratic surface is fitted in each point's tangent frame. Eigenvectors
    of the fitted 2-D Hessian provide the maximum- and minimum-curvature axes.
    The returned array has shape ``(N, 2, 3)`` in world coordinates.
    """
    xyz = point_array(points, minimum_points=6)
    surface_normals = np.asarray(normals, dtype=np.float64)
    if surface_normals.shape != xyz.shape or not np.all(np.isfinite(surface_normals)):
        raise ValueError("normals must be finite and have the same shape as points")
    if neighbors < 6:
        raise ValueError("principal-curvature estimation requires at least 6 neighbors")

    # Query all neighborhoods once to avoid rebuilding a tree per sampled point.
    count = min(int(neighbors), len(xyz))
    tree = cKDTree(xyz)
    _, neighbor_indices = tree.query(xyz, k=count)
    directions = np.empty((len(xyz), 2, 3), dtype=np.float64)

    for index, indices in enumerate(neighbor_indices):
        normal = surface_normals[index].copy()
        normal /= np.linalg.norm(normal)
        reference = np.eye(3)[int(np.argmin(np.abs(normal)))]
        tangent_u = np.cross(normal, reference)
        tangent_u /= np.linalg.norm(tangent_u)
        tangent_v = np.cross(normal, tangent_u)

        # Fit height over the tangent plane with a local quadratic polynomial.
        offsets = xyz[np.atleast_1d(indices)] - xyz[index]
        coordinate_u = offsets @ tangent_u
        coordinate_v = offsets @ tangent_v
        height = offsets @ normal
        design = np.column_stack(
            (
                0.5 * coordinate_u**2,
                coordinate_u * coordinate_v,
                0.5 * coordinate_v**2,
                coordinate_u,
                coordinate_v,
                np.ones_like(coordinate_u),
            )
        )
        coefficients, _, _, _ = np.linalg.lstsq(design, height, rcond=None)
        hessian = np.array(
            [[coefficients[0], coefficients[1]],
             [coefficients[1], coefficients[2]]],
            dtype=np.float64,
        )
        curvatures, tangent_directions = np.linalg.eigh(hessian)
        order = np.argsort(np.abs(curvatures))[::-1]

        # Convert both principal axes from tangent coordinates into world axes.
        for output_index, curvature_index in enumerate(order):
            direction_2d = tangent_directions[:, curvature_index]
            direction = (
                direction_2d[0] * tangent_u + direction_2d[1] * tangent_v
            )
            directions[index, output_index] = direction / np.linalg.norm(direction)
    return directions


def read_point_cloud(path: Path | str) -> o3d.geometry.PointCloud:
    """Read a non-empty PCD/PLY point cloud with Open3D."""
    cloud_path = Path(path).expanduser()
    cloud = o3d.io.read_point_cloud(str(cloud_path))
    if not cloud.has_points():
        raise ValueError(f"point cloud is empty or unreadable: {cloud_path}")
    return cloud


def save_point_cloud_json(
    path: Path | str,
    point_cloud: o3d.geometry.PointCloud,
    *,
    frame_id: str = "world",
) -> Path:
    """Save Open3D points and optional attributes in a portable JSON file."""
    if not point_cloud.has_points():
        raise ValueError("cannot save an empty point cloud")

    # Convert Open3D buffers into JSON-compatible arrays.
    points = np.asarray(point_cloud.points, dtype=np.float64)
    record: dict[str, Any] = {
        "format": "open3d-point-cloud-json-v1",
        "frame_id": str(frame_id),
        "point_count": int(len(points)),
        "points": points.tolist(),
    }
    if point_cloud.has_colors():
        record["colors"] = np.asarray(
            point_cloud.colors, dtype=np.float64
        ).tolist()
    if point_cloud.has_normals():
        record["normals"] = np.asarray(
            point_cloud.normals, dtype=np.float64
        ).tolist()

    # Create the destination only after the cloud has passed validation.
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return output_path


def load_point_cloud_json(path: Path | str) -> o3d.geometry.PointCloud:
    """Load a point cloud written by :func:`save_point_cloud_json`."""
    input_path = Path(path).expanduser()
    try:
        record = json.loads(input_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read point cloud JSON {input_path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid point cloud JSON {input_path}: {error}") from error
    if (
        not isinstance(record, dict)
        or record.get("format") != "open3d-point-cloud-json-v1"
    ):
        raise ValueError(f"unsupported point cloud JSON format: {input_path}")

    # Validate the main XYZ array before attaching optional Open3D buffers.
    points = point_array(record.get("points"), minimum_points=1)
    if int(record.get("point_count", -1)) != len(points):
        raise ValueError(f"point_count does not match points in {input_path}")
    cloud = point_cloud_from_array(points)
    for field, target in (("colors", "colors"), ("normals", "normals")):
        if field not in record:
            continue
        values = np.asarray(record[field], dtype=np.float64)
        if values.shape != points.shape or not np.all(np.isfinite(values)):
            raise ValueError(f"{field} must have shape {points.shape} in {input_path}")
        setattr(cloud, target, o3d.utility.Vector3dVector(values))
    return cloud


def rgbd_to_pcd(color_path, depth_path, TF_base_cam, intrinsic):
    # Load color and depth images
    color_image = cv2.imread(color_path, cv2.IMREAD_COLOR)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # Convert depth to float32 meters if needed
    if depth_image.dtype == np.uint16:
        depth_image = depth_image.astype(np.float32) / 1000.0

    # Convert BGR to RGB for Open3D
    color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # Create Open3D images
    o3d_color = o3d.geometry.Image(color_image_rgb)
    o3d_depth = o3d.geometry.Image(depth_image)

    # Create RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, o3d_depth,
        convert_rgb_to_intensity=False,
        depth_scale=DEPTH_SCALE,  # already in meters
        depth_trunc=DEPTH_TRUNC   # max depth in meters
    )

    # Creat pcd
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic
    )

    # Voxel grid filter
    voxel_size = 0.005
    pcd = pcd.voxel_down_sample(voxel_size)

    # Transform 
    # TF_base_cam[:3,3] = [0,0,0]
    # pcd.transform(np.linalg.inv(TF_base_cam))
    pcd.transform(TF_base_cam)

    # Generate point cloud
    return pcd

def rgbd_to_pcd_mask(color_path, depth_path, mask, TF_base_cam, intrinsic):
    # Load color and depth images
    color_image = cv2.imread(color_path, cv2.IMREAD_COLOR)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # Convert depth to float32 meters if needed
    if depth_image.dtype == np.uint16:
        depth_image = depth_image.astype(np.float32) / 1000.0

    # Convert BGR to RGB for Open3D
    color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # bitwise mask 
    color_image_rgb = cv2.bitwise_and(color_image_rgb, color_image_rgb, mask=mask)
    depth_image = cv2.bitwise_and(depth_image, depth_image, mask=mask)

    # Create Open3D images
    o3d_color = o3d.geometry.Image(color_image_rgb)
    o3d_depth = o3d.geometry.Image(depth_image)

    # Create RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, o3d_depth,
        convert_rgb_to_intensity=False,
        depth_scale=DEPTH_SCALE,  # already in meters
        depth_trunc=DEPTH_TRUNC   # max depth in meters
    )

    # Creat pcd
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic
    )

    # Voxel grid filter

    voxel_size = 0.005
    pcd = pcd.voxel_down_sample(voxel_size)
    # print(pcd.get_center())

    # Transform 
    # TF_base_cam[:3,3] = [0,0,0]
    # pcd.transform(np.linalg.inv(TF_base_cam))
    pcd.transform(TF_base_cam)

    # Generate point cloud
    return pcd

def remove_outlier_o3d(pcd, nb_neighbors=20, std_ratio=2.0):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
    return cl, ind


def _box_at_gripper_pose(
    transform: np.ndarray,
    size: Sequence[float],
    local_min_corner: Sequence[float],
    color: Sequence[float],
) -> o3d.geometry.TriangleMesh:
    """Create an oriented box positioned in a gripper coordinate frame."""
    box = o3d.geometry.TriangleMesh.create_box(
        width=float(size[0]), height=float(size[1]), depth=float(size[2])
    )
    box.paint_uniform_color(color)
    box.compute_vertex_normals()
    local_transform = np.eye(4, dtype=np.float64)
    local_transform[:3, 3] = np.asarray(local_min_corner, dtype=np.float64)
    box.transform(transform @ local_transform)
    return box


def create_grasp_visualization(
    scene_cloud: o3d.geometry.PointCloud,
    candidate: Any,
    *,
    object_cloud: o3d.geometry.PointCloud | None = None,
    geometry: Any = None,
) -> list[o3d.geometry.Geometry3D]:
    """Create Open3D geometries showing a sampled grasp in a scene cloud."""
    hand_depth = float(getattr(geometry, "hand_depth", 0.060))
    finger_reach = float(getattr(geometry, "finger_reach", 0.060))
    outer_width = float(getattr(geometry, "outer_width", 0.105))
    grasp_depth = float(getattr(geometry, "grasp_depth", 0.020))
    finger_base_z = grasp_depth - finger_reach
    scene_display = copy.deepcopy(scene_cloud)
    scene_display.paint_uniform_color([0.55, 0.58, 0.62])
    visualizations: list[o3d.geometry.Geometry3D] = []
    if object_cloud is not None:
        object_display = copy.deepcopy(object_cloud)
        # object_display.paint_uniform_color([1.0, 0.55, 0.05])
        visualizations.append(object_display)

    grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.075)
    grasp_frame.transform(candidate.transform)
    pregrasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.045)
    pregrasp_frame.transform(candidate.pregrasp_transform)
    visualizations.extend((grasp_frame, pregrasp_frame))

    finger_thickness = 0.008
    half_opening = candidate.required_opening / 2.0
    for direction in (-1.0, 1.0):
        finger_x = direction * (half_opening + finger_thickness / 2.0)
        visualizations.append(
            _box_at_gripper_pose(
                candidate.transform,
                (finger_thickness, hand_depth, finger_reach),
                (
                    finger_x - finger_thickness / 2.0,
                    -hand_depth / 2.0,
                    finger_base_z,
                ),
                (0.1, 0.35, 0.95),
            )
        )

    palm_thickness = 0.012
    visualizations.append(
        _box_at_gripper_pose(
            candidate.transform,
            (outer_width, hand_depth, palm_thickness),
            (
                -outer_width / 2.0,
                -hand_depth / 2.0,
                finger_base_z - palm_thickness,
            ),
            (0.08, 0.2, 0.65),
        )
    )

    # Contact markers remain available for candidates from the temporary sampler.
    contacts = [
        contact
        for contact in (candidate.contact_a, candidate.contact_b)
        if contact is not None
    ]
    for contact in contacts:
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
        marker.translate(contact)
        marker.paint_uniform_color([0.95, 0.1, 0.1])
        marker.compute_vertex_normals()
        visualizations.append(marker)

    # Always draw the green pregrasp path; add the red contact line when present.
    line_points = [
        candidate.pregrasp_transform[:3, 3],
        candidate.transform[:3, 3],
    ]
    line_indices = [[0, 1]]
    line_colors = [[0.1, 0.9, 0.2]]
    if len(contacts) == 2:
        line_points.extend(contacts)
        line_indices.append([2, 3])
        line_colors.append([1.0, 0.1, 0.1])
    lines = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(line_points)),
        lines=o3d.utility.Vector2iVector(line_indices),
    )
    lines.colors = o3d.utility.Vector3dVector(line_colors)
    visualizations.append(lines)
    return visualizations


def visualize_grasp(
    scene_cloud: o3d.geometry.PointCloud,
    candidate: Any,
    *,
    object_cloud: o3d.geometry.PointCloud | None = None,
    geometry: Any = None,
    window_name: str = "Sampled grasp in reconstructed scene",
) -> None:
    """Open an interactive Open3D window for one scene-level grasp."""
    geometries = create_grasp_visualization(
        scene_cloud,
        candidate,
        object_cloud=object_cloud,
        geometry=geometry,
    )
    o3d.visualization.draw_geometries(geometries, window_name=window_name)
