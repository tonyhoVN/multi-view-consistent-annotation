#!/usr/bin/env python3
"""Reconstruct a world-frame RGB-D point cloud from ``scan_output``.

Only manifest entries whose status is ``captured`` are used. Per-view moving
optical-frame extrinsics come from ``manifest.json``. Pinhole intrinsics and
the static hand-to-camera mount calibration come from ``camera.yaml``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from multi_view_scan.aux_math import (  # noqa: E402
    matrix_from_pose,
    transform_from_euler,
)


DEFAULT_CAMERA_YAML = Path(
    "/home/hier-tony/Projects/dual_manipulation_isaac_sim/env/urdf/camera.yaml"
)


@dataclass(frozen=True)
class IntrinsicCalibration:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class CaptureView:
    step: int
    side: str
    color_path: Path
    depth_path: Path
    world_from_camera: np.ndarray


def load_yaml_calibration(
    path: Path,
) -> tuple[IntrinsicCalibration, dict[str, np.ndarray], dict[str, str]]:
    """Load shared intrinsics and left/right hand-to-camera mount transforms."""
    if not path.is_file():
        raise FileNotFoundError(f"camera calibration does not exist: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"camera calibration is not a mapping: {path}")

    try:
        intrinsics = data["intrinsics"]
        width, height = intrinsics["resolution"]
        intrinsic = IntrinsicCalibration(
            width=int(width),
            height=int(height),
            fx=float(intrinsics["fx"]),
            fy=float(intrinsics["fy"]),
            cx=float(intrinsics["cx"]),
            cy=float(intrinsics["cy"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid intrinsics in {path}: {error}") from error

    if intrinsic.width <= 0 or intrinsic.height <= 0:
        raise ValueError("camera resolution must be positive")
    if intrinsic.fx <= 0.0 or intrinsic.fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")

    mounts: dict[str, np.ndarray] = {}
    frame_ids: dict[str, str] = {}
    for side in ("left", "right"):
        camera_name = f"{side}_handeye"
        try:
            camera = data["cameras"][camera_name]
            extrinsics = camera["extrinsics"]
            translation = extrinsics["translation_xyz_m"]
            rpy_degrees = extrinsics["rotation_rpy_degrees"]
            frame_ids[side] = str(camera["frame_id"])
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"missing {camera_name} extrinsics in {path}: {error}"
            ) from error

        # camera.yaml expresses the optical camera frame in its mounting/TCP
        # parent using XYZ translation and fixed-axis XYZ roll/pitch/yaw.
        mounts[side] = transform_from_euler(
            "xyz", rpy_degrees, translation=translation, degrees=True
        )

    return intrinsic, mounts, frame_ids


def pose_record_to_matrix(record: dict[str, Any], description: str) -> np.ndarray:
    """Convert a manifest pose record into a homogeneous transform."""
    try:
        return matrix_from_pose(
            record["position_xyz"], record["quaternion_xyzw"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {description}: {error}") from error


def collect_capture_views(
    scan_dir: Path,
    manifest: dict[str, Any],
    pose_source: str,
) -> tuple[list[CaptureView], list[str]]:
    """Resolve captured image pairs and their world optical-camera poses."""
    views: list[CaptureView] = []
    warnings: list[str] = []

    for capture in manifest.get("captures", []):
        if capture.get("status") != "captured":
            continue
        step = int(capture["step"])
        for side in ("left", "right"):
            side_capture = capture.get(side, {})
            pose_key = f"{pose_source}_camera_pose"
            pose_record = side_capture.get(pose_key)
            if pose_record is None and pose_source == "actual":
                pose_key = "desired_camera_pose"
                pose_record = side_capture.get(pose_key)
                warnings.append(
                    f"step {step} {side}: actual pose missing; using desired pose"
                )
            if pose_record is None:
                warnings.append(f"step {step} {side}: missing {pose_key}; skipped")
                continue

            try:
                color_path = scan_dir / side_capture["color_image"]
                depth_path = scan_dir / side_capture["depth_image"]
            except KeyError as error:
                warnings.append(f"step {step} {side}: missing image path {error}; skipped")
                continue
            if not color_path.is_file() or not depth_path.is_file():
                warnings.append(
                    f"step {step} {side}: image pair does not exist; skipped"
                )
                continue

            recorded_world_from_camera = pose_record_to_matrix(
                pose_record, f"step {step} {side} {pose_key}"
            )
            # This is already the moving ROS optical frame (+X right, +Y down,
            # +Z forward). The YAML hand-eye transform is relative to the hand
            # mount, so applying it again here would double-transform the cloud.
            views.append(
                CaptureView(
                    step=step,
                    side=side,
                    color_path=color_path,
                    depth_path=depth_path,
                    world_from_camera=recorded_world_from_camera,
                )
            )

    if not views:
        raise ValueError("manifest contains no usable captured RGB-D views")
    return views, warnings


def image_size(path: Path) -> tuple[int, int]:
    """Return image width and height without changing its encoding."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"unable to read image: {path}")
    height, width = image.shape[:2]
    return width, height


def make_open3d_intrinsic(o3d, calibration: IntrinsicCalibration, size):
    """Create an Open3D intrinsic, scaling calibration for image resolution."""
    width, height = size
    scale_x = width / calibration.width
    scale_y = height / calibration.height
    if not math.isclose(scale_x, scale_y, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"image aspect ratio {width}x{height} differs from calibration "
            f"{calibration.width}x{calibration.height}"
        )
    return o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        calibration.fx * scale_x,
        calibration.fy * scale_y,
        calibration.cx * scale_x,
        calibration.cy * scale_y,
    )


def reconstruct_scene(
    views: list[CaptureView],
    calibration: IntrinsicCalibration,
    voxel_size: float,
    remove_outliers: bool,
    outlier_neighbors: int,
    outlier_std_ratio: float,
):
    """Back-project, world-transform, merge, and filter all RGB-D views."""
    try:
        import open3d as o3d
        from scene_reconstruction.o3d_process import rgbd_to_pcd
    except ImportError as error:
        raise RuntimeError(
            "scene reconstruction requires Open3D; install the 'open3d' Python package"
        ) from error

    combined = o3d.geometry.PointCloud()
    expected_size = None
    for number, view in enumerate(views, start=1):
        size = image_size(view.color_path)
        if image_size(view.depth_path) != size:
            raise ValueError(
                f"color/depth resolution mismatch at step {view.step} {view.side}"
            )
        if expected_size is None:
            expected_size = size
        elif size != expected_size:
            raise ValueError(
                f"inconsistent image resolution at step {view.step} {view.side}: "
                f"{size} versus {expected_size}"
            )

        intrinsic = make_open3d_intrinsic(o3d, calibration, size)
        cloud = rgbd_to_pcd(
            str(view.color_path),
            str(view.depth_path),
            view.world_from_camera,
            intrinsic,
        )
        cloud = cloud.voxel_down_sample(voxel_size)
        combined += cloud
        print(
            f"[{number}/{len(views)}] added step {view.step:03d} {view.side}: "
            f"{len(cloud.points):,} points"
        )

    if not combined.has_points():
        raise RuntimeError("RGB-D conversion produced an empty point cloud")
    combined = combined.voxel_down_sample(voxel_size)
    print(f"Merged/downsampled cloud: {len(combined.points):,} points")

    if remove_outliers:
        combined, _ = combined.remove_statistical_outlier(
            nb_neighbors=outlier_neighbors, std_ratio=outlier_std_ratio
        )
        print(f"After outlier removal: {len(combined.points):,} points")
    return combined, o3d


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, default=Path("scan_output"))
    parser.add_argument("--camera-yaml", type=Path, default=DEFAULT_CAMERA_YAML)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="output PLY path (default: <scan-dir>/scene_reconstruction.ply)",
    )
    parser.add_argument(
        "--pose-source", choices=("actual", "desired"), default="actual",
        help="manifest camera poses used to recover each TCP pose",
    )
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--outlier-neighbors", type=int, default=20)
    parser.add_argument("--outlier-std-ratio", type=float, default=2.0)
    parser.add_argument("--no-outlier-removal", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate calibration, manifest, poses, and images without Open3D",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not math.isfinite(args.voxel_size) or args.voxel_size <= 0.0:
        raise ValueError("voxel-size must be finite and greater than zero")
    if args.outlier_neighbors <= 0:
        raise ValueError("outlier-neighbors must be greater than zero")
    if not math.isfinite(args.outlier_std_ratio) or args.outlier_std_ratio <= 0.0:
        raise ValueError("outlier-std-ratio must be finite and greater than zero")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    scan_dir = args.scan_dir.expanduser().resolve()
    manifest_path = scan_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"scan manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    calibration_path = args.camera_yaml.expanduser().resolve()
    intrinsics, yaml_mounts, camera_frame_ids = load_yaml_calibration(
        calibration_path
    )
    views, warnings = collect_capture_views(scan_dir, manifest, args.pose_source)
    for warning in warnings:
        print(f"warning: {warning}")
    print(
        f"Validated {len(views)} RGB-D views using calibration {calibration_path}"
    )
    if args.validate_only:
        return

    cloud, o3d = reconstruct_scene(
        views,
        intrinsics,
        args.voxel_size,
        not args.no_outlier_removal,
        args.outlier_neighbors,
        args.outlier_std_ratio,
    )
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else scan_dir / "scene_reconstruction.ply"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(output_path), cloud):
        raise RuntimeError(f"failed to write point cloud: {output_path}")

    metadata = {
        "point_cloud": str(output_path),
        "point_count": len(cloud.points),
        "view_count": len(views),
        "world_frame": manifest.get("world_frame", "world"),
        "pose_source": args.pose_source,
        "camera_yaml": str(calibration_path),
        "intrinsics": vars(intrinsics),
        "camera_frame_ids": camera_frame_ids,
        "yaml_hand_from_camera_mount": {
            side: transform.tolist() for side, transform in yaml_mounts.items()
        },
        "voxel_size": args.voxel_size,
        "outlier_removal": not args.no_outlier_removal,
        "views": [
            {
                "step": view.step,
                "side": view.side,
                "color_image": str(view.color_path.relative_to(scan_dir)),
                "depth_image": str(view.depth_path.relative_to(scan_dir)),
                "world_from_camera": view.world_from_camera.tolist(),
            }
            for view in views
        ],
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved reconstructed scene: {output_path}")
    print(f"Saved reconstruction metadata: {metadata_path}")

    if args.visualize:
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries(
            [cloud, coordinate_frame], window_name="RGB-D scene reconstruction"
        )


if __name__ == "__main__":
    main()
