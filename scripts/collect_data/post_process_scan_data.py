#!/usr/bin/env python3
"""Remove missing transforms and implausible pose jumps from scan trajectories.

Examples:
    python3 scripts/collect_data/post_process_scan_data.py \
        scan_output/manifest_run_3.json

    python3 scripts/collect_data/post_process_scan_data.py \
        scan_output/manifest_run_3.json --route hamilton_2opt
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Sequence

import numpy as np
import yaml

SCRIPTS_DIRECTORY = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from path_planning_single.planning import sample_hemisphere


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and minimally validate one scan manifest."""
    if not path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest root must be a JSON object: {path}")
    if not isinstance(manifest.get("routes"), dict):
        raise ValueError(f"manifest has no routes object: {path}")
    if not isinstance(manifest.get("captures"), list):
        raise ValueError(f"manifest has no captures list: {path}")
    return manifest


def captured_transforms(
    manifest: dict[str, Any], manifest_path: Path
) -> dict[int, np.ndarray]:
    """Load finite 4x4 transforms belonging to successful captures."""
    transforms: dict[int, np.ndarray] = {}
    for capture in manifest["captures"]:
        if not isinstance(capture, dict) or capture.get("status") != "captured":
            continue
        transform_value = capture.get("camera_transform")
        try:
            sample_index = int(capture["sample_index"])
        except (KeyError, TypeError, ValueError):
            continue
        if not isinstance(transform_value, str) or not transform_value:
            continue
        transform_path = Path(transform_value).expanduser()
        if not transform_path.is_absolute():
            transform_path = manifest_path.parent / transform_path
        if not transform_path.is_file():
            continue
        try:
            transform = np.asarray(np.load(transform_path), dtype=np.float64)
        except (OSError, ValueError):
            continue
        if transform.shape == (4, 4) and np.all(np.isfinite(transform)):
            transforms[sample_index] = transform
    return transforms


def planned_camera_poses(
    manifest: dict[str, Any], manifest_path: Path, config_override: Path | None
) -> dict[int, np.ndarray]:
    """Recreate the planned hemisphere pose associated with each sample index."""
    config_value: str | Path | None = config_override or manifest.get("configuration")
    if config_value is None:
        raise ValueError(
            "manifest has no configuration path; pass --config or --no-jump-filter"
        )
    config_path = Path(config_value).expanduser()
    if config_override is not None:
        config_path = config_path.resolve()
    elif not config_path.is_absolute():
        config_path = manifest_path.parent / config_path
    if not config_path.is_file():
        raise FileNotFoundError(
            f"scan configuration does not exist: {config_path}; "
            "pass --config or --no-jump-filter"
        )
    with config_path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    planning = document.get("single_path_planning") if isinstance(document, dict) else None
    if not isinstance(planning, dict):
        raise ValueError(f"configuration has no single_path_planning section: {config_path}")

    # Counts are saved in the manifest and therefore survive later config edits.
    latitude_layers = int(
        manifest.get("latitude_layers", planning["latitude_layers"])
    )
    azimuth_samples = int(
        manifest.get("azimuth_samples", planning["azimuth_samples"])
    )
    viewpoints = sample_hemisphere(
        planning["center"],
        float(planning["radius"]),
        latitude_layers,
        azimuth_samples,
        planning["elevation_bounds"],
        float(planning.get("azimuth_offset_deg", 0.0)),
    )
    index_base = int(manifest.get("sampling_index_base", 1))
    return {
        viewpoint.source_index + index_base: viewpoint.camera_pose
        for viewpoint in viewpoints
    }


def rotation_error_degrees(actual: np.ndarray, planned: np.ndarray) -> float:
    """Return the shortest angular distance between two rotation matrices."""
    relative = planned[:3, :3].T @ actual[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def find_pose_jumps(
    transforms: dict[int, np.ndarray],
    planned_poses: dict[int, np.ndarray],
    translation_threshold_m: float,
    rotation_threshold_deg: float,
) -> dict[int, dict[str, float]]:
    """Find measured poses that disagree substantially with their planned poses."""
    jumps: dict[int, dict[str, float]] = {}
    for sample_index, actual in transforms.items():
        planned = planned_poses.get(sample_index)
        if planned is None:
            # Index zero is the intentionally captured initial robot view.
            continue
        translation_error = float(
            np.linalg.norm(actual[:3, 3] - planned[:3, 3])
        )
        rotation_error = rotation_error_degrees(actual, planned)
        if (
            translation_error > translation_threshold_m
            or rotation_error > rotation_threshold_deg
        ):
            jumps[sample_index] = {
                "translation_error_m": translation_error,
                "rotation_error_deg": rotation_error,
            }
    return jumps


def clean_routes(
    manifest: dict[str, Any], available: set[int], selected_routes: set[str] | None
) -> dict[str, list[int]]:
    """Remove unavailable indices while retaining the relative order of each route."""
    removed_by_route: dict[str, list[int]] = {}
    routes = manifest["routes"]
    if selected_routes is None:
        selected_routes = {
            name for name in routes if name.startswith("hamilton_2opt")
        }
        if not selected_routes:
            raise ValueError("manifest contains no Hamilton 2-opt route")
    unknown = selected_routes - set(routes)
    if unknown:
        raise ValueError(f"unknown route(s): {', '.join(sorted(unknown))}")

    for route_name, route in routes.items():
        if route_name not in selected_routes:
            continue
        if not isinstance(route, dict) or "sample_indices" not in route:
            continue
        sample_indices = route["sample_indices"]
        if not isinstance(sample_indices, list):
            raise ValueError(f"route {route_name!r} sample_indices must be a list")
        try:
            normalized = [int(index) for index in sample_indices]
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"route {route_name!r} contains a non-integer sample index"
            ) from error

        removed = [index for index in normalized if index not in available]
        route["sample_indices"] = [
            index for index in normalized if index in available
        ]
        removed_by_route[route_name] = removed
    return removed_by_route


def write_manifest_atomically(
    manifest_path: Path, manifest: dict[str, Any], create_backup: bool
) -> Path | None:
    """Back up the original once, then atomically replace the JSON document."""
    backup_path = manifest_path.with_suffix(manifest_path.suffix + ".bak")
    if create_backup and not backup_path.exists():
        shutil.copy2(manifest_path, backup_path)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=manifest_path.parent,
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, manifest_path.stat().st_mode)
        os.replace(temporary_path, manifest_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return backup_path if create_backup else None


def process_manifest(
    manifest_path: Path,
    selected_routes: set[str] | None,
    *,
    dry_run: bool,
    create_backup: bool,
    config_override: Path | None,
    filter_jumps: bool,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
) -> tuple[dict[str, list[int]], set[int], dict[int, dict[str, float]]]:
    """Clean selected routes in one manifest and record the operation."""
    manifest_path = manifest_path.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    transforms = captured_transforms(manifest, manifest_path)
    available = set(transforms)

    # Reject TFs far from their indexed planned pose; this catches recorded jumps.
    jumps: dict[int, dict[str, float]] = {}
    if filter_jumps:
        planned = planned_camera_poses(manifest, manifest_path, config_override)
        jumps = find_pose_jumps(
            transforms,
            planned,
            translation_threshold_m,
            rotation_threshold_deg,
        )
    valid = available - set(jumps)
    removed_by_route = clean_routes(manifest, valid, selected_routes)

    manifest["trajectory_postprocess"] = {
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "available_transform_count": len(available),
        "jump_filter_enabled": filter_jumps,
        "translation_threshold_m": translation_threshold_m,
        "rotation_threshold_deg": rotation_threshold_deg,
        "route_metrics_recomputed": False,
        "routes": {
            route_name: {
                "removed_count": len(indices),
                "removed_missing_sample_indices": [
                    index for index in indices if index not in available
                ],
                "removed_jump_samples": {
                    str(index): jumps[index]
                    for index in indices
                    if index in jumps
                },
            }
            for route_name, indices in removed_by_route.items()
        },
    }
    if not dry_run:
        backup_path = write_manifest_atomically(
            manifest_path, manifest, create_backup=create_backup
        )
        print(f"Updated manifest: {manifest_path}")
        if backup_path is not None:
            print(f"Original backup: {backup_path}")
    return removed_by_route, available, jumps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="manifest_<suffix>.json to update")
    parser.add_argument(
        "--route",
        action="append",
        default=None,
        help=(
            "route to clean; repeat for multiple routes "
            "(default: all Hamilton 2-opt routes)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report changes without writing the manifest",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="do not create the default .json.bak backup",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="scan config YAML (default: the manifest's configuration field)",
    )
    parser.add_argument(
        "--translation-threshold",
        type=float,
        default=0.03,
        metavar="METERS",
        help="maximum measured-to-planned position error (default: 0.03)",
    )
    parser.add_argument(
        "--rotation-threshold-deg",
        type=float,
        default=10.0,
        metavar="DEGREES",
        help="maximum measured-to-planned orientation error (default: 10)",
    )
    parser.add_argument(
        "--no-jump-filter",
        action="store_true",
        help="only remove samples with missing or invalid transform files",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    if args.translation_threshold <= 0.0:
        raise ValueError("--translation-threshold must be positive")
    if args.rotation_threshold_deg <= 0.0:
        raise ValueError("--rotation-threshold-deg must be positive")
    selected_routes = set(args.route) if args.route else None
    removed_by_route, available, jumps = process_manifest(
        args.manifest,
        selected_routes,
        dry_run=args.dry_run,
        create_backup=not args.no_backup,
        config_override=args.config,
        filter_jumps=not args.no_jump_filter,
        translation_threshold_m=args.translation_threshold,
        rotation_threshold_deg=args.rotation_threshold_deg,
    )
    for route_name, removed in removed_by_route.items():
        missing = [index for index in removed if index not in available]
        jumped = [index for index in removed if index in jumps]
        if missing:
            print(
                f"{route_name}: removed {len(missing)} missing sample(s): "
                + ", ".join(map(str, missing))
            )
        if jumped:
            details = ", ".join(
                f"{index} ({jumps[index]['translation_error_m']:.3f} m, "
                f"{jumps[index]['rotation_error_deg']:.1f} deg)"
                for index in jumped
            )
            print(f"{route_name}: removed {len(jumped)} pose jump(s): {details}")
        if not missing and not jumped:
            print(f"{route_name}: no missing samples or pose jumps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
