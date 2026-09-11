#!/usr/bin/env python3
"""Clean a scan trajectory and renumber its saved files into refined path order.

Examples:
    python3 scripts/collect_data/post_process_scan_data.py \
        scan_output/run_3/manifest.json

    python3 scripts/collect_data/post_process_scan_data.py \
        scan_output/run_3/manifest.json --route hamilton_2opt
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
from uuid import uuid4

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


def selected_route_names(
    manifest: dict[str, Any], selected_routes: set[str] | None
) -> set[str]:
    """Resolve the routes selected by CLI defaults and validate their names."""
    routes = manifest["routes"]
    names = selected_routes
    if names is None:
        names = {name for name in routes if name.startswith("hamilton_2opt")}
        if not names:
            raise ValueError("manifest contains no Hamilton 2-opt route")
    unknown = names - set(routes)
    if unknown:
        raise ValueError(f"unknown route(s): {', '.join(sorted(unknown))}")
    return names


def resolve_manifest_path(value: str, manifest_path: Path) -> Path:
    """Resolve a manifest path relative to the directory containing the manifest."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else manifest_path.parent / path


def _link_or_copy(source: str, destination: str) -> str:
    """Hard-link immutable scan data when possible, falling back to a copy."""
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def _safe_output_directory(
    manifest: dict[str, Any], manifest_path: Path, key: str
) -> Path:
    """Resolve and constrain one generated-data directory to the scan root."""
    directories = manifest.get("directories")
    if not isinstance(directories, dict) or not isinstance(directories.get(key), str):
        raise ValueError(f"manifest has no directories.{key} path")
    path = resolve_manifest_path(directories[key], manifest_path).resolve()
    try:
        relative = path.relative_to(manifest_path.parent)
    except ValueError as error:
        raise ValueError(f"directories.{key} is outside the scan root: {path}") from error
    if not relative.parts or relative == Path("."):
        raise ValueError(f"directories.{key} cannot be the scan root")
    if not path.is_dir():
        raise FileNotFoundError(f"directories.{key} does not exist: {path}")
    return path


def _copy_file_to_staging(source: Path, destination: Path) -> None:
    """Validate and stage one file under its canonical destination name."""
    if not source.is_file():
        raise FileNotFoundError(f"recorded scan file does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _link_or_copy(str(source), str(destination))


def canonicalize_route_files(
    manifest: dict[str, Any],
    manifest_path: Path,
    route_name: str,
    *,
    dry_run: bool,
) -> tuple[list[dict[str, int]], Path | None, list[tuple[Path, Path]]]:
    """Rename scan products to consecutive route order and update capture records.

    The replacement directories are assembled first. Original directories are
    then moved to a backup before the replacements are installed. The returned
    swap list allows the caller to roll back if writing the manifest fails.
    """
    sample_indices = [
        int(index) for index in manifest["routes"][route_name]["sample_indices"]
    ]
    captures = {
        int(capture["sample_index"]): capture
        for capture in manifest["captures"]
        if isinstance(capture, dict)
        and capture.get("status") == "captured"
        and "sample_index" in capture
    }
    missing = [index for index in sample_indices if index not in captures]
    if missing:
        raise ValueError(
            f"refined route contains samples without capture records: {missing}"
        )

    roots = {
        key: _safe_output_directory(manifest, manifest_path, key)
        for key in ("images", "segments", "transforms")
    }
    mapping = [
        {"path_index": path_index, "sample_index": sample_index}
        for path_index, sample_index in enumerate(sample_indices)
    ]
    if dry_run:
        return mapping, None, []

    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{manifest_path.stem}_refined_", dir=manifest_path.parent)
    )
    updated_captures: list[dict[str, Any]] = []
    swaps: list[tuple[Path, Path]] = []
    backup_root: Path | None = None
    try:
        for original in roots.values():
            (staging_root / original.relative_to(manifest_path.parent)).mkdir(
                parents=True, exist_ok=True
            )

        # Build complete replacement directories before touching recorded data.
        for path_index, sample_index in enumerate(sample_indices):
            capture = dict(captures[sample_index])
            color_source = resolve_manifest_path(capture["color_image"], manifest_path)
            depth_source = resolve_manifest_path(capture["depth_image"], manifest_path)
            transform_source = resolve_manifest_path(
                capture["camera_transform"], manifest_path
            )
            color_target = roots["images"] / f"color_{path_index}{color_source.suffix}"
            depth_target = roots["images"] / f"depth_{path_index}{depth_source.suffix}"
            transform_target = roots["transforms"] / f"T_base_cam_{path_index}.npy"
            for source, target in (
                (color_source, color_target),
                (depth_source, depth_target),
                (transform_source, transform_target),
            ):
                staged = staging_root / target.relative_to(manifest_path.parent)
                _copy_file_to_staging(source, staged)

            capture["path_index"] = path_index
            capture["color_image"] = str(color_target.relative_to(manifest_path.parent))
            capture["depth_image"] = str(depth_target.relative_to(manifest_path.parent))
            capture["camera_transform"] = str(
                transform_target.relative_to(manifest_path.parent)
            )

            # Preserve Isaac's subtree while renaming its segment_<sample> root.
            segmentation = capture.get("segmentation")
            if isinstance(segmentation, dict) and segmentation.get("status") == "saved":
                segmentation = dict(segmentation)
                source_directory = resolve_manifest_path(
                    segmentation["directory"], manifest_path
                )
                try:
                    relative_source = source_directory.relative_to(roots["segments"])
                except ValueError as error:
                    raise ValueError(
                        f"unexpected segmentation directory for sample {sample_index}: "
                        f"{source_directory}"
                    ) from error
                if not relative_source.parts or not relative_source.parts[0].startswith(
                    "segment_"
                ):
                    raise ValueError(
                        f"segmentation directory has no segment_<index> component: "
                        f"{source_directory}"
                    )
                suffix = Path(*relative_source.parts[1:])
                target_directory = roots["segments"] / f"segment_{path_index}" / suffix
                staged_directory = (
                    staging_root / target_directory.relative_to(manifest_path.parent)
                )
                if not source_directory.is_dir():
                    raise FileNotFoundError(
                        f"segmentation directory does not exist: {source_directory}"
                    )
                shutil.copytree(
                    source_directory,
                    staged_directory,
                    copy_function=_link_or_copy,
                )
                source_manifest = resolve_manifest_path(
                    segmentation["manifest"], manifest_path
                )
                manifest_suffix = source_manifest.relative_to(source_directory)
                target_manifest = target_directory / manifest_suffix
                segmentation["directory"] = str(
                    target_directory.relative_to(manifest_path.parent)
                )
                segmentation["manifest"] = str(
                    target_manifest.relative_to(manifest_path.parent)
                )
                capture["segmentation"] = segmentation
            updated_captures.append(capture)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_root = manifest_path.parent / (
            f"postprocess_backup_{manifest_path.stem}_{timestamp}_{uuid4().hex[:8]}"
        )
        backup_root.mkdir()

        # Swap each complete artifact directory; originals remain recoverable.
        for key, original in roots.items():
            backup = backup_root / original.name
            replacement = staging_root / original.relative_to(manifest_path.parent)
            original.rename(backup)
            swaps.append((original, backup))
            try:
                replacement.rename(original)
            except Exception:
                backup.rename(original)
                swaps.pop()
                raise
        manifest["captures"] = updated_captures
    except Exception:
        for original, backup in reversed(swaps):
            if original.exists():
                original.rename(staging_root / original.relative_to(manifest_path.parent))
            if backup.exists():
                backup.rename(original)
        if backup_root is not None and backup_root.exists():
            backup_root.rmdir()
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    shutil.rmtree(staging_root, ignore_errors=True)
    return mapping, backup_root, swaps


def rollback_directory_swaps(
    swaps: list[tuple[Path, Path]], backup_root: Path | None
) -> None:
    """Restore original artifact directories after a manifest-write failure."""
    rollback_root = Path(tempfile.mkdtemp(prefix=".postprocess_rollback_", dir=swaps[0][0].parent))
    for original, backup in reversed(swaps):
        if original.exists():
            original.rename(rollback_root / original.name)
        backup.rename(original)
    shutil.rmtree(rollback_root, ignore_errors=True)
    if backup_root is not None and backup_root.exists():
        backup_root.rmdir()


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
    rename_files: bool,
) -> tuple[
    dict[str, list[int]],
    set[int],
    dict[int, dict[str, float]],
    list[dict[str, int]],
    Path | None,
]:
    """Clean selected routes in one manifest and record the operation."""
    manifest_path = manifest_path.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    route_names = selected_route_names(manifest, selected_routes)
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
    removed_by_route = clean_routes(manifest, valid, route_names)

    renumbering: list[dict[str, int]] = []
    artifact_backup: Path | None = None
    swaps: list[tuple[Path, Path]] = []
    if rename_files:
        if len(route_names) != 1:
            raise ValueError(
                "saved files can follow only one route; pass exactly one --route"
            )
        route_name = next(iter(route_names))
        renumbering, artifact_backup, swaps = canonicalize_route_files(
            manifest, manifest_path, route_name, dry_run=dry_run
        )

    manifest["trajectory_postprocess"] = {
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "available_transform_count": len(available),
        "jump_filter_enabled": filter_jumps,
        "translation_threshold_m": translation_threshold_m,
        "rotation_threshold_deg": rotation_threshold_deg,
        "files_renamed_to_route_order": rename_files,
        "renumbering": renumbering,
        "artifact_backup": (
            str(artifact_backup.relative_to(manifest_path.parent))
            if artifact_backup is not None and create_backup
            else None
        ),
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
        try:
            backup_path = write_manifest_atomically(
                manifest_path, manifest, create_backup=create_backup
            )
        except Exception:
            if swaps:
                rollback_directory_swaps(swaps, artifact_backup)
            raise

        # The artifact directory is needed transactionally until the manifest
        # commits; --no-backup removes it only after that commit succeeds.
        if artifact_backup is not None and not create_backup:
            shutil.rmtree(artifact_backup)
            artifact_backup = None
        print(f"Updated manifest: {manifest_path}")
        if backup_path is not None:
            print(f"Original backup: {backup_path}")
        if artifact_backup is not None:
            print(f"Original artifact directories: {artifact_backup}")
    return removed_by_route, available, jumps, renumbering, artifact_backup


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="<run>/manifest.json to update")
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
        help="do not retain manifest or renamed-artifact backups",
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
    parser.add_argument(
        "--no-rename-files",
        action="store_true",
        help="keep sample-index filenames instead of renumbering into route order",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    if args.translation_threshold <= 0.0:
        raise ValueError("--translation-threshold must be positive")
    if args.rotation_threshold_deg <= 0.0:
        raise ValueError("--rotation-threshold-deg must be positive")
    selected_routes = set(args.route) if args.route else None
    removed_by_route, available, jumps, renumbering, _ = process_manifest(
        args.manifest,
        selected_routes,
        dry_run=args.dry_run,
        create_backup=not args.no_backup,
        config_override=args.config,
        filter_jumps=not args.no_jump_filter,
        translation_threshold_m=args.translation_threshold,
        rotation_threshold_deg=args.rotation_threshold_deg,
        rename_files=not args.no_rename_files,
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
    if renumbering:
        verb = "Would renumber" if args.dry_run else "Renumbered"
        print(
            f"{verb} {len(renumbering)} captured views to consecutive "
            f"path indices 0..{len(renumbering) - 1}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
