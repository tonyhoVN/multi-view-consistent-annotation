#!/usr/bin/env python3
"""Move legacy scan artifacts into prefix-scoped run directories.

Example:
    python3 scripts/collect_data/migrate_scan_layout.py 1 24
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Sequence

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scan_layout import ScanRunLayout  # noqa: E402


def legacy_moves(scan_root: Path, run_name: str) -> list[tuple[Path, Path]]:
    """Return every existing legacy artifact and its canonical destination."""
    layout = ScanRunLayout(scan_root, run_name)
    candidates = [
        (scan_root / f"manifest_{run_name}.json", layout.manifest),
        (scan_root / f"manifest_{run_name}.json.bak", layout.root / "manifest.json.bak"),
        (scan_root / f"save_images_{run_name}", layout.images),
        (scan_root / f"save_segment_{run_name}", layout.segments),
        (scan_root / f"save_TF_{run_name}", layout.transforms),
        (scan_root / "collection_logs" / run_name, layout.collection_log),
        (scan_root / f"transfer_segment_{run_name}", layout.transfer_segment),
        (
            scan_root / f"naive_vlm_zeroshot_{run_name}",
            layout.baseline_segment("naive_vlm", "zeroshot"),
        ),
        (
            scan_root / f"naive_vlm_multi_shot_{run_name}",
            layout.baseline_segment("naive_vlm", "multi_shot"),
        ),
        (
            scan_root / f"naive_vlm_segment_{run_name}",
            layout.root / "baseline_segment" / "naive_vlm_legacy",
        ),
        # Also migrate the short-lived first version of the prefix layout.
        (
            layout.root / "naive_vlm" / "zeroshot",
            layout.baseline_segment("naive_vlm", "zeroshot"),
        ),
        (
            layout.root / "naive_vlm" / "multi_shot",
            layout.baseline_segment("naive_vlm", "multi_shot"),
        ),
        (
            layout.root / "naive_vlm" / "legacy",
            layout.root / "baseline_segment" / "naive_vlm_legacy",
        ),
    ]

    # Keep every post-processing backup, since these are recovery data.
    backup_pattern = f"postprocess_backup_manifest_{run_name}_*"
    candidates.extend(
        (source, layout.root / "postprocess_backups" / source.name)
        for source in sorted(scan_root.glob(backup_pattern))
    )
    return [(source, destination) for source, destination in candidates if source.exists()]


def validate_moves(moves: Sequence[tuple[Path, Path]]) -> None:
    """Refuse ambiguous sources, symlinks, and destination collisions."""
    destinations: set[Path] = set()
    for source, destination in moves:
        if source.is_symlink():
            raise RuntimeError(f"refusing to migrate symlink: {source}")
        if destination in destinations:
            raise RuntimeError(f"multiple artifacts target {destination}")
        destinations.add(destination)
        if destination.exists():
            raise FileExistsError(
                f"destination already exists: {destination}; no files were moved"
            )


def replace_path_string(
    value: str, moves: Sequence[tuple[Path, Path]], old_manifest: Path, new_manifest: Path
) -> str:
    """Translate absolute and manifest-relative legacy paths in JSON documents."""
    replacements = [*moves, (old_manifest, new_manifest)]
    for source, destination in replacements:
        source_absolute = str(source.resolve())
        destination_absolute = str(destination.resolve())
        if value == source_absolute or value.startswith(source_absolute + os.sep):
            return destination_absolute + value[len(source_absolute) :]

        source_relative = source.name
        try:
            destination_relative = str(destination.relative_to(new_manifest.parent))
        except ValueError:
            continue
        if value == source_relative or value.startswith(source_relative + "/"):
            return destination_relative + value[len(source_relative) :]
    return value


def rewrite_json_value(
    value: Any,
    moves: Sequence[tuple[Path, Path]],
    old_manifest: Path,
    new_manifest: Path,
) -> Any:
    """Recursively update stored paths without changing unrelated JSON values."""
    if isinstance(value, dict):
        return {
            key: rewrite_json_value(item, moves, old_manifest, new_manifest)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            rewrite_json_value(item, moves, old_manifest, new_manifest)
            for item in value
        ]
    if isinstance(value, str):
        return replace_path_string(value, moves, old_manifest, new_manifest)
    return value


def rewrite_moved_json(
    run_root: Path,
    moves: Sequence[tuple[Path, Path]],
    old_manifest: Path,
    new_manifest: Path,
) -> None:
    """Make manifests and reports portable relative to the new run directory."""
    for path in run_root.rglob("*.json"):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        updated = rewrite_json_value(document, moves, old_manifest, new_manifest)
        if updated != document:
            path.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")


def migrate_run(scan_root: Path, run_name: str, dry_run: bool) -> int:
    """Validate, move, and rewrite one run, rolling moves back on failure."""
    layout = ScanRunLayout(scan_root, run_name)
    moves = legacy_moves(scan_root, run_name)
    if not moves:
        print(f"{run_name}: no legacy artifacts found")
        return 0
    validate_moves(moves)
    print(f"{run_name}: {len(moves)} artifact(s)")
    for source, destination in moves:
        print(f"  {source} -> {destination}")
    if dry_run:
        return len(moves)

    completed: list[tuple[Path, Path]] = []
    old_manifest = scan_root / f"manifest_{run_name}.json"
    try:
        layout.root.mkdir(parents=True, exist_ok=True)
        for source, destination in moves:
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            completed.append((source, destination))
        rewrite_moved_json(layout.root, moves, old_manifest, layout.manifest)
    except Exception:
        # Rename completed artifacts back so a failed migration is recoverable.
        for source, destination in reversed(completed):
            if destination.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                destination.rename(source)
        raise
    return len(moves)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start_run", type=int, help="first run index, inclusive")
    parser.add_argument("end_run", type=int, help="last run index, inclusive")
    parser.add_argument("--prefix", default="run_", help="run directory prefix")
    parser.add_argument("--scan-dir", type=Path, default=Path("scan_output"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    if args.start_run < 0 or args.end_run < args.start_run:
        raise ValueError("require 0 <= start_run <= end_run")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.prefix):
        raise ValueError("--prefix must contain only safe filename characters")
    scan_root = args.scan_dir.expanduser().resolve()
    moved = 0
    for index in range(args.start_run, args.end_run + 1):
        moved += migrate_run(scan_root, f"{args.prefix}{index}", args.dry_run)
    print(f"{'Would migrate' if args.dry_run else 'Migrated'} {moved} artifact(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
