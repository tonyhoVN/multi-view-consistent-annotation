#!/usr/bin/env python3
"""Convert repository-absolute paths in existing scan JSON files to relative paths."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Sequence

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scan_layout import serialized_relative_path  # noqa: E402


def relativize_value(
    value: Any,
    repository_root: Path,
    record_directory: Path,
    destination_repository_root: Path | None = None,
) -> tuple[Any, int]:
    """Recursively convert paths from a source checkout to relative paths."""
    destination_root = destination_repository_root or repository_root
    if isinstance(value, dict):
        updated: dict[str, Any] = {}
        replacements = 0
        for key, item in value.items():
            updated_item, count = relativize_value(
                item, repository_root, record_directory, destination_root
            )
            updated[key] = updated_item
            replacements += count
        return updated, replacements
    if isinstance(value, list):
        updated_items = []
        replacements = 0
        for item in value:
            updated_item, count = relativize_value(
                item, repository_root, record_directory, destination_root
            )
            updated_items.append(updated_item)
            replacements += count
        return updated_items, replacements
    if not isinstance(value, str):
        return value, 0

    # Only rewrite complete absolute filesystem paths below the repository.
    # This preserves ROS prim paths, URLs, messages, and external-workspace paths.
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        return value, 0
    normalized = candidate.resolve(strict=False)
    try:
        repository_relative = normalized.relative_to(repository_root)
    except ValueError:
        return value, 0
    rebased = destination_root / repository_relative
    return serialized_relative_path(rebased, record_directory), 1


def write_json_atomically(path: Path, document: Any) -> None:
    """Replace one JSON document without leaving a partially written target."""
    temporary = path.with_name(f".{path.name}.relative-paths.tmp")
    try:
        temporary.write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def relativize_json_file(
    path: Path,
    repository_root: Path,
    *,
    dry_run: bool,
    backup: bool,
    destination_repository_root: Path | None = None,
) -> int:
    """Rewrite one JSON file and return its number of converted path strings."""
    if path.is_symlink():
        raise RuntimeError(f"refusing to rewrite symlink: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        # Interrupted evaluation can leave an empty report; preserve it and
        # continue converting valid JSON files in the same run.
        print(f"warning: skipping unreadable JSON {path}: {error}", file=sys.stderr)
        return 0

    updated, replacements = relativize_value(
        document,
        repository_root,
        path.parent,
        destination_repository_root,
    )
    if replacements == 0 or dry_run:
        return replacements

    if backup:
        backup_path = path.with_suffix(path.suffix + ".absolute-paths.bak")
        if backup_path.exists():
            raise FileExistsError(f"backup already exists: {backup_path}")
        shutil.copy2(path, backup_path)
    write_json_atomically(path, updated)
    return replacements


def relativize_run(
    run_directory: Path,
    repository_root: Path,
    *,
    dry_run: bool,
    backup: bool,
    verbose: bool,
    destination_repository_root: Path | None = None,
) -> tuple[int, int]:
    """Process every JSON document below one run directory."""
    changed_files = 0
    replacement_count = 0
    for path in sorted(run_directory.rglob("*.json")):
        replacements = relativize_json_file(
            path,
            repository_root,
            dry_run=dry_run,
            backup=backup,
            destination_repository_root=destination_repository_root,
        )
        if replacements:
            changed_files += 1
            replacement_count += replacements
            if verbose:
                action = "would rewrite" if dry_run else "rewrote"
                relative = path.relative_to(run_directory)
                print(f"  {action} {relative}: {replacements} path(s)")
    return changed_files, replacement_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start_run", type=int, help="first run index, inclusive")
    parser.add_argument(
        "end_run",
        type=int,
        nargs="?",
        help="last run index, inclusive (default: start_run)",
    )
    parser.add_argument("--prefix", default="run_", help="run directory prefix")
    parser.add_argument("--scan-dir", type=Path, default=Path("scan_output"))
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help=(
            "old absolute checkout prefix stored in JSON "
            "(default: current repository root)"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--backup",
        action="store_true",
        help="retain each original as <name>.json.absolute-paths.bak",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    end_run = args.start_run if args.end_run is None else args.end_run
    if args.start_run < 0 or end_run < args.start_run:
        raise ValueError("require 0 <= start_run <= end_run")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.prefix):
        raise ValueError("--prefix must contain only safe filename characters")

    scan_root = args.scan_dir.expanduser().resolve()
    repository_root = args.repository_root.expanduser().resolve()
    current_repository_root = Path(__file__).resolve().parents[2]
    total_files = 0
    total_paths = 0

    # Missing run indices are expected in partially collected datasets.
    for index in range(args.start_run, end_run + 1):
        run_directory = scan_root / f"{args.prefix}{index}"
        if not run_directory.is_dir():
            print(f"Skipping missing run directory: {run_directory}")
            continue
        changed_files, replacements = relativize_run(
            run_directory,
            repository_root,
            dry_run=args.dry_run,
            backup=args.backup,
            verbose=args.verbose,
            destination_repository_root=current_repository_root,
        )
        total_files += changed_files
        total_paths += replacements
        action = "would update" if args.dry_run else "updated"
        print(
            f"{run_directory.name}: {action} {changed_files} JSON file(s), "
            f"{replacements} path(s)"
        )

    action = "Would update" if args.dry_run else "Updated"
    print(f"{action} {total_files} JSON file(s), {total_paths} path(s) total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
