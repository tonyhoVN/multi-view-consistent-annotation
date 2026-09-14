#!/usr/bin/env python3
"""Move a run-level annotation directory into ``baseline_segment``."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start_run", type=int, help="first scan run number")
    parser.add_argument(
        "end_run",
        type=int,
        nargs="?",
        help="last scan run number, inclusive; defaults to start_run",
    )
    parser.add_argument(
        "annotation_name",
        help="run-level directory to move, for example transfer_sam_vith_segment",
    )
    parser.add_argument("--scan-root", type=Path, default=Path("scan_output"))
    parser.add_argument(
        "--dry-run", action="store_true", help="show the move without changing files"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if Path(args.annotation_name).name != args.annotation_name:
        raise ValueError("annotation_name must be one directory name")

    end_run = args.start_run if args.end_run is None else args.end_run
    if args.start_run < 0 or end_run < args.start_run:
        raise ValueError("run range must satisfy 0 <= start_run <= end_run")

    scan_root = args.scan_root.expanduser().resolve()
    moves: list[tuple[Path, Path, Path]] = []
    for run_index in range(args.start_run, end_run + 1):
        run_root = scan_root / f"run_{run_index}"
        source = run_root / args.annotation_name
        destination_root = run_root / "baseline_segment"
        destination = destination_root / args.annotation_name
        if not source.is_dir():
            print(f"Warning: run {run_index} source does not exist; skipped: {source}")
            continue
        if destination.exists():
            raise FileExistsError(f"destination already exists: {destination}")
        moves.append((source, destination_root, destination))

    if not moves:
        print("No annotation directories found to move")
        return 0

    # Validate the complete range before changing any directory.
    for source, _, destination in moves:
        print(f"Move: {source}")
        print(f"  to: {destination}")
    if args.dry_run:
        print(f"Dry run: no files changed ({len(moves)} move(s) found)")
        return 0

    for source, destination_root, destination in moves:
        destination_root.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
    print(f"Moved {len(moves)} annotation directorie(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
