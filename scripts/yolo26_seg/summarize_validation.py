#!/usr/bin/env python3
"""Combine YOLO segmentation validation metrics into one JSON report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Sequence


METRIC_KEYS = (
    "box_mAP50",
    "box_mAP50_95",
    "mask_mAP50",
    "mask_mAP50_95",
)


def read_report(path: Path) -> dict[str, Any]:
    """Read one validation report and require finite numeric metrics."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read validation JSON {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"validation JSON root must be an object: {path}")
    metrics = {}
    for key in METRIC_KEYS:
        value = document.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{path} has no finite numeric {key}")
        metrics[key] = float(value)
    return {**document, **metrics}


def discover_reports(root: Path, recursive: bool) -> list[Path]:
    """Find per-experiment reports without including the combined output."""
    pattern = "**/ground_truth_validation.json" if recursive else "*/ground_truth_validation.json"
    return sorted(path for path in root.glob(pattern) if path.is_file())


def build_summary(root: Path, reports: Sequence[Path]) -> dict[str, Any]:
    """Return per-experiment metrics and their unweighted macro averages."""
    experiments = {}
    for path in reports:
        document = read_report(path)
        experiment = path.parent.relative_to(root).as_posix()
        experiments[experiment] = {
            key: document[key] for key in METRIC_KEYS
        } | {
            "model": document.get("model"),
            "dataset": document.get("dataset"),
            "report": path.relative_to(root).as_posix(),
        }
    averages = {
        key: mean(record[key] for record in experiments.values())
        for key in METRIC_KEYS
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "validation_root": root.as_posix(),
        "experiment_count": len(experiments),
        "metrics": list(METRIC_KEYS),
        "experiments": experiments,
        "macro_average": averages,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-root", type=Path, default=Path("scan_output/yolo26_seg")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scan_output/yolo26_seg_validation_summary.json"),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="also include nested evaluation sets such as eval_50_60",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    root = args.validation_root.expanduser().resolve()
    reports = discover_reports(root, args.recursive)
    if not reports:
        raise ValueError(f"no ground_truth_validation.json reports under {root}")
    summary = build_summary(root, reports)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Combined {len(reports)} validation reports into {output}")
    for name, metrics in summary["experiments"].items():
        print(
            f"{name}: box={metrics['box_mAP50']:.4f}/"
            f"{metrics['box_mAP50_95']:.4f}, mask={metrics['mask_mAP50']:.4f}/"
            f"{metrics['mask_mAP50_95']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
