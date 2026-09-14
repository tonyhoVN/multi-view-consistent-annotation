#!/usr/bin/env python3
"""Validate a trained YOLO segmentation checkpoint against ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def scalar(value: Any) -> float | None:
    """Convert an Ultralytics scalar or tensor to a JSON-compatible float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("model", type=Path, help="checkpoint, normally weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
        help="dataset split to evaluate; final reporting should use test",
    )
    parser.add_argument("--project", type=Path)
    parser.add_argument("--name", default="ground_truth_validation")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--exist-ok", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    from ultralytics import YOLO

    dataset = args.dataset.expanduser().resolve()
    checkpoint = args.model.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"model checkpoint does not exist: {checkpoint}")
    project = args.project.expanduser().resolve() if args.project else checkpoint.parents[1]

    # Both held-out splits use Isaac ground truth; test is never used by training.
    metrics = YOLO(str(checkpoint)).val(
        data=str(dataset), split=args.split, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers, project=str(project),
        name=args.name, exist_ok=args.exist_ok,
    )
    report = {
        "model": str(checkpoint), "dataset": str(dataset),
        "validation_split": args.split,
        "validation_annotation_source": "Isaac ground truth",
        "mask_mAP50": scalar(metrics.seg.map50),
        "mask_mAP50_95": scalar(metrics.seg.map),
        "box_mAP50": scalar(metrics.box.map50),
        "box_mAP50_95": scalar(metrics.box.map),
        "image_size": args.imgsz,
    }
    report_path = args.output.expanduser().resolve() if args.output else project / "ground_truth_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
