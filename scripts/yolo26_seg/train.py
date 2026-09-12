#!/usr/bin/env python3
"""Train YOLO26 nano segmentation on one prepared pseudo-label dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", default="yolo26n-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=Path, default=Path("scan_output/yolo26_seg"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exist-ok", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    from ultralytics import YOLO

    dataset = args.dataset.expanduser().resolve()
    project = args.project.expanduser().resolve()
    YOLO(args.model).train(
        data=str(dataset), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers, project=str(project), name=args.name,
        patience=args.patience, seed=args.seed, exist_ok=args.exist_ok, task="segment",
    )
    print(f"Training complete. Best checkpoint: {project / args.name / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
