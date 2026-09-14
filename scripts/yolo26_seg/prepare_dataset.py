#!/usr/bin/env python3
"""Build an Ultralytics segmentation dataset from collected scan runs.

Training labels come from one transfer/baseline prediction directory. Validation
and test labels always come from saved Isaac segmentation ground truth.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scan_layout import resolve_saved_path  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object with an error that identifies the damaged file."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return document


def normalized_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def object_label(item: dict[str, Any]) -> str:
    """Read a prediction label or remove Isaac's object/instance prefix."""
    if item.get("label"):
        return normalized_label(str(item["label"]))
    instance = str(item.get("instance", ""))
    return normalized_label(re.sub(r"^object_\d+_(?:\d+_)?", "", instance))


def parse_runs(values: Sequence[str]) -> list[int]:
    """Expand values such as ``1 3-5`` into sorted unique run indices."""
    result: set[int] = set()
    for value in values:
        match = re.fullmatch(r"(\d+)-(\d+)", value)
        if match:
            start, end = map(int, match.groups())
            if end < start:
                raise ValueError(f"invalid descending run range: {value}")
            result.update(range(start, end + 1))
        else:
            result.add(int(value))
    if not result:
        raise ValueError("at least one run is required")
    return sorted(result)


def captured_frames(run_root: Path, route: str) -> list[tuple[dict[str, Any], Path]]:
    """Return successful captures in route order with resolved color images."""
    manifest_path = run_root / "manifest.json"
    manifest = read_json(manifest_path)
    route_record = manifest.get("routes", {}).get(route, {})
    order = route_record.get("sample_indices")
    if not isinstance(order, list):
        raise ValueError(f"{manifest_path} has no route {route!r}")
    captures = {
        int(item["sample_index"]): item
        for item in manifest.get("captures", [])
        if isinstance(item, dict)
        and item.get("status") == "captured"
        and "sample_index" in item
    }
    frames = []
    for sample_index in map(int, order):
        capture = captures.get(sample_index)
        if capture is None or not isinstance(capture.get("color_image"), str):
            continue
        image = resolve_saved_path(capture["color_image"], manifest_path.parent)
        if image.is_file():
            frames.append((capture, image))
        else:
            print(f"warning: missing image {image}; frame skipped")
    return frames


def ground_truth_manifest(run_root: Path, capture: dict[str, Any]) -> Path | None:
    segmentation = capture.get("segmentation")
    value = segmentation.get("manifest") if isinstance(segmentation, dict) else None
    if not isinstance(value, str):
        return None
    path = resolve_saved_path(value, run_root)
    return path if path.is_file() else None


def prediction_manifests(source_root: Path) -> dict[int, Path]:
    """Index per-frame prediction manifests primarily by saved sample index."""
    result: dict[int, Path] = {}
    for path in sorted(source_root.rglob("manifest.json")):
        if not any(re.fullmatch(r"segment_\d+", part) for part in path.parts):
            continue
        document = read_json(path)
        sample_index = document.get("sample_index")
        if sample_index is None:
            segment = next(
                (part for part in reversed(path.parts) if re.fullmatch(r"segment_\d+", part)),
                None,
            )
            if segment is None:
                continue
            sample_index = int(segment.split("_", 1)[1])
        result[int(sample_index)] = path
    return result


def source_directory(run_root: Path, source: str) -> Path:
    """Map an experiment source name to its per-run annotation directory."""
    if source == "transfer":
        return run_root / "transfer_segment"
    return run_root / "baseline_segment" / source


def labels_in_manifest(path: Path) -> set[str]:
    return {
        object_label(item)
        for item in read_json(path).get("objects", [])
        if isinstance(item, dict) and object_label(item)
    }


def discover_classes(scan_root: Path, runs: Iterable[int], route: str) -> list[str]:
    """Build a stable class vocabulary exclusively from ground truth."""
    classes: set[str] = set()
    for run_index in runs:
        run_root = scan_root / f"run_{run_index}"
        for capture, _ in captured_frames(run_root, route):
            path = ground_truth_manifest(run_root, capture)
            if path is not None:
                classes.update(labels_in_manifest(path))
    if not classes:
        raise ValueError("no ground-truth classes found in selected runs")
    return sorted(classes)


def mask_polygons(mask: np.ndarray, epsilon: float, minimum_area: float) -> list[np.ndarray]:
    """Convert foreground components to normalized YOLO polygon coordinates."""
    height, width = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < minimum_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon * perimeter, True).reshape(-1, 2)
        if len(polygon) < 3:
            continue
        normalized = polygon.astype(np.float64)
        normalized[:, 0] = np.clip(normalized[:, 0] / width, 0.0, 1.0)
        normalized[:, 1] = np.clip(normalized[:, 1] / height, 0.0, 1.0)
        polygons.append(normalized)
    return polygons


def manifest_to_yolo(path: Path | None, class_ids: dict[str, int], epsilon: float, minimum_area: float) -> list[str]:
    """Convert every readable object mask in one manifest to YOLO-seg rows."""
    if path is None:
        return []
    lines = []
    document = read_json(path)
    for item in document.get("objects", []):
        if not isinstance(item, dict) or not isinstance(item.get("mask"), str):
            continue
        label = object_label(item)
        if label not in class_ids:
            continue
        mask_path = resolve_saved_path(item["mask"], path.parent)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"warning: cannot read mask {mask_path}; object skipped")
            continue
        for polygon in mask_polygons(mask, epsilon, minimum_area):
            coordinates = " ".join(f"{value:.6f}" for value in polygon.reshape(-1))
            lines.append(f"{class_ids[label]} {coordinates}")
    return lines


def link_image(source: Path, destination: Path) -> None:
    """Create a portable relative symlink without duplicating RGB data."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source, destination.parent))


def build_split(
    scan_root: Path,
    dataset_root: Path,
    split: str,
    runs: Sequence[int],
    route: str,
    source: str,
    class_ids: dict[str, int],
    epsilon: float,
    minimum_area: float,
) -> dict[str, int]:
    """Write one image/label split and return auditable frame statistics."""
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    frame_count = annotated_count = 0
    for run_index in runs:
        run_root = scan_root / f"run_{run_index}"
        predictions = prediction_manifests(source_directory(run_root, source)) if split == "train" else {}
        for capture, image in captured_frames(run_root, route):
            sample_index = int(capture["sample_index"])
            annotation = (
                predictions.get(sample_index)
                if split == "train"
                else ground_truth_manifest(run_root, capture)
            )
            if annotation is None:
                print(
                    f"warning: run {run_index} sample {sample_index} has no "
                    f"{source if split == 'train' else 'ground-truth'} manifest; "
                    "frame skipped"
                )
                continue
            stem = f"run_{run_index:03d}_sample_{sample_index:04d}"
            link_image(image, image_dir / f"{stem}{image.suffix.lower()}")
            lines = manifest_to_yolo(annotation, class_ids, epsilon, minimum_area)
            (label_dir / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            frame_count += 1
            annotated_count += bool(lines)
    return {"frames": frame_count, "frames_with_labels": annotated_count}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", type=Path, default=Path("scan_output"))
    parser.add_argument("--train-runs", nargs="+", required=True, metavar="RUN")
    parser.add_argument("--val-runs", nargs="+", required=True, metavar="RUN")
    parser.add_argument("--test-runs", nargs="+", required=True, metavar="RUN")
    parser.add_argument("--source", required=True, help="transfer or a baseline_segment child directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route", default="hamilton_2opt")
    parser.add_argument("--polygon-epsilon", type=float, default=0.002)
    parser.add_argument("--minimum-component-area", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    scan_root = args.scan_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    train_runs = parse_runs(args.train_runs)
    val_runs = parse_runs(args.val_runs)
    test_runs = parse_runs(args.test_runs)
    split_runs = {"train": set(train_runs), "val": set(val_runs), "test": set(test_runs)}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_runs[first] & split_runs[second]
        if overlap:
            raise ValueError(
                f"{first}/{second} run leakage: {sorted(overlap)}"
            )
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"dataset exists (use --overwrite): {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    # One ground-truth-derived vocabulary keeps every pseudo-label source comparable.
    names = discover_classes(
        scan_root, [*train_runs, *val_runs, *test_runs], args.route
    )
    class_ids = {name: index for index, name in enumerate(names)}
    train_stats = build_split(
        scan_root, output, "train", train_runs, args.route, args.source,
        class_ids, args.polygon_epsilon, args.minimum_component_area,
    )
    val_stats = build_split(
        scan_root, output, "val", val_runs, args.route, args.source,
        class_ids, args.polygon_epsilon, args.minimum_component_area,
    )
    test_stats = build_split(
        scan_root, output, "test", test_runs, args.route, args.source,
        class_ids, args.polygon_epsilon, args.minimum_component_area,
    )
    # Omitting ``path`` makes Ultralytics resolve splits beside this YAML file,
    # keeping the generated dataset portable across repository locations.
    dataset_yaml = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    (output / "dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")
    metadata = {
        "source": args.source, "route": args.route, "train_runs": train_runs,
        "val_runs": val_runs, "test_runs": test_runs, "classes": names,
        "train": train_stats, "val": val_stats, "test": test_stats,
        "val_annotation_source": "Isaac ground truth",
        "test_annotation_source": "Isaac ground truth",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Created {output / 'dataset.yaml'}")
    print(
        f"Classes: {len(names)}; train: {train_stats}; "
        f"validation: {val_stats}; test: {test_stats}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
