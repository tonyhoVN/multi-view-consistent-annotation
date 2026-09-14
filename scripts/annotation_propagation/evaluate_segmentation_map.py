#!/usr/bin/env python3
"""Measure COCO-style mask and bounding-box mAP for one annotation run.

Ground-truth manifests are resolved through the scan manifest's capture records,
so evaluation remains correct after trajectory filtering and file renumbering.
Prediction manifests are matched by ``sample_index`` when available and by
``path_index`` as a fallback.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scan_layout import resolve_saved_path, serialized_relative_path  # noqa: E402


IOU_THRESHOLDS = np.arange(0.50, 0.951, 0.05)


@dataclass(frozen=True)
class MaskRecord:
    """One class-labeled binary mask in one captured frame."""

    frame_key: int
    class_name: str
    instance: str
    mask_path: Path
    confidence: float


def normalized_label(value: str) -> str:
    """Normalize spelling differences without merging genuinely different classes."""
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def label_from_object(record: dict[str, Any]) -> str:
    """Read an explicit label or derive one from an Isaac instance name."""
    if record.get("label"):
        return normalized_label(str(record["label"]))
    instance = str(record.get("instance", ""))
    label = re.sub(r"^object_\d+_(?:\d+_)?", "", instance)
    return normalized_label(label or instance)


def resolve_path(value: str, base: Path) -> Path:
    """Resolve a path stored relative to a known manifest directory."""
    return resolve_saved_path(value, base)


def read_json(path: Path) -> dict[str, Any]:
    """Load and validate a JSON object."""
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    if not isinstance(record, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return record


def route_captures(
    scan_manifest: dict[str, Any], route_name: str
) -> list[dict[str, Any]]:
    """Return successful capture records in the selected refined-route order."""
    route = scan_manifest.get("routes", {}).get(route_name)
    if not isinstance(route, dict) or not isinstance(route.get("sample_indices"), list):
        raise ValueError(f"scan manifest has no usable route {route_name!r}")
    captures = {
        int(capture["sample_index"]): capture
        for capture in scan_manifest.get("captures", [])
        if isinstance(capture, dict)
        and capture.get("status") == "captured"
        and "sample_index" in capture
    }
    ordered = []
    for sample_index in map(int, route["sample_indices"]):
        capture = captures.get(sample_index)
        if capture is None:
            print(f"warning: sample {sample_index} has no captured record; skipped")
            continue
        ordered.append(capture)
    if not ordered:
        raise ValueError(f"route {route_name!r} contains no captured records")
    return ordered


def first_view_classes(
    scan_manifest_path: Path, captures: Sequence[dict[str, Any]]
) -> set[str]:
    """Return canonical object classes visible in the first route capture."""
    first_capture = captures[0]
    sample_index = int(first_capture["sample_index"])
    segmentation = first_capture.get("segmentation")
    if not isinstance(segmentation, dict) or segmentation.get("status") != "saved":
        raise ValueError(
            f"first route capture (sample {sample_index}) has no saved segmentation"
        )
    manifest_value = segmentation.get("manifest")
    if not isinstance(manifest_value, str):
        raise ValueError(
            f"first route capture (sample {sample_index}) has no segmentation manifest"
        )

    segmentation_path = resolve_path(manifest_value, scan_manifest_path.parent)
    document = read_json(segmentation_path)
    classes = {
        label_from_object(item)
        for item in document.get("objects", [])
        if isinstance(item, dict)
    }
    classes.discard("")
    if not classes:
        raise ValueError(
            f"first-view segmentation contains no object classes: {segmentation_path}"
        )
    return classes


def records_from_segmentation_manifest(
    segmentation_manifest: Path,
    frame_key: int,
    selected_classes: set[str] | None,
) -> list[MaskRecord]:
    """Load object masks referenced by one Isaac-compatible manifest."""
    document = read_json(segmentation_manifest)
    records = []
    for item in document.get("objects", []):
        if not isinstance(item, dict) or not isinstance(item.get("mask"), str):
            continue
        class_name = label_from_object(item)
        if selected_classes is not None and class_name not in selected_classes:
            continue
        mask_path = resolve_path(item["mask"], segmentation_manifest.parent)
        if not mask_path.is_file():
            print(f"warning: missing mask {mask_path}; annotation skipped")
            continue
        confidence = float(item.get("confidence", item.get("score", 1.0)))
        if not np.isfinite(confidence):
            raise ValueError(f"non-finite confidence in {segmentation_manifest}")
        records.append(
            MaskRecord(
                frame_key=frame_key,
                class_name=class_name,
                instance=str(item.get("instance", class_name)),
                mask_path=mask_path,
                confidence=confidence,
            )
        )
    return records


def load_ground_truth(
    scan_manifest: dict[str, Any],
    scan_manifest_path: Path,
    captures: Sequence[dict[str, Any]],
    selected_classes: set[str] | None,
) -> list[MaskRecord]:
    """Load ground truth using each capture's saved Isaac segmentation manifest."""
    records = []
    for capture in captures:
        sample_index = int(capture["sample_index"])
        segmentation = capture.get("segmentation")
        if not isinstance(segmentation, dict) or segmentation.get("status") != "saved":
            print(f"warning: sample {sample_index} has no ground-truth segmentation")
            continue
        manifest_value = segmentation.get("manifest")
        if not isinstance(manifest_value, str):
            print(f"warning: sample {sample_index} has no segmentation manifest path")
            continue
        path = resolve_path(manifest_value, scan_manifest_path.parent)
        records.extend(
            records_from_segmentation_manifest(path, sample_index, selected_classes)
        )
    if not records:
        raise ValueError("no ground-truth masks were found for the selected route")
    return records


def discover_prediction_manifests(prediction_root: Path) -> list[Path]:
    """Find per-frame prediction manifests while excluding the run summary."""
    paths = [
        path
        for path in prediction_root.rglob("manifest.json")
        if any(part.startswith("segment_") for part in path.parts)
    ]
    if not paths:
        raise ValueError(f"no prediction manifests found under {prediction_root}")
    return sorted(paths)


def segment_path_index(path: Path) -> int | None:
    """Extract a path index from a segment_<index> ancestor."""
    for part in reversed(path.parts):
        match = re.fullmatch(r"segment_(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def load_predictions(
    prediction_root: Path,
    captures: Sequence[dict[str, Any]],
    selected_classes: set[str] | None,
) -> list[MaskRecord]:
    """Load predictions and map their path indices back to sample indices."""
    path_to_sample = {
        int(capture.get("path_index", position)): int(capture["sample_index"])
        for position, capture in enumerate(captures)
    }
    records = []
    for path in discover_prediction_manifests(prediction_root):
        document = read_json(path)
        sample_value = document.get("sample_index")
        if sample_value is not None:
            frame_key = int(sample_value)
        else:
            path_index = document.get("path_index", segment_path_index(path))
            if path_index is None or int(path_index) not in path_to_sample:
                print(f"warning: cannot associate prediction manifest {path}; skipped")
                continue
            frame_key = path_to_sample[int(path_index)]
        records.extend(
            records_from_segmentation_manifest(path, frame_key, selected_classes)
        )
    return records


def load_mask(record: MaskRecord, cache: dict[Path, np.ndarray]) -> np.ndarray:
    """Load one binary mask once and retain it for all IoU thresholds."""
    if record.mask_path not in cache:
        mask = cv2.imread(str(record.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"cannot read mask: {record.mask_path}")
        cache[record.mask_path] = mask > 0
    return cache[record.mask_path]


def mask_iou(
    prediction: MaskRecord,
    ground_truth: MaskRecord,
    cache: dict[Path, np.ndarray],
) -> float:
    """Calculate binary-mask intersection over union with shape validation."""
    predicted_mask = load_mask(prediction, cache)
    ground_truth_mask = load_mask(ground_truth, cache)
    if predicted_mask.shape != ground_truth_mask.shape:
        raise ValueError(
            f"mask shape mismatch: {prediction.mask_path} {predicted_mask.shape} "
            f"versus {ground_truth.mask_path} {ground_truth_mask.shape}"
        )
    intersection = np.count_nonzero(predicted_mask & ground_truth_mask)
    union = np.count_nonzero(predicted_mask | ground_truth_mask)
    return float(intersection / union) if union else 1.0


def mask_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return an exclusive-maximum ``(x1, y1, x2, y2)`` mask bounding box."""
    rows, columns = np.nonzero(mask)
    if not len(columns):
        return None
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


def box_iou(
    prediction: MaskRecord,
    ground_truth: MaskRecord,
    cache: dict[Path, np.ndarray],
) -> float:
    """Calculate IoU between boxes tightly enclosing two binary masks."""
    predicted_mask = load_mask(prediction, cache)
    ground_truth_mask = load_mask(ground_truth, cache)
    if predicted_mask.shape != ground_truth_mask.shape:
        raise ValueError(
            f"mask shape mismatch: {prediction.mask_path} {predicted_mask.shape} "
            f"versus {ground_truth.mask_path} {ground_truth_mask.shape}"
        )
    predicted_box = mask_box(predicted_mask)
    ground_truth_box = mask_box(ground_truth_mask)
    if predicted_box is None or ground_truth_box is None:
        return 0.0

    px1, py1, px2, py2 = predicted_box
    gx1, gy1, gx2, gy2 = ground_truth_box
    intersection_width = max(0, min(px2, gx2) - max(px1, gx1))
    intersection_height = max(0, min(py2, gy2) - max(py1, gy1))
    intersection = intersection_width * intersection_height
    predicted_area = (px2 - px1) * (py2 - py1)
    ground_truth_area = (gx2 - gx1) * (gy2 - gy1)
    union = predicted_area + ground_truth_area - intersection
    return float(intersection / union) if union else 0.0


def interpolated_ap(
    recalls: np.ndarray, precisions: np.ndarray
) -> float:
    """Calculate COCO's 101-point interpolated average precision."""
    values = []
    for recall_level in np.linspace(0.0, 1.0, 101):
        eligible = precisions[recalls >= recall_level]
        values.append(float(eligible.max()) if len(eligible) else 0.0)
    return float(np.mean(values))


def class_ap(
    ground_truth: Sequence[MaskRecord],
    predictions: Sequence[MaskRecord],
    threshold: float,
    cache: dict[Path, np.ndarray],
    iou_metric: Callable[
        [MaskRecord, MaskRecord, dict[Path, np.ndarray]], float
    ] = mask_iou,
) -> float:
    """Evaluate one class at one IoU threshold with per-frame greedy matching."""
    ground_truth_by_frame: dict[int, list[MaskRecord]] = defaultdict(list)
    for record in ground_truth:
        ground_truth_by_frame[record.frame_key].append(record)
    matched: dict[int, set[int]] = defaultdict(set)

    # Stable secondary keys make equal-confidence predictions reproducible.
    ordered_predictions = sorted(
        predictions,
        key=lambda record: (
            -record.confidence,
            record.frame_key,
            record.instance,
            str(record.mask_path),
        ),
    )
    true_positive = []
    false_positive = []
    for prediction in ordered_predictions:
        candidates = ground_truth_by_frame.get(prediction.frame_key, [])
        available = [
            (index, target)
            for index, target in enumerate(candidates)
            if index not in matched[prediction.frame_key]
        ]
        scored = [
            (iou_metric(prediction, target, cache), index)
            for index, target in available
        ]
        best_iou, best_index = max(scored, default=(0.0, -1))
        is_match = best_iou >= threshold
        true_positive.append(float(is_match))
        false_positive.append(float(not is_match))
        if is_match:
            matched[prediction.frame_key].add(best_index)

    if not ordered_predictions:
        return 0.0
    cumulative_tp = np.cumsum(true_positive)
    cumulative_fp = np.cumsum(false_positive)
    recalls = cumulative_tp / len(ground_truth)
    precisions = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1.0)
    return interpolated_ap(recalls, precisions)


def evaluate(
    ground_truth: Sequence[MaskRecord], predictions: Sequence[MaskRecord]
) -> dict[str, Any]:
    """Compute per-class and macro mask/box AP50 and AP50:95."""
    classes = sorted({record.class_name for record in ground_truth})
    cache: dict[Path, np.ndarray] = {}
    per_class = {}
    for class_name in classes:
        class_ground_truth = [
            record for record in ground_truth if record.class_name == class_name
        ]
        class_predictions = [
            record for record in predictions if record.class_name == class_name
        ]
        mask_threshold_scores = {}
        box_threshold_scores = {}
        for threshold in IOU_THRESHOLDS:
            key = f"{threshold:.2f}"
            mask_threshold_scores[key] = class_ap(
                class_ground_truth, class_predictions, float(threshold), cache, mask_iou
            )
            box_threshold_scores[key] = class_ap(
                class_ground_truth, class_predictions, float(threshold), cache, box_iou
            )
        per_class[class_name] = {
            "ground_truth_count": len(class_ground_truth),
            "prediction_count": len(class_predictions),
            # Legacy keys remain aliases for mask metrics.
            "AP50": mask_threshold_scores["0.50"],
            "AP50_95": float(np.mean(list(mask_threshold_scores.values()))),
            "AP_by_IoU": mask_threshold_scores,
            "mask_AP50": mask_threshold_scores["0.50"],
            "mask_AP50_95": float(np.mean(list(mask_threshold_scores.values()))),
            "mask_AP_by_IoU": mask_threshold_scores,
            "box_AP50": box_threshold_scores["0.50"],
            "box_AP50_95": float(np.mean(list(box_threshold_scores.values()))),
            "box_AP_by_IoU": box_threshold_scores,
        }

    mask_map50 = float(np.mean([per_class[name]["mask_AP50"] for name in classes]))
    mask_map50_95 = float(
        np.mean([per_class[name]["mask_AP50_95"] for name in classes])
    )
    return {
        "class_count": len(classes),
        "ground_truth_count": len(ground_truth),
        "prediction_count": len(predictions),
        # Legacy keys remain aliases for mask metrics.
        "mAP50": mask_map50,
        "mAP50_95": mask_map50_95,
        "mask_mAP50": mask_map50,
        "mask_mAP50_95": mask_map50_95,
        "box_mAP50": float(np.mean([per_class[name]["box_AP50"] for name in classes])),
        "box_mAP50_95": float(
            np.mean([per_class[name]["box_AP50_95"] for name in classes])
        ),
        "per_class": per_class,
    }


def print_report(report: dict[str, Any]) -> None:
    """Print a compact run summary and per-class AP table."""
    print(
        f"Masks: {report['prediction_count']} predicted / "
        f"{report['ground_truth_count']} ground truth; "
        f"classes: {report['class_count']}"
    )
    print(f"Mask mAP50:     {report['mask_mAP50']:.4f}")
    print(f"Mask mAP50-95:  {report['mask_mAP50_95']:.4f}")
    print(f"Box mAP50:      {report['box_mAP50']:.4f}")
    print(f"Box mAP50-95:   {report['box_mAP50_95']:.4f}")
    print(
        "\nClass                                GT  Pred  "
        "Mask50 Mask50-95   Box50  Box50-95"
    )
    for class_name, metrics in report["per_class"].items():
        print(
            f"{class_name:<35} {metrics['ground_truth_count']:>3} "
            f"{metrics['prediction_count']:>5} {metrics['mask_AP50']:>7.4f} "
            f"{metrics['mask_AP50_95']:>9.4f} {metrics['box_AP50']:>7.4f} "
            f"{metrics['box_AP50_95']:>9.4f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="scan_output/<run>/manifest.json")
    parser.add_argument(
        "--predictions",
        type=Path,
        help="prediction root (default: <run>/transfer_segment)",
    )
    parser.add_argument("--route", default="hamilton_2opt")
    parser.add_argument("--objects", nargs="+", help="optional class-name subset")
    parser.add_argument("--output", type=Path, help="optional detailed JSON report")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    manifest_path = args.manifest.expanduser().resolve()
    scan_manifest = read_json(manifest_path)
    captures = route_captures(scan_manifest, args.route)
    initial_classes = first_view_classes(manifest_path, captures)
    if args.objects:
        selected = {normalized_label(value) for value in args.objects}
        unavailable = selected - initial_classes
        if unavailable:
            raise ValueError(
                "--objects must be a subset of first-view classes; unavailable: "
                + ", ".join(sorted(unavailable))
            )
        selection_source = "explicit_objects"
    else:
        selected = initial_classes
        selection_source = "first_route_capture_segmentation"
    print(
        "Evaluating classes selected from "
        f"{selection_source}: {', '.join(sorted(selected))}"
    )
    prediction_root = args.predictions
    if prediction_root is None:
        prediction_root = manifest_path.parent / "transfer_segment"
    prediction_root = prediction_root.expanduser().resolve()

    # Load both sides before evaluation so all missing-data warnings are visible.
    ground_truth = load_ground_truth(
        scan_manifest, manifest_path, captures, selected
    )
    predictions = load_predictions(prediction_root, captures, selected)
    report = evaluate(ground_truth, predictions)
    output = args.output.expanduser().resolve() if args.output else None
    record_directory = output.parent if output is not None else manifest_path.parent
    report.update(
        {
            "scan_manifest": serialized_relative_path(
                manifest_path, record_directory
            ),
            "prediction_root": serialized_relative_path(
                prediction_root, record_directory
            ),
            "route": args.route,
            "evaluated_objects": sorted(selected),
            "object_selection_source": selection_source,
            "first_view_sample_index": int(captures[0]["sample_index"]),
            "iou_thresholds": [float(value) for value in IOU_THRESHOLDS],
            "default_prediction_confidence": 1.0,
        }
    )
    print_report(report)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved detailed report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
