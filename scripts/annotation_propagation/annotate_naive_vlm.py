#!/usr/bin/env python3
"""Annotate every scan image independently with Grounding DINO and SAM.

This is the naive vision-only baseline: it deliberately ignores depth, camera
transforms, neighboring frames, and all propagated masks. Grounding DINO finds
boxes for each requested class, and SAM produces the class segmentation.
Outputs use the same layout as geometric propagation for direct mAP comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from transfer_annotations_test import (
    MaskAnnotation,
    VisionModels,
    keep_largest_component,
    load_scan_manifest,
    object_label,
    reset_output_directory,
    resolve_record_path,
    save_annotation_visualizations,
    save_frame_masks,
    serialized_relative_path,
    slug,
    synchronize_device,
)


@dataclass(frozen=True)
class ObjectDefinition:
    """Stable class and instance metadata used in every independently scored view."""

    label: str
    instance: str
    prim_path: str
    segmentation_id: int


@dataclass(frozen=True)
class ImageFrame:
    """Only the image metadata needed by the vision-only baseline."""

    path_index: int
    sample_index: int
    color_path: Path
    segmentation_manifest: Path | None


def load_image_frames(
    manifest: dict[str, Any], manifest_path: Path, route_name: str
) -> list[ImageFrame]:
    """Load route-ordered RGB frames without requiring depth or camera TF files."""
    route = manifest.get("routes", {}).get(route_name)
    if not isinstance(route, dict) or not isinstance(route.get("sample_indices"), list):
        raise ValueError(f"manifest has no usable route {route_name!r}")
    captures = {
        int(capture["sample_index"]): capture
        for capture in manifest.get("captures", [])
        if isinstance(capture, dict)
        and capture.get("status") == "captured"
        and "sample_index" in capture
    }
    frames = []
    for route_position, sample_index in enumerate(map(int, route["sample_indices"])):
        capture = captures.get(sample_index)
        if capture is None or not isinstance(capture.get("color_image"), str):
            print(f"warning: sample {sample_index} has no captured RGB image; skipped")
            continue
        color_path = resolve_record_path(capture["color_image"], manifest_path)
        if not color_path.is_file():
            print(f"warning: missing RGB image {color_path}; sample skipped")
            continue
        segmentation = capture.get("segmentation")
        segmentation_value = (
            segmentation.get("manifest") if isinstance(segmentation, dict) else None
        )
        segmentation_manifest = (
            resolve_record_path(segmentation_value, manifest_path)
            if isinstance(segmentation_value, str)
            else None
        )
        frames.append(
            ImageFrame(
                path_index=int(capture.get("path_index", route_position)),
                sample_index=sample_index,
                color_path=color_path,
                segmentation_manifest=segmentation_manifest,
            )
        )
    if not frames:
        raise ValueError(f"route {route_name!r} contains no readable RGB frames")
    return frames


def object_definitions(
    frames: Sequence[ImageFrame], requested_objects: Sequence[str] | None
) -> list[ObjectDefinition]:
    """Resolve the evaluated class list without using later-frame ground truth."""
    requested = {slug(value) for value in requested_objects} if requested_objects else None
    initial_manifest = frames[0].segmentation_manifest
    if initial_manifest is not None and initial_manifest.is_file():
        with initial_manifest.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        definitions = []
        for record in document.get("objects", []):
            label = object_label(str(record["instance"]))
            if requested is not None and slug(label) not in requested:
                continue
            definitions.append(
                ObjectDefinition(
                    label=label,
                    instance=str(record["instance"]),
                    prim_path=str(record.get("prim_path", "")),
                    segmentation_id=int(record["segmentation_id"]),
                )
            )
        if not definitions:
            raise ValueError("initial segmentation manifest has no selected classes")
        return definitions

    # Real scans need an explicit class list; otherwise every default class would
    # require a detector query in every frame and would not define scene contents.
    if not requested_objects:
        raise ValueError(
            "real scan has no initial segmentation manifest; pass --objects"
        )
    return [
        ObjectDefinition(
            label=label,
            instance=f"object_{index:03d}_{slug(label)}",
            prim_path="",
            segmentation_id=index,
        )
        for index, label in enumerate(requested_objects, start=1)
    ]


def failure_record(
    frame: ImageFrame, definition: ObjectDefinition, reason: str
) -> dict[str, Any]:
    """Create a consistent failure record for either inference mode."""
    return {
        "path_index": frame.path_index,
        "sample_index": frame.sample_index,
        "object": definition.label,
        "reason": reason,
    }


def annotate_frame(
    frame: ImageFrame,
    objects: Sequence[ObjectDefinition],
    models: VisionModels,
    minimum_mask_pixels: int,
    detection_mode: str,
    filter_candidates: bool,
) -> tuple[list[MaskAnnotation], list[dict[str, Any]]]:
    """Run independent text-box-mask inference for all classes in one frame."""
    image = Image.open(frame.color_path).convert("RGB")
    annotations = []
    failures = []
    labels = [definition.label for definition in objects]

    if detection_mode == "zeroshot":
        # Detect all object classes together while keeping proposals class-scoped.
        detections_by_class = models.detect_class_boxes(image, labels)
    else:
        # Multi-shot runs one class-specific DINO query for each object.
        detections_by_class = {
            definition.label: models.detect_boxes(image, definition.label)
            for definition in objects
        }

    for definition in objects:
        detections = detections_by_class[definition.label]
        if not detections:
            failures.append(
                failure_record(frame, definition, "Grounding DINO found no box")
            )
            continue

        # Filter mode preserves the original baseline: keep only the strongest
        # box, its largest SAM component, and enforce minimum mask area.
        selected = (
            [max(detections, key=lambda item: item[1])]
            if filter_candidates
            else list(detections)
        )
        masks = models.segment_boxes(image, [box for box, _ in selected])
        if not masks:
            failures.append(failure_record(frame, definition, "SAM returned no mask"))
            continue
        if filter_candidates:
            mask = keep_largest_component(masks[0])
            source = "grounding_dino+sam_box_filtered"
        else:
            # No-filter mode accepts every detector box for this class and joins
            # all corresponding SAM regions into one semantic class mask.
            mask = np.logical_or.reduce(
                [np.asarray(candidate, dtype=bool) for candidate in masks]
            )
            source = "grounding_dino+sam_all_boxes_union"
        area = int(mask.sum())
        if filter_candidates and area < minimum_mask_pixels:
            failures.append(
                {
                    "path_index": frame.path_index,
                    "sample_index": frame.sample_index,
                    "object": definition.label,
                    "reason": f"SAM mask too small ({area} pixels)",
                }
            )
            continue
        annotations.append(
            MaskAnnotation(
                label=definition.label,
                instance=definition.instance,
                prim_path=definition.prim_path,
                segmentation_id=definition.segmentation_id,
                mask=mask,
                source=source,
                confidence=float(max(score for _, score in selected)),
            )
        )
    return annotations, failures


def run(args: argparse.Namespace) -> None:
    """Load one manifest route, annotate every image, and write comparable masks."""
    total_started = time.perf_counter()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_scan_manifest(manifest_path)
    frames = load_image_frames(manifest, manifest_path, args.route)
    objects = object_definitions(frames, args.objects)
    labels = [definition.label for definition in objects]
    print(
        f"Loaded {len(frames)} frames and {len(objects)} classes: "
        + ", ".join(labels)
    )
    if args.validate_only:
        return

    output = args.output_dir
    if output is None:
        mode_directory = args.detection_mode.replace("-", "_")
        output = (
            manifest_path.parent
            / "baseline_segment"
            / (
                f"naive_vlm_{mode_directory}"
                if args.filter_candidates
                else f"naive_vlm_{mode_directory}_no_filter"
            )
        )
    output = output.expanduser().resolve()
    reset_output_directory(output)
    camera_frame = str(manifest.get("segmentation_camera_frame", "camera"))
    models = VisionModels(args)
    failures: list[dict[str, Any]] = []
    annotations_by_path: dict[int, list[MaskAnnotation]] = {}
    synchronize_device(models.device)
    annotation_started = time.perf_counter()

    # Each iteration starts only from I_t and class text: no temporal state exists.
    for ordinal, frame in enumerate(frames, start=1):
        annotations, frame_failures = annotate_frame(
            frame,
            objects,
            models,
            args.minimum_mask_pixels,
            args.detection_mode,
            args.filter_candidates,
        )
        failures.extend(frame_failures)
        annotations_by_path[frame.path_index] = annotations
        save_frame_masks(
            output,
            frame,
            camera_frame,
            annotations,
            Image.open(frame.color_path).size,
        )
        print(
            f"[{ordinal}/{len(frames)}] sample {frame.sample_index}: "
            f"{len(annotations)}/{len(objects)} objects segmented"
        )

    synchronize_device(models.device)
    annotation_runtime = time.perf_counter() - annotation_started
    total_runtime = time.perf_counter() - total_started
    object_frame_count = len(frames) * len(objects)

    # Reuse transfer visualization after timing; baseline annotations have no
    # point prompts, so the shared renderer draws only masks and boundary boxes.
    visualization_directory = None
    if args.save_visualizations:
        visualization_directory = save_annotation_visualizations(
            output, frames, annotations_by_path
        )

    summary = {
        "method": "naive_grounding_dino_plus_sam",
        "detection_mode": args.detection_mode,
        "source_manifest": serialized_relative_path(manifest_path, output),
        "route": args.route,
        "frame_count": len(frames),
        "objects": [asdict(definition) for definition in objects],
        "dino_model": args.dino_model,
        "sam_model": args.sam_model,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "minimum_mask_pixels": args.minimum_mask_pixels,
        "candidate_filter_enabled": args.filter_candidates,
        "visualizations": (
            None
            if visualization_directory is None
            else serialized_relative_path(visualization_directory, output)
        ),
        "runtime": {
            "total_seconds": total_runtime,
            "annotation_seconds": annotation_runtime,
            "average_seconds_per_frame": annotation_runtime / len(frames),
            "average_seconds_per_object_frame": (
                annotation_runtime / object_frame_count
                if object_frame_count
                else 0.0
            ),
            "includes_model_loading": True,
            "includes_mask_writing": True,
            "excludes_visualization": True,
            "device": str(models.device),
        },
        "failure_count": len(failures),
        "failures": failures,
        "total_time_minutes": round(annotation_runtime / 60, 3),
    }
    (output / "naive_vlm_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved naive VLM annotations to {output}")
    print(
        f"Annotation runtime: {annotation_runtime:.3f} s "
        f"({annotation_runtime / len(frames):.3f} s/frame)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="scan_output/<run>/manifest.json")
    parser.add_argument("--route", default="hamilton_2opt")
    parser.add_argument(
        "--detection-mode",
        choices=("zeroshot", "multi-shot"),
        default="zeroshot",
        help=(
            "zeroshot detects all objects together; multi-shot detects and "
            "segments one object at a time (default: zeroshot)"
        ),
    )
    parser.add_argument("--objects", nargs="+", help="classes for a real scan")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dino-model", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--sam-model", default="facebook/sam-vit-base")
    parser.add_argument("--box-threshold", type=float, default=0.20)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--minimum-mask-pixels", type=int, default=25)
    parser.add_argument(
        "--filter-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "keep only the strongest box and filter its mask; "
            "--no-filter-candidates unions masks from every returned box"
        ),
    )
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="save mask and boundary-box overlays after runtime measurement",
    )
    # VisionModels also accepts this field, although this baseline never loads Qwen.
    parser.set_defaults(qwen_model="Qwen/Qwen3-VL-4B-Instruct")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    if not 0.0 <= args.box_threshold <= 1.0:
        raise ValueError("--box-threshold must be between zero and one")
    if not 0.0 <= args.text_threshold <= 1.0:
        raise ValueError("--text-threshold must be between zero and one")
    if args.minimum_mask_pixels < 1:
        raise ValueError("--minimum-mask-pixels must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
