#!/usr/bin/env python3
"""Annotate every scan image independently with Grounding DINO and SAM.

This is the naive vision-only baseline: it deliberately ignores depth, camera
transforms, neighboring frames, and all propagated masks. Grounding DINO finds
one box per requested object class in each image, and SAM segments that box.
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
    save_frame_masks,
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
) -> tuple[list[MaskAnnotation], list[dict[str, Any]]]:
    """Run independent text-box-mask inference for all classes in one frame."""
    image = Image.open(frame.color_path).convert("RGB")
    annotations = []
    failures = []
    labels = [definition.label for definition in objects]

    if detection_mode == "zeroshot":
        # Detect all object classes together, then segment all boxes together.
        detections_by_class = models.detect_class_boxes(image, labels)
        selected = []
        for definition in objects:
            detections = detections_by_class[definition.label]
            if not detections:
                failures.append(
                    failure_record(
                        frame, definition, "Grounding DINO found no box"
                    )
                )
                continue
            box, confidence = max(detections, key=lambda item: item[1])
            selected.append((definition, confidence, box))
        masks = models.segment_boxes(image, [item[2] for item in selected])
        candidates = [
            (definition, confidence, mask)
            for (definition, confidence, _), mask in zip(selected, masks)
        ]
    else:
        # Detect and segment exactly one object at a time, like text fallback.
        candidates = []
        for definition in objects:
            detections = models.detect_boxes(image, definition.label)
            if not detections:
                failures.append(
                    failure_record(
                        frame, definition, "Grounding DINO found no box"
                    )
                )
                continue
            box, confidence = max(detections, key=lambda item: item[1])
            candidates.append(
                (definition, confidence, models.segment_box(image, box))
            )

    for definition, confidence, raw_mask in candidates:
        mask = keep_largest_component(raw_mask)
        area = int(mask.sum())
        if area < minimum_mask_pixels:
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
                source="grounding_dino+sam_box",
                confidence=float(confidence),
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
        suffix = str(manifest.get("output_suffix", "scan"))
        output = manifest_path.parent / f"naive_vlm_segment_{suffix}"
    output = output.expanduser().resolve()
    reset_output_directory(output)
    camera_frame = str(manifest.get("segmentation_camera_frame", "camera"))
    models = VisionModels(args)
    failures: list[dict[str, Any]] = []
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
        )
        failures.extend(frame_failures)
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

    summary = {
        "method": "naive_grounding_dino_plus_sam",
        "detection_mode": args.detection_mode,
        "source_manifest": str(manifest_path),
        "route": args.route,
        "frame_count": len(frames),
        "objects": [asdict(definition) for definition in objects],
        "dino_model": args.dino_model,
        "sam_model": args.sam_model,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "minimum_mask_pixels": args.minimum_mask_pixels,
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
    parser.add_argument("manifest", type=Path, help="scan_output/manifest_<run>.json")
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
