#!/usr/bin/env python3
"""SAM2 video tracking with transfer-mask anchors for lost objects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import annotate_sam2 as sam2_base  # noqa: E402
import transfer_annotations_test as transfer  # noqa: E402


def transfer_manifests(root: Path) -> dict[int, Path]:
    """Index transfer frame manifests by scan sample index."""
    result: dict[int, Path] = {}
    for path in sorted(root.rglob("manifest.json")):
        if not any(part.startswith("segment_") for part in path.parts):
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("sample_index") is not None:
            result[int(document["sample_index"])] = path
    return result


def transfer_anchor_points(path: Path | None) -> dict[str, tuple[float, float]]:
    """Return median foreground points keyed by stable instance name."""
    if path is None:
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    anchors: dict[str, tuple[float, float]] = {}
    for item in document.get("objects", []):
        if not isinstance(item, dict) or not isinstance(item.get("mask"), str):
            continue
        mask_path = transfer.resolve_record_path(item["mask"], path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"warning: cannot read transfer anchor mask {mask_path}")
            continue
        rows, columns = np.nonzero(mask)
        if not len(columns):
            continue
        instance = str(item.get("instance", ""))
        if instance:
            anchors[instance] = (float(np.median(columns)), float(np.median(rows)))
    return anchors


def lost_prediction(
    prediction: tuple[np.ndarray, float] | None,
    minimum_pixels: int,
    minimum_confidence: float,
) -> bool:
    """Decide whether a SAM2 object needs a transfer-derived corrective prompt."""
    if prediction is None:
        return True
    mask, confidence = prediction
    return int(mask.sum()) < minimum_pixels or confidence < minimum_confidence


def run(args: argparse.Namespace) -> Path:
    """Track all objects and inject transfer anchors whenever SAM2 loses one."""
    total_started = time.perf_counter()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = transfer.load_scan_manifest(manifest_path)
    frames = transfer.load_frames(manifest, manifest_path, args.route)
    wanted = {transfer.slug(value) for value in args.objects} if args.objects else None
    seeds = transfer.load_simulation_seeds(frames[0], wanted)
    video = [Image.open(frame.color_path).convert("RGB") for frame in frames]
    width, height = video[0].size
    if any(image.size != (width, height) for image in video):
        raise ValueError("SAM2 video requires every scan frame to have one resolution")

    transfer_root = (
        args.transfer_dir.expanduser().resolve()
        if args.transfer_dir is not None
        else manifest_path.parent / "transfer_segment"
    )
    if not transfer_root.is_dir():
        raise FileNotFoundError(f"transfer annotation directory not found: {transfer_root}")
    anchor_manifests = transfer_manifests(transfer_root)

    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else manifest_path.parent / "baseline_segment" / "sam2_transfer_reanchor"
    )
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite to replace it: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    from transformers import Sam2VideoModel, Sam2VideoProcessor

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Loading SAM2 transfer-reanchor baseline: {args.model} on {device} ({dtype})")
    model = Sam2VideoModel.from_pretrained(args.model, dtype=dtype).to(device).eval()
    processor = Sam2VideoProcessor.from_pretrained(args.model)
    session = processor.init_video_session(
        video=video,
        inference_device=device,
        inference_state_device=args.inference_state_device,
        processing_device=device,
        video_storage_device=args.video_storage_device,
        max_vision_features_cache_size=args.vision_feature_cache_size,
        dtype=dtype,
    )

    object_ids = list(range(1, len(seeds) + 1))
    seed_points = [sam2_base.median_mask_point(seed) for seed in seeds]
    processor.add_inputs_to_inference_session(
        inference_session=session,
        frame_idx=0,
        obj_ids=list(object_ids),
        input_points=[[[[x, y]] for x, y in seed_points]],
        input_labels=[[[1] for _ in seed_points]],
    )

    predictions: dict[int, dict[int, tuple[np.ndarray, float]]] = {}
    reanchors: list[dict[str, Any]] = []
    with torch.inference_mode():
        # Run sequentially so a correction becomes video memory for later frames.
        for frame_index, frame in enumerate(frames):
            result = model(inference_session=session, frame_idx=frame_index)
            current = sam2_base.output_masks(processor, result, height, width)
            anchors = transfer_anchor_points(anchor_manifests.get(frame.sample_index))
            lost_ids = []
            corrective_points = []
            for object_id, seed in zip(object_ids, seeds):
                prediction = current.get(object_id)
                anchor = anchors.get(str(seed.get("instance", "")))
                if anchor is not None and lost_prediction(
                    prediction, args.minimum_track_pixels, args.minimum_track_confidence
                ):
                    lost_ids.append(object_id)
                    corrective_points.append(anchor)
                    reanchors.append(
                        {
                            "path_index": frame.path_index,
                            "sample_index": frame.sample_index,
                            "object_id": object_id,
                            "instance": str(seed.get("instance", "")),
                            "anchor_point_xy": list(anchor),
                            "reason": "missing_or_weak_sam2_track",
                        }
                    )

            # Re-run this frame after adding positive prompts for only lost objects.
            if lost_ids:
                processor.add_inputs_to_inference_session(
                    inference_session=session,
                    frame_idx=frame_index,
                    obj_ids=list(lost_ids),
                    input_points=[[[[x, y]] for x, y in corrective_points]],
                    input_labels=[[[1] for _ in corrective_points]],
                )
                corrected = model(inference_session=session, frame_idx=frame_index)
                current.update(sam2_base.output_masks(processor, corrected, height, width))
                print(
                    f"SAM2 hybrid [{frame_index + 1}/{len(frames)}]: "
                    f"reanchored {len(lost_ids)} object(s)"
                )
            else:
                print(f"SAM2 hybrid [{frame_index + 1}/{len(frames)}]")
            predictions[frame_index] = current

    transfer.synchronize_device(device)
    annotation_runtime = time.perf_counter() - total_started
    camera_frame = str(manifest.get("segmentation_camera_frame", "camera"))
    annotations_by_path: dict[int, list[transfer.MaskAnnotation]] = {}
    failures = []

    # Preserve all SAM2 predictions; transfer masks supply prompts, not output masks.
    for frame_index, frame in enumerate(frames):
        annotations = []
        corrected_ids = {
            int(item["object_id"])
            for item in reanchors
            if int(item["path_index"]) == frame.path_index
        }
        for object_id, seed in zip(object_ids, seeds):
            prediction = predictions.get(frame_index, {}).get(object_id)
            if prediction is None:
                failures.append(
                    {
                        "path_index": frame.path_index,
                        "sample_index": frame.sample_index,
                        "object": seed["label"],
                        "reason": "tracker returned no object and no usable transfer anchor",
                    }
                )
                continue
            mask, confidence = prediction
            annotations.append(
                sam2_base.tracked_annotation(
                    seed,
                    object_id,
                    mask,
                    confidence,
                    "sam2_transfer_reanchor" if object_id in corrected_ids else "sam2_video",
                    None,
                )
            )
        annotations_by_path[frame.path_index] = annotations
        sam2_base.save_unfiltered_frame_masks(
            output, frame, camera_frame, annotations, (width, height)
        )

    visualization_directory = None
    if args.save_visualizations:
        visualizable = {
            index: [item for item in items if np.any(item.mask)]
            for index, items in annotations_by_path.items()
        }
        visualization_directory = transfer.save_annotation_visualizations(
            output, frames, visualizable
        )

    summary = {
        "method": "sam2_video_with_transfer_reanchoring",
        "source_manifest": transfer.serialized_relative_path(manifest_path, output),
        "transfer_source": transfer.serialized_relative_path(transfer_root, output),
        "route": args.route,
        "frame_count": len(frames),
        "objects": [str(seed["label"]) for seed in seeds],
        "model": args.model,
        "initialization": "first_view_simulation_mask_median_points",
        "reanchor_policy": {
            "minimum_track_pixels": args.minimum_track_pixels,
            "minimum_track_confidence": args.minimum_track_confidence,
            "anchor": "median_foreground_point_of_same-frame_transfer_mask",
        },
        "reanchor_count": len(reanchors),
        "reanchors": reanchors,
        "failures": failures,
        "visualizations": (
            None
            if visualization_directory is None
            else transfer.serialized_relative_path(visualization_directory, output)
        ),
        "runtime": {
            "total_seconds": annotation_runtime,
            "average_seconds_per_frame": annotation_runtime / len(frames),
            "includes_model_loading": True,
            "excludes_visualization": True,
            "device": str(device),
            "dtype": str(dtype),
        },
    }
    (output / "sam2_transfer_reanchor_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved SAM2 transfer-reanchor annotations to {output}")
    print(f"Reanchored {len(reanchors)} lost object track(s)")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--route", default="hamilton_2opt")
    parser.add_argument("--objects", nargs="+")
    parser.add_argument("--transfer-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default="facebook/sam2.1-hiera-small")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--inference-state-device", default="cpu")
    parser.add_argument("--video-storage-device", default="cpu")
    parser.add_argument("--vision-feature-cache-size", type=int, default=1)
    parser.add_argument("--minimum-track-pixels", type=int, default=25)
    parser.add_argument("--minimum-track-confidence", type=float, default=0.5)
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    if args.minimum_track_pixels < 0:
        raise ValueError("--minimum-track-pixels cannot be negative")
    if not 0.0 <= args.minimum_track_confidence <= 1.0:
        raise ValueError("--minimum-track-confidence must be in [0, 1]")
    if args.vision_feature_cache_size < 1:
        raise ValueError("--vision-feature-cache-size must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
