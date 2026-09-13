#!/usr/bin/env python3
"""Annotate a scan route by propagating first-view masks with SAM 2 video.

This baseline derives one median foreground point from every saved segmentation
mask in route frame zero and uses those points as SAM 2 video prompts. It tracks
all seeded instances jointly and writes the same
per-frame manifest layout as the transfer and naive-VLM annotation methods.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import transfer_annotations_test as transfer  # noqa: E402


def output_masks(
    processor: Any,
    output: Any,
    height: int,
    width: int,
    apply_non_overlapping_constraints: bool,
) -> dict[int, tuple[np.ndarray, float]]:
    """Convert one SAM 2 video output to object-ID-keyed full-size masks."""
    masks = processor.post_process_masks(
        [output.pred_masks],
        original_sizes=[[height, width]],
        binarize=True,
        apply_non_overlapping_constraints=apply_non_overlapping_constraints,
    )[0]
    object_ids = [int(value) for value in output.object_ids]
    logits = output.object_score_logits.detach().float().cpu().reshape(-1)
    confidences = torch.sigmoid(logits).numpy()
    result = {}
    for index, object_id in enumerate(object_ids):
        mask = masks[index].detach().cpu().numpy().squeeze().astype(bool)
        confidence = float(confidences[index]) if index < len(confidences) else 1.0
        result[object_id] = (mask, confidence)
    return result


def median_mask_point(seed: dict[str, Any]) -> tuple[float, float]:
    """Return the median foreground pixel of one absolute Isaac seed mask."""
    rows, columns = np.nonzero(np.asarray(seed["mask_array"], dtype=bool))
    if not len(columns):
        raise ValueError(f"seed mask is empty: {seed.get('label', 'unknown')}")
    return float(np.median(columns)), float(np.median(rows))


def tracked_annotation(
    seed: dict[str, Any],
    object_id: int,
    mask: np.ndarray,
    confidence: float,
    source: str,
    prompt_point: tuple[float, float] | None = None,
) -> transfer.MaskAnnotation:
    """Attach stable seed metadata to one propagated SAM 2 mask."""
    label = str(seed["label"])
    return transfer.MaskAnnotation(
        label=label,
        instance=str(seed.get("instance", f"object_{object_id:03d}_{transfer.slug(label)}")),
        prim_path=str(seed.get("prim_path", "")),
        segmentation_id=int(seed.get("segmentation_id", object_id)),
        mask=mask,
        source=source,
        confidence=confidence,
        prompt_point_xy=prompt_point,
    )


def run(args: argparse.Namespace) -> Path:
    """Load one scan, propagate its initial masks, and save baseline annotations."""
    total_started = time.perf_counter()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = transfer.load_scan_manifest(manifest_path)
    frames = transfer.load_frames(manifest, manifest_path, args.route)
    wanted = {transfer.slug(value) for value in args.objects} if args.objects else None
    seeds = transfer.load_simulation_seeds(frames[0], wanted)
    video = [Image.open(frame.color_path).convert("RGB") for frame in frames]
    width, height = video[0].size
    if any(image.size != (width, height) for image in video):
        raise ValueError("SAM 2 video requires every scan frame to have one resolution")

    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else manifest_path.parent / "baseline_segment" / "sam2_video"
    )
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite to replace it: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    from transformers import Sam2VideoModel, Sam2VideoProcessor

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Loading SAM 2 video: {args.model} on {device} ({dtype})")
    model = Sam2VideoModel.from_pretrained(args.model, dtype=dtype).to(device).eval()
    processor = Sam2VideoProcessor.from_pretrained(args.model)

    # Store video pixels on the requested device while model computation remains
    # on ``device``; CPU storage substantially lowers long-route GPU memory use.
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
    seed_points = [median_mask_point(seed) for seed in seeds]
    processor.add_inputs_to_inference_session(
        inference_session=session,
        frame_idx=0,
        obj_ids=object_ids,
        input_points=[[[[x, y]] for x, y in seed_points]],
        input_labels=[[[1] for _ in seed_points]],
    )

    # Segment view zero from median points, then encode those predictions into
    # video memory before following the camera trajectory.
    with torch.inference_mode():
        initial_output = model(inference_session=session, frame_idx=0)
        propagated = model.propagate_in_video_iterator(
            session, start_frame_idx=0, show_progress_bar=args.show_progress
        )
        predictions = {
            0: output_masks(
                processor,
                initial_output,
                height,
                width,
                args.apply_non_overlapping_constraints,
            )
        }
        for result in propagated:
            frame_index = int(result.frame_idx)
            if frame_index == 0:
                continue
            predictions[frame_index] = output_masks(
                processor,
                result,
                height,
                width,
                args.apply_non_overlapping_constraints,
            )
            print(f"SAM 2 video [{frame_index + 1}/{len(frames)}]")

    transfer.synchronize_device(device)
    annotation_runtime = time.perf_counter() - total_started
    camera_frame = str(manifest.get("segmentation_camera_frame", "camera"))
    annotations_by_path: dict[int, list[transfer.MaskAnnotation]] = {}
    failures = []

    # Convert tracker object IDs back to stable dataset instances and manifests.
    for frame_index, frame in enumerate(frames):
        annotations = []
        for object_id, seed in zip(object_ids, seeds):
            prediction = predictions.get(frame_index, {}).get(object_id)
            if prediction is None:
                failures.append(
                    {"path_index": frame.path_index, "sample_index": frame.sample_index,
                     "object": seed["label"], "reason": "tracker returned no object"}
                )
                continue
            mask, confidence = prediction
            if int(mask.sum()) < args.minimum_mask_pixels:
                failures.append(
                    {"path_index": frame.path_index, "sample_index": frame.sample_index,
                     "object": seed["label"], "reason": "mask below minimum pixels"}
                )
                continue
            annotation = tracked_annotation(
                seed,
                object_id,
                mask,
                confidence,
                "sam2_video_point_seed" if frame_index == 0 else "sam2_video",
                seed_points[object_id - 1] if frame_index == 0 else None,
            )
            annotations.append(annotation)
        annotations_by_path[frame.path_index] = annotations
        transfer.save_frame_masks(
            output, frame, camera_frame, annotations, (width, height)
        )

    visualization_directory = None
    if args.save_visualizations:
        visualization_directory = transfer.save_annotation_visualizations(
            output, frames, annotations_by_path
        )
    summary = {
        "method": "sam2_video",
        "source_manifest": transfer.serialized_relative_path(manifest_path, output),
        "route": args.route,
        "frame_count": len(frames),
        "objects": [str(seed["label"]) for seed in seeds],
        "model": args.model,
        "initialization": "median_points_from_first_route_capture_simulation_masks",
        "seed_points_xy": {
            str(seed["label"]): list(point)
            for seed, point in zip(seeds, seed_points)
        },
        "minimum_mask_pixels": args.minimum_mask_pixels,
        "apply_non_overlapping_constraints": args.apply_non_overlapping_constraints,
        "inference_state_device": args.inference_state_device,
        "video_storage_device": args.video_storage_device,
        "failures": failures,
        "visualizations": (
            None if visualization_directory is None
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
    (output / "sam2_video_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved SAM 2 video annotations to {output}")
    print(f"Runtime: {annotation_runtime:.3f} s ({annotation_runtime / len(frames):.3f} s/frame)")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="scan_output/run_N/manifest.json")
    parser.add_argument("--route", default="hamilton_2opt")
    parser.add_argument("--objects", nargs="+", help="optional first-view class subset")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default="facebook/sam2.1-hiera-small")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--inference-state-device", default="cpu")
    parser.add_argument("--video-storage-device", default="cpu")
    parser.add_argument("--vision-feature-cache-size", type=int, default=1)
    parser.add_argument("--minimum-mask-pixels", type=int, default=25)
    parser.add_argument("--apply-non-overlapping-constraints", action="store_true")
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    if args.minimum_mask_pixels < 1:
        raise ValueError("--minimum-mask-pixels must be positive")
    if args.vision_feature_cache_size < 1:
        raise ValueError("--vision-feature-cache-size must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
