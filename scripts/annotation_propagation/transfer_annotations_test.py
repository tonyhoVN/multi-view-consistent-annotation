#!/usr/bin/env python3
"""Propagate an initial object mask through a recorded camera trajectory.

The scan manifest is the authority for frame order and file association. The
algorithm projects the previous segmented point cloud, prompts SAM at the
median projected pixel, falls back to Grounding DINO plus SAM when area is
implausible, and finally enforces 3-D centroid consistency.
Simulation runs use the Isaac segmentation of captured view zero as the seed;
real runs use Qwen3-VL bounding boxes followed by SAM.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
import torch
import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Reuse the project's point-cloud conversion rather than maintaining a copy.
from scene_reconstruction.o3d_process import (
    pixels_to_point_cloud,
    remove_outlier_o3d,
    rgbd_to_pcd_mask,
    trim_point_cloud_above_plane,
)
from scan_layout import resolve_saved_path, serialized_relative_path  # noqa: E402


DEFAULT_OBJECTS = (
    "Apple", "Banana", "Brick", "Biscuit_Box", "Hammer", "Clamp", "Cup",
    "Mug", "Mustard_Bottle", "Pear", "Power_Drill", "Screwdriver",
    "Scissor", "Strawberry", "Tennis_Ball", "Soup_Can",
)


@dataclass(frozen=True)
class Frame:
    """Files and camera pose for one successful capture in route order."""

    path_index: int
    sample_index: int
    color_path: Path
    depth_path: Path
    transform_path: Path
    transform: np.ndarray
    segmentation_manifest: Path | None


@dataclass
class ObjectState:
    """Previous accepted mask/cloud plus the best accepted area reference."""

    label: str
    instance: str
    prim_path: str
    segmentation_id: int
    mask: np.ndarray
    point_cloud: o3d.geometry.PointCloud
    best_mask: np.ndarray
    best_area: int


@dataclass(frozen=True)
class MaskAnnotation:
    """Immutable accepted output for one object in one camera frame."""

    label: str
    instance: str
    prim_path: str
    segmentation_id: int
    mask: np.ndarray
    source: str
    confidence: float = 1.0
    prompt_point_xy: tuple[float, float] | None = None


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole calibration and Open3D representation."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def open3d(self) -> o3d.camera.PinholeCameraIntrinsic:
        return o3d.camera.PinholeCameraIntrinsic(
            self.width, self.height, self.fx, self.fy, self.cx, self.cy
        )


def resolve_record_path(value: str, manifest_path: Path) -> Path:
    """Resolve a path stored relative to the scan manifest directory."""
    return resolve_saved_path(value, manifest_path.parent)


def load_scan_manifest(path: Path) -> dict[str, Any]:
    """Load a scan manifest and validate its frame-bearing fields."""
    manifest_path = path.expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest.get("routes"), dict):
        raise ValueError("scan manifest has no routes object")
    if not isinstance(manifest.get("captures"), list):
        raise ValueError("scan manifest has no captures list")
    return manifest


def load_frames(
    manifest: dict[str, Any], manifest_path: Path, route_name: str
) -> list[Frame]:
    """Join captures to the selected manifest route without filename sorting."""
    route = manifest["routes"].get(route_name)
    if not isinstance(route, dict) or not isinstance(route.get("sample_indices"), list):
        raise ValueError(f"manifest has no usable route {route_name!r}")
    captures = {
        int(capture["sample_index"]): capture
        for capture in manifest["captures"]
        if isinstance(capture, dict)
        and capture.get("status") == "captured"
        and "sample_index" in capture
    }

    frames: list[Frame] = []
    for route_position, raw_index in enumerate(route["sample_indices"]):
        sample_index = int(raw_index)
        capture = captures.get(sample_index)
        if capture is None:
            print(f"Skipping sample {sample_index}: no successful capture")
            continue
        try:
            color = resolve_record_path(capture["color_image"], manifest_path)
            depth = resolve_record_path(capture["depth_image"], manifest_path)
            transform_path = resolve_record_path(
                capture["camera_transform"], manifest_path
            )
        except (KeyError, TypeError) as error:
            print(f"Skipping sample {sample_index}: incomplete file record ({error})")
            continue
        missing = [path for path in (color, depth, transform_path) if not path.is_file()]
        if missing:
            print(f"Skipping sample {sample_index}: missing {missing[0]}")
            continue
        transform = np.asarray(np.load(transform_path), dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            print(f"Skipping sample {sample_index}: invalid camera transform")
            continue
        segmentation = capture.get("segmentation", {})
        segmentation_value = (
            segmentation.get("manifest") if isinstance(segmentation, dict) else None
        )
        segmentation_path = (
            resolve_record_path(segmentation_value, manifest_path)
            if isinstance(segmentation_value, str)
            else None
        )
        frames.append(
            Frame(
                path_index=int(capture.get("path_index", route_position)),
                sample_index=sample_index,
                color_path=color,
                depth_path=depth,
                transform_path=transform_path,
                transform=transform,
                segmentation_manifest=segmentation_path,
            )
        )
    if not frames:
        raise ValueError(f"route {route_name!r} contains no complete captured frames")
    if frames[0].sample_index != 0:
        raise ValueError("route must begin with captured initial view sample 0")
    return frames


def load_intrinsics(camera_yaml: Path) -> Intrinsics:
    """Load this project's camera.yaml intrinsic schema."""
    with camera_yaml.expanduser().open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    intrinsics = data["intrinsics"]
    width, height = map(int, intrinsics["resolution"])
    return Intrinsics(
        width=width,
        height=height,
        fx=float(intrinsics["fx"]),
        fy=float(intrinsics["fy"]),
        cx=float(intrinsics["cx"]),
        cy=float(intrinsics["cy"]),
    )


def camera_yaml_from_manifest(manifest: dict[str, Any], manifest_path: Path) -> Path:
    """Read camera_yaml from the scan configuration referenced by the manifest."""
    config = resolve_record_path(str(manifest["configuration"]), manifest_path)
    with config.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    value = document["single_path_planning"]["camera_yaml"]
    path = Path(value).expanduser()
    return path if path.is_absolute() else config.parent / path


def object_label(instance: str) -> str:
    """Convert an Isaac instance name into a stable human-readable label."""
    label = re.sub(r"^object_\d+_(?:\d+_)?", "", instance)
    return label or instance


def slug(value: str) -> str:
    """Return a filesystem-safe object identifier."""
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return normalized or "object"


def load_simulation_seeds(
    frame: Frame, wanted_objects: set[str] | None
) -> list[dict[str, Any]]:
    """Load initial masks and object metadata from Isaac segmentation output."""
    path = frame.segmentation_manifest
    if path is None or not path.is_file():
        raise ValueError("initial view has no saved Isaac segmentation manifest")
    with path.open("r", encoding="utf-8") as stream:
        segmentation = json.load(stream)
    seeds = []
    for record in segmentation.get("objects", []):
        label = object_label(str(record["instance"]))
        if wanted_objects and slug(label) not in wanted_objects:
            continue
        mask_path = path.parent / record["mask"]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or not np.any(mask):
            print(f"Skipping empty simulation seed: {mask_path}")
            continue
        seeds.append({**record, "label": label, "mask_array": mask > 0})
    if not seeds:
        raise ValueError("Isaac segmentation manifest contains no selected valid masks")
    return seeds


def parse_json_response(text: str) -> Any:
    """Parse JSON from a plain or fenced VLM response."""
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    return json.loads(stripped)


class VisionModels:
    """Lazy Qwen3-VL, Grounding DINO, and SAM inference wrappers."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)
        self._dino_processor = None
        self._dino_model = None
        self._sam_processor = None
        self._sam_model = None
        self._qwen_processor = None
        self._qwen_model = None

    def _load_sam(self) -> None:
        if self._sam_model is not None:
            return
        from transformers import SamModel, SamProcessor

        print(f"Loading SAM: {self.args.sam_model}")
        self._sam_processor = SamProcessor.from_pretrained(self.args.sam_model)
        self._sam_model = SamModel.from_pretrained(self.args.sam_model)
        self._sam_model.to(self.device).eval()

    def _load_dino(self) -> None:
        if self._dino_model is not None:
            return
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        print(f"Loading Grounding DINO: {self.args.dino_model}")
        self._dino_processor = AutoProcessor.from_pretrained(self.args.dino_model)
        self._dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.args.dino_model
        ).to(self.device).eval()

    def _load_qwen(self) -> None:
        if self._qwen_model is not None:
            return
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        print(f"Loading Qwen3-VL: {self.args.qwen_model}")
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self._qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.args.qwen_model, dtype=dtype, device_map=str(self.device)
        ).eval()
        self._qwen_processor = AutoProcessor.from_pretrained(self.args.qwen_model)

    def _sam_masks(self, inputs: Any) -> list[np.ndarray]:
        """Run SAM once and select the best mask for every prompt batch."""
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self._sam_model(**inputs)
        masks = self._sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]
        scores = outputs.iou_scores.detach().cpu()[0]
        if masks.ndim == 3:
            masks = masks.unsqueeze(0)
        if scores.ndim == 1:
            scores = scores.unsqueeze(0)
        selected = []
        for prompt_index in range(masks.shape[0]):
            best = int(torch.argmax(scores[prompt_index]))
            selected.append(masks[prompt_index, best].numpy().astype(bool))
        return selected

    def segment_point(self, image: Image.Image, point: Sequence[float]) -> np.ndarray:
        """Run SAM with one positive point prompt at the projected median."""
        self._load_sam()
        inputs = self._sam_processor(
            image,
            input_points=[[[float(point[0]), float(point[1])]]],
            input_labels=[[1]],
            return_tensors="pt",
        )
        return self._sam_masks(inputs)[0]

    def segment_box(self, image: Image.Image, box: Sequence[float]) -> np.ndarray:
        """Run SAM for one Grounding-DINO pixel-coordinate bounding box."""
        self._load_sam()
        inputs = self._sam_processor(
            image, input_boxes=[[[float(value) for value in box]]], return_tensors="pt"
        )
        return self._sam_masks(inputs)[0]

    def segment_boxes(
        self, image: Image.Image, boxes: Sequence[Sequence[float]]
    ) -> list[np.ndarray]:
        """Segment multiple DINO boxes with one shared SAM image encoding."""
        if not boxes:
            return []
        self._load_sam()
        input_boxes = [
            [[float(value) for value in box] for box in boxes]
        ]
        inputs = self._sam_processor(
            image, input_boxes=input_boxes, return_tensors="pt"
        )
        return self._sam_masks(inputs)

    def detect_boxes(self, image: Image.Image, label: str) -> list[tuple[np.ndarray, float]]:
        """Return Grounding-DINO boxes and scores for one label."""
        return self.detect_class_boxes(image, [label])[label]

    def detect_class_boxes(
        self, image: Image.Image, labels: Sequence[str]
    ) -> dict[str, list[tuple[np.ndarray, float]]]:
        """Detect multiple class prompts with one Grounding-DINO image pass."""
        self._load_dino()
        if not labels:
            return {}
        prompt_labels = [label.replace("_", " ").lower() for label in labels]
        prompt = ". ".join(prompt_labels) + "."
        inputs = self._dino_processor(images=image, text=prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self._dino_model(**inputs)
        result = self._dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.args.box_threshold,
            text_threshold=self.args.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        grouped: dict[str, list[tuple[np.ndarray, float]]] = {
            label: [] for label in labels
        }
        normalized = {slug(label): label for label in labels}
        result_labels = result.get("text_labels", [])
        for box, score, result_label in zip(
            result["boxes"], result["scores"], result_labels
        ):
            detected = slug(str(result_label))
            matched = normalized.get(detected)
            if matched is None:
                matched = next(
                    (
                        original
                        for key, original in normalized.items()
                        if key in detected or detected in key
                    ),
                    None,
                )
            if matched is not None:
                grouped[matched].append(
                    (box.detach().cpu().numpy(), float(score))
                )
        return grouped

    def qwen_seed_boxes(
        self, image_path: Path, allowed_objects: Sequence[str]
    ) -> list[dict[str, Any]]:
        """Ask Qwen3-VL for labeled pixel boxes in the initial real image."""
        self._load_qwen()
        image = Image.open(image_path).convert("RGB")
        prompt = (
            f"Image size is {image.width}x{image.height}. Find all visible objects "
            f"from this allowed list: {list(allowed_objects)}. Return only a JSON "
            "array of objects with keys label and bbox. bbox must be integer pixel "
            "coordinates [xmin,ymin,xmax,ymax]. Return [] if none are visible."
        )
        messages = [{"role": "user", "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": prompt},
        ]}]
        inputs = self._qwen_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._qwen_model.device)
        with torch.no_grad():
            generated = self._qwen_model.generate(**inputs, max_new_tokens=512)
        trimmed = generated[:, inputs.input_ids.shape[1]:]
        response = self._qwen_processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        records = parse_json_response(response)
        if not isinstance(records, list):
            raise ValueError("Qwen3-VL seed response is not a JSON array")
        return records


def project_cloud(
    cloud: o3d.geometry.PointCloud, transform: np.ndarray, intrinsics: Intrinsics
) -> np.ndarray:
    """Project a base-frame point cloud into an image as floating-point UV pixels."""
    points = np.asarray(cloud.points, dtype=np.float64)
    if not len(points):
        return np.empty((0, 2), dtype=np.float64)
    camera_points = (
        np.linalg.inv(transform) @ np.column_stack((points, np.ones(len(points)))).T
    ).T[:, :3]
    camera_points = camera_points[camera_points[:, 2] > 0.02]
    if not len(camera_points):
        return np.empty((0, 2), dtype=np.float64)
    pixels = np.column_stack(
        (
            intrinsics.fx * camera_points[:, 0] / camera_points[:, 2] + intrinsics.cx,
            intrinsics.fy * camera_points[:, 1] / camera_points[:, 2] + intrinsics.cy,
        )
    )
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < intrinsics.width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < intrinsics.height)
    )
    return pixels[inside]


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Remove isolated SAM regions while keeping its largest foreground component."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if count <= 1:
        return mask.astype(bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def mask_bounding_box(mask: np.ndarray) -> list[int]:
    """Return the B_t^k XYXY box enclosing a non-empty accepted mask."""
    rows, columns = np.nonzero(mask)
    if not len(columns):
        raise ValueError("accepted mask is empty")
    return [
        int(columns.min()),
        int(rows.min()),
        int(columns.max()),
        int(rows.max()),
    ]


def cloud_from_mask(
    frame: Frame, mask: np.ndarray, intrinsic: o3d.camera.PinholeCameraIntrinsic
) -> o3d.geometry.PointCloud:
    """Create the segmented base-frame cloud using scene reconstruction helpers."""
    return rgbd_to_pcd_mask(
        str(frame.color_path),
        str(frame.depth_path),
        (mask.astype(np.uint8) * 255),
        frame.transform,
        intrinsic,
    )


def output_frame_directory(output: Path, frame: Frame, camera_frame: str) -> Path:
    """Mirror Isaac's per-view segmentation directory layout."""
    return output / f"segment_{frame.path_index}" / camera_frame / "capture_000000"


def save_frame_masks(
    output: Path,
    frame: Frame,
    camera_frame: str,
    annotations: Iterable[MaskAnnotation],
    resolution: tuple[int, int],
) -> None:
    """Write masks and an Isaac-compatible per-view segmentation manifest."""
    directory = output_frame_directory(output, frame, camera_frame)
    directory.mkdir(parents=True, exist_ok=True)
    annotations = list(annotations)
    objects = []
    for annotation in annotations:
        filename = f"{annotation.instance}.png"
        cv2.imwrite(
            str(directory / filename), annotation.mask.astype(np.uint8) * 255
        )
        objects.append(
            {
                "instance": annotation.instance,
                "prim_path": annotation.prim_path,
                "segmentation_id": annotation.segmentation_id,
                "visible_pixels": int(annotation.mask.sum()),
                "mask": filename,
                "bbox_xyxy": mask_bounding_box(annotation.mask),
                "label": annotation.label,
                "source": annotation.source,
                "confidence": float(annotation.confidence),
                "prompt_point_xy": (
                    None
                    if annotation.prompt_point_xy is None
                    else list(annotation.prompt_point_xy)
                ),
            }
        )
    record = {
        "camera_frame": camera_frame,
        "resolution": [int(resolution[0]), int(resolution[1])],
        "sample_index": frame.sample_index,
        "path_index": frame.path_index,
        "visible_object_count": len(objects),
        "objects": objects,
    }
    (directory / "manifest.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )


def initialize_states(
    seeds: list[dict[str, Any]],
    frame: Frame,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
) -> list[ObjectState]:
    """Convert seed masks into the state used by kinematic propagation."""
    states = []
    for index, seed in enumerate(seeds, start=1):
        mask = np.asarray(seed["mask_array"], dtype=bool)
        cloud = cloud_from_mask(frame, mask, intrinsic)
        if cloud.is_empty():
            print(f"Skipping seed {seed['label']}: its depth point cloud is empty")
            continue
        label = str(seed["label"])
        states.append(
            ObjectState(
                label=label,
                instance=str(seed.get("instance", f"object_{index:03d}_{slug(label)}")),
                prim_path=str(seed.get("prim_path", "")),
                segmentation_id=int(seed.get("segmentation_id", index)),
                mask=mask,
                point_cloud=cloud,
                best_mask=mask.copy(),
                best_area=int(mask.sum()),
            )
        )
    if not states:
        raise ValueError("no seed has both a valid mask and depth point cloud")
    return states


def create_vlm_seeds(
    frame: Frame, models: VisionModels, allowed_objects: Sequence[str]
) -> list[dict[str, Any]]:
    """Create real-robot view-zero seeds with Qwen3-VL and SAM."""
    image = Image.open(frame.color_path).convert("RGB")
    records = models.qwen_seed_boxes(frame.color_path, allowed_objects)
    allowed_slugs = {slug(value) for value in allowed_objects}
    seeds = []
    for index, record in enumerate(records, start=1):
        try:
            label = str(record["label"])
            if slug(label) not in allowed_slugs:
                print(f"Skipping Qwen label outside the allowed list: {label}")
                continue
            box = np.asarray(record["bbox"], dtype=np.float64).reshape(4)
            if not np.all(np.isfinite(box)):
                raise ValueError("bbox is not finite")
            box[[0, 2]] = np.clip(box[[0, 2]], 0, image.width - 1)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, image.height - 1)
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("bbox has no positive area")
            mask = keep_largest_component(models.segment_box(image, box))
        except (KeyError, TypeError, ValueError) as error:
            print(f"Skipping invalid Qwen seed {record!r}: {error}")
            continue
        if np.any(mask):
            seeds.append(
                {
                    "label": label,
                    "instance": f"object_{index:03d}_{slug(label)}",
                    "prim_path": "",
                    "segmentation_id": index,
                    "mask_array": mask,
                }
            )
    if not seeds:
        raise ValueError("Qwen3-VL and SAM produced no usable initial seeds")
    return seeds


def annotation_from_state(
    state: ObjectState,
    source: str,
    prompt_point: Sequence[float] | None = None,
) -> MaskAnnotation:
    """Snapshot mutable tracking state for later per-frame output."""
    prompt_xy = (
        None
        if prompt_point is None
        else (float(prompt_point[0]), float(prompt_point[1]))
    )
    return MaskAnnotation(
        label=state.label,
        instance=state.instance,
        prim_path=state.prim_path,
        segmentation_id=state.segmentation_id,
        mask=state.mask.copy(),
        source=source,
        prompt_point_xy=prompt_xy,
    )


def valid_area_ratio(
    mask: np.ndarray, state: ObjectState, args: argparse.Namespace
) -> tuple[bool, float]:
    """Evaluate rho(M_t, M_best) against configured lower and upper bounds."""
    ratio = int(mask.sum()) / max(state.best_area, 1)
    valid = (
        args.no_area_filter
        or args.minimum_area_ratio <= ratio <= args.maximum_area_ratio
    )
    return valid, ratio


def text_prompt_mask(
    state: ObjectState,
    image: Image.Image,
    models: VisionModels,
    prompt_point: Sequence[float],
    args: argparse.Namespace,
) -> np.ndarray | None:
    """Return the first text candidate passing prompt and area consistency."""
    detections = models.detect_boxes(image, state.label)
    if not detections:
        return None

    # Segment every DINO proposal. Confidence alone cannot distinguish another
    # nearby instance from the object propagated by the geometric prompt.
    boxes = [box for box, _ in detections]
    masks = models.segment_boxes(image, boxes)
    column = int(float(prompt_point[0]))
    row = int(float(prompt_point[1]))
    for (_, _confidence), raw_mask in zip(detections, masks):
        mask = keep_largest_component(raw_mask)
        height, width = mask.shape

        # Match get_validated_mask_vlm_first: the projected median must fall
        # inside this candidate and its area must agree with M_best.
        if not (0 <= column < width and 0 <= row < height):
            continue
        if not mask[row, column]:
            continue
        if int(mask.sum()) < args.minimum_mask_pixels:
            continue
        area_valid, _ = valid_area_ratio(mask, state, args)
        if area_valid:
            return mask
    return None


def clean_spatial_cloud(
    cloud: o3d.geometry.PointCloud, args: argparse.Namespace
) -> o3d.geometry.PointCloud:
    """Remove neighbor outliers and points on/below the configured table plane."""
    filtered, _ = remove_outlier_o3d(
        cloud,
        nb_neighbors=args.outlier_neighbors,
        std_ratio=args.outlier_std_ratio,
    )
    return trim_point_cloud_above_plane(
        filtered,
        args.table_origin,
        args.table_z_axis,
        args.table_clearance,
    )


def get_spatial_drift(
    reference_cloud: o3d.geometry.PointCloud,
    prompt_points: np.ndarray,
    frame: Frame,
    intrinsics: Intrinsics,
    args: argparse.Namespace,
) -> tuple[float, o3d.geometry.PointCloud | None]:
    """Measure robust pre-segmentation drift using current depth at projected UVs."""
    depth_map = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
    if depth_map is None:
        return float("inf"), None

    # Reconstruct what the previous object occupies in the current depth view.
    current_cloud = pixels_to_point_cloud(
        prompt_points,
        depth_map,
        frame.transform,
        intrinsics.open3d(),
    )
    current_cloud = clean_spatial_cloud(current_cloud, args)
    cleaned_reference = clean_spatial_cloud(reference_cloud, args)
    if current_cloud.is_empty() or cleaned_reference.is_empty():
        return float("inf"), None

    distance = float(
        np.linalg.norm(current_cloud.get_center() - cleaned_reference.get_center())
    )
    return distance, current_cloud


def propagate_to_frame(
    state: ObjectState,
    frame: Frame,
    image: Image.Image,
    models: VisionModels,
    intrinsics: Intrinsics,
    args: argparse.Namespace,
) -> tuple[bool, str, np.ndarray | None]:
    """Apply the paper algorithm once from the prior accepted view to frame t."""
    # --- 1. PROJECT THE PREVIOUS ACCEPTED OBJECT ---
    # P_prev already lives in the base frame. Project it into candidate frame t.
    pixels = project_cloud(state.point_cloud, frame.transform, intrinsics)
    if not len(pixels):
        return False, "no projected points", None

    # --- 2. SPATIAL PRE-FILTER ---
    # Reject weak projections, then back-project their current depths. Statistical
    # filtering removes isolated depth noise; the user-provided table +Z axis
    # removes the table plane before the two robust centroids are compared.
    if not args.no_spatial_filter and len(pixels) < args.minimum_projected_points:
        return False, f"only {len(pixels)} projected points", None
    prompt_point = np.median(pixels, axis=0)
    if not args.no_drift_filter:
        projected_drift, _ = get_spatial_drift(
            state.point_cloud, pixels, frame, intrinsics, args
        )
        if projected_drift > args.maximum_center_distance:
            return (
                False,
                f"projected-depth drift {projected_drift:.4f} m",
                prompt_point,
            )

    # --- 3. POINT-PROMPT SEGMENTATION ---
    # Re-prompt SAM at the median, which is robust to a minority of projected noise.
    mask = keep_largest_component(models.segment_point(image, prompt_point))
    area = int(mask.sum())
    if area < args.minimum_mask_pixels:
        point_valid, ratio = False, area / max(state.best_area, 1)
    else:
        point_valid, ratio = valid_area_ratio(mask, state, args)

    # --- 4. TEXT-PROMPT FALLBACK AND AREA VALIDATION ---
    # Retry with Grounding DINO + SAM only when the point mask is implausible.
    source = "sam_point"
    if not point_valid:
        text_mask = text_prompt_mask(
            state, image, models, prompt_point, args
        )
        if text_mask is None:
            return (
                False,
                f"point area ratio {ratio:.3f}; text detection failed",
                prompt_point,
            )
        mask = text_mask
        area = int(mask.sum())
        text_valid, ratio = valid_area_ratio(mask, state, args)
        if area < args.minimum_mask_pixels or not text_valid:
            return (
                False,
                f"text mask area ratio {ratio:.3f} outside limits",
                prompt_point,
            )
        source = "dino_text+sam_box"

    # --- 5. SEGMENTED-CLOUD SPATIAL VALIDATION ---
    # Clean P_t with the same outlier and table-plane rules used by the pre-filter.
    cloud = cloud_from_mask(frame, mask, intrinsics.open3d())
    cloud = clean_spatial_cloud(cloud, args)
    reference_cloud = clean_spatial_cloud(state.point_cloud, args)
    if cloud.is_empty() or reference_cloud.is_empty():
        return False, "segmented depth cloud is empty after filtering", prompt_point
    center_distance = float(
        np.linalg.norm(reference_cloud.get_center() - cloud.get_center())
    )
    if not args.no_drift_filter and center_distance > args.maximum_center_distance:
        return False, f"3D center drift {center_distance:.4f} m", prompt_point

    # --- 6. COMMIT THE ACCEPTED TRACKING STATE ---
    # A rejected frame never replaces M_prev or its filtered object point cloud.
    state.mask = mask
    if area > state.best_area:
        state.point_cloud = cloud
        state.best_mask = mask.copy()
        state.best_area = area
    return True, source, prompt_point


def propagate_object(
    state: ObjectState,
    frames: Sequence[Frame],
    models: VisionModels,
    intrinsics: Intrinsics,
    args: argparse.Namespace,
    seed_source: str,
) -> tuple[dict[int, MaskAnnotation], list[dict[str, Any]]]:
    """Follow Algorithm 1 for one class across the manifest-ordered trajectory."""
    accepted = {frames[0].path_index: annotation_from_state(state, seed_source)}
    failures: list[dict[str, Any]] = []

    # M_prev remains the last accepted state when a candidate frame is rejected.
    for ordinal, frame in enumerate(frames[1:], start=1):
        image = Image.open(frame.color_path).convert("RGB")
        success, detail, prompt_point = propagate_to_frame(
            state, frame, image, models, intrinsics, args
        )
        if not success:
            failures.append(
                {
                    "path_index": frame.path_index,
                    "sample_index": frame.sample_index,
                    "object": state.label,
                    "reason": detail,
                }
            )
            print(
                f"[{state.label} {ordinal}/{len(frames)-1}] rejected: {detail}"
            )
            continue
        accepted[frame.path_index] = annotation_from_state(
            state, detail, prompt_point
        )
        print(f"[{state.label} {ordinal}/{len(frames)-1}] accepted via {detail}")
    return accepted, failures


def update_object_scenes(path: Path, run_key: str, labels: Sequence[str]) -> None:
    """Create or update the run-to-object index from seed segmentation labels."""
    document: dict[str, Any] = {}
    if path.is_file():
        with path.open("r", encoding="utf-8") as stream:
            loaded = json.load(stream)
        if isinstance(loaded, dict):
            document = loaded
    document[run_key] = list(dict.fromkeys(labels))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def reset_output_directory(path: Path) -> None:
    """Replace only the explicitly selected annotation output directory."""
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"output path exists and is not a directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def synchronize_device(device: torch.device) -> None:
    """Finish queued CUDA kernels before reading a wall-clock timestamp."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def save_annotation_visualizations(
    output: Path,
    frames: Sequence[Frame],
    annotations_by_path: dict[int, list[MaskAnnotation]],
) -> Path:
    """Save mask, box, label, and point-prompt overlays on image copies."""
    visualization_directory = output / "visualizations"
    visualization_directory.mkdir(parents=True, exist_ok=True)
    palette = (
        (40, 40, 255),
        (40, 220, 40),
        (255, 120, 20),
        (220, 40, 220),
        (20, 220, 220),
    )

    for frame in frames:
        original = cv2.imread(str(frame.color_path), cv2.IMREAD_COLOR)
        if original is None:
            print(f"Visualization skipped; cannot read {frame.color_path}")
            continue
        annotations = annotations_by_path.get(frame.path_index, [])

        # Paint translucent masks on a copy; the captured color image is untouched.
        mask_layer = original.copy()
        for index, annotation in enumerate(annotations):
            color = palette[index % len(palette)]
            mask_layer[annotation.mask.astype(bool)] = color
        annotated = cv2.addWeighted(original, 0.60, mask_layer, 0.40, 0.0)

        # Draw each boundary box, label/source, and geometric point prompt last.
        for index, annotation in enumerate(annotations):
            color = palette[index % len(palette)]
            xmin, ymin, xmax, ymax = mask_bounding_box(annotation.mask)
            cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(
                annotated,
                f"{annotation.label} ({annotation.source})",
                (xmin, max(18, ymin - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )
            if annotation.prompt_point_xy is not None:
                point = tuple(
                    int(round(value)) for value in annotation.prompt_point_xy
                )
                cv2.circle(annotated, point, 6, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(annotated, point, 7, (0, 0, 0), 1, cv2.LINE_AA)

        destination = visualization_directory / f"frame_{frame.path_index:05d}.jpg"
        if not cv2.imwrite(str(destination), annotated):
            raise OSError(f"failed to save annotation visualization: {destination}")
    return visualization_directory


def run(args: argparse.Namespace) -> None:
    """Execute manifest loading, seed creation, and sequential mask propagation."""
    total_started = time.perf_counter()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_scan_manifest(manifest_path)
    frames = load_frames(manifest, manifest_path, args.route)
    camera_yaml = (
        args.camera_yaml.expanduser().resolve()
        if args.camera_yaml
        else camera_yaml_from_manifest(manifest, manifest_path)
    )
    intrinsics = load_intrinsics(camera_yaml)
    wanted = {slug(value) for value in args.objects} if args.objects else None
    simulation_available = (
        frames[0].segmentation_manifest is not None
        and frames[0].segmentation_manifest.is_file()
    )
    seed_source = args.seed_source
    if seed_source == "auto":
        seed_source = "simulation" if simulation_available else "vlm"
    print(
        f"Loaded {len(frames)} frames in {args.route!r} order; "
        f"initial seed source: {seed_source}"
    )
    if args.validate_only:
        if seed_source == "simulation":
            seeds = load_simulation_seeds(frames[0], wanted)
            print("Simulation seed objects: " + ", ".join(seed["label"] for seed in seeds))
        return

    models = VisionModels(args)
    synchronize_device(models.device)
    annotation_started = time.perf_counter()
    if seed_source == "simulation":
        seeds = load_simulation_seeds(frames[0], wanted)
    else:
        allowed = args.objects or list(DEFAULT_OBJECTS)
        seeds = create_vlm_seeds(frames[0], models, allowed)
    states = initialize_states(seeds, frames[0], intrinsics.open3d())

    output = args.output_dir
    if output is None:
        output = manifest_path.parent / "transfer_segment"
    output = output.expanduser().resolve()
    reset_output_directory(output)
    camera_frame = str(manifest.get("segmentation_camera_frame", "camera"))
    labels = [state.label for state in states]
    run_key = str(manifest.get("output_suffix", manifest_path.stem))
    object_scenes_path = args.object_scenes_json or output / "obj_scenes.json"
    update_object_scenes(object_scenes_path.expanduser().resolve(), run_key, labels)

    # The outer object loop intentionally mirrors ``for each class c_k``.
    annotations_by_path: dict[int, list[MaskAnnotation]] = {
        frame.path_index: [] for frame in frames
    }
    failures: list[dict[str, Any]] = []
    for state in states:
        accepted, object_failures = propagate_object(
            state,
            frames,
            models,
            intrinsics,
            args,
            f"{seed_source}_seed",
        )
        failures.extend(object_failures)
        for path_index, annotation in accepted.items():
            annotations_by_path[path_index].append(annotation)

    # Write combined per-frame manifests after every class has been propagated.
    for frame in frames:
        save_frame_masks(
            output,
            frame,
            camera_frame,
            annotations_by_path[frame.path_index],
            (intrinsics.width, intrinsics.height),
        )
    synchronize_device(models.device)
    annotation_runtime = time.perf_counter() - annotation_started
    total_runtime = time.perf_counter() - total_started
    object_frame_count = len(frames) * len(states)

    # Visualization is deliberately outside both recorded runtime measurements.
    visualization_directory = None
    if args.save_visualizations:
        visualization_directory = save_annotation_visualizations(
            output, frames, annotations_by_path
        )
    summary = {
        "source_manifest": serialized_relative_path(manifest_path, output),
        "route": args.route,
        "seed_source": seed_source,
        "camera_yaml": serialized_relative_path(camera_yaml, output),
        "frame_count": len(frames),
        "objects": labels,
        "criteria": {
            "minimum_projected_points": args.minimum_projected_points,
            "maximum_center_distance_m": args.maximum_center_distance,
            "minimum_area_ratio": args.minimum_area_ratio,
            "maximum_area_ratio": args.maximum_area_ratio,
            "minimum_mask_pixels": args.minimum_mask_pixels,
            "outlier_neighbors": args.outlier_neighbors,
            "outlier_std_ratio": args.outlier_std_ratio,
            "table_origin": list(args.table_origin),
            "table_z_axis": list(args.table_z_axis),
            "table_clearance_m": args.table_clearance,
            "spatial_filter_enabled": not args.no_spatial_filter,
            "drift_filter_enabled": not args.no_drift_filter,
        },
        "visualizations": (
            None
            if visualization_directory is None
            else serialized_relative_path(visualization_directory, output)
        ),
        "failures": failures,
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
        "total_time_minutes": round(annotation_runtime / 60, 3),
    }
    (output / "transfer_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved transferred segmentations to {output}")
    print(
        f"Annotation runtime: {annotation_runtime:.3f} s "
        f"({annotation_runtime / len(frames):.3f} s/frame)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="scan_output/<run>/manifest.json")
    parser.add_argument("--route", default="hamilton_2opt")
    parser.add_argument("--seed-source", choices=("auto", "simulation", "vlm"), default="auto")
    parser.add_argument("--objects", nargs="+", help="optional object labels to propagate")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--object-scenes-json", type=Path)
    parser.add_argument("--camera-yaml", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--dino-model", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--sam-model", default="facebook/sam-vit-base")
    parser.add_argument("--box-threshold", type=float, default=0.20)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument(
        "--minimum-projected-points",
        type=int,
        default=10,
        help="tau_proj: minimum projected P_prev pixels (default: 10)",
    )
    parser.add_argument(
        "--maximum-center-distance",
        type=float,
        default=0.05,
        metavar="METERS",
        help="tau_dist: maximum 3D centroid drift (default: 0.05)",
    )
    parser.add_argument("--minimum-mask-pixels", type=int, default=25)
    parser.add_argument("--minimum-area-ratio", type=float, default=0.20)
    parser.add_argument("--maximum-area-ratio", type=float, default=5.0)
    parser.add_argument("--no-area-filter", action="store_true")
    parser.add_argument(
        "--table-origin",
        type=float,
        nargs=3,
        default=(0.0, 0.0, -0.10),
        metavar=("X", "Y", "Z"),
        help="a point on the table plane in the base frame",
    )
    parser.add_argument(
        "--table-z-axis",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 1.0),
        metavar=("ZX", "ZY", "ZZ"),
        help="table-local +Z direction in the base frame",
    )
    parser.add_argument(
        "--table-clearance",
        type=float,
        default=0.0,
        metavar="METERS",
        help="keep points at least this far above the table plane",
    )
    parser.add_argument("--outlier-neighbors", type=int, default=20)
    parser.add_argument("--outlier-std-ratio", type=float, default=2.0)
    parser.add_argument("--no-spatial-filter", action="store_true")
    parser.add_argument("--no-drift-filter", action="store_true")
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="save annotated color-image copies after runtime measurement",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    if args.minimum_projected_points < 1:
        raise ValueError("--minimum-projected-points must be positive")
    if args.maximum_center_distance <= 0.0:
        raise ValueError("--maximum-center-distance must be positive")
    if args.minimum_mask_pixels < 1:
        raise ValueError("--minimum-mask-pixels must be positive")
    if not 0.0 < args.minimum_area_ratio <= args.maximum_area_ratio:
        raise ValueError("area ratios must be positive and ordered")
    if args.outlier_neighbors < 1 or args.outlier_std_ratio <= 0.0:
        raise ValueError("outlier parameters must be positive")
    table_origin = np.asarray(args.table_origin, dtype=np.float64)
    table_z_axis = np.asarray(args.table_z_axis, dtype=np.float64)
    if (
        not np.all(np.isfinite(table_origin))
        or not np.all(np.isfinite(table_z_axis))
        or np.linalg.norm(table_z_axis) < 1e-9
        or not np.isfinite(args.table_clearance)
        or args.table_clearance < 0.0
    ):
        raise ValueError("table plane parameters must be finite with nonzero Z axis")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
