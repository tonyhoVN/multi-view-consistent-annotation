#!/usr/bin/env python3
"""Create a route-ordered GIF from one run's annotation visualizations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Sequence

from PIL import Image


try:
    LANCZOS = Image.Resampling.LANCZOS
    ADAPTIVE = Image.Palette.ADAPTIVE
except AttributeError:  # Pillow before the Resampling/Palette enum migration.
    LANCZOS = Image.LANCZOS
    ADAPTIVE = Image.ADAPTIVE


METHOD_DIRECTORIES = {
    "transfer": Path("transfer_segment"),
    "transfer_spiral": Path("transfer_spiral"),
    "sam2_video": Path("baseline_segment/sam2_video"),
    "naive_vlm_zeroshot": Path("baseline_segment/naive_vlm_zeroshot"),
    "naive_vlm_multi_shot": Path("baseline_segment/naive_vlm_multi_shot"),
    "naive_vlm_zeroshot_no_filter": Path(
        "baseline_segment/naive_vlm_zeroshot_no_filter"
    ),
    "naive_vlm_multi_shot_no_filter": Path(
        "baseline_segment/naive_vlm_multi_shot_no_filter"
    ),
}
SUMMARY_NAMES = (
    "transfer_manifest.json",
    "naive_vlm_manifest.json",
    "sam2_video_manifest.json",
)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object and identify malformed files clearly."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return document


def resolve_run(value: str, scan_root: Path) -> Path:
    """Resolve an integer, run name, or explicit run-directory path."""
    supplied = Path(value).expanduser()
    if supplied.is_dir():
        return supplied.resolve()
    if re.fullmatch(r"\d+", value):
        candidate = scan_root.expanduser().resolve() / f"run_{int(value)}"
    elif re.fullmatch(r"run_\d+", value):
        candidate = scan_root.expanduser().resolve() / value
    else:
        candidate = supplied.resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(f"scan run directory does not exist: {candidate}")
    return candidate


def resolve_method_directory(run_root: Path, method: str) -> Path:
    """Resolve known method aliases while allowing future method directories."""
    relative = METHOD_DIRECTORIES.get(method)
    candidates = []
    if relative is not None:
        candidates.append(run_root / relative)
    candidates.extend((run_root / method, run_root / "baseline_segment" / method))
    directory = next((path for path in candidates if path.is_dir()), None)
    if directory is None:
        attempted = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"annotation method {method!r} not found; tried {attempted}")
    return directory.resolve()


def method_route(method_root: Path, override: str | None) -> str:
    """Read the exact route recorded by the annotation method."""
    if override is not None:
        return override
    for name in SUMMARY_NAMES:
        path = method_root / name
        if path.is_file():
            route = read_json(path).get("route")
            if isinstance(route, str) and route:
                return route
    raise ValueError(
        f"no method summary with a route under {method_root}; pass --route"
    )


def route_path_indices(scan_manifest: Path, route: str) -> list[int]:
    """Map manifest route sample IDs to saved capture path indices."""
    manifest = read_json(scan_manifest)
    route_record = manifest.get("routes", {}).get(route)
    sample_indices = route_record.get("sample_indices") if isinstance(route_record, dict) else None
    if not isinstance(sample_indices, list):
        raise ValueError(f"{scan_manifest} has no routes.{route}.sample_indices")
    capture_by_sample = {
        int(capture["sample_index"]): int(capture.get("path_index", position))
        for position, capture in enumerate(manifest.get("captures", []))
        if isinstance(capture, dict)
        and capture.get("status") == "captured"
        and "sample_index" in capture
    }
    ordered = [
        capture_by_sample[sample]
        for sample in map(int, sample_indices)
        if sample in capture_by_sample
    ]
    if not ordered:
        raise ValueError(f"route {route!r} contains no captured path indices")
    return ordered


def visualization_paths(
    method_root: Path, ordered_indices: Sequence[int]
) -> tuple[list[Path], list[int]]:
    """Return available visualization files in trajectory order."""
    directory = method_root / "visualizations"
    if not directory.is_dir():
        raise FileNotFoundError(
            f"visualizations do not exist: {directory}; rerun annotation with "
            "--save-visualizations"
        )
    paths, missing = [], []
    for index in ordered_indices:
        path = directory / f"frame_{index:05d}.jpg"
        if path.is_file():
            paths.append(path)
        else:
            missing.append(index)
    if not paths:
        raise ValueError(f"no route visualization frames found under {directory}")
    return paths, missing


def gif_frame(path: Path, max_width: int | None) -> Image.Image:
    """Load, optionally resize, and palette-convert one GIF frame."""
    image = Image.open(path).convert("RGB")
    if max_width is not None and image.width > max_width:
        height = max(1, round(image.height * max_width / image.width))
        image = image.resize((max_width, height), LANCZOS)
    return image.convert("P", palette=ADAPTIVE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="run index, run_N name, or run directory")
    parser.add_argument("method", help="annotation method, for example transfer or sam2_video")
    parser.add_argument("--scan-root", type=Path, default=Path("scan_output"))
    parser.add_argument("--route", choices=("hamilton_2opt", "random", "spiral"))
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--max-width", type=int, default=640)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-loop", action="store_true", help="play once instead of looping")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    if args.fps <= 0.0:
        raise ValueError("--fps must be positive")
    if args.max_width is not None and args.max_width <= 0:
        raise ValueError("--max-width must be positive")

    # Resolve ordering from manifests rather than lexicographically sorting files.
    run_root = resolve_run(args.run, args.scan_root)
    method_root = resolve_method_directory(run_root, args.method)
    route = method_route(method_root, args.route)
    indices = route_path_indices(run_root / "manifest.json", route)
    paths, missing = visualization_paths(method_root, indices)
    frames = [gif_frame(path, args.max_width) for path in paths]
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else method_root / f"annotation_sequence_{route}.gif"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=max(1, round(1000.0 / args.fps)),
        loop=1 if args.no_loop else 0,
        optimize=False,
        disposal=2,
    )
    print(f"Saved {len(frames)} frames in {route!r} order to {output}")
    if missing:
        print("Missing visualization path indices skipped: " + ", ".join(map(str, missing)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
