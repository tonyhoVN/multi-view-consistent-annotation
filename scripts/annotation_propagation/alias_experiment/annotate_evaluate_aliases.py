#!/usr/bin/env python3
"""Run one alias-only annotation method and evaluate it immediately."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import yaml

ANNOTATION_DIR = Path(__file__).resolve().parents[1]
if str(ANNOTATION_DIR) not in sys.path:
    sys.path.insert(0, str(ANNOTATION_DIR))

import annotate_naive_vlm as naive  # noqa: E402
import evaluate_segmentation_map as evaluation  # noqa: E402
import transfer_annotations_test as transfer  # noqa: E402


DEFAULT_ALIASES = Path(__file__).with_name("object_aliases.yaml")
METHODS = ("transfer", "naive_vlm_zeroshot", "naive_vlm_multi_shot")
OUTPUT_DIRECTORIES = {
    "transfer": "transfer_segment",
    "naive_vlm_zeroshot": "naive_vlm_zeroshot",
    "naive_vlm_multi_shot": "naive_vlm_multi_shot",
}


def load_aliases(path: Path) -> dict[str, str]:
    """Load canonical-label to Grounding-DINO-prompt mappings."""
    document = yaml.safe_load(path.expanduser().read_text(encoding="utf-8"))
    raw_aliases = document.get("aliases") if isinstance(document, dict) else None
    if not isinstance(raw_aliases, dict) or not raw_aliases:
        raise ValueError(f"alias config has no non-empty aliases mapping: {path}")
    aliases: dict[str, str] = {}
    for canonical, prompt in raw_aliases.items():
        key = transfer.slug(str(canonical))
        value = str(prompt).strip()
        if not value:
            raise ValueError(f"empty Grounding DINO alias for {canonical!r}")
        aliases[key] = value
    return aliases


def alias_model_class(
    aliases: dict[str, str],
) -> type[transfer.VisionModels]:
    """Create a model adapter that changes prompts but preserves output labels."""

    class AliasVisionModels(transfer.VisionModels):
        def prompt_for(self, canonical_label: str) -> str:
            return aliases.get(transfer.slug(canonical_label), canonical_label)

        def detect_boxes(self, image, label):
            """Detect one canonical class using only its configured alias prompt."""
            prompt = self.prompt_for(label)
            detected = super().detect_class_boxes(image, [prompt])
            return detected[prompt]

        def detect_class_boxes(self, image, labels):
            """Alias a zero-shot request and map results back to canonical keys."""
            prompts = [self.prompt_for(label) for label in labels]
            normalized = [transfer.slug(prompt) for prompt in prompts]
            if len(set(normalized)) != len(normalized):
                raise ValueError(
                    "aliases in one zero-shot request must be unique: "
                    f"{prompts}"
                )
            detected = super().detect_class_boxes(image, prompts)
            return {
                canonical: detected[prompt]
                for canonical, prompt in zip(labels, prompts)
            }

    return AliasVisionModels


def default_output(
    manifest: Path, method: str, filter_candidates: bool = True
) -> Path:
    """Place alias annotations beside canonical results without overwriting them."""
    directory = OUTPUT_DIRECTORIES[method]
    if method != "transfer" and not filter_candidates:
        directory += "_no_filter"
    return manifest.parent / "alias_segment" / directory


def annotation_arguments(args: argparse.Namespace, output: Path) -> list[str]:
    """Build the existing pipeline's CLI while keeping experiment policy explicit."""
    common = [
        str(args.manifest),
        "--route",
        args.route,
        "--output-dir",
        str(output),
    ]
    if args.save_visualizations:
        common.append("--save-visualizations")
    if args.method == "transfer":
        transfer_arguments = [
            *common,
            "--table-origin",
            *map(str, args.table_origin),
            "--table-z-axis",
            *map(str, args.table_z_axis),
            "--table-clearance",
            str(args.table_clearance),
            "--outlier-neighbors",
            str(args.outlier_neighbors),
            "--outlier-std-ratio",
            str(args.outlier_std_ratio),
            "--maximum-center-distance",
            str(args.maximum_center_distance),
            "--minimum-area-ratio",
            str(args.minimum_area_ratio),
            "--maximum-area-ratio",
            str(args.maximum_area_ratio),
        ]
        if args.camera_yaml is not None:
            transfer_arguments.extend(["--camera-yaml", str(args.camera_yaml)])
        return transfer_arguments
    mode = "zeroshot" if args.method.endswith("zeroshot") else "multi-shot"
    baseline = [*common, "--detection-mode", mode]
    if not args.filter_candidates:
        baseline.append("--no-filter-candidates")
    return baseline


def record_alias_metadata(
    output: Path, method: str, alias_path: Path, aliases: dict[str, str]
) -> None:
    """Record the exact prompt mapping inside the generated method manifest."""
    name = "transfer_manifest.json" if method == "transfer" else "naive_vlm_manifest.json"
    path = output / name
    document = json.loads(path.read_text(encoding="utf-8"))
    document["grounding_dino_alias_config"] = transfer.serialized_relative_path(
        alias_path, path.parent
    )
    document["grounding_dino_aliases"] = aliases
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    """Inject aliases, annotate once, then evaluate the saved canonical labels."""
    manifest = args.manifest.expanduser().resolve()
    alias_path = args.aliases.expanduser().resolve()
    aliases = load_aliases(alias_path)
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else default_output(manifest, args.method, args.filter_candidates)
    )

    # Both pipelines construct VisionModels through their module-level symbol.
    # Replacing those symbols affects this experiment process only.
    model_class = alias_model_class(aliases)
    transfer.VisionModels = model_class
    naive.VisionModels = model_class
    pipeline = transfer if args.method == "transfer" else naive
    pipeline.main(annotation_arguments(args, output))
    record_alias_metadata(output, args.method, alias_path, aliases)

    # Canonical labels were preserved, so the unchanged evaluator can compare
    # alias-prompt predictions directly with Isaac ground truth.
    report = output / "map_report.json"
    evaluation.main(
        [
            str(manifest),
            "--route",
            args.route,
            "--predictions",
            str(output),
            "--output",
            str(report),
        ]
    )
    print(f"Alias annotation and evaluation saved to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--route", default="hamilton_2opt")
    parser.add_argument(
        "--filter-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-visualizations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--table-origin", type=float, nargs=3, default=(0, 0, -0.20))
    parser.add_argument("--table-z-axis", type=float, nargs=3, default=(0, 0, 1))
    parser.add_argument("--table-clearance", type=float, default=0.003)
    parser.add_argument("--outlier-neighbors", type=int, default=20)
    parser.add_argument("--outlier-std-ratio", type=float, default=2.0)
    parser.add_argument("--maximum-center-distance", type=float, default=0.1)
    parser.add_argument("--minimum-area-ratio", type=float, default=0.2)
    parser.add_argument("--maximum-area-ratio", type=float, default=2.5)
    parser.add_argument(
        "--camera-yaml",
        type=Path,
        help="override camera calibration when the scan config references another host",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    if not args.manifest.expanduser().is_file():
        raise FileNotFoundError(f"scan manifest does not exist: {args.manifest}")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
