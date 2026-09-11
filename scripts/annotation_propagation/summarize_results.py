"""Aggregate mAP and runtime across manifest runs for the three annotation
methods: proposed transfer, naive VLM zeroshot, naive VLM multi-shot.

Reads, for each run in [start, end] and each method:
  - <output_dir>/map_report.json   for mAP50 / mAP50_95
  - <output_dir>/transfer_manifest.json or naive_vlm_manifest.json
    for runtime.total_seconds

Usage:
  python scripts/annotation_propagation/summarize_results.py [start] [end]
  python scripts/annotation_propagation/summarize_results.py 1 24 --scan-dir scan_output
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple


class Method(NamedTuple):
    key: str
    label: str
    output_dir_template: str
    runtime_manifest_name: str


METHODS = [
    Method(
        "transfer",
        "Proposed (transfer)",
        "transfer_segment_run_{run}",
        "transfer_manifest.json",
    ),
    Method(
        "zeroshot",
        "Naive VLM (zeroshot)",
        "naive_vlm_zeroshot_run_{run}",
        "naive_vlm_manifest.json",
    ),
    Method(
        "multi_shot",
        "Naive VLM (multi-shot)",
        "naive_vlm_multi_shot_run_{run}",
        "naive_vlm_manifest.json",
    ),
]


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def collect(scan_dir: Path, start: int, end: int) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for method in METHODS:
        map50_values = []
        map50_95_values = []
        runtime_values = []
        missing_runs = []

        for run in range(start, end + 1):
            output_dir = scan_dir / method.output_dir_template.format(run=run)
            report = load_json(output_dir / "map_report.json")
            runtime_doc = load_json(output_dir / method.runtime_manifest_name)

            if report is None and runtime_doc is None:
                missing_runs.append(run)
                continue

            if report is not None:
                if "mAP50" in report:
                    map50_values.append(report["mAP50"])
                if "mAP50_95" in report:
                    map50_95_values.append(report["mAP50_95"])

            if runtime_doc is not None:
                seconds = runtime_doc.get("runtime", {}).get("total_seconds")
                if seconds is not None:
                    runtime_values.append(seconds)

        results[method.key] = {
            "label": method.label,
            "runs_with_map": len(map50_values),
            "runs_with_runtime": len(runtime_values),
            "missing_runs": missing_runs,
            "avg_mAP50": (sum(map50_values) / len(map50_values)) if map50_values else None,
            "avg_mAP50_95": (
                sum(map50_95_values) / len(map50_95_values) if map50_95_values else None
            ),
            "avg_runtime_seconds": (
                sum(runtime_values) / len(runtime_values) if runtime_values else None
            ),
        }
    return results


def format_value(value: float | None, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if value is not None else "n/a"


def print_report(results: dict[str, dict], start: int, end: int) -> None:
    print(f"Summary over runs {start}-{end}\n")
    header = f"{'Method':<24}{'avg mAP50':<14}{'avg mAP50:95':<16}{'avg runtime (s)':<18}{'runs (map/rt)'}"
    print(header)
    print("-" * len(header))
    for method in METHODS:
        r = results[method.key]
        print(
            f"{r['label']:<24}"
            f"{format_value(r['avg_mAP50']):<14}"
            f"{format_value(r['avg_mAP50_95']):<16}"
            f"{format_value(r['avg_runtime_seconds'], 2):<18}"
            f"{r['runs_with_map']}/{r['runs_with_runtime']}"
        )
        if r["missing_runs"]:
            print(f"    missing runs: {r['missing_runs']}")
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start", type=int, nargs="?", default=1)
    parser.add_argument("end", type=int, nargs="?", default=24)
    parser.add_argument(
        "--scan-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "scan_output",
        help="directory containing manifest_run_<n>.json and method output dirs",
    )
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    return parser


def main(arguments: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(arguments)

    results = collect(args.scan_dir, args.start, args.end)
    print_report(results, args.start, args.end)

    if args.output:
        args.output.write_text(
            json.dumps(
                {"start": args.start, "end": args.end, "results": results},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
