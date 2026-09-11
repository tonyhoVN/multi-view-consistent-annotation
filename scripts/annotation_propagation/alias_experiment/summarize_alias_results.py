"""Aggregate mAP and runtime for the Grounding DINO alias experiment.

Unlike summarize_results.py (which reads a fixed method list), this scans
each run_<n>/alias_segment/ directory and discovers whatever method
subdirectories are actually present, so partially generated experiments
(e.g. only transfer_segment for one run, all five variants for another) are
summarized without extra flags.

For each discovered subdirectory it reads:
  - map_report.json               for mAP50 / mAP50_95
  - transfer_manifest.json (transfer_segment*) or
    naive_vlm_manifest.json (naive_vlm_*)        for runtime.total_seconds

Usage:
  python scripts/annotation_propagation/summarize_alias_results.py [start] [end]
  python scripts/annotation_propagation/summarize_alias_results.py 1 24 --scan-dir scan_output
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def runtime_manifest_name(method_dir_name: str) -> str:
    return "transfer_manifest.json" if method_dir_name.startswith("transfer") else "naive_vlm_manifest.json"


def discover_methods(scan_dir: Path, start: int, end: int) -> list[str]:
    names: set[str] = set()
    for run in range(start, end + 1):
        alias_dir = scan_dir / f"run_{run}" / "alias_segment"
        if not alias_dir.is_dir():
            continue
        for child in sorted(alias_dir.iterdir()):
            if child.is_dir():
                names.add(child.name)
    return sorted(names)


def collect(scan_dir: Path, start: int, end: int) -> dict[str, dict]:
    methods = discover_methods(scan_dir, start, end)
    results: dict[str, dict] = {}

    for method in methods:
        map50_values = []
        map50_95_values = []
        runtime_values = []
        missing_runs = []

        for run in range(start, end + 1):
            output_dir = scan_dir / f"run_{run}" / "alias_segment" / method
            report = load_json(output_dir / "map_report.json")
            runtime_doc = load_json(output_dir / runtime_manifest_name(method))

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

        results[method] = {
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
    print(f"Alias experiment summary over runs {start}-{end}\n")
    if not results:
        print("No alias_segment directories found in this range.")
        return

    label_width = max((len(name) for name in results), default=24) + 2
    header = (
        f"{'Method':<{label_width}}{'avg mAP50':<14}{'avg mAP50:95':<16}"
        f"{'avg runtime (s)':<18}{'runs (map/rt)'}"
    )
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(
            f"{name:<{label_width}}"
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
        default=Path(__file__).resolve().parents[3] / "scan_output",
        help="directory containing prefix-scoped run_<n> directories",
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
