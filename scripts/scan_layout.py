#!/usr/bin/env python3
"""Canonical paths for one prefix-scoped scan run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


SAFE_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class ScanRunLayout:
    """Paths stored beneath ``<scan_root>/<run_name>``."""

    scan_root: Path
    run_name: str

    def __post_init__(self) -> None:
        if not SAFE_RUN_NAME.fullmatch(self.run_name):
            raise ValueError(f"unsafe scan run name: {self.run_name!r}")

    @property
    def root(self) -> Path:
        return self.scan_root.expanduser().resolve() / self.run_name

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def collection_log(self) -> Path:
        return self.root / "collection_log"

    @property
    def images(self) -> Path:
        return self.root / "save_images"

    @property
    def segments(self) -> Path:
        return self.root / "save_segment"

    @property
    def transforms(self) -> Path:
        return self.root / "save_TF"

    @property
    def transfer_segment(self) -> Path:
        return self.root / "transfer_segment"

    def baseline_segment(self, baseline: str, detection_mode: str) -> Path:
        """Return one baseline's isolated segmentation output directory."""
        if baseline != "naive_vlm":
            raise ValueError(f"unknown segmentation baseline: {baseline}")
        mode = detection_mode.replace("-", "_")
        if mode not in {"zeroshot", "multi_shot"}:
            raise ValueError(f"unknown naive VLM detection mode: {detection_mode}")
        return self.root / "baseline_segment" / f"{baseline}_{mode}"
