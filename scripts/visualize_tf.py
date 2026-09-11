#!/usr/bin/env python3
"""Visualize saved TF matrices and the Hamilton trajectory from a scan manifest.

Usage:
    python visualize_tf.py [directory] [--manifest manifest_run.json]
"""
import argparse
import glob
import json
import os
from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def natural_key(path):
    name = os.path.basename(path)
    nums = re.findall(r"\d+", name)
    return (name if not nums else "", [int(n) for n in nums], name)


def load_tf_matrices(directory):
    paths = sorted(glob.glob(os.path.join(directory, "*.npy")), key=natural_key)
    names, matrices = [], []
    for p in paths:
        m = np.load(p)
        if m.shape != (4, 4):
            print(f"skipping {p}: shape {m.shape} != (4, 4)")
            continue
        names.append(os.path.splitext(os.path.basename(p))[0])
        matrices.append(m)
    return names, matrices


def infer_manifest_path(directory):
    """Infer a canonical or legacy manifest from a transform directory."""
    directory = Path(directory).expanduser().resolve()
    match = re.fullmatch(r"save_TF_(.+)", directory.name)
    candidates = []
    if directory.name == "save_TF":
        candidates.append(directory.parent / "manifest.json")
    if match:
        candidates.append(directory.parent / f"manifest_{match.group(1)}.json")
    candidates.append(directory.parent / "manifest_run.json")
    return next((path for path in candidates if path.is_file()), None)


def load_hamilton_trajectory(manifest_path):
    """Load captured camera transforms in the manifest's Hamilton path order."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)

    try:
        sample_order = manifest["routes"]["hamilton_2opt"]["sample_indices"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"{manifest_path} has no routes.hamilton_2opt.sample_indices"
        ) from error
    if not isinstance(sample_order, list):
        raise ValueError("Hamilton sample_indices must be a list")

    # A failed motion has no camera_transform, so only plot captured viewpoints.
    transform_by_sample = {}
    for capture in manifest.get("captures", []):
        if not isinstance(capture, dict) or "camera_transform" not in capture:
            continue
        try:
            sample_index = int(capture["sample_index"])
        except (KeyError, TypeError, ValueError):
            continue
        transform_by_sample[sample_index] = manifest_path.parent / capture["camera_transform"]

    indices, matrices, missing = [], [], []
    for raw_index in sample_order:
        try:
            sample_index = int(raw_index)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid Hamilton sample index {raw_index!r}") from error
        transform_path = transform_by_sample.get(sample_index)
        if transform_path is None or not transform_path.is_file():
            missing.append(sample_index)
            continue
        matrix = np.load(transform_path)
        if matrix.shape != (4, 4):
            print(f"skipping {transform_path}: shape {matrix.shape} != (4, 4)")
            missing.append(sample_index)
            continue
        indices.append(sample_index)
        matrices.append(matrix)
    return indices, matrices, missing


def plot_frames(
    names,
    matrices,
    axis_len=0.05,
    max_frames=None,
    trajectory_matrices=None,
):
    if max_frames is not None:
        names = names[:max_frames]
        matrices = matrices[:max_frames]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = {"x": "r", "y": "g", "z": "b"}
    origins = np.array([m[:3, 3] for m in matrices])

    for m in matrices:
        origin = m[:3, 3]
        rot = m[:3, :3]
        for i, axis in enumerate(["x", "y", "z"]):
            direction = rot[:, i] * axis_len
            ax.plot(
                [origin[0], origin[0] + direction[0]],
                [origin[1], origin[1] + direction[1]],
                [origin[2], origin[2] + direction[2]],
                color=colors[axis],
                linewidth=1.2,
            )

    ax.scatter(origins[:, 0], origins[:, 1], origins[:, 2], color="k", s=8, alpha=0.6)

    if trajectory_matrices:
        trajectory_origins = np.asarray(
            [matrix[:3, 3] for matrix in trajectory_matrices], dtype=np.float64
        )
        ax.plot(
            trajectory_origins[:, 0],
            trajectory_origins[:, 1],
            trajectory_origins[:, 2],
            color="#19ff00",
            linewidth=3.0,
            marker="o",
            markersize=3.5,
            label="Hamilton + 2-opt",
            zorder=10,
        )
        ax.scatter(
            *trajectory_origins[0], color="#00ffff", s=45, label="trajectory start", zorder=11
        )
        ax.scatter(
            *trajectory_origins[-1], color="#ff00ff", s=45, label="trajectory end", zorder=11
        )
        ax.legend(loc="best")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    title = f"TF frames ({len(matrices)} matrices)"
    if trajectory_matrices:
        title += f"; Hamilton path ({len(trajectory_matrices)} captured views)"
    ax.set_title(title)

    # Include the route in the equal-aspect bounds even when --max-frames is used.
    all_pts = origins
    if trajectory_matrices:
        all_pts = np.vstack((all_pts, trajectory_origins))
    center = all_pts.mean(axis=0)
    spread = np.max(np.ptp(all_pts, axis=0)) / 2 + axis_len
    spread = max(spread, 1e-3)
    ax.set_xlim(center[0] - spread, center[0] + spread)
    ax.set_ylim(center[1] - spread, center[1] + spread)
    ax.set_zlim(center[2] - spread, center[2] + spread)

    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", nargs="?", default="scan_output/run_1/save_TF", help="directory with .npy TF matrices"
    )
    parser.add_argument("--max-frames", type=int, default=None, help="limit number of frames plotted")
    parser.add_argument("--axis-len", type=float, default=0.05, help="length of drawn axes")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "scan manifest containing routes.hamilton_2opt; by default infer "
            "<run>/manifest.json (legacy names are also supported)"
        ),
    )
    parser.add_argument("--save", type=str, default=None, help="path to save figure instead of showing")
    args = parser.parse_args()

    names, matrices = load_tf_matrices(args.directory)
    if not matrices:
        print(f"No valid 4x4 .npy files found in {args.directory}")
        return

    manifest_path = args.manifest or infer_manifest_path(args.directory)
    trajectory_indices, trajectory_matrices = [], []
    if manifest_path is None:
        print("No scan manifest found; plotting TF frames without a Hamilton trajectory")
    else:
        trajectory_indices, trajectory_matrices, missing = load_hamilton_trajectory(
            manifest_path
        )
        print(
            f"Loaded Hamilton trajectory with {len(trajectory_matrices)} captured "
            f"views from {manifest_path}"
        )
        if missing:
            print(
                "Hamilton samples without a saved transform (skipped): "
                + ", ".join(map(str, missing))
            )

    print(f"Loaded {len(matrices)} TF matrices from {args.directory}")
    fig = plot_frames(
        names,
        matrices,
        axis_len=args.axis_len,
        max_frames=args.max_frames,
        trajectory_matrices=trajectory_matrices,
    )

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
