#!/usr/bin/env python3
"""Visualize 4x4 TF matrices stored as .npy files in a directory.

Usage:
    python visualize_tf.py [directory] [--max-frames N] [--axis-len L]
"""
import argparse
import glob
import os
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


def plot_frames(names, matrices, axis_len=0.05, max_frames=None):
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

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"TF frames ({len(matrices)} matrices)")

    # keep equal aspect ratio
    all_pts = origins
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
        "directory", nargs="?", default="scan_output/save_TF_run1", help="directory with .npy TF matrices"
    )
    parser.add_argument("--max-frames", type=int, default=None, help="limit number of frames plotted")
    parser.add_argument("--axis-len", type=float, default=0.05, help="length of drawn axes")
    parser.add_argument("--save", type=str, default=None, help="path to save figure instead of showing")
    args = parser.parse_args()

    names, matrices = load_tf_matrices(args.directory)
    if not matrices:
        print(f"No valid 4x4 .npy files found in {args.directory}")
        return

    print(f"Loaded {len(matrices)} TF matrices from {args.directory}")
    fig = plot_frames(names, matrices, axis_len=args.axis_len, max_frames=args.max_frames)

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
