#!/usr/bin/env python3
"""Convert NPZ shards to directories of .npy files for true mmap support.

Usage:
    python convert_npz_to_npy.py /path/to/npz_shards /path/to/npy_shards

Each training_shard_000000.npz becomes a directory training_shard_000000/
containing one .npy file per array key (ok.npy, theta.npy, x_lc.npy, etc.).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


def convert_shard(npz_path: Path, out_dir: Path) -> None:
    shard_dir = out_dir / npz_path.stem
    shard_dir.mkdir(parents=True, exist_ok=True)

    with np.load(npz_path, allow_pickle=True) as z:
        for key in z.files:
            np.save(shard_dir / f"{key}.npy", z[key])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path, help="Directory containing .npz shards")
    parser.add_argument("dst", type=Path, help="Output directory for .npy shard dirs")
    parser.add_argument("--glob", default="*.npz", help="Glob pattern (default: *.npz)")
    args = parser.parse_args()

    npz_files = sorted(args.src.glob(args.glob))
    if not npz_files:
        print(f"No files matching {args.glob} in {args.src}", file=sys.stderr)
        sys.exit(1)

    args.dst.mkdir(parents=True, exist_ok=True)
    print(f"Converting {len(npz_files)} shards: {args.src} -> {args.dst}")

    for npz_path in tqdm(npz_files, desc="Converting"):
        convert_shard(npz_path, args.dst)

    print("Done.")


if __name__ == "__main__":
    main()
