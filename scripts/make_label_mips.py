#!/usr/bin/env python
"""Precompute 2D label MIPs (max over z) for every frame of a (T, Z, Y, X) label array.

Writes one compressed file per frame, <out>/tNNNNN.npz (key "mip"), the same cache
format movie_explorer.ipynb reads, so the notebook never has to touch
the 3D masks once this has run. Resumable: frames already written are skipped.

Each frame is read z-slab by z-slab. A partial zarr read still decompresses the
whole chunk, so peak RAM ~ one chunk + one slab (≈2.7 GB for chunks of
1x253x1456x1456 int32 with --slab-gb 0.5). Run several processes on disjoint
frame ranges to parallelise if the workstation has the RAM.

Example:
    python scripts/make_label_mips.py \
        --masks data/<dataset>/segments.zarr \
        --out cache/<dataset>/label_mips \
        --start 0 --stop 1600
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def open_labels(path: Path):
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    import zarr
    return zarr.open(str(path), mode="r")


def label_mip(arr, t: int, slab_gb: float) -> np.ndarray:
    _, Z, Y, X = arr.shape
    itemsize = np.dtype(arr.dtype).itemsize
    chunks = getattr(arr, "chunks", None) or (1, Z, Y, X)
    slab = int(max(1, min(chunks[1], slab_gb * 1024**3 // (Y * X * itemsize))))
    mip = np.zeros((Y, X), dtype=arr.dtype)
    for z0 in range(0, Z, slab):
        block = np.asarray(arr[t, z0:z0 + slab])
        np.maximum(mip, block.max(axis=0), out=mip)
        del block
    return mip


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None, help="exclusive; default = all frames")
    ap.add_argument("--slab-gb", type=float, default=0.5)
    a = ap.parse_args(argv)

    arr = open_labels(a.masks)
    T = arr.shape[0]
    stop = T if a.stop is None else min(a.stop, T)
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"masks {arr.shape} {arr.dtype}, chunks {getattr(arr, 'chunks', None)}; frames {a.start}..{stop - 1}")
    t0 = time.time(); done = 0
    for t in range(a.start, stop):
        f = a.out / f"t{t:05d}.npz"
        if f.exists():
            continue
        mip = label_mip(arr, t, a.slab_gb)
        tmp = f.with_name(f.stem + ".tmp.npz")
        np.savez_compressed(tmp, mip=mip)
        tmp.replace(f)                       # atomic: no half-written frames if interrupted
        done += 1
        if done % 25 == 0:
            rate = (time.time() - t0) / done
            print(f"t={t}: {done} written, {rate:.1f} s/frame, ~{rate * (stop - t - 1) / 3600:.1f} h left", flush=True)
    print(f"finished: {done} frames written in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
