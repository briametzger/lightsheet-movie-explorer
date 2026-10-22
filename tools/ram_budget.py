#!/usr/bin/env python
"""Tiny RAM guard for a 16 GB laptop.

Command line
------------
    python tools/ram_budget.py free
    python tools/ram_budget.py estimate 460,967,967 uint16 --copies 4
    python tools/ram_budget.py zarr path/to/array.zarr          # shape, chunk bytes, frame bytes

In code
-------
    from ram_budget import check_fits
    check_fits((460, 967, 967), "float32", copies=3)   # raises MemoryError if over budget

Limits: SOFT_GB warns, HARD_GB refuses (edit below if your machine changes).
Only the standard library is required; psutil and zarr are used if installed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

SOFT_GB = 4.0   # warn above this per-process peak
HARD_GB = 6.0   # refuse above this; treat as a workstation job

_DTYPE_BYTES = {
    "bool": 1, "uint8": 1, "int8": 1, "uint16": 2, "int16": 2, "float16": 2,
    "uint32": 4, "int32": 4, "float32": 4, "uint64": 8, "int64": 8,
    "float64": 8, "complex64": 8, "complex128": 16,
}
_NUMPY_CODES = {"u1": "uint8", "i1": "int8", "u2": "uint16", "i2": "int16",
                "f2": "float16", "u4": "uint32", "i4": "int32", "f4": "float32",
                "u8": "uint64", "i8": "int64", "f8": "float64", "b1": "bool",
                "c8": "complex64", "c16": "complex128"}


def _itemsize(dtype) -> int:
    name = str(dtype).lstrip("<>|=")
    name = _NUMPY_CODES.get(name, name)
    if name not in _DTYPE_BYTES:
        try:
            import numpy as np
            return np.dtype(dtype).itemsize
        except Exception as exc:  # pragma: no cover
            raise ValueError(f"Unknown dtype {dtype!r}") from exc
    return _DTYPE_BYTES[name]


def nbytes(shape, dtype) -> int:
    return int(math.prod(int(s) for s in shape)) * _itemsize(dtype)


def gb(n: float) -> float:
    return n / 1024**3


def available_gb() -> float | None:
    """Available memory in GB (psutil if present, else macOS vm_stat, else /proc)."""
    try:
        import psutil
        return gb(psutil.virtual_memory().available)
    except ImportError:
        pass
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
            page = int(out.split("page size of")[1].split()[0])
            stats = {}
            for line in out.splitlines()[1:]:
                if ":" in line:
                    k, v = line.split(":", 1)
                    stats[k.strip()] = int(v.strip().rstrip("."))
            free_pages = sum(stats.get(k, 0) for k in
                             ("Pages free", "Pages inactive", "Pages speculative",
                              "Pages purgeable"))
            return gb(free_pages * page)
        except Exception:
            return None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1024**2
    except Exception:
        return None
    return None


def estimate(shape, dtype, copies: float = 1.0) -> dict:
    one = nbytes(shape, dtype)
    peak = one * copies
    verdict = "ok"
    if gb(peak) > HARD_GB:
        verdict = "WORKSTATION (exceeds hard limit)"
    elif gb(peak) > SOFT_GB:
        verdict = "caution (above soft limit; close other apps, or crop/downsample)"
    return {"shape": list(shape), "dtype": str(dtype), "one_copy_gb": round(gb(one), 3),
            "copies": copies, "peak_gb": round(gb(peak), 3),
            "soft_gb": SOFT_GB, "hard_gb": HARD_GB, "verdict": verdict,
            "available_gb_now": None if available_gb() is None else round(available_gb(), 2)}


def check_fits(shape, dtype, copies: float = 1.0, label: str = "") -> dict:
    """Raise MemoryError above HARD_GB; print a warning above SOFT_GB."""
    est = estimate(shape, dtype, copies)
    tag = f"[{label}] " if label else ""
    if est["peak_gb"] > HARD_GB:
        raise MemoryError(
            f"{tag}Estimated peak {est['peak_gb']} GB > {HARD_GB} GB. "
            "Crop/downsample/chunk it, or add it to WORKSTATION_JOBS.md.")
    if est["peak_gb"] > SOFT_GB:
        print(f"WARNING {tag}estimated peak {est['peak_gb']} GB (> {SOFT_GB} GB soft limit)",
              file=sys.stderr)
    return est


def describe_zarr(path: str) -> dict:
    """Shape, dtype, bytes per chunk, bytes per leading index (e.g. per timepoint).

    Reads only metadata (.zarray / zarr.json), so it is safe on huge arrays.
    """
    p = Path(path)
    meta = None
    if (p / ".zarray").exists():
        meta = json.loads((p / ".zarray").read_text())
        shape, chunks, dtype = meta["shape"], meta["chunks"], meta["dtype"]
    elif (p / "zarr.json").exists():
        meta = json.loads((p / "zarr.json").read_text())
        if meta.get("node_type") != "array":
            return {"path": str(p), "type": "group",
                    "children": sorted(c.name for c in p.iterdir() if c.is_dir())}
        shape = meta["shape"]
        chunks = meta["chunk_grid"]["configuration"]["chunk_shape"]
        dtype = meta["data_type"]
    else:
        children = sorted(c.name for c in p.iterdir() if c.is_dir()) if p.is_dir() else []
        return {"path": str(p), "type": "group (or not zarr)", "children": children}
    return {
        "path": str(p), "shape": shape, "chunks": chunks, "dtype": dtype,
        "chunk_gb": round(gb(nbytes(chunks, dtype)), 3),
        "per_leading_index_gb": round(gb(nbytes(shape[1:], dtype)), 3) if len(shape) > 1 else None,
        "total_uncompressed_gb": round(gb(nbytes(shape, dtype)), 2),
        "note": "Any slice decompresses every chunk it touches; chunk_gb is the minimum "
                "transient cost of reading even one voxel.",
    }


def _parse_shape(s: str):
    return tuple(int(x) for x in s.replace("x", ",").split(",") if x)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("free", help="report available RAM")
    e = sub.add_parser("estimate", help="estimate peak RAM for an array")
    e.add_argument("shape", help="e.g. 460,967,967")
    e.add_argument("dtype", help="e.g. uint16, float32")
    e.add_argument("--copies", type=float, default=1.0,
                   help="full-size copies the step makes (filters/FFT often 3-6)")
    z = sub.add_parser("zarr", help="describe a zarr array without loading it")
    z.add_argument("path")
    a = ap.parse_args(argv)
    if a.cmd == "free":
        print(json.dumps({"available_gb": available_gb(), "soft_gb": SOFT_GB, "hard_gb": HARD_GB}))
    elif a.cmd == "estimate":
        print(json.dumps(estimate(_parse_shape(a.shape), a.dtype, a.copies), indent=2))
    elif a.cmd == "zarr":
        print(json.dumps(describe_zarr(a.path), indent=2))


if __name__ == "__main__":
    main()
