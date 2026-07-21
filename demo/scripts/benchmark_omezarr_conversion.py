#################################################################
# File        : benchmark_omezarr_conversion.py
# Author      : sebi06
#
# Benchmark the read vs. write split of the CZI -> OME-ZARR
# conversion backends. Run this BEFORE attempting to parallelize
# writing, so effort is spent on the real bottleneck.
#################################################################

"""Benchmark CZI -> OME-ZARR conversion (read vs. write split).

For each backend the total conversion time is measured and the (identical)
standalone read time is subtracted to estimate the write-only time:

    write_time  ~=  total_convert_time  -  standalone_read_time

The standalone read uses the exact same ``read_tools.read_6darray`` call the
HCS converters use internally, so the subtraction is a good approximation.

Usage:
    pixi run python demo/scripts/benchmark_omezarr_conversion.py
    pixi run python demo/scripts/benchmark_omezarr_conversion.py --czi <path> --repeats 3
"""

from __future__ import annotations

import argparse
import gc
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np

from czitools.read_tools import read_tools
from czitools.metadata_tools.czi_metadata import CziMetadata
from czitools.export_tools.conversion import (
    convert_czi2hcs_ngff,
    convert_czi2hcs_omezarr,
    write_omezarr,
    write_omezarr_ngff,
)
from czitools.export_tools.resolver import resolve_hcs_layout

# Silence the converters' very chatty per-plane / per-field logging so the
# benchmark output stays readable. Timing is unaffected.
logging.getLogger("czitools").setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)

DEFAULT_CZI = r"F:\Testdata_Zeiss\CZI_Testfiles\WP96_4Pos_B4-10_DAPI.czi"


def _timeit(func: Callable[[], object], repeats: int) -> tuple[float, float]:
    """Run ``func`` ``repeats`` times and return (best, mean) wall-clock seconds."""
    timings: list[float] = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        func()
        timings.append(time.perf_counter() - start)
    return min(timings), float(np.mean(timings))


def _measure_read(czi: str, repeats: int) -> tuple[float, float, tuple, np.dtype, float]:
    """Measure the standalone full-plate read and report array stats."""
    holder: dict[str, object] = {}

    def _do() -> None:
        array6d, _ = read_tools.read_6darray(czi, use_xarray=True)
        holder["array"] = array6d

    best, mean = _timeit(_do, repeats)
    array6d = holder["array"]
    shape = tuple(array6d.shape)  # type: ignore[union-attr]
    dtype = array6d.dtype  # type: ignore[union-attr]
    size_mb = float(np.prod(shape)) * np.dtype(dtype).itemsize / (1024**2)
    del holder
    gc.collect()
    return best, mean, shape, dtype, size_mb


def _time_convert(kind: str, czi: str, repeats: int, zarr_format: int = 3) -> float:
    """Time a full HCS conversion, isolating each run in a fresh temp dir.

    The CZI is copied into a unique temp directory (untimed) so every timed run
    writes to a clean location -- this avoids Windows overwrite/file-lock races
    and never touches the user's test files. Returns the best wall-clock time.
    """
    timings: list[float] = []
    czi_name = Path(czi).name
    for _ in range(repeats):
        work = Path(tempfile.mkdtemp(prefix="omezarr_bench_"))
        czi_copy = work / czi_name
        shutil.copy2(czi, czi_copy)  # untimed setup
        gc.collect()
        start = time.perf_counter()
        if kind == "ngff":
            convert_czi2hcs_ngff(czi_filepath=str(czi_copy), overwrite=True)
        else:
            convert_czi2hcs_omezarr(czi_filepath=str(czi_copy), overwrite=True, zarr_format=zarr_format)
        timings.append(time.perf_counter() - start)
        shutil.rmtree(work, ignore_errors=True)
    return min(timings)


def _is_hcs(czi: str) -> bool:
    """Return True if the CZI resolves to an HCS plate layout."""
    try:
        mdata = CziMetadata(czi)
        resolve_hcs_layout(mdata)
        return True
    except Exception:
        return False


def _time_standard(kind: str, array5d, mdata, repeats: int, zarr_format: int = 3) -> float:
    """Time a standard (single-image) write using a pre-read 5D array.

    The array is already in memory, so this measures write-only time directly
    (no read is bundled in). Each run writes to a fresh temp dir. Returns the
    best wall-clock time.
    """
    timings: list[float] = []
    for _ in range(repeats):
        work = Path(tempfile.mkdtemp(prefix="omezarr_bench_"))
        out = work / "image.ome.zarr"
        gc.collect()
        start = time.perf_counter()
        if kind == "ngff":
            write_omezarr_ngff(array5d, out, mdata, scale_factors=None, overwrite=True)
        else:
            write_omezarr(array5d, zarr_path=out, metadata=mdata, overwrite=True, zarr_format=zarr_format)
        timings.append(time.perf_counter() - start)
        shutil.rmtree(work, ignore_errors=True)
    return min(timings)


def _run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--czi", default=DEFAULT_CZI, help="Path to the input CZI file.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of timed repetitions (best reported).")
    args = parser.parse_args()

    czi = str(Path(args.czi))
    if not Path(czi).exists():
        raise SystemExit(f"CZI file not found: {czi}")

    repeats = max(1, int(args.repeats))
    is_hcs = _is_hcs(czi)

    print("=" * 78)
    print("CZI -> OME-ZARR conversion benchmark")
    print("=" * 78)
    print(f"CZI file : {czi}")
    print(f"Repeats  : {repeats}")
    print(f"Layout   : {'HCS plate' if is_hcs else 'single image (standard)'}")
    print("-" * 78)

    # ---- 1) Standalone read (dominant-cost suspect) -------------------------
    r_best, r_mean, shape, dtype, size_mb = _measure_read(czi, repeats)
    read_mb_s = size_mb / r_best if r_best > 0 else float("nan")
    print(f"Array    : shape={shape} dtype={dtype} size={size_mb:.1f} MB")
    print(f"READ     : best={r_best:7.3f}s  mean={r_mean:7.3f}s  ({read_mb_s:6.1f} MB/s)")
    print("-" * 78)

    if is_hcs:
        # ---- HCS: measure full convert, subtract standalone read ------------
        backends: list[tuple[str, str, int]] = [
            ("ngff-zarr HCS (v0.5 / zarr v3)", "ngff", 3),
            ("ome-zarr-py HCS (v0.4-attrs / zarr v3)", "omezarr", 3),
            ("ome-zarr-py HCS (v0.4 / zarr v2)", "omezarr", 2),
        ]
        print(f"{'Backend':40s} {'total(s)':>9s} {'write~(s)':>10s} {'write%':>7s}")
        print("-" * 78)
        for name, kind, zfmt in backends:
            t_best = _time_convert(kind, czi, repeats, zarr_format=zfmt)
            write_s = max(0.0, t_best - r_best)
            write_pct = 100.0 * write_s / t_best if t_best > 0 else 0.0
            print(f"{name:40s} {t_best:9.3f} {write_s:10.3f} {write_pct:6.1f}%")
    else:
        # ---- Standard: read once, then measure write directly (clean split) -
        array6d, mdata = read_tools.read_6darray(czi, use_xarray=True)
        array5d = array6d.squeeze("S") if "S" in getattr(array6d, "dims", ()) else array6d
        backends_std: list[tuple[str, str, int]] = [
            ("ngff-zarr (v0.5 / zarr v3)", "ngff", 3),
            ("ome-zarr-py (v0.4-attrs / zarr v3)", "omezarr", 3),
            ("ome-zarr-py (v0.4 / zarr v2)", "omezarr", 2),
        ]
        print(f"{'Backend':40s} {'write(s)':>9s}  (read={r_best:.2f}s, {'write%':>6s})")
        print("-" * 78)
        for name, kind, zfmt in backends_std:
            w_best = _time_standard(kind, array5d, mdata, repeats, zarr_format=zfmt)
            total = w_best + r_best
            write_pct = 100.0 * w_best / total if total > 0 else 0.0
            print(f"{name:40s} {w_best:9.3f}  ({write_pct:6.1f}% of read+write)")

    print("-" * 78)
    print("Interpretation:")
    print("  If write dominates -> parallelize the WRITE (tensorstore/dask/threads).")
    print("  If read dominates  -> parallelize the READ (per-scene/plane, own readers).")
    print("=" * 78)


if __name__ == "__main__":
    _run()
