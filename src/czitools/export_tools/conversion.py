# -*- coding: utf-8 -*-
"""Core CZI -> OME-Zarr conversion functions (czitools Stage 5).

Adapted from ``czi_omezarr_utils.conversion`` in the ``omezarr_playground``
repository. The two HCS pipelines now consume the canonical layout produced by
:func:`czitools.export_tools.resolver.resolve_hcs_layout`, so they support the
Stage 1 HCS model (preferred), a ``CziSampleInfo`` fallback, sparse plates, and
variable field counts per well.

  - ``convert_czi2hcs_omezarr`` — HCS pipeline using ome-zarr-py (OME-NGFF v0.4)
  - ``convert_czi2hcs_ngff``    — HCS pipeline using ngff-zarr (OME-NGFF v0.5)
  - ``write_omezarr``           — write a single 5D image using ome-zarr-py
  - ``write_omezarr_ngff``      — write a single 5D image using ngff-zarr with pyramid
"""

import gc
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import xarray as xr
import dask
import dask.array as da
import zarr
import ngff_zarr as nz
from ngff_zarr.v04.zarr_metadata import Plate, PlateColumn, PlateRow, PlateWell
from ngff_zarr.hcs import HCSPlate, HCSPlateWriter, to_hcs_zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_plate_metadata, write_well_metadata
import ome_zarr.writer
import ome_zarr.format

from czitools.read_tools import read_tools
from czitools.metadata_tools.czi_metadata import CziMetadata

from ._logging import setup_logging
from .plate import convert_hcs_omezarr2ozx
from .display import get_fieldimage, create_channel_list, create_ngff_omero_channels, compute_pyramid_scale_factors
from .resolver import resolve_hcs_layout

logger = logging.getLogger(__name__)

# Optional progress bar (progressbar2) for long-running steps. Guarded so the
# module still imports if the package is unavailable.
try:
    import progressbar  # type: ignore

    HAS_PROGRESSBAR = True
except ImportError:  # pragma: no cover
    HAS_PROGRESSBAR = False

# Optional tensorstore backend for ngff-zarr writes (async parallel chunk I/O).
# Enabled automatically when the package is installed; harmless when absent.
try:
    import tensorstore  # type: ignore  # noqa: F401

    HAS_TENSORSTORE = True
except ImportError:  # pragma: no cover
    HAS_TENSORSTORE = False


def _to_ome_zarr_image(array: Union[np.ndarray, xr.DataArray, da.Array]) -> Union[np.ndarray, da.Array]:
    """Return an array type accepted by ome-zarr writer functions."""
    if isinstance(array, xr.DataArray):
        data = array.data
        if isinstance(data, (np.ndarray, da.Array)):
            return data
        return np.asarray(data)
    return array


def _write_image_delayed(image, group, axes: str, chunks: tuple, fmt) -> list:
    """Schedule an ome-zarr-py image write as parallel (dask) chunk-write tasks.

    The array is wrapped in a dask array (chunked to ``chunks``) so ome-zarr-py's
    ``compute=False`` path produces a chunk-parallel write graph instead of writing
    synchronously. The returned dask-delayed tasks are NOT executed here -- the
    caller batches them and runs a single :func:`dask.compute`, which parallelizes
    chunk writes (and compression, which releases the GIL) across threads.

    Args:
        image: Array data (numpy or dask) for one image.
        group: Target zarr group.
        axes (str): Axis string, e.g. ``"tczyx"``.
        chunks (tuple): Chunk shape used for both the dask array and zarr storage.
        fmt: ome-zarr ``Format`` instance.

    Returns:
        list: A list of dask-delayed write tasks (possibly empty).
    """
    if not isinstance(image, da.Array):
        image = da.from_array(image, chunks=chunks)
    delayed = ome_zarr.writer.write_image(
        image=image,
        group=group,
        axes=axes,
        storage_options=dict(chunks=chunks),
        fmt=fmt,
        compute=False,
    )
    return list(delayed) if delayed else []


def _ensure_plate_version_metadata(zarr_path: Union[str, os.PathLike, Path], version: str) -> None:
    """Ensure nested ome.plate.version exists in root metadata."""
    parsed = parse_url(Path(zarr_path), mode="r+")
    assert parsed is not None, f"Failed to open zarr store at {zarr_path}"

    root = zarr.group(store=parsed.store)
    attrs = root.attrs.asdict()
    ome_attrs = attrs.get("ome")
    if not isinstance(ome_attrs, dict):
        return

    plate_attrs = ome_attrs.get("plate")
    if not isinstance(plate_attrs, dict) or plate_attrs.get("version") is not None:
        return

    plate_attrs["version"] = version
    ome_attrs["plate"] = plate_attrs
    attrs["ome"] = ome_attrs
    root.attrs.update(attrs)


def _normalize_multiscale_level_names_v2(store_path: Union[str, os.PathLike, Path]) -> None:
    """Rename zarr v2 multiscale levels from ``sN`` to ``N`` in a directory store.

    ome-zarr 0.18 names multiscale datasets ``s0, s1, ...``. Some legacy OME-NGFF
    v0.4 viewers (notably vizarr-based web tools such as Find-Nuclei) only resolve
    numeric level names (``0, 1, ...``). This walks a directory-based (zarr v2)
    store, renames each ``sN`` level directory to ``N`` and rewrites the matching
    ``multiscales[].datasets[].path`` entries in every group's ``.zattrs``.

    Args:
        store_path (Union[str, os.PathLike, Path]): Path to the zarr v2 store root.
    """
    root = Path(store_path)
    pattern = re.compile(r"^s(\d+)$")
    zattrs_paths = list(root.rglob(".zattrs"))
    total = len(zattrs_paths)

    logger.info("Finalizing v2 legacy store: normalizing %d group(s) (renaming pyramid levels sN -> N)...", total)

    iterator: object = zattrs_paths
    if HAS_PROGRESSBAR and total > 0:
        widgets = [
            "Finalizing v2 store ",
            progressbar.Percentage(),
            " ",
            progressbar.Bar(),
            " ",
            progressbar.SimpleProgress(),
        ]
        iterator = progressbar.progressbar(zattrs_paths, widgets=widgets, max_value=total, term_width=80)

    for zattrs_path in iterator:  # type: ignore[assignment]
        try:
            attrs = json.loads(zattrs_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        multiscales = attrs.get("multiscales")
        if not multiscales:
            continue
        group_dir = zattrs_path.parent
        changed = False
        for multiscale in multiscales:
            for dataset in multiscale.get("datasets", []):
                match = pattern.match(str(dataset.get("path", "")))
                if match is None:
                    continue
                new_name = match.group(1)
                src = group_dir / dataset["path"]
                dst = group_dir / new_name
                if src.is_dir() and not dst.exists():
                    src.rename(dst)
                dataset["path"] = new_name
                changed = True
        if changed:
            zattrs_path.write_text(json.dumps(attrs), encoding="utf-8")


# ---------------------------------------------------------------------------
# ome-zarr-py HCS conversion
# ---------------------------------------------------------------------------


def convert_czi2hcs_omezarr(
    czi_filepath: Union[str, os.PathLike, Path],
    overwrite: bool = True,
    log_file_path: Optional[Union[str, os.PathLike, Path]] = None,
    pad_columns: bool = True,
    zarr_format: int = 3,
) -> Path:
    """Convert a CZI file to OME-Zarr HCS format using the ome-zarr-py backend.

    Args:
        czi_filepath (Union[str, os.PathLike, Path]): Path to the input CZI file.
        overwrite (bool): Remove existing output directory if True.
        log_file_path (Optional[Union[str, os.PathLike, Path]]): Log file path.
            Defaults to ``<stem>_hcs_omezarr.log``.
        pad_columns (bool): Zero-pad column numbers in well paths (e.g. ``"04"``).
        zarr_format (int): Zarr storage format to write. ``3`` (default) writes a
            zarr v3 store; ``2`` writes an OME-NGFF v0.4 / zarr v2 store for
            compatibility with legacy readers (e.g. vizarr-based web viewers) that
            do not yet support zarr v3.

    Returns:
        Path: Output OME-Zarr HCS directory (``<stem>_HCSplate.ome.zarr``).
    """
    czi_path = Path(czi_filepath)
    if log_file_path is None:
        log_file_path = czi_path.parent / f"{czi_path.stem}_hcs_omezarr.log"
    else:
        log_file_path = Path(log_file_path)

    setup_logging(log_file_path)

    logger.info("=" * 80)
    logger.info("CZI to HCS OME-ZARR Conversion Started (ome-zarr-py backend)")
    logger.info("=" * 80)
    logger.info(f"Input CZI file: {czi_path.absolute()}")

    zarr_output_path = czi_path.parent / f"{czi_path.stem}_HCSplate.ome.zarr"

    if zarr_output_path.exists():
        if overwrite:
            logger.info(f"Removing existing directory: {zarr_output_path}")
            shutil.rmtree(zarr_output_path)
        else:
            logger.info(f"File exists at {zarr_output_path}. Set overwrite=True to remove.")
            return zarr_output_path

    array6d, mdata = read_tools.read_6darray(str(czi_path), use_xarray=True)
    assert isinstance(array6d, xr.DataArray), "Expected xarray DataArray from read_6darray with use_xarray=True"

    layout = resolve_hcs_layout(mdata, pad_columns=pad_columns)
    logger.info(f"Resolved plate layout from '{layout.source}': {len(layout.wells)} well(s)")

    # Select the OME-Zarr format. FormatV04 forces an OME-NGFF v0.4 / zarr v2 store
    # (legacy-reader compatible); CurrentFormat() is ome-zarr's default (zarr v3).
    # fmt is threaded through every writer call so the whole store is written in a
    # single, consistent format.
    _fmt = ome_zarr.format.FormatV04() if zarr_format == 2 else ome_zarr.format.CurrentFormat()
    logger.info(f"Zarr storage format: v{zarr_format}" + (" (OME-NGFF v0.4)" if zarr_format == 2 else ""))

    parsed = parse_url(zarr_output_path, mode="w", fmt=_fmt)
    assert parsed is not None, f"Failed to open zarr store at {zarr_output_path}"
    root = zarr.group(store=parsed.store)

    well_paths = [w.path for w in layout.wells]
    write_plate_metadata(root, layout.row_names, layout.col_names, well_paths, fmt=_fmt)  # type: ignore[arg-type]

    plate_attrs = root.attrs.asdict()
    plate_attrs["rows"] = [{"name": r} for r in sorted(layout.row_names)]
    plate_attrs["columns"] = [{"name": c} for c in sorted(layout.col_names, key=int)]
    root.attrs.update(plate_attrs)

    # OMERO channel metadata is written on every field image so that readers such
    # as ngio / napari-ome-zarr-navigator can resolve per-channel display settings.
    channels_list = create_channel_list(mdata)

    chunks = (1, 1, array6d.sizes["Z"], array6d.sizes["Y"], array6d.sizes["X"])
    # Collect chunk-parallel write tasks across ALL fields, then execute once with a
    # single dask.compute so fields (and their chunks) are written in parallel.
    delayed_writes: list = []

    for well in layout.wells:
        well_group = root.require_group(well.row).require_group(well.column)
        field_paths = [str(field_index) for field_index, _ in well.fields]
        write_well_metadata(well_group, field_paths, fmt=_fmt)  # type: ignore[arg-type]

        for field_index, scene_index in well.fields:
            image_group = well_group.require_group(str(field_index))
            logger.info(f"Scheduling Well: {well.path}, Field: {field_index}, Scene Index: {scene_index}")
            image = array6d[scene_index, ...]
            delayed_writes.extend(
                _write_image_delayed(
                    _to_ome_zarr_image(image),
                    image_group,
                    "".join(str(d).lower() for d in image.dims),
                    chunks,
                    _fmt,
                )
            )
            ome_zarr.writer.add_metadata(
                image_group,
                {"omero": {"name": f"{well.path}/{field_index}", "channels": channels_list}},
                fmt=_fmt,
            )

    if delayed_writes:
        logger.info("Writing %d field-pyramid task(s) in parallel (dask)...", len(delayed_writes))
        dask.compute(*delayed_writes)

    if zarr_format == 2:
        # Normalize multiscale level names (sN -> N) for legacy v0.4 readers.
        _normalize_multiscale_level_names_v2(zarr_output_path)

    logger.info("=" * 80)
    logger.info("Conversion completed successfully!")
    logger.info(f"Output HCS OME-ZARR file: {zarr_output_path}")
    logger.info("=" * 80)

    return zarr_output_path


# ---------------------------------------------------------------------------
# ngff-zarr HCS conversion
# ---------------------------------------------------------------------------


def convert_czi2hcs_ngff(
    czi_filepath: Union[str, os.PathLike, Path],
    plate_name: str = "Automated Plate",
    overwrite: bool = True,
    log_file_path: Optional[Union[str, os.PathLike, Path]] = None,
    write_ozx_directly: bool = False,
    version: str = "0.5",
    output_dir: Optional[Union[str, os.PathLike, Path]] = None,
    pad_columns: bool = True,
) -> Path:
    """Convert a CZI file to OME-Zarr HCS format using the ngff-zarr backend.

    Args:
        czi_filepath (Union[str, os.PathLike, Path]): Path to the input CZI file.
        plate_name (str): Name for the well plate in metadata.
        overwrite (bool): Remove existing output if True.
        log_file_path (Optional[Union[str, os.PathLike, Path]]): Log file path.
            Defaults to ``<stem>_hcs_ngff.log``.
        write_ozx_directly (bool): Write a single-file ``.ozx`` archive directly.
        version (str): NGFF version string. Defaults to ``"0.5"``.
        output_dir (Optional[Union[str, os.PathLike, Path]]): Output directory.
            Defaults to the CZI file's parent directory.
        pad_columns (bool): Zero-pad column numbers in well paths (e.g. ``"04"``).

    Returns:
        Path: Output HCS directory (``<stem>_ngff_plate.ome.zarr``) or ``.ozx`` file.
    """
    czi_path = Path(czi_filepath)
    output_path_obj: Optional[Path] = Path(output_dir) if output_dir is not None else None

    if log_file_path is None:
        base = output_path_obj if output_path_obj is not None else czi_path.parent
        log_file_path = base / f"{czi_path.stem}_hcs_ngff.log"
    else:
        log_file_path = Path(log_file_path)

    setup_logging(log_file_path)

    logger.info("=" * 80)
    logger.info("CZI to HCS OME-ZARR Conversion Started (ngff-zarr backend)")
    logger.info("=" * 80)
    logger.info(f"Input CZI file: {czi_path.absolute()}")
    logger.info(f"Plate name: {plate_name}")

    stem = czi_path.stem
    suffix = "_ngff_plate.ozx" if write_ozx_directly else "_ngff_plate.ome.zarr"
    base_dir = output_path_obj if output_path_obj is not None else czi_path.parent
    zarr_output_path = base_dir / f"{stem}{suffix}"

    if zarr_output_path.exists():
        if overwrite:
            logger.info(f"Removing existing file/directory: {zarr_output_path}")
            if zarr_output_path.is_dir():
                shutil.rmtree(zarr_output_path)
            else:
                os.remove(zarr_output_path)
            gc.collect()
            time.sleep(0.5)
        else:
            logger.info(f"File exists at {zarr_output_path}. Set overwrite=True to remove.")
            return zarr_output_path

    array6d, mdata = read_tools.read_6darray(str(czi_path), use_xarray=True)
    assert isinstance(array6d, xr.DataArray), "Expected xarray DataArray from read_6darray with use_xarray=True"

    layout = resolve_hcs_layout(mdata, pad_columns=pad_columns)
    logger.info(f"Resolved plate layout from '{layout.source}': {len(layout.wells)} well(s)")

    columns = [PlateColumn(name=c) for c in sorted(layout.col_names, key=int)]
    rows = [PlateRow(name=r) for r in sorted(layout.row_names)]
    # Per the OME-NGFF plate spec, well ``rowIndex``/``columnIndex`` are indices
    # INTO the (possibly sparse) ``rows``/``columns`` arrays above, not absolute
    # 96-well-plate coordinates. Build position maps from the sorted labels so a
    # sparse plate (e.g. only row "B", columns "04".."10") gets consistent
    # indices; using absolute coordinates here produces out-of-bounds indices
    # that break spec-compliant readers such as ngio / napari-ome-zarr-navigator.
    row_pos = {row.name: idx for idx, row in enumerate(rows)}
    col_pos = {col.name: idx for idx, col in enumerate(columns)}
    wells_meta = [
        PlateWell(
            path=well.path,
            rowIndex=row_pos[well.row],
            columnIndex=col_pos[well.column],
        )
        for well in layout.wells
    ]

    plate_metadata = Plate(
        columns=columns,
        rows=rows,
        wells=wells_meta,
        name=plate_name,
        field_count=layout.field_count,
        version=version,
    )

    # On Windows, HCSPlateWriter.__exit__ zips while the temp store is still open,
    # causing a PermissionError (ngff-zarr issue #241). Workaround: write a .ome.zarr
    # directory first, then zip afterwards.
    _win_ozx_workaround = write_ozx_directly and sys.platform == "win32"
    if _win_ozx_workaround:
        logger.warning(
            "write_ozx_directly=True is not supported on Windows (ngff-zarr issue #241). "
            "Writing to .ome.zarr first, then converting to .ozx."
        )
        write_path = base_dir / f"{stem}_ngff_plate.ome.zarr"
        if write_path.exists():
            shutil.rmtree(write_path)
            gc.collect()
            time.sleep(0.2)
    else:
        write_path = zarr_output_path

    hcs_plate = HCSPlate(store=write_path, plate_metadata=plate_metadata)
    to_hcs_zarr(hcs_plate, write_path)

    # OMERO channel metadata attached to each field image so readers such as ngio /
    # napari-ome-zarr-navigator can resolve per-channel display settings.
    omero_channels = create_ngff_omero_channels(mdata)

    with HCSPlateWriter(str(write_path), plate_metadata) as writer:
        for well in layout.wells:
            logger.info(f"Creating Well: {well.well_id} (Row: {well.row}, Column: {well.column})")
            for field_index, scene_index in well.fields:
                logger.info(f"Writing Well: {well.path}, Field: {field_index}, Scene Index: {scene_index}")
                multiscales = get_fieldimage(array6d, scene_index, mdata)
                if omero_channels:
                    multiscales.metadata.omero = nz.Omero(channels=omero_channels)
                writer.write_well_image(
                    multiscales=multiscales,
                    row_name=well.row,
                    column_name=well.column,
                    field_index=field_index,
                )

    _ensure_plate_version_metadata(write_path, version)

    if _win_ozx_workaround:
        logger.info("Converting intermediate .ome.zarr to .ozx (Windows workaround)...")
        gc.collect()
        time.sleep(0.5)
        result = convert_hcs_omezarr2ozx(write_path, remove_omezarr=True)
        if result is not None:
            zarr_output_path = result

    logger.info("=" * 80)
    logger.info("Conversion completed successfully!")
    logger.info(f"Output HCS OME-ZARR file: {zarr_output_path}")
    logger.info("=" * 80)

    return zarr_output_path


# ---------------------------------------------------------------------------
# write_omezarr (ome-zarr-py single image)
# ---------------------------------------------------------------------------


def write_omezarr(
    array5d: Union[np.ndarray, xr.DataArray, da.Array],
    zarr_path: Union[str, Path],
    metadata: CziMetadata,
    overwrite: bool = False,
    log_file_path: Optional[Union[str, Path]] = None,
    zarr_format: int = 3,
) -> Optional[Path]:
    """Write a single 5D image to OME-Zarr using the ome-zarr-py backend.

    Args:
        array5d (Union[np.ndarray, xr.DataArray, da.Array]): Input xarray DataArray
            with named dimensions ``(T, C, Z, Y, X)``.
        zarr_path (Union[str, Path]): Output path for the OME-Zarr file.
        metadata (CziMetadata): Metadata with channel and scale information.
        overwrite (bool): Remove existing output if True.
        log_file_path (Optional[Union[str, Path]]): Log file path. Defaults to
            ``<stem>_omezarr.log``.
        zarr_format (int): Zarr storage format to write. ``3`` (default) writes a
            zarr v3 store; ``2`` writes an OME-NGFF v0.4 / zarr v2 store for
            compatibility with legacy readers that do not yet support zarr v3.

    Returns:
        Optional[Path]: Path to the written OME-Zarr file, or ``None`` on failure.
    """
    if log_file_path is None:
        zarr_path_obj = Path(zarr_path)
        log_file_path = zarr_path_obj.parent / f"{zarr_path_obj.stem}_omezarr.log"

    setup_logging(log_file_path)

    logger.info("=" * 80)
    logger.info("Writing OME-ZARR (ome-zarr-py backend)")
    logger.info("=" * 80)
    logger.info(f"Input array shape: {array5d.shape}")
    logger.info(f"Output path: {zarr_path}")

    assert isinstance(array5d, xr.DataArray), "write_omezarr requires an xarray DataArray"

    zarr_path = Path(zarr_path)

    if len(array5d.shape) > 5:
        logger.info("Input array has more than 5 dimensions.")
        return None

    if zarr_path.exists() and overwrite:
        logger.info(f"Removing existing file/directory: {zarr_path}")
        if zarr_path.is_dir():
            shutil.rmtree(zarr_path)
        else:
            os.remove(zarr_path)
    elif zarr_path.exists() and not overwrite:
        logger.info(f"File already exists at {zarr_path}. Set overwrite=True to remove.")
        return None

    # FormatV04 forces an OME-NGFF v0.4 / zarr v2 store; CurrentFormat() is the
    # ome-zarr default (zarr v3). fmt must be passed to parse_url so the group is
    # created in the matching zarr format (otherwise write_image(fmt=...) raises
    # "Group is zarr_format: 3 but OME-Zarr 0.4 is 2").
    _fmt = ome_zarr.format.FormatV04() if zarr_format == 2 else ome_zarr.format.CurrentFormat()
    logger.info(f"Zarr storage format: v{zarr_format}" + (" (OME-NGFF v0.4)" if zarr_format == 2 else ""))

    parsed = parse_url(zarr_path, mode="w", fmt=_fmt)
    assert parsed is not None, f"Failed to open zarr store at {zarr_path}"
    # Do NOT pass overwrite=overwrite here: in zarr 3.x that recreates the group
    # with the default zarr_format (3), overriding the v2 store from parse_url and
    # causing "Group is zarr_format: 3 but OME-Zarr 0.4 is 2". Any existing store
    # was already removed above when overwrite=True. Set zarr_format explicitly to
    # match the chosen OME-Zarr format.
    root = zarr.group(store=parsed.store, zarr_format=(2 if zarr_format == 2 else 3))

    # Chunk the full Z-stack per (T, C) instead of one XY plane per chunk. Single-
    # plane chunks explode the file count (T*C*Z chunks/level), which makes writing
    # large images pathologically slow on Windows.
    chunks = (1, 1, array5d.sizes["Z"], array5d.sizes["Y"], array5d.sizes["X"])
    axes = "".join(str(d).lower() for d in array5d.dims)

    # Parallel write: dask-wrap + compute=False yields a chunk-parallel write graph
    # that we execute with a single dask.compute (threads release the GIL during
    # zarr chunk writes + compression), roughly halving write time on large images.
    delayed = _write_image_delayed(_to_ome_zarr_image(array5d), root, axes, chunks, _fmt)
    if delayed:
        logger.info("Writing %d pyramid level(s) in parallel (dask)...", len(delayed))
        dask.compute(*delayed)

    channels_list = create_channel_list(metadata)
    ome_zarr.writer.add_metadata(
        root,
        {
            "omero": {
                "name": metadata.filename,
                "channels": channels_list,
            }
        },
        fmt=_fmt,
    )

    if zarr_format == 2:
        # Normalize multiscale level names (sN -> N) for legacy v0.4 readers.
        _normalize_multiscale_level_names_v2(zarr_path)

    logger.info("OME-ZARR writing completed successfully!")
    logger.info(f"Output file: {zarr_path}")

    return zarr_path


# ---------------------------------------------------------------------------
# write_omezarr_ngff (ngff-zarr single image with pyramid)
# ---------------------------------------------------------------------------


def write_omezarr_ngff(
    array5d: Union[np.ndarray, xr.DataArray, da.Array],
    zarr_path: Union[Path, str],
    metadata: CziMetadata,
    scale_factors: Optional[list] = None,
    overwrite: bool = False,
    version: str = "0.5",
    chunks: Union[tuple, None] = None,
    chunks_per_shard: Union[Dict[str, int], int, None] = 2,
    log_file_path: Union[Path, str, None] = None,
    min_size: int = 512,
    max_levels: int = 6,
    use_tensorstore: Optional[bool] = None,
) -> Optional["nz.NgffImage"]:
    """Write a single 5D image to OME-Zarr NGFF format with multi-scale pyramids.

    Args:
        array5d (Union[np.ndarray, xr.DataArray, da.Array]): Input 5D array with
            dimensions ``(t, c, z, y, x)``.
        zarr_path (Union[Path, str]): Output path for the OME-Zarr NGFF file.
        metadata (CziMetadata): Metadata with scale and channel information.
        scale_factors (Optional[list]): Downscaling factors for the pyramid. When
            None (default), size-aware Y/X-only factors are computed from the plane
            size via :func:`compute_pyramid_scale_factors` (``min_size``/``max_levels``).
        overwrite (bool): Remove existing output if True.
        version (str): NGFF version string. Defaults to ``"0.5"``.
        chunks (Union[tuple, None]): Explicit chunk shape (auto-computed if None).
        chunks_per_shard (Union[Dict[str, int], int, None]): Chunks per shard.
        log_file_path (Union[Path, str, None]): Log file path. Defaults to
            ``<stem>_ngff.log``.
        min_size (int): Target maximum XY size of the coarsest pyramid level, used
            only when ``scale_factors`` is None. Defaults to 512.
        max_levels (int): Hard cap on the number of pyramid levels, used only when
            ``scale_factors`` is None. Defaults to 6.
        use_tensorstore (Optional[bool]): Use the tensorstore backend for async
            parallel chunk I/O. When None (default), it is enabled automatically if
            the ``tensorstore`` package is installed, otherwise disabled.

    Returns:
        Optional[nz.NgffImage]: The written NgffImage, or ``None`` on failure.
    """
    if scale_factors is None:
        # Size-aware, Y/X-only pyramid depth derived from the plane size.
        scale_factors = compute_pyramid_scale_factors(
            int(array5d.shape[-2]), int(array5d.shape[-1]), min_size=min_size, max_levels=max_levels
        )

    if log_file_path is None:
        zarr_path_obj = Path(zarr_path)
        log_file_path = zarr_path_obj.parent / f"{zarr_path_obj.stem}_ngff.log"

    setup_logging(log_file_path)

    logger.info("=" * 80)
    logger.info("Writing OME-ZARR NGFF format with multiscale")
    logger.info("=" * 80)
    logger.info(f"Input array shape: {array5d.shape}")
    logger.info(f"Output path: {zarr_path}")
    logger.info(f"Scale factors: {scale_factors}")

    if len(array5d.shape) > 5:
        logger.info("Input array has more than 5 dimensions.")
        return None

    if Path(zarr_path).exists() and overwrite:
        shutil.rmtree(zarr_path)
    elif Path(zarr_path).exists() and not overwrite:
        logger.info(f"File already exists at {zarr_path}. Set overwrite=True to remove.")
        return None

    _scale = metadata.scale
    _filename = metadata.filename or "image.czi"

    image = nz.to_ngff_image(
        array5d.data if isinstance(array5d, xr.DataArray) else array5d,  # type: ignore[arg-type]
        dims=["t", "c", "z", "y", "x"],
        scale={
            "y": float(_scale.Y) if (_scale is not None and _scale.Y is not None) else 1.0,
            "x": float(_scale.X) if (_scale is not None and _scale.X is not None) else 1.0,
            "z": float(_scale.Z) if (_scale is not None and _scale.Z is not None) else 1.0,
        },
        name=_filename[:-4] + ".ome.zarr",
    )

    if chunks is None:
        chunks = (1, array5d.shape[1], array5d.shape[2], array5d.shape[3], array5d.shape[4])  # type: ignore[misc]

    multiscales = nz.to_multiscales(
        image,
        scale_factors=scale_factors,
        chunks=chunks,
        method=nz.Methods.DASK_IMAGE_GAUSSIAN,  # type: ignore[attr-defined]
    )

    channels_list = create_channel_list(metadata)
    channels = []
    for ch in channels_list:
        omero_channel = nz.OmeroChannel(
            color=ch["color"],
            window=nz.OmeroWindow(
                min=ch["window"]["min"],
                max=ch["window"]["max"],
                start=ch["window"]["start"],
                end=ch["window"]["end"],
            ),
            label=ch["label"],
        )
        channels.append(omero_channel)
    multiscales.metadata.omero = nz.Omero(channels=channels)

    _use_ts = HAS_TENSORSTORE if use_tensorstore is None else bool(use_tensorstore)
    logger.info("Writing NGFF pyramid (tensorstore=%s)...", _use_ts)

    nz.to_ngff_zarr(
        zarr_path,
        version=version,
        chunks_per_shard=chunks_per_shard,
        use_tensorstore=_use_ts,
        multiscales=multiscales,
    )

    logger.info("NGFF OME-ZARR writing completed successfully!")
    logger.info(f"Output file: {zarr_path}")

    return image
