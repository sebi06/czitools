"""Core CZI -> OME-Zarr conversion functions.

The HCS pipelines consume the canonical layout produced by
:func:`czitools.export_tools.resolver.resolve_hcs_layout`, supporting the
Stage 1 HCS model, sparse plates, and variable field counts per well. All
OME-Zarr outputs use Zarr 3 and retain their backend's native pyramid paths.
"""

import gc
import logging
import os
import shutil
import time
from pathlib import Path

import dask
import dask.array as da
import ngff_zarr as nz
import numpy as np
import ome_zarr.format
import ome_zarr.writer
import xarray as xr
import zarr
from ngff_zarr.hcs import HCSPlateWriter
from ngff_zarr.v04.zarr_metadata import Plate, PlateColumn, PlateRow, PlateWell
from numcodecs import Blosc as NumcodecsBlosc
from numcodecs import Zstd as NumcodecsZstd
from ome_zarr.io import parse_url
from ome_zarr.writer import write_plate_metadata, write_well_metadata
from zarr.codecs import Blosc, Zstd

from czitools.metadata_tools.czi_metadata import CziMetadata
from czitools.read_tools import read_tools

from ._logging import compression_type, setup_logging
from .display import compute_pyramid_scale_factors, create_channel_list, create_ngff_omero_channels, get_fieldimage
from .resolver import resolve_hcs_layout

logger = logging.getLogger(__name__)

# Optional tensorstore backend for ngff-zarr writes (async parallel chunk I/O).
# Enabled automatically when the package is installed; harmless when absent.
try:
    import tensorstore  # type: ignore  # noqa: F401

    HAS_TENSORSTORE = True
except ImportError:  # pragma: no cover
    HAS_TENSORSTORE = False


def _to_ome_zarr_image(array: np.ndarray | xr.DataArray | da.Array) -> np.ndarray | da.Array:
    """Return an array type accepted by ome-zarr writer functions."""
    if isinstance(array, xr.DataArray):
        data = array.data
        if isinstance(data, (np.ndarray, da.Array)):
            return data
        return np.asarray(data)
    return array


def _retry_io(func, *args, _attempts: int = 5, _base_delay: float = 0.2, **kwargs):
    """Run a file-writing callable, retrying on transient Windows ``PermissionError``.

    Windows can raise ``PermissionError`` during zarr's atomic rename when
    antivirus or the search indexer briefly holds a handle to a newly written
    file. A short exponential backoff makes idempotent writes robust.

    Args:
        func: The file-writing callable to invoke.
        *args: Positional arguments forwarded to ``func``.
        _attempts (int): Maximum number of attempts before giving up.
        _base_delay (float): Base delay in seconds for exponential backoff.
        **kwargs: Keyword arguments forwarded to ``func``.

    Returns:
        Any: The return value of ``func``.

    Raises:
        PermissionError: If all attempts fail.
    """
    last_exc: PermissionError | None = None
    for attempt in range(_attempts):
        try:
            return func(*args, **kwargs)
        except PermissionError as exc:
            last_exc = exc
            if attempt == _attempts - 1:
                break
            delay = _base_delay * (2**attempt)
            logger.warning(
                "Transient file lock (%s); retrying in %.2fs (attempt %d/%d)...",
                exc,
                delay,
                attempt + 1,
                _attempts,
            )
            gc.collect()
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _read_single_scene(czi_path: str | os.PathLike | Path, scene_index: int) -> xr.DataArray:
    """Read a single CZI scene as a 6D xarray DataArray.

    Args:
        czi_path (str | os.PathLike | Path): Path to the input CZI file.
        scene_index (int): Zero-based scene index to read.

    Returns:
        xr.DataArray: Array with a length-one scene dimension.

    Raises:
        TypeError: If the scene cannot be read as an xarray DataArray.
    """
    array6d, _ = read_tools.read_6darray(str(czi_path), planes={"S": (scene_index, scene_index)}, use_xarray=True)
    if not isinstance(array6d, xr.DataArray):
        raise TypeError(f"Failed to read scene {scene_index} from {czi_path} as an xarray DataArray")
    return array6d


def _write_image_delayed(
    image,
    group,
    axes: str,
    chunks: str | tuple[int, ...],
    compression: compression_type | None,
    fmt,
    scale: dict[str, float] | None = None,
    axes_units: dict[str, str] | None = None,
) -> list:
    """Schedule an ome-zarr-py image write as parallel Dask tasks.

    Args:
        image: Array data for one image.
        group: Target Zarr group.
        axes (str): Axis string, such as ``"tczyx"``.
        chunks (str | tuple[int, ...]): Dask and Zarr chunk shape.
        compression (compression_type | None): Compression selection.
        fmt: OME-Zarr format instance.
        scale (dict[str, float] | None): Physical pixel sizes by axis.
        axes_units (dict[str, str] | None): Units by axis.

    Returns:
        list: Delayed write tasks, possibly empty.
    """
    if not isinstance(image, da.Array):
        image = da.from_array(image, chunks=chunks)  # type: ignore[arg-type]

    compressor = None
    if compression == compression_type.BLOSC:
        compressor = Blosc()
    elif compression == compression_type.ZSTD:
        compressor = Zstd()
    # compression_type.NONE or None leaves compression disabled.

    delayed = _retry_io(
        ome_zarr.writer.write_image,
        image=image,
        group=group,
        axes=axes,
        method="nearest",
        storage_options={"chunks": chunks, "overwrite": True, "compressors": compressor},
        fmt=fmt,
        scale_factors=[2, 4, 8, 16],
        scale=scale,
        axes_units=axes_units,
        compute=False,
        storage_options_kwargs={"use_tensorstore": HAS_TENSORSTORE},
    )
    return list(delayed) if delayed else []


def _ensure_plate_version_metadata(zarr_path: str | os.PathLike | Path, version: str) -> None:
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
    _retry_io(root.attrs.update, attrs)


# ---------------------------------------------------------------------------
# ome-zarr-py HCS conversion
# ---------------------------------------------------------------------------


def convert_czi2hcs_omezarr(
    czi_filepath: str | os.PathLike | Path,
    overwrite: bool = True,
    log_file_path: str | os.PathLike | Path | None = None,
    pad_columns: bool = True,
    compression: compression_type | None = compression_type.BLOSC,
) -> Path:
    """Convert a CZI file to OME-Zarr HCS format using the ome-zarr-py backend.

    Args:
        czi_filepath (Union[str, os.PathLike, Path]): Path to the input CZI file.
        overwrite (bool): Remove existing output directory if True.
        log_file_path (Optional[Union[str, os.PathLike, Path]]): Log file path.
            Defaults to ``<stem>_hcs_omezarr.log``.
        pad_columns (bool): Zero-pad column numbers in well paths (e.g. ``"04"``).
        compression (Optional[compression_type]): Chunk compression type.
            Defaults to ``compression_type.BLOSC``. Set to ``None`` for no compression

    Returns:
        Path: Output OME-Zarr HCS directory
            (``<stem>_HCSplate_zarr3.ome.zarr``).
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

    zarr_output_path = czi_path.parent / f"{czi_path.stem}_HCSplate_zarr3.ome.zarr"

    if zarr_output_path.exists():
        if overwrite:
            logger.info(f"Removing existing directory: {zarr_output_path}")
            shutil.rmtree(zarr_output_path)
        else:
            logger.info(f"File exists at {zarr_output_path}. Set overwrite=True to remove.")
            return zarr_output_path

    # Read metadata once; scenes are read individually in the field loop below so
    # plates with inconsistent scene shapes (variable well/field sizes) are
    # supported -- reading the whole plate at once returns None in that case.
    mdata = CziMetadata(str(czi_path))

    layout = resolve_hcs_layout(mdata, pad_columns=pad_columns)
    logger.info(f"Resolved plate layout from '{layout.source}': {len(layout.wells)} well(s)")

    _fmt = ome_zarr.format.CurrentFormat()
    logger.info("Zarr storage format: v3")

    parsed = parse_url(zarr_output_path, mode="w", fmt=_fmt)
    assert parsed is not None, f"Failed to open zarr store at {zarr_output_path}"
    root = zarr.group(store=parsed.store)

    well_paths = [w.path for w in layout.wells]
    _retry_io(write_plate_metadata, root, layout.row_names, layout.col_names, well_paths, fmt=_fmt)  # type: ignore[arg-type]

    plate_attrs = root.attrs.asdict()
    plate_attrs["rows"] = [{"name": r} for r in sorted(layout.row_names)]
    plate_attrs["columns"] = [{"name": c} for c in sorted(layout.col_names, key=int)]
    _retry_io(root.attrs.update, plate_attrs)

    # OMERO channel metadata is written on every field image so that readers such
    # as ngio / napari-ome-zarr-navigator can resolve per-channel display settings.
    channels_list = create_channel_list(mdata)

    _mscale = mdata.scale
    _phys_scale = {
        "t": 1.0,
        "c": 1.0,
        "z": float(_mscale.Z) if (_mscale is not None and _mscale.Z is not None) else 1.0,
        "y": float(_mscale.Y) if (_mscale is not None and _mscale.Y is not None) else 1.0,
        "x": float(_mscale.X) if (_mscale is not None and _mscale.X is not None) else 1.0,
    }
    _phys_units = {"t": "second", "z": "micrometer", "y": "micrometer", "x": "micrometer"}
    logger.info(
        "Physical scale (um): X=%.6f  Y=%.6f  Z=%.6f",
        _phys_scale["x"],
        _phys_scale["y"],
        _phys_scale["z"],
    )

    # Collect chunk-parallel write tasks across ALL fields, then execute once with a
    # single dask.compute so fields (and their chunks) are written in parallel.
    delayed_writes: list = []

    for well in layout.wells:
        well_group = root.require_group(well.row).require_group(well.column)
        field_paths = [str(field_index) for field_index, _ in well.fields]
        _retry_io(write_well_metadata, well_group, field_paths, fmt=_fmt)  # type: ignore[arg-type]

        for field_index, scene_index in well.fields:
            image_group = well_group.require_group(str(field_index))
            logger.info(f"Scheduling Well: {well.path}, Field: {field_index}, Scene Index: {scene_index}")
            # Read one scene at a time (scenes may differ in Y/X size across the plate).
            image = _read_single_scene(czi_path, scene_index).isel(S=0)
            # Full Z-stack per (T, C); computed per scene since sizes may vary.
            chunks = (1, 1, image.sizes["Z"], image.sizes["Y"], image.sizes["X"])
            delayed_writes.extend(
                _write_image_delayed(
                    _to_ome_zarr_image(image),
                    image_group,
                    "".join(str(d).lower() for d in image.dims),
                    chunks,
                    compression=compression,
                    fmt=_fmt,
                    scale=_phys_scale,
                    axes_units=_phys_units,
                )
            )
            _retry_io(
                ome_zarr.writer.add_metadata,
                image_group,
                {"omero": {"name": f"{well.path}/{field_index}", "channels": channels_list}},
                fmt=_fmt,
            )

    if delayed_writes:
        logger.info("Writing %d field-pyramid task(s) in parallel (dask)...", len(delayed_writes))
        _retry_io(dask.compute, *delayed_writes)

    logger.info("=" * 80)
    logger.info("Conversion completed successfully!")
    logger.info(f"Output HCS OME-ZARR file: {zarr_output_path}")
    logger.info("=" * 80)

    return zarr_output_path


# ---------------------------------------------------------------------------
# ngff-zarr HCS conversion
# ---------------------------------------------------------------------------


def convert_czi2hcs_ngff(
    czi_filepath: str | os.PathLike | Path,
    plate_name: str = "Automated Plate",
    overwrite: bool = True,
    log_file_path: str | os.PathLike | Path | None = None,
    write_ozx_directly: bool = False,
    version: str = "0.5",
    output_dir: str | os.PathLike | Path | None = None,
    pad_columns: bool = True,
    compression: compression_type | None = compression_type.BLOSC,
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
        compression (Optional[compression_type]): Chunk compression type.
            Defaults to ``compression_type.BLOSC``. Set to ``None`` for no compression.
    Returns:
        Path: Output HCS directory (``<stem>_ngff_plate_zarr3.ome.zarr``) or ``.ozx`` file.
    """
    czi_path = Path(czi_filepath)
    output_path_obj: Path | None = Path(output_dir) if output_dir is not None else None

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
    suffix = "_ngff_plate.ozx" if write_ozx_directly else "_ngff_plate_zarr3.ome.zarr"
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

    # Read metadata once; scenes are read individually in the writer loop below so
    # plates with inconsistent scene shapes (variable well/field sizes) are
    # supported -- reading the whole plate at once returns None in that case.
    mdata = CziMetadata(str(czi_path))

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

    # OMERO channel metadata attached to each field image so readers such as ngio /
    # napari-ome-zarr-navigator can resolve per-channel display settings.
    omero_channels = create_ngff_omero_channels(mdata)

    with HCSPlateWriter(str(zarr_output_path), plate_metadata, overwrite=overwrite) as writer:
        for well in layout.wells:
            logger.info(f"Creating Well: {well.well_id} (Row: {well.row}, Column: {well.column})")
            for field_index, scene_index in well.fields:
                logger.info(f"Writing Well: {well.path}, Field: {field_index}, Scene Index: {scene_index}")
                # Read one scene at a time (scenes may differ in Y/X size across the plate).
                array6d_scene = _read_single_scene(czi_path, scene_index)
                multiscales = get_fieldimage(array6d_scene, 0, mdata)
                if omero_channels:
                    multiscales.metadata.omero = nz.Omero(channels=omero_channels)
                writer.write_well_image(
                    multiscales=multiscales,
                    row_name=well.row,
                    column_name=well.column,
                    field_index=field_index,
                )

    if not write_ozx_directly:
        _ensure_plate_version_metadata(zarr_output_path, version)

    logger.info("=" * 80)
    logger.info("Conversion completed successfully!")
    logger.info(f"Output HCS OME-ZARR file: {zarr_output_path}")
    logger.info("=" * 80)

    return zarr_output_path


# ---------------------------------------------------------------------------
# write_omezarr (ome-zarr-py single image)
# ---------------------------------------------------------------------------


def write_omezarr(
    array5d: np.ndarray | xr.DataArray | da.Array,
    zarr_path: str | Path,
    metadata: CziMetadata,
    overwrite: bool = False,
    log_file_path: str | Path | None = None,
    compression: compression_type | None = compression_type.BLOSC,
) -> Path | None:
    """Write a single 5D image to OME-Zarr using the ome-zarr-py backend.

    Args:
        array5d (Union[np.ndarray, xr.DataArray, da.Array]): Input xarray DataArray
            with named dimensions ``(T, C, Z, Y, X)``.
        zarr_path (Union[str, Path]): Output path for the OME-Zarr file.
        metadata (CziMetadata): Metadata with channel and scale information.
        overwrite (bool): Remove existing output if True.
        log_file_path (Optional[Union[str, Path]]): Log file path. Defaults to
            ``<stem>_omezarr.log``.
        compression (Optional[compression_type]): Chunk compression type.
            Defaults to ``compression_type.BLOSC``. Set to ``None`` for no compression

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

    _fmt = ome_zarr.format.CurrentFormat()
    logger.info("Zarr storage format: v3")

    parsed = parse_url(zarr_path, mode="w", fmt=_fmt)
    assert parsed is not None, f"Failed to open zarr store at {zarr_path}"
    root = zarr.group(store=parsed.store, zarr_format=3)

    # Chunk the full Z-stack per (T, C) instead of one XY plane per chunk. Single-
    # plane chunks explode the file count (T*C*Z chunks/level), which makes writing
    # large images pathologically slow on Windows.
    chunks = (1, 1, array5d.sizes["Z"], array5d.sizes["Y"], array5d.sizes["X"])
    axes = "".join(str(d).lower() for d in array5d.dims)

    _mscale = metadata.scale
    _phys_scale = {
        "t": 1.0,
        "c": 1.0,
        "z": float(_mscale.Z) if (_mscale is not None and _mscale.Z is not None) else 1.0,
        "y": float(_mscale.Y) if (_mscale is not None and _mscale.Y is not None) else 1.0,
        "x": float(_mscale.X) if (_mscale is not None and _mscale.X is not None) else 1.0,
    }
    _phys_units = {"t": "second", "z": "micrometer", "y": "micrometer", "x": "micrometer"}
    logger.info(
        "Physical scale (um): X=%.6f  Y=%.6f  Z=%.6f",
        _phys_scale["x"],
        _phys_scale["y"],
        _phys_scale["z"],
    )

    # Parallel write: dask-wrap + compute=False yields a chunk-parallel write graph
    # that we execute with a single dask.compute (threads release the GIL during
    # zarr chunk writes + compression), roughly halving write time on large images.
    delayed = _write_image_delayed(
        _to_ome_zarr_image(array5d),
        root,
        axes,
        chunks,
        compression=compression,
        fmt=_fmt,
        scale=_phys_scale,
        axes_units=_phys_units,
    )
    if delayed:
        logger.info("Writing %d pyramid level(s) in parallel (dask)...", len(delayed))
        _retry_io(dask.compute, *delayed)

    channels_list = create_channel_list(metadata)
    _retry_io(
        ome_zarr.writer.add_metadata,
        root,
        {
            "omero": {
                "name": metadata.filename,
                "channels": channels_list,
            }
        },
        fmt=_fmt,
    )

    logger.info("OME-ZARR writing completed successfully!")
    logger.info(f"Output file: {zarr_path}")

    return zarr_path


# ---------------------------------------------------------------------------
# write_omezarr_ngff (ngff-zarr single image with pyramid)
# ---------------------------------------------------------------------------


def write_omezarr_ngff(
    array5d: np.ndarray | xr.DataArray | da.Array,
    zarr_path: Path | str,
    metadata: CziMetadata,
    scale_factors: list | None = None,
    overwrite: bool = False,
    version: str = "0.5",
    chunks: tuple | None = None,
    chunks_per_shard: dict[str, int] | int | None = 2,
    compression: compression_type | None = compression_type.BLOSC,
    log_file_path: Path | str | None = None,
    min_size: int = 512,
    max_levels: int = 6,
    use_tensorstore: bool | None = None,
) -> "nz.NgffImage | None":
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
        compression (Optional[compression_type]): Chunk compression type. Defaults to
            ``compression_type.BLOSC``. Set to ``None`` for no compression.
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

    image_data = array5d.data if isinstance(array5d, xr.DataArray) else array5d
    if isinstance(image_data, da.Array):
        image_data = image_data.rechunk({1: 1})
    else:
        image_data = da.from_array(image_data, chunks={1: 1})  # type: ignore[arg-type]

    image = nz.to_ngff_image(
        image_data,
        dims=["t", "c", "z", "y", "x"],
        scale={
            "t": 1.0,
            "c": 1.0,
            "z": float(_scale.Z) if (_scale is not None and _scale.Z is not None) else 1.0,
            "y": float(_scale.Y) if (_scale is not None and _scale.Y is not None) else 1.0,
            "x": float(_scale.X) if (_scale is not None and _scale.X is not None) else 1.0,
        },
        axes_units={
            "t": "second",
            "z": "micrometer",
            "y": "micrometer",
            "x": "micrometer",
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

    # Convert compression_type enum to actual codec instance
    compressor = None
    if compression == compression_type.BLOSC:
        compressor = NumcodecsBlosc()
    elif compression == compression_type.ZSTD:
        compressor = NumcodecsZstd()
    # compression_type.NONE or None → compressor stays None

    nz.to_ngff_zarr(
        zarr_path,
        version=version,
        chunks_per_shard=chunks_per_shard,
        compressor=compressor,
        use_tensorstore=_use_ts,
        multiscales=multiscales,
    )

    logger.info("NGFF OME-ZARR writing completed successfully!")
    logger.info(f"Output file: {zarr_path}")

    return image
