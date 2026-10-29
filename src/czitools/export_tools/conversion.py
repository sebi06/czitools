"""Core CZI -> OME-Zarr conversion functions.

The HCS pipelines consume the canonical layout produced by
:func:`czitools.export_tools.resolver.resolve_hcs_layout`, supporting the
Stage 1 HCS model, sparse plates, and variable field counts per well. All
OME-Zarr outputs use Zarr 3 and retain their backend's native pyramid paths.
"""

import gc
import json
import logging
import os
import shutil
import sys
import threading
import time
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator, Literal

import dask
import dask.array as da
import ngff_zarr as nz
from ngff_zarr.config import config as ngff_config
import numpy as np
import ome_zarr.format
import ome_zarr.writer
import psutil
import xarray as xr
import zarr
from ngff_zarr.hcs import HCSPlateWriter
from ngff_zarr.rfc9_zip import write_store_to_zip
from ngff_zarr.rich_dask_progress import NgffProgressCallback
from ngff_zarr.v04.zarr_metadata import (
    Axis,
    Dataset,
    Metadata,
    Plate,
    PlateColumn,
    PlateRow,
    PlateWell,
    Scale,
    Translation,
)
from numcodecs import Blosc as NumcodecsBlosc
from numcodecs import Zstd as NumcodecsZstd
from ome_zarr import OMEZarrImage, OMEZarrMultiscale
from ome_zarr.io import parse_url
from ome_zarr.writer import write_plate_metadata, write_well_metadata
from pylibCZIrw import czi as pyczi
from zarr.codecs import Blosc, Zstd

from czitools.metadata_tools.czi_metadata import CziMetadata
from czitools.read_tools import read_tools

from ._logging import compression_type, setup_logging
from .display import compute_pyramid_scale_factors, create_channel_list, create_ngff_omero_channels
from .resolver import resolve_hcs_layout

logger = logging.getLogger(__name__)

LoggingDetail = Literal["basic", "full"]

DEFAULT_PYRAMID_LEVELS = 3
QUALITY_CHUNKS = (1, 1, 8, 512, 512)
QUALITY_CHUNKS_PER_SHARD = {"z": 4, "y": 4, "x": 4}
_NGFF_CACHE_LOCK = threading.RLock()


def _write_original_czi_metadata(czi_filepath: str | os.PathLike, zarr_path: Path) -> None:
    """Preserve source XML in a directory store without reserializing it."""
    with pyczi.open_czi(str(czi_filepath)) as source:
        xml = source.raw_metadata
    if not isinstance(xml, str) or not xml.strip():
        raise ValueError(f"Original CZI XML metadata is unavailable: {czi_filepath}")

    root = zarr.open_group(zarr_path, mode="r+")
    root.require_group("CZI")
    (zarr_path / "CZI" / "metadata.xml").write_bytes(xml.encode("utf-8"))
    attributes = dict(root.attrs.get("czitools", {}))
    attributes["originalMetadata"] = {
        "path": "CZI/metadata.xml",
        "format": "ZEISS-CZI-XML",
        "mediaType": "application/xml",
        "sourceFilename": Path(czi_filepath).name,
    }
    root.attrs["czitools"] = attributes
    logger.info("Preserved original CZI XML at CZI/metadata.xml")


@contextmanager
def _original_metadata_output(
    czi_filepath: str | os.PathLike,
    output_path: Path,
    preserve_original_metadata: bool,
    version: str,
) -> Iterator[Path]:
    """Attach source XML before packaging an optional RFC-9 archive."""
    if not preserve_original_metadata:
        yield output_path
        return

    is_archive = output_path.suffix.lower() == ".ozx"
    if is_archive and version not in {"0.5", "0.6"}:
        raise ValueError("RFC-9 .ozx output requires OME-NGFF version 0.5 or 0.6.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = (
        TemporaryDirectory(prefix=".czitools-metadata-", dir=output_path.parent) if is_archive else nullcontext(None)
    )
    with staging as directory:
        store_path = Path(directory) / "image.ome.zarr" if directory is not None else output_path
        yield store_path
        _write_original_czi_metadata(czi_filepath, store_path)
        if is_archive:
            try:
                with _logged_progress_bar("OZX archive finalization"):
                    write_store_to_zip(store_path, output_path, version=version)
            except BaseException:
                output_path.unlink(missing_ok=True)
                raise


def _resolve_worker_count(worker_count: int = 0) -> int:
    """Choose a conservative thread count from available RAM, or honor an override."""
    if worker_count < 0:
        raise ValueError("worker_count must be non-negative (0 selects Auto).")
    if worker_count:
        return worker_count
    cpu_limit = max(1, (os.cpu_count() or 2) // 2)
    try:
        memory = psutil.virtual_memory()
    except (OSError, psutil.Error):
        workers = min(2, cpu_limit)
        logger.warning("Cannot determine available RAM; Auto uses %d Dask worker(s).", workers)
        return workers
    reserve = max(2 * 2**30, memory.total // 5)
    memory_limit = max(1, (memory.available - reserve) // 2**30)
    workers = min(8, cpu_limit, memory_limit)
    logger.info(
        "Auto Dask workers: %d (available RAM %.1f GiB; reserve %.1f GiB; 1 GiB/worker; CPU limit %d; cap 8)",
        workers,
        memory.available / 2**30,
        reserve / 2**30,
        cpu_limit,
    )
    return workers


@contextmanager
def _ngff_conversion_resources(output_path: Path, use_cache: bool = True, worker_count: int = 0) -> Iterator[None]:
    """Limit conversion concurrency and clean up its output-local disk cache."""
    worker_count = _resolve_worker_count(worker_count)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_context = (
        TemporaryDirectory(prefix=".czitools-cache-", dir=output_path.parent) if use_cache else nullcontext(None)
    )
    with _NGFF_CACHE_LOCK, cache_context as cache_dir:
        previous_store = ngff_config.cache_store
        previous_target = ngff_config.memory_target
        try:
            if cache_dir is not None:
                ngff_config.cache_store = Path(cache_dir)
                logger.info("Temporary NGFF cache: %s", cache_dir)
            ngff_config.memory_target = 256 * 1024 * 1024
            logger.info("Dask worker count: %d", worker_count)
            with dask.config.set(scheduler="threads", num_workers=worker_count):
                yield
        finally:
            ngff_config.cache_store = previous_store
            ngff_config.memory_target = previous_target


def _cache_ngff_image(
    image: nz.NgffImage, chunks: tuple, method: nz.Methods, label: str = "base image"
) -> nz.NgffImage:
    """Stage an image to disk, using strips for singleton-Z images."""
    dims = list(image.dims)
    singleton_z = "z" in dims and image.data.shape[dims.index("z")] == 1
    z_index = dims.index("z") if singleton_z else None
    cache_dims = [dim for dim in dims if dim != "z"] if singleton_z else dims
    cache_data = image.data.squeeze(axis=z_index) if singleton_z else image.data
    logger.info(
        "Staging %s to disk: shape=%s, uncompressed=%.2f GiB", label, image.data.shape, image.data.nbytes / 2**30
    )
    with _logged_progress_bar(f"NGFF cache {label}") as progress:
        cached = nz.to_multiscales(
            nz.NgffImage(
                data=cache_data,
                dims=cache_dims,
                scale={dim: image.scale[dim] for dim in cache_dims},
                translation={dim: image.translation.get(dim, 0.0) for dim in cache_dims},
                axes_units=image.axes_units,
                name=image.name,
            ),
            scale_factors=[],
            chunks=tuple(chunk for dim, chunk in zip(dims, chunks) if dim in cache_dims),
            method=method,
            cache=True,
            progress=progress,
        ).images[0]
    return nz.NgffImage(
        data=da.expand_dims(cached.data, axis=z_index) if singleton_z else cached.data,
        dims=dims,
        scale=dict(image.scale),
        translation=dict(image.translation),
        axes_units=image.axes_units,
        name=image.name,
    )


def _bounded_ngff_multiscales(
    image: nz.NgffImage, scale_factors: list, chunks: tuple, method: nz.Methods
) -> nz.NgffMultiscales:
    """Build each pyramid level from the preceding disk-backed level."""
    images = [_cache_ngff_image(image, chunks, method)]
    previous_factors = dict.fromkeys(image.dims, 1)
    for index, factor in enumerate(scale_factors, start=1):
        absolute = factor if isinstance(factor, dict) else {dim: factor for dim in ("z", "y", "x")}
        relative = {dim: value / previous_factors[dim] for dim, value in absolute.items()}
        logger.info("Building disk-backed pyramid level %d of %d", index, len(scale_factors))
        step = nz.to_multiscales(images[-1], scale_factors=[relative], chunks=chunks, method=method, cache=False)
        level = step.images[-1]
        level.name = image.name
        images.append(_cache_ngff_image(level, chunks, method, label=f"level {index}/{len(scale_factors)}"))
        previous_factors.update(absolute)
    return _ngff_multiscales_metadata(images, scale_factors, chunks, method)


def _ngff_multiscales_metadata(
    images: list[nz.NgffImage], scale_factors: list, chunks: tuple, method: nz.Methods
) -> nz.NgffMultiscales:
    """Describe completed levels without constructing additional pixel graphs."""
    image = images[0]
    dims = list(image.dims)
    datasets = [
        Dataset(
            path=f"scale{index}/{image.name}",
            coordinateTransformations=[
                Scale([level.scale[dim] for dim in dims]),
                Translation([level.translation.get(dim, 0.0) for dim in dims]),
            ],
        )
        for index, level in enumerate(images)
    ]
    axes = [
        Axis(
            name=dim,
            type="space" if dim in {"z", "y", "x"} else "channel" if dim == "c" else "time",
            unit=(image.axes_units or {}).get(dim),
        )
        for dim in dims
    ]
    return nz.NgffMultiscales(
        images=images,
        metadata=Metadata(axes=axes, datasets=datasets, name=image.name, coordinateTransformations=None),
        scale_factors=scale_factors,
        method=method,
        chunks=chunks,
    )


def _write_ngff_directory(
    output_path: Path,
    image: nz.NgffImage,
    scale_factors: list,
    chunks: tuple,
    method: nz.Methods,
    version: str,
    chunks_per_shard: dict[str, int] | int | None,
    compressor,
    channels: list,
    source_multiscales: nz.NgffMultiscales | None = None,
) -> None:
    """Write supplied levels directly, or derive each from its stored predecessor."""
    images = [image]
    previous_factors = dict.fromkeys(image.dims, 1)
    level_count = len(source_multiscales.images) if source_multiscales is not None else len(scale_factors) + 1
    for index in range(level_count):
        if index and source_multiscales is not None:
            images.append(source_multiscales.images[index])
        elif index:
            factor = scale_factors[index - 1]
            absolute = factor if isinstance(factor, dict) else {dim: factor for dim in ("z", "y", "x")}
            relative = {dim: value / previous_factors[dim] for dim, value in absolute.items()}
            logger.info("Building level %d from the preceding written level", index)
            step = nz.to_multiscales(images[-1], scale_factors=[relative], chunks=chunks, method=method, cache=False)
            level = step.images[-1]
            level.name = image.name
            images.append(level)
            previous_factors.update(absolute)
        multiscales = _ngff_multiscales_metadata(
            images, [] if source_multiscales is not None else scale_factors[:index], chunks, method
        )
        multiscales.metadata.omero = nz.Omero(channels=channels)
        logger.info("Writing level %d/%d directly to final store", index, level_count - 1)
        with _logged_progress_bar(f"NGFF write level {index}/{level_count - 1}") as progress:
            nz.to_ngff_zarr(
                output_path,
                multiscales=multiscales,
                version=version,
                overwrite=index == 0,
                start_level=index,
                chunks_per_shard=chunks_per_shard,
                compressor=compressor,
                progress=progress,
            )
        images[-1].data = da.from_zarr(str(output_path), component=multiscales.metadata.datasets[-1].path)


def _pyramid_scale_factors(pyramid_levels: int) -> list[dict[str, int]]:
    """Return cumulative Y/X scale factors for additional pyramid levels."""
    if pyramid_levels < 0:
        raise ValueError("pyramid_levels must be at least 0.")
    return [{"z": 1, "y": 2**level, "x": 2**level} for level in range(1, pyramid_levels + 1)]


def _shard_shape(
    chunks: tuple[int, ...],
    chunks_per_shard: dict[str, int],
) -> tuple[int, ...]:
    """Return a TCZYX shard shape while preserving T/C independence."""
    dims = ("t", "c", "z", "y", "x")
    return tuple(chunk * chunks_per_shard.get(dim, 1) for dim, chunk in zip(dims, chunks, strict=True))


def _log_omezarr_layout(zarr_path: str | os.PathLike | Path) -> None:
    """Log the effective chunk and shard layout of every Zarr v3 array."""
    path = Path(zarr_path)
    metadata_entries: list[tuple[str, dict]] = []
    try:
        if path.is_dir():
            for metadata_path in sorted(path.rglob("zarr.json")):
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata_entries.append((metadata_path.relative_to(path).as_posix(), metadata))
        elif path.is_file() and zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                for name in sorted(name for name in archive.namelist() if name.endswith("zarr.json")):
                    metadata_entries.append((name, json.loads(archive.read(name))))
    except (OSError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        logger.warning("Could not verify OME-Zarr chunk and shard layout for %s: %s", path, exc)
        return

    array_count = 0
    for name, metadata in metadata_entries:
        if metadata.get("node_type") != "array":
            continue
        array_count += 1
        shard_codec = next(
            (codec for codec in metadata.get("codecs", []) if codec.get("name") == "sharding_indexed"),
            None,
        )
        outer_shape = tuple(metadata["chunk_grid"]["configuration"]["chunk_shape"])
        inner_shape = tuple(shard_codec["configuration"]["chunk_shape"]) if shard_codec is not None else outer_shape
        if shard_codec is None:
            logger.warning(
                "OME-Zarr layout is unsharded: path=%s shape=%s chunk=%s codecs=%s",
                name,
                tuple(metadata["shape"]),
                inner_shape,
                [codec.get("name") for codec in metadata.get("codecs", [])],
            )
            continue
        logger.info(
            "OME-Zarr layout verified: path=%s shape=%s inner_chunk=%s shard=%s codec=sharding_indexed",
            name,
            tuple(metadata["shape"]),
            inner_shape,
            outer_shape,
        )

    if array_count == 0:
        logger.warning("No Zarr v3 array metadata found for layout verification: %s", path)


def _validate_logging_detail(logging_detail: LoggingDetail) -> bool:
    """Return whether full HCS logging is enabled."""
    if logging_detail not in {"basic", "full"}:
        raise ValueError("logging_detail must be 'basic' or 'full'.")
    return logging_detail == "full"


class _LoggedNgffProgress(NgffProgressCallback):
    """Adapt ngff-zarr scale and Dask task updates to the conversion log."""

    def __init__(self, label: str, started: float) -> None:
        super().__init__(self)
        self.label = label
        self.started = started
        self.completed = 0
        self.total = 0
        self.width = 30
        self._task_counter = 0
        self._task_description = ""
        self._task_completed = 0
        self._task_total = 0
        self._task_visible = False
        self._finalizing_storage = False
        self._last_task_percent = -1

    def add_task(self, description: str, total: int | None = None) -> int:
        """Create a log-backed task for ngff-zarr's Dask callback."""
        self._task_counter += 1
        self._task_description = description.replace("[green]", "").replace("[blue]", "")
        self._task_completed = 0
        self._task_total = total or 0
        self._task_visible = True
        self._finalizing_storage = False
        self._last_task_percent = -1
        return self._task_counter

    def update(
        self,
        task_id: int,
        *,
        total: int | None = None,
        completed: int | None = None,
        visible: bool | None = None,
        **kwargs,
    ) -> None:
        """Receive task updates from ngff-zarr's registered Dask callback."""
        del task_id, kwargs
        if total is not None:
            self._task_total = total
        if completed is not None:
            self._task_completed = completed
        if visible is not None:
            self._task_visible = visible
            if not visible and self._task_total:
                self._finalizing_storage = True
                self._log_progress()
                return

        if self._task_visible and self._task_total:
            percent = min(99, 100 * self._task_completed // self._task_total)
            crossed_ten_percent = percent // 10 > self._last_task_percent // 10
            reached_final_task_percent = percent == 99 and self._last_task_percent != 99
            if self._last_task_percent < 0 or crossed_ten_percent or reached_final_task_percent:
                self._last_task_percent = percent
                self._log_progress()

    def add_multiscales_task(self, description: str, scales: int) -> None:
        del description
        self.total = scales
        self._log_progress()

    def update_multiscales_task_completed(self, completed: int) -> None:
        # ngff-zarr sends the one-based scale index before writing that scale.
        self.completed = min(max(completed - 1, 0), self.total)
        self._log_progress(active_scale=completed)

    def _log_progress(self, active_scale: int | None = None) -> None:
        if self._task_visible and self._task_total:
            completed = min(self._task_completed, self._task_total)
            percent = min(99, 100 * completed // self._task_total)
            filled = min(self.width - 1, self.width * completed // self._task_total)
            bar = "=" * filled + "." * (self.width - filled)
            logger.info(
                "%s progress: [%s] %d%% (%d/%d tasks, %s, elapsed %.1f s)",
                self.label,
                bar,
                percent,
                completed,
                self._task_total,
                self._task_description,
                time.perf_counter() - self.started,
            )
        elif self._finalizing_storage:
            logger.info(
                "%s progress: [%s.] finalizing storage (Dask tasks complete, elapsed %.1f s)",
                self.label,
                "=" * (self.width - 1),
                time.perf_counter() - self.started,
            )
        elif self.total:
            percent = round(100 * self.completed / self.total)
            filled = self.width * self.completed // self.total
            bar = "=" * filled + "." * (self.width - filled)
            scale_status = f", scale {active_scale}/{self.total}" if active_scale else ""
            logger.info(
                "%s progress: [%s] %d%% (%d/%d scales%s, elapsed %.1f s)",
                self.label,
                bar,
                percent,
                self.completed,
                self.total,
                scale_status,
                time.perf_counter() - self.started,
            )
        else:
            logger.info(
                "%s progress: [>%s] elapsed %.1f s",
                self.label,
                "." * (self.width - 1),
                time.perf_counter() - self.started,
            )


@contextmanager
def _logged_progress_bar(label: str) -> Iterator[_LoggedNgffProgress]:
    """Log meaningful NGFF progress transitions without periodic duplicates."""
    started = time.perf_counter()
    progress = _LoggedNgffProgress(label, started)

    progress._log_progress()
    progress.register()
    try:
        yield progress
    finally:
        progress.unregister()

    logger.info(
        "%s progress: [%s] 100%% (elapsed %.1f s)",
        label,
        "=" * progress.width,
        time.perf_counter() - started,
    )


@contextmanager
def _hcs_plate_writer(
    zarr_output_path: Path,
    plate_metadata: Plate,
    overwrite: bool,
    write_ozx_directly: bool,
    version: str,
) -> Iterator[HCSPlateWriter]:
    """Open an HCS writer and report otherwise-silent OZX finalization."""
    writer = HCSPlateWriter(
        str(zarr_output_path),
        plate_metadata,
        version=version,
        overwrite=overwrite,
    )
    writer.__enter__()
    try:
        yield writer
    except BaseException:
        writer.__exit__(*sys.exc_info())
        raise
    else:
        if not write_ozx_directly:
            writer.__exit__(None, None, None)
            return

        logger.info("All HCS fields written. Finalizing the OZX archive...")
        try:
            with _logged_progress_bar("OZX archive finalization"):
                writer.__exit__(None, None, None)
        except BaseException:
            if zarr_output_path.is_file():
                zarr_output_path.unlink()
                logger.warning("Removed incomplete OZX archive: %s", zarr_output_path)
            raise


def _log_hcs_progress(completed: int, total: int, started: float, full_logging: bool = False) -> None:
    """Log HCS field progress, limiting basic mode to ten-percent steps."""
    if not full_logging and 0 < completed < total:
        current_step = completed * 10 // total
        previous_step = (completed - 1) * 10 // total
        if current_step == previous_step:
            return

    width = 30
    percent = round(100 * completed / total) if total else 100
    filled = width * completed // total if total else width
    bar = "=" * filled + "." * (width - filled)
    logger.info(
        "HCS conversion progress: [%s] %d%% (%d/%d fields, elapsed %.1f s)",
        bar,
        percent,
        completed,
        total,
        time.perf_counter() - started,
    )


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


def _validate_h_size(size_h: int | None, location: str) -> None:
    """Reject CZI phase dimensions that cannot be exported interoperably."""
    if size_h is None or size_h <= 1:
        return
    raise ValueError(
        f"CZI {location} has SizeH={size_h}. The H dimension is a CZI-specific phase axis "
        "that cannot be represented alongside the channel axis in a standard OME-NGFF multiscale image, "
        "and common OME-Zarr viewers do not support it. Only SizeH=1 is supported and dropped during export."
    )


def _validate_metadata_h_dimension(metadata: CziMetadata) -> None:
    """Validate the H dimension reported by CZI metadata before writing."""
    image = metadata.image
    _validate_h_size(image.SizeH if image is not None else None, "metadata")


def _log_hcs_scene_dimensions(metadata: CziMetadata) -> None:
    """Log the metadata-reported dimensions of an individual HCS scene."""
    image = metadata.image
    if image is None:
        logger.warning("Individual scene dimensions unavailable in CZI metadata.")
        return

    scene_x = getattr(image, "SizeX_scene", None) or getattr(image, "SizeX", None)
    scene_y = getattr(image, "SizeY_scene", None) or getattr(image, "SizeY", None)
    logger.info(
        "Individual scene dimensions: X=%s, Y=%s, C=%s, Z=%s, T=%s",
        scene_x,
        scene_y,
        getattr(image, "SizeC", None),
        getattr(image, "SizeZ", None),
        getattr(image, "SizeT", None),
    )


def _normalize_h_dimension(array: xr.DataArray, scene_index: int) -> xr.DataArray:
    """Drop a singleton CZI H dimension or reject unsupported H phases."""
    if "H" not in array.dims:
        return array
    _validate_h_size(array.sizes["H"], f"scene {scene_index}")
    return array.isel(H=0, drop=True)


def _read_single_scene_multiscales(
    czi_path: str | os.PathLike | Path,
    scene_index: int,
    metadata: CziMetadata,
    pyramid_zooms: list[float],
    spatial_chunk_size: int,
    planes_per_chunk: int,
    full_logging: bool = False,
    tile_size: int = 4096,
    chunk_memory_limit: int = 256 * 1024 * 1024,
) -> nz.NgffMultiscales:
    """Read a scene's stored CZI levels, extending only the coarse tail if needed."""
    levels, level_infos, _, _, _ = read_tools.read_stacks_multiscale(
        filepath=czi_path,
        use_xarray=True,
        stack_scenes=True,
        planes={"S": (scene_index, scene_index)},
        zooms=pyramid_zooms,
        max_coarse_edge=512,
        planes_per_chunk=planes_per_chunk,
        metadata=metadata,
        tile_size=tile_size,
        chunk_memory_limit=chunk_memory_limit,
    )

    level_arrays: list[xr.DataArray] = []
    for level in levels:
        if isinstance(level, list):
            if len(level) != 1 or not isinstance(level[0], xr.DataArray):
                raise TypeError(f"Failed to read scene {scene_index} pyramid level as an xarray DataArray")
            level_array = level[0]
        elif isinstance(level, xr.DataArray):
            level_array = level.isel(S=0, drop=True) if "S" in level.dims else level
        else:
            raise TypeError(f"Failed to read scene {scene_index} pyramid level as an xarray DataArray")
        level_array = _normalize_h_dimension(level_array, scene_index)
        level_arrays.append(
            level_array.chunk({dim: spatial_chunk_size if dim in {"Y", "X"} else 1 for dim in level_array.dims})
        )

    if not level_arrays:
        raise ValueError(f"No pyramid levels found for scene {scene_index} in {czi_path}")

    dims = [str(dim).lower() for dim in level_arrays[0].dims]
    unsupported_dims = set(dims) - {"t", "c", "z", "y", "x"}
    if unsupported_dims:
        raise ValueError(f"Unsupported HCS image dimensions: {sorted(unsupported_dims)}")

    physical_scale = metadata.scale
    base_scale = {
        "t": 1.0,
        "c": 1.0,
        "z": float(physical_scale.Z) if physical_scale is not None and physical_scale.Z is not None else 1.0,
        "y": float(physical_scale.Y) if physical_scale is not None and physical_scale.Y is not None else 1.0,
        "x": float(physical_scale.X) if physical_scale is not None and physical_scale.X is not None else 1.0,
    }
    axes_units = {"t": "second", "z": "micrometer", "y": "micrometer", "x": "micrometer"}
    base_shape = dict(zip(dims, level_arrays[0].shape))
    images: list[nz.NgffImage] = []
    datasets: list[Dataset] = []
    image_name = metadata.filename or "image.czi"

    for index, level_array in enumerate(level_arrays):
        level_shape = dict(zip(dims, level_array.shape))
        scale = dict(base_scale)
        translation = dict.fromkeys(dims, 0.0)
        for dim in ("y", "x"):
            if dim in level_shape:
                factor = base_shape[dim] / level_shape[dim]
                scale[dim] = base_scale[dim] * factor
                translation[dim] = 0.5 * (factor - 1.0) * base_scale[dim]

        image = nz.NgffImage(
            data=level_array.data,
            dims=dims,
            scale={dim: scale[dim] for dim in dims},
            translation=translation,
            name=image_name,
            axes_units={dim: unit for dim, unit in axes_units.items() if dim in dims},
        )
        images.append(image)
        datasets.append(
            Dataset(
                path=f"scale{index}/{image_name}",
                coordinateTransformations=[
                    Scale([image.scale[dim] for dim in dims]),
                    Translation([image.translation[dim] for dim in dims]),
                ],
            )
        )

    axes = [
        Axis(
            name=dim,
            type="space" if dim in {"z", "y", "x"} else "channel" if dim == "c" else "time",
            unit=axes_units.get(dim),
        )
        for dim in dims
    ]
    multiscales_metadata = Metadata(
        axes=axes,
        datasets=datasets,
        coordinateTransformations=None,
        name=image_name,
    )
    if full_logging:
        logger.info(
            "Using %d CZI-backed pyramid level(s) for scene %d: %s",
            len(level_infos),
            scene_index,
            [f"{level.zoom:g} ({'stored' if level.stored else 'native zoom'})" for level in level_infos],
        )
    return nz.NgffMultiscales(images=images, metadata=multiscales_metadata)


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
    )
    return list(delayed) if delayed else []


def _write_czi_multiscales_delayed(
    multiscales: nz.NgffMultiscales,
    group,
    chunks: tuple[int, ...],
    compression: compression_type | None,
    fmt,
    shards: tuple[int, ...] | None = None,
) -> list:
    """Schedule supplied CZI levels without ome-zarr-py downsampling."""
    image = multiscales.images[0]
    compressor = (
        Blosc() if compression == compression_type.BLOSC else Zstd() if compression == compression_type.ZSTD else None
    )
    storage_options = {"chunks": chunks, "overwrite": True, "compressors": compressor}
    if shards is not None:
        storage_options["shards"] = shards
    transforms = [
        [
            {"type": "scale", "scale": [level.scale[dim] for dim in image.dims]},
            {"type": "translation", "translation": [level.translation.get(dim, 0.0) for dim in image.dims]},
        ]
        for level in multiscales.images
    ]
    delayed = _retry_io(
        ome_zarr.writer.write_multiscale,
        pyramid=[level.data.rechunk(chunks) for level in multiscales.images],
        group=group,
        axes=list(image.dims),
        scale=image.scale,
        axes_units=image.axes_units,
        coordinate_transformations=transforms,
        storage_options=storage_options,
        name=image.name,
        fmt=fmt,
        compute=False,
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
    logging_detail: LoggingDetail = "basic",
    preserve_original_metadata: bool = False,
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
        logging_detail (Literal["basic", "full"]): Logging detail. Basic logging
            reports milestones; full logging also reports every field. Defaults
            to ``"basic"``.
        preserve_original_metadata (bool): Save the original CZI XML at
            ``CZI/metadata.xml`` with a root ``czitools.originalMetadata`` reference.
            Defaults to False; the XML is not converted to OME-XML.

    Returns:
        Path: Output OME-Zarr HCS directory
            (``<stem>_HCSplate_zarr3.ome.zarr``).
    """
    full_logging = _validate_logging_detail(logging_detail)
    czi_path = Path(czi_filepath)
    if log_file_path is None:
        log_file_path = czi_path.parent / f"{czi_path.stem}_hcs_omezarr.log"
    else:
        log_file_path = Path(log_file_path)

    setup_logging(log_file_path, include_internal_info=full_logging)

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
    _validate_metadata_h_dimension(mdata)
    _log_hcs_scene_dimensions(mdata)

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
    logger.info(
        "Physical scale (um): X=%.6f  Y=%.6f  Z=%.6f",
        _phys_scale["x"],
        _phys_scale["y"],
        _phys_scale["z"],
    )

    # Collect chunk-parallel write tasks across ALL fields, then execute once with a
    # single dask.compute so fields (and their chunks) are written in parallel.
    delayed_writes: list = []
    pyramid_zooms = read_tools.get_pyramid_zooms(str(czi_path))

    for well in layout.wells:
        well_group = root.require_group(well.row).require_group(well.column)
        field_paths = [str(field_index) for field_index, _ in well.fields]
        _retry_io(write_well_metadata, well_group, field_paths, fmt=_fmt)  # type: ignore[arg-type]

        for field_index, scene_index in well.fields:
            image_group = well_group.require_group(str(field_index))
            if full_logging:
                logger.info(f"Scheduling Well: {well.path}, Field: {field_index}, Scene Index: {scene_index}")
            multiscales = _read_single_scene_multiscales(
                czi_path,
                scene_index,
                mdata,
                pyramid_zooms,
                512,
                1,
                full_logging=full_logging,
                chunk_memory_limit=32 * 1024 * 1024,
            )
            chunks = tuple(512 if dim in {"y", "x"} else 1 for dim in multiscales.images[0].dims)
            delayed_writes.extend(_write_czi_multiscales_delayed(multiscales, image_group, chunks, compression, _fmt))
            _retry_io(
                ome_zarr.writer.add_metadata,
                image_group,
                {"omero": {"name": f"{well.path}/{field_index}", "channels": channels_list}},
                fmt=_fmt,
            )

    if delayed_writes:
        logger.info("Writing %d field-pyramid task(s) in parallel (dask)...", len(delayed_writes))
        _retry_io(dask.compute, *delayed_writes)

    if preserve_original_metadata:
        _write_original_czi_metadata(czi_path, zarr_output_path)
    _log_omezarr_layout(zarr_output_path)
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
    spatial_chunk_size: int = 512,
    chunks_per_shard: dict[str, int] | int | tuple[int, ...] | None = {"y": 4, "x": 4},
    planes_per_chunk: int = 64,
    max_workers: int = 4,
    logging_detail: LoggingDetail = "basic",
    preserve_original_metadata: bool = False,
) -> Path:
    """Convert a CZI file to OME-Zarr HCS format using the ngff-zarr backend.

    Args:
        czi_filepath (Union[str, os.PathLike, Path]): Path to the input CZI file.
        plate_name (str): Name for the well plate in metadata.
        overwrite (bool): Remove existing output if True.
        log_file_path (Optional[Union[str, os.PathLike, Path]]): Log file path.
            Defaults to ``<stem>_hcs_ngff.log``.
        write_ozx_directly (bool): Write a single-file ``.ozx`` archive directly.
        version (str): NGFF version string. Defaults to ``"0.5"``. The
            ngff-zarr HCS writer supports versions 0.4, 0.5, and 0.6.
        output_dir (Optional[Union[str, os.PathLike, Path]]): Output directory.
            Defaults to the CZI file's parent directory.
        pad_columns (bool): Zero-pad column numbers in well paths (e.g. ``"04"``).
        compression (Optional[compression_type]): Chunk compression type.
            Defaults to ``compression_type.BLOSC``. Set to ``None`` for no compression.
        spatial_chunk_size (int): Y/X chunk edge in pixels. Defaults to 512.
        chunks_per_shard (Optional[Union[dict[str, int], int, tuple[int, ...]]]):
            Number of chunks per shard. Defaults to four chunks along Y and X,
            yielding one spatial shard for a typical 2000 x 2000 field.
        planes_per_chunk (int): Maximum T/C/Z planes read per CZI file open.
            Defaults to 64.
        max_workers (int): Maximum number of fields read and written concurrently.
            Defaults to 4 to overlap CZI reads, compression, and writes while
            keeping memory bounded to a small number of fields. Use 1 for very
            large fields or deep T/Z stacks when minimizing peak memory matters.
        logging_detail (Literal["basic", "full"]): Logging detail. Basic logging
            reports milestones and bounded progress; full logging also reports
            every well, field, and scene pyramid. Defaults to ``"basic"``.
        preserve_original_metadata (bool): Save the original CZI XML at
            ``CZI/metadata.xml`` with a root ``czitools.originalMetadata`` reference,
            including in ``.ozx`` archives. Defaults to False.
    Returns:
        Path: Output HCS directory (``<stem>_ngff_plate_zarr3.ome.zarr``) or ``.ozx`` file.
    """
    conversion_started = time.perf_counter()
    full_logging = _validate_logging_detail(logging_detail)
    if version not in {"0.4", "0.5", "0.6"}:
        raise ValueError("ngff-zarr HCS writing currently supports OME-NGFF versions 0.4, 0.5, and 0.6.")
    if spatial_chunk_size < 1:
        raise ValueError("spatial_chunk_size must be at least 1.")
    if planes_per_chunk < 1:
        raise ValueError("planes_per_chunk must be at least 1.")
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1.")

    czi_path = Path(czi_filepath)
    output_path_obj: Path | None = Path(output_dir) if output_dir is not None else None

    if log_file_path is None:
        base = output_path_obj if output_path_obj is not None else czi_path.parent
        log_file_path = base / f"{czi_path.stem}_hcs_ngff.log"
    else:
        log_file_path = Path(log_file_path)

    setup_logging(log_file_path, include_internal_info=full_logging)

    logger.info("=" * 80)
    logger.info("CZI to HCS OME-ZARR Conversion Started (ngff-zarr backend)")
    logger.info("=" * 80)
    logger.info(f"Input CZI file: {czi_path.absolute()}")
    logger.info(f"Plate name: {plate_name}")
    logger.info(
        "NGFF HCS tuning: chunk=%dx%d, chunks_per_shard=%s, planes_per_chunk=%d, max_workers=%d",
        spatial_chunk_size,
        spatial_chunk_size,
        chunks_per_shard,
        planes_per_chunk,
        max_workers,
    )

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
    _validate_metadata_h_dimension(mdata)
    _log_hcs_scene_dimensions(mdata)

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
    pyramid_zooms = read_tools.get_pyramid_zooms(czi_path)
    logger.info("Detected stored CZI pyramid zooms: %s", pyramid_zooms)
    if pyramid_zooms == [1.0]:
        logger.warning(
            "The CZI has no stored lower-resolution pyramid levels. "
            "Coarse levels will be read with libCZI native zoom instead of Gaussian resampling."
        )

    total_fields = sum(len(well.fields) for well in layout.wells)
    completed_fields = 0
    _log_hcs_progress(completed_fields, total_fields, conversion_started, full_logging)

    compressor = None
    if compression == compression_type.BLOSC:
        compressor = NumcodecsBlosc()
    elif compression == compression_type.ZSTD:
        compressor = NumcodecsZstd()

    write_options = {"compressor": compressor}
    if version != "0.4":
        write_options["chunks_per_shard"] = chunks_per_shard

    with (
        _original_metadata_output(czi_path, zarr_output_path, preserve_original_metadata, version) as output_store,
        _hcs_plate_writer(
            output_store,
            plate_metadata,
            overwrite,
            write_ozx_directly and not preserve_original_metadata,
            version,
        ) as writer,
    ):
        field_jobs = []
        for well in layout.wells:
            if full_logging:
                logger.info(f"Creating Well: {well.well_id} (Row: {well.row}, Column: {well.column})")
            for field_index, scene_index in well.fields:
                field_jobs.append((well, field_index, scene_index))

        def write_field(well, field_index: int, scene_index: int) -> None:
            if full_logging:
                logger.info(f"Writing Well: {well.path}, Field: {field_index}, Scene Index: {scene_index}")
            multiscales = _read_single_scene_multiscales(
                czi_path,
                scene_index,
                mdata,
                pyramid_zooms,
                spatial_chunk_size,
                planes_per_chunk,
                full_logging,
            )
            if omero_channels:
                multiscales.metadata.omero = nz.Omero(channels=omero_channels)
            writer.write_well_image(
                multiscales=multiscales,
                row_name=well.row,
                column_name=well.column,
                field_index=field_index,
                **write_options,
            )

        logger.info("Writing %d HCS fields with up to %d workers...", total_fields, max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(write_field, *job) for job in field_jobs]
            for future in as_completed(futures):
                future.result()
                completed_fields += 1
                _log_hcs_progress(completed_fields, total_fields, conversion_started, full_logging)

    if not write_ozx_directly:
        _ensure_plate_version_metadata(zarr_output_path, version)

    _log_omezarr_layout(zarr_output_path)
    logger.info("=" * 80)
    logger.info("Conversion completed successfully!")
    logger.info(f"Output HCS OME-ZARR file: {zarr_output_path}")
    logger.info("Total conversion time: %.2f seconds", time.perf_counter() - conversion_started)
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
    version: str = "0.6",
    pyramid_levels: int = DEFAULT_PYRAMID_LEVELS,
    spatial_chunk_size: int = 512,
    source_multiscales: nz.NgffMultiscales | None = None,
    worker_count: int = 0,
    preserve_original_metadata: bool = False,
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
        version (str): OME-NGFF specification version. Defaults to ``"0.6"``.
        pyramid_levels (int): Number of additional 2x Y/X pyramid levels.
            Defaults to 3.
        spatial_chunk_size (int): Y/X chunk edge in pixels. Defaults to 512.
        source_multiscales (Optional[nz.NgffMultiscales]): Lazy CZI levels to
            write as supplied, preserving their physical transforms.
        worker_count (int): Concurrent Dask threads; 0 (default) selects Auto
            using available RAM and CPU capacity.
        preserve_original_metadata (bool): Save the original XML from
            ``metadata.filepath`` at ``CZI/metadata.xml``, referenced by root
            ``czitools.originalMetadata`` attributes. Defaults to False.

    Returns:
        Optional[Path]: Path to the written OME-Zarr file, or ``None`` on failure.
    """
    if worker_count < 0:
        raise ValueError("worker_count must be non-negative (0 selects Auto).")
    if log_file_path is None:
        zarr_path_obj = Path(zarr_path)
        log_file_path = zarr_path_obj.parent / f"{zarr_path_obj.stem}_omezarr.log"

    setup_logging(log_file_path)
    worker_count = _resolve_worker_count(worker_count)

    logger.info("=" * 80)
    logger.info("Writing OME-ZARR (ome-zarr-py backend)")
    logger.info("=" * 80)
    logger.info(f"Input array shape: {array5d.shape}")
    logger.info(f"Output path: {zarr_path}")

    assert (
        isinstance(array5d, xr.DataArray) or source_multiscales is not None
    ), "write_omezarr requires an xarray DataArray"

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

    logger.info("Zarr storage format: v3")

    chunks = (*QUALITY_CHUNKS[:3], spatial_chunk_size, spatial_chunk_size)
    shards = _shard_shape(chunks, QUALITY_CHUNKS_PER_SHARD)
    axes = (
        "".join(source_multiscales.images[0].dims)
        if source_multiscales is not None
        else "".join(str(d).lower() for d in array5d.dims)
    )

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

    compressor = None
    if compression == compression_type.BLOSC:
        compressor = Blosc()
    elif compression == compression_type.ZSTD:
        compressor = Zstd()

    if source_multiscales is not None:
        parsed = parse_url(zarr_path, mode="w", fmt=ome_zarr.format.CurrentFormat())
        assert parsed is not None, f"Failed to open zarr store at {zarr_path}"
        root = zarr.group(store=parsed.store)
        delayed = _write_czi_multiscales_delayed(
            source_multiscales,
            root,
            chunks,
            compression,
            ome_zarr.format.CurrentFormat(),
            shards=shards,
        )
        logger.info(
            "Reusing %d CZI-backed pyramid levels without ome-zarr-py downsampling", len(source_multiscales.images)
        )
    else:
        image = OMEZarrImage(
            data=_to_ome_zarr_image(array5d),
            axes=axes,
            scale=_phys_scale,
            axes_units=_phys_units,
            name=metadata.filename,
        )
        multiscales = OMEZarrMultiscale(
            image=image,
            scale_factors=[factor["y"] for factor in _pyramid_scale_factors(pyramid_levels)],
            method="nearest",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Pydantic serializer warnings:\s*PydanticSerializationUnexpectedValue\(Expected `tuple\[Dataset, \.\.\.\]`",
                category=UserWarning,
            )
            delayed = _retry_io(
                multiscales.to_ome_zarr,
                str(zarr_path),
                storage_options={"chunks": chunks, "shards": shards, "compressors": compressor},
                version=version,
                compute=False,
                overwrite=True,
            )
    if delayed:
        logger.info("Writing %d pyramid level(s) in parallel (dask)...", len(delayed))
        logger.info("Dask worker count: %d", worker_count)
        with dask.config.set(scheduler="threads", num_workers=worker_count):
            _retry_io(dask.compute, *delayed)

    root = zarr.open_group(zarr_path, mode="r+")
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
        fmt=ome_zarr.format.CurrentFormat(),
    )

    if preserve_original_metadata:
        _write_original_czi_metadata(metadata.filepath, zarr_path)
    _log_omezarr_layout(zarr_path)
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
    version: str = "0.6",
    chunks: tuple | None = None,
    chunks_per_shard: dict[str, int] | int | None = QUALITY_CHUNKS_PER_SHARD,
    compression: compression_type | None = compression_type.BLOSC,
    log_file_path: Path | str | None = None,
    min_size: int = 512,
    max_levels: int = 6,
    downsampling_method: "nz.Methods" = nz.Methods.DASK_IMAGE_GAUSSIAN,
    pyramid_levels: int | None = DEFAULT_PYRAMID_LEVELS,
    source_multiscales: nz.NgffMultiscales | None = None,
    worker_count: int = 0,
    preserve_original_metadata: bool = False,
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
        version (str): NGFF version string. Defaults to ``"0.6"``. RFC-9
            ``.ozx`` outputs always use the required version ``"0.5"``.
        chunks (Union[tuple, None]): Explicit ``TCZYX`` chunk shape. Defaults
            to ``(1, 1, 8, 512, 512)``.
        chunks_per_shard (Union[Dict[str, int], int, None]): Chunks per shard.
        compression (Optional[compression_type]): Chunk compression type. Defaults to
            ``compression_type.BLOSC``. Set to ``None`` for no compression.
        log_file_path (Union[Path, str, None]): Log file path. Defaults to
            ``<stem>_ngff.log``.
        min_size (int): Target maximum XY size of the coarsest pyramid level, used
            only when ``scale_factors`` is None. Defaults to 512.
        max_levels (int): Hard cap on the number of pyramid levels, used only when
            ``scale_factors`` is None. Defaults to 6.
        downsampling_method (nz.Methods): Pyramid downsampling method. Defaults
            to ``DASK_IMAGE_GAUSSIAN``.
        pyramid_levels (Optional[int]): Number of additional 2x Y/X pyramid
            levels used when ``scale_factors`` is None. Defaults to 3. Set to
            ``None`` to retain size-aware level selection using ``min_size``
            and ``max_levels``.
        source_multiscales (Optional[nz.NgffMultiscales]): Preloaded lazy CZI
            pyramid levels. When supplied, preserve these levels and their
            transforms instead of generating a new pyramid.
        worker_count (int): Concurrent Dask threads; 0 (default) selects Auto
            using available RAM and CPU capacity.
        preserve_original_metadata (bool): Save the original XML from
            ``metadata.filepath`` at ``CZI/metadata.xml``, referenced by root
            ``czitools.originalMetadata`` attributes. Also applies to ``.ozx``
            archives. Defaults to False.
    Notes:
        Directory output writes each level directly and builds the next from
        the preceding stored level, without an intermediate pyramid cache.
        Archive output stages levels to a temporary cache beside the output,
        removed on success or failure. Both paths use configurable Dask workers and a
        256 MiB region target; archives require additional temporary disk space.

    Returns:
        Optional[nz.NgffImage]: The written NgffImage, or ``None`` on failure.
    """
    conversion_started = time.perf_counter()
    if worker_count < 0:
        raise ValueError("worker_count must be non-negative (0 selects Auto).")

    if source_multiscales is not None:
        scale_factors = []
    elif scale_factors is None:
        if pyramid_levels is None:
            scale_factors = compute_pyramid_scale_factors(
                int(array5d.shape[-2]),
                int(array5d.shape[-1]),
                min_size=min_size,
                max_levels=max_levels,
            )
        else:
            scale_factors = _pyramid_scale_factors(pyramid_levels)

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
    logger.info("Downsampling method: %s", downsampling_method.value)

    if len(array5d.shape) > 5:
        logger.info("Input array has more than 5 dimensions.")
        return None

    output_path = Path(zarr_path)
    if output_path.suffix.lower() == ".ozx" and version != "0.5":
        logger.info(
            "Using OME-NGFF 0.5 for RFC-9 .ozx output instead of requested version %s.",
            version,
        )
        version = "0.5"
    if output_path.exists() and overwrite:
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    elif output_path.exists() and not overwrite:
        logger.info(f"File already exists at {zarr_path}. Set overwrite=True to remove.")
        return None

    _scale = metadata.scale
    _filename = metadata.filename or "image.czi"

    if chunks is None:
        chunks = QUALITY_CHUNKS
    logger.info("Chunk shape (TCZYX): %s", chunks)

    image_data = array5d.data if isinstance(array5d, xr.DataArray) else array5d
    if isinstance(image_data, da.Array):
        image_data = image_data.rechunk(chunks)
    else:
        image_data = da.from_array(image_data, chunks=chunks)  # type: ignore[arg-type]

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
    if source_multiscales is not None:
        image = source_multiscales.images[0]
        for level in source_multiscales.images:
            level.data = level.data.rechunk(chunks)
        logger.info("Reusing %d CZI-backed pyramid levels without NGFF downsampling", len(source_multiscales.images))

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
    # Convert compression_type enum to actual codec instance
    compressor = None
    if compression == compression_type.BLOSC:
        compressor = NumcodecsBlosc()
    elif compression == compression_type.ZSTD:
        compressor = NumcodecsZstd()
    # compression_type.NONE or None → compressor stays None

    is_archive = output_path.suffix.lower() == ".ozx"
    with (
        _ngff_conversion_resources(
            output_path, use_cache=is_archive and source_multiscales is None, worker_count=worker_count
        ),
        _original_metadata_output(
            metadata.filepath if preserve_original_metadata else "", output_path, preserve_original_metadata, version
        ) as output_store,
    ):
        if is_archive:
            multiscales = (
                source_multiscales
                if source_multiscales is not None
                else _bounded_ngff_multiscales(image, scale_factors, chunks, downsampling_method)
            )
            multiscales.metadata.omero = nz.Omero(channels=channels)
            logger.info("Writing NGFF pyramid with the zarrista backend...")
            with _logged_progress_bar("NGFF write") as progress:
                nz.to_ngff_zarr(
                    output_store,
                    version=version,
                    chunks_per_shard=chunks_per_shard,
                    compressor=compressor,
                    multiscales=multiscales,
                    progress=progress,
                )
        else:
            directory_options = {"source_multiscales": source_multiscales} if source_multiscales is not None else {}
            _write_ngff_directory(
                output_path,
                image,
                scale_factors,
                chunks,
                downsampling_method,
                version,
                chunks_per_shard,
                compressor,
                channels,
                **directory_options,
            )

    if version == "0.6" and output_path.is_dir():
        # ngff-zarr 0.46.1 writes the finalized 0.6 model with its former
        # release-candidate tag. Publish the requested final version instead.
        root = zarr.open_group(zarr_path, mode="r+")
        ome_attrs = dict(root.attrs["ome"])
        ome_attrs["version"] = "0.6"
        root.attrs["ome"] = ome_attrs

    _log_omezarr_layout(output_path)
    logger.info("NGFF OME-ZARR writing completed successfully!")
    logger.info(f"Output file: {zarr_path}")
    logger.info("Total conversion time: %.2f seconds", time.perf_counter() - conversion_started)

    return image
