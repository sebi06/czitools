"""CZI pyramid discovery and multiscale reading.

Pyramid-level detection uses :mod:`czifile`'s per-scene level structure
(scene.levels + directory_entry.is_pyramid) for local files, which correctly
distinguishes full-resolution subblocks from stored pyramid levels and
plate-wide thumbnails. URL paths fall back to :mod:`pylibCZIrw` enumeration
restricted to scene 0's bounding rectangle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import czifile as czifile_module
import xarray as xr
from pylibCZIrw import czi as pyczi

from czitools.metadata_tools import czi_metadata as czimd
from czitools.utils import misc

from ._helpers import (
    CziPath,
    LazyReadStrategy,
    StackArray,
    StackList,
    logger,
)
from .stacks import read_stacks

# Rounding precision for zoom-factor deduplication. Six digits distinguishes
# well-formed CZI pyramid steps (1.0, 0.5, 0.333..., 0.25, 0.125, 0.0625, ...)
# without treating equal levels as different due to double-precision noise.
_ZOOM_ROUND = 6

# Minimum zoom that ``pylibCZIrw.CziReader.read(zoom=...)`` accepts. Lower
# values are clamped by the library, so a "stored" level below this cannot be
# read at its native resolution and is dropped from the detected set.
_MIN_READ_ZOOM = 0.01


@dataclass(frozen=True)
class PyramidLevel:
    """Description of one level in a CZI multiscale pyramid.

    Attributes:
        zoom: Zoom factor relative to layer 0 (``1.0`` for layer 0).
        stored: True if the level exists as its own subblock set in the CZI
            and therefore reads without resampling. False for synthetic
            levels emitted by :func:`read_stacks_multiscale` when the
            coarsest stored level is still too large for a single GPU
            texture.
        y: Full-plane height in pixels at this level (rounded down).
        x: Full-plane width in pixels at this level (rounded down).
    """

    zoom: float
    stored: bool
    y: int
    x: int


def get_pyramid_zooms(filepath: CziPath) -> list[float]:
    """Return the zoom factors of pyramid levels stored in the CZI file.

    For local files uses :mod:`czifile`'s ``scene.levels`` structure.
    Each level's zoom is derived from its stored X size relative to level 0,
    matching the approach in ``check_PYL.py``. The ``directory_entry`` of
    every subblock carries a unitless scale (physicalSize / logicalSize);
    subblocks with ``is_pyramid=True`` have a scale < 1.0 and belong to a
    stored pyramid level. Files without a pyramid return ``[1.0]``.

    For URL paths (not supported by :mod:`czifile`), falls back to
    :mod:`pylibCZIrw` enumeration restricted to scene 0's bounding rectangle
    so that plate-wide thumbnail subblocks are excluded.

    Levels whose zoom is below :data:`_MIN_READ_ZOOM` (0.01) are dropped
    because ``pylibCZIrw.CziReader.read`` clamps such zooms to 0.01.

    Args:
        filepath: Path to the CZI file (local path or URL).

    Returns:
        Sorted list of unique zoom factors, largest first — for example
        ``[1.0, 0.5, 0.25, 0.125, 0.0625]`` for standard pyramids, or
        ``[1.0, 0.333333]`` for ZEN's 3x pyramid variant.
    """
    path_str = str(filepath)
    is_url = path_str.startswith(("http://", "https://"))

    if not is_url:
        # czifile correctly identifies pyramid levels via directory_entry.is_pyramid
        # and exposes them through scene.levels — same API used by check_PYL.py.
        try:
            with czifile_module.CziFile(path_str) as czi:
                scenes = czi.scenes
                if scenes and scenes[0].levels:
                    levels = scenes[0].levels
                    base_x = levels[0].shape[-1]
                    if base_x > 0:
                        zooms = {round(float(lvl.shape[-1]) / base_x, _ZOOM_ROUND) for lvl in levels}
                        readable = sorted((z for z in zooms if z >= _MIN_READ_ZOOM), reverse=True)
                        return readable or [1.0]
        except Exception:
            pass  # fall through to pylibCZIrw

    # Fallback for URLs: pylibCZIrw enumeration filtered to scene 0's ROI.
    readertype, _ = misc.get_pyczi_readertype(path_str)
    zooms_pyl: set[float] = set()

    def _collect(_idx: int, info) -> bool:
        try:
            zooms_pyl.add(round(float(info.get_zoom()), _ZOOM_ROUND))
        except Exception:
            pass
        return True

    with pyczi.open_czi(path_str, readertype) as doc:
        scene_rects = doc.scenes_bounding_rectangle_no_pyramid
        if scene_rects:
            s0 = scene_rects[min(scene_rects.keys())]
            doc.enumerate_subblocks_subset(_collect, roi=(s0.x, s0.y, s0.w, s0.h))
        else:
            doc.enumerate_subblocks(_collect)

    readable = sorted((z for z in zooms_pyl if z >= _MIN_READ_ZOOM), reverse=True)
    return readable or [1.0]


def _synthesize_coarser_zooms(
    stored_zooms: list[float],
    layer0_edge: int,
    max_coarse_edge: int,
) -> list[float]:
    """Return additional synthetic zoom factors below the coarsest stored one.

    Starts from the smallest stored zoom and halves until the coarsest edge
    (``layer0_edge * zoom``) is at or below ``max_coarse_edge``. Synthetic
    levels are what makes napari happy for gigapixel files whose stored
    pyramid stops at a level that is still bigger than one GPU texture.

    Args:
        stored_zooms: All zoom factors that already exist on disk.
        layer0_edge: Longest edge of layer 0 in pixels (``max(y, x)``).
        max_coarse_edge: Target maximum edge for the coarsest returned
            level, in pixels. Should be safely below the GPU's texture
            limit (e.g. 8192 when the GPU accepts 16k textures).

    Returns:
        List of synthetic zoom factors, largest first, that extend
        ``stored_zooms`` towards zero. Empty when the coarsest stored level
        already fits.
    """
    if layer0_edge <= 0 or max_coarse_edge <= 0:
        return []
    coarsest = min(stored_zooms)
    edge = int(layer0_edge * coarsest)
    synth: list[float] = []
    zoom = coarsest
    # Halve until either the edge fits or we would drop below the reader's
    # minimum accepted zoom. ``_MIN_READ_ZOOM`` mirrors pylibCZIrw's clamp;
    # values below it cannot be read at their native resolution.
    while edge > max_coarse_edge and zoom / 2.0 >= _MIN_READ_ZOOM:
        zoom = zoom / 2.0
        edge = int(layer0_edge * zoom)
        synth.append(round(zoom, _ZOOM_ROUND))
    return synth


def read_stacks_multiscale(
    filepath: CziPath,
    use_xarray: bool = True,
    stack_scenes: bool = True,
    planes: dict[str, tuple[int, int]] | None = None,
    zooms: list[float] | None = None,
    max_coarse_edge: int = 8192,
    tile_size: int = 4096,
    chunk_memory_limit: int = 256 * 1024 * 1024,
    lazy_read_strategy: LazyReadStrategy = "chunk",
    planes_per_chunk: int = 64,
    scene_stack_tolerance: int = 0,
) -> tuple[
    list[StackArray | StackList],
    list[PyramidLevel],
    list[str],
    int,
    czimd.CziMetadata,
]:
    """Read a CZI as a multiscale pyramid ready for ``napari.add_image``.

    Each returned array shares the same S/T/C/Z shape but has progressively
    smaller Y/X, matching the shape napari expects for
    ``add_image(data, multiscale=True)``.

    Level selection:

    1. If ``zooms`` is not given, use :func:`get_pyramid_zooms` to detect the
       levels stored on disk. Reads at those zooms are served directly from
       the matching subblocks (fast, no resampling).
    2. If the coarsest stored level is still larger than
       ``max_coarse_edge`` on either edge, additional synthetic coarser
       levels are appended by halving the zoom. These use libCZI's C++
       resampler at read time — slower than stored levels but still cheaper
       than materialising layer 0 in Python.

    Each level is built by delegating to :func:`read_stacks` with the chosen
    zoom, so spatial Y/X tiling (item 3b) still applies per level.

    Args:
        filepath: CZI file to read.
        use_xarray: If True (default), levels are ``xarray.DataArray`` with
            labelled dimensions. Pass False for raw ``dask.array``.
        stack_scenes: Forwarded to :func:`read_stacks`. Set True to receive
            one stacked array per level; set False for a per-scene list per
            level.
        planes: Optional S/T/C/Z subset (see :func:`read_stacks`).
        zooms: Override the detected pyramid list entirely. Rarely needed;
            useful for testing or when the file's stored pyramid metadata
            is broken.
        max_coarse_edge: If the coarsest stored level's longer edge exceeds
            this, synthetic coarser levels are appended until the edge
            fits. Default 8192 is safely under typical 16k GPU texture
            limits.
        tile_size: Forwarded to :func:`read_stacks`. Controls Y/X tile size
            for very large planes.
        chunk_memory_limit: Forwarded to :func:`read_stacks`. Threshold that
            decides when Y/X tiling activates for a single plane.
        lazy_read_strategy: Forwarded to :func:`read_stacks`.
        planes_per_chunk: Forwarded to :func:`read_stacks`.

    Returns:
        Tuple ``(levels, level_infos, dims, num_stacks, mdata)``:

        - ``levels``: list of arrays (or per-scene lists), coarsest last.
        - ``level_infos``: matching :class:`PyramidLevel` descriptors.
        - ``dims``: canonical dimension labels of each level.
        - ``num_stacks``: number of scenes in the read (same for every level).
        - ``mdata``: shared :class:`CziMetadata` for the file.

    Notes:
        Every returned dask array is lazy — nothing is read until napari
        (or the caller) triggers computation on the tiles it needs.
    """
    filepath = str(filepath)

    # Detect stored levels once via subblock enumeration.
    detected = get_pyramid_zooms(filepath) if zooms is None else list(zooms)
    if not detected:
        detected = [1.0]
    detected = sorted({round(float(z), _ZOOM_ROUND) for z in detected}, reverse=True)

    # Determine layer-0 edge so we know whether to synthesize extra levels.
    # Use per-scene dimensions (SizeY_scene / SizeX_scene), not the total
    # plate bounding box (SizeY / SizeX), which spans all scenes and would
    # trigger unnecessary synthetic coarse levels for multi-scene files.
    mdata_probe = czimd.CziMetadata(filepath)
    image = mdata_probe.image_required
    layer0_y = int(image.SizeY_scene or image.SizeY or 0)
    layer0_x = int(image.SizeX_scene or image.SizeX or 0)
    layer0_edge = max(layer0_y, layer0_x)

    synth = _synthesize_coarser_zooms(detected, layer0_edge, max_coarse_edge) if layer0_edge > 0 else []
    all_zooms = detected + synth
    if synth:
        logger.info(
            "read_stacks_multiscale: stored zooms=%s; appended synthetic zooms=%s "
            "so the coarsest level fits within max_coarse_edge=%d px.",
            detected,
            synth,
            max_coarse_edge,
        )
    else:
        logger.info("read_stacks_multiscale: stored zooms=%s (no synthesis needed).", detected)

    levels: list[StackArray | StackList] = []
    infos: list[PyramidLevel] = []
    dims_out: list[str] = []
    num_stacks_out = 0
    mdata_out: czimd.CziMetadata | None = None

    for z in all_zooms:
        # Delegate the per-level read to the fully featured stack reader; it
        # already handles scenes, tiling, planes selection, xarray labels,
        # and coordinate scaling for the zoom factor.
        result, dims, num_stacks, mdata = read_stacks(
            filepath=filepath,
            use_dask=True,
            use_xarray=use_xarray,
            stack_scenes=stack_scenes,
            planes=planes,
            zoom=z,
            tile_size=tile_size,
            chunk_memory_limit=chunk_memory_limit,
            lazy_read_strategy=lazy_read_strategy,
            planes_per_chunk=planes_per_chunk,
            scene_stack_tolerance=scene_stack_tolerance,
        )
        levels.append(result)
        dims_out = dims
        num_stacks_out = num_stacks
        mdata_out = mdata

        # Extract level shape for the PyramidLevel record. Handle both the
        # stacked-array and per-scene-list return shapes uniformly.
        probe = result[0] if isinstance(result, list) else result
        # Trailing dims of the array are (..., Y, X) for grayscale or
        # (..., Y, X, A) for RGB; look up Y/X by name when using xarray, or
        # by position otherwise.
        if isinstance(probe, xr.DataArray):
            y_size = int(probe.sizes.get("Y", 0))
            x_size = int(probe.sizes.get("X", 0))
        else:
            shape = getattr(probe, "shape", ())
            # For RGB the last axis is A; grayscale ends with X.
            if len(shape) >= 3 and shape[-1] in (3, 4):
                y_size, x_size = int(shape[-3]), int(shape[-2])
            elif len(shape) >= 2:
                y_size, x_size = int(shape[-2]), int(shape[-1])
            else:
                y_size = x_size = 0

        infos.append(
            PyramidLevel(
                zoom=z,
                stored=z in set(detected),
                y=y_size,
                x=x_size,
            )
        )

    # ``mdata_out`` can only be None when the loop didn't run, which cannot
    # happen because ``all_zooms`` always contains at least the layer-0 entry.
    assert mdata_out is not None
    return levels, infos, dims_out, num_stacks_out, cast(czimd.CziMetadata, mdata_out)
