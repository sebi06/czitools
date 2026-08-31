"""Read CZI scenes as regular or irregular eager/lazy stacks."""

import copy
import itertools
import math
from typing import Any, cast

import dask.array as da
from dask.delayed import delayed as dask_delayed
import numpy as np
from pylibCZIrw import czi as pyczi
import xarray as xr

from czitools.metadata_tools import czi_metadata as czimd
from czitools.utils import misc

from ._helpers import (
    CziPath,
    LazyReadStrategy,
    ReadStacksWithMetaReturn,
    StackArray,
    StackList,
    _as_float,
    _channel_names_or_default,
    _CORE_DIMS,
    _EXTRA_DIMS,
    _get_axis_coord_step,
    _PLANE_DIMS_READ,
    logger,
)


@dask_delayed
def _read_plane_delayed(
    filepath: str,
    plane: dict[str, int],
    stack_idx: int | None,
    squeeze_grayscale: bool,
    zoom: float = 1.0,
    readertype: pyczi.ReaderFileInputTypes = pyczi.ReaderFileInputTypes.Standard,
) -> np.ndarray:
    """Delayed function to read a single 2D plane from a CZI file.

    This function is called lazily by dask when the data is actually needed.
    Supports both local files and URLs.

    Args:
        filepath: Path to the CZI file (local path or URL).
        plane: Dictionary mapping dimension names to coordinate values.
        stack_idx: Index of the stack to read (same as scene index in CZIs).
        squeeze_grayscale: If True, squeeze the trailing dimension for grayscale images.
        zoom: Downscale factor for 2D plane reads [0.01 - 1.0].
        readertype: The pylibCZIrw reader type (Standard for local files, Curl for URLs).

    Returns:
        2D numpy array of the plane data.
    """
    with pyczi.open_czi(filepath, readertype) as czidoc:
        # If stack_idx is None the CZI has no scenes; read without scene arg
        if stack_idx is None:
            img2d = czidoc.read(plane=plane, zoom=zoom)
        else:
            img2d = czidoc.read(plane=plane, scene=stack_idx, zoom=zoom)

        if squeeze_grayscale:
            img2d = img2d[..., 0]
    return img2d


def _read_plane_chunk(
    filepath: str,
    chunk_indices: list[tuple[int, ...]],
    read_dims: list[str],
    read_starts: list[int],
    stack_idx: int | None,
    squeeze_grayscale: bool,
    plane_shape: tuple[int, ...],
    dtype: np.dtype,
    zoom: float = 1.0,
    readertype: pyczi.ReaderFileInputTypes = pyczi.ReaderFileInputTypes.Standard,
) -> np.ndarray:
    """Read a chunk of multiple planes from a CZI file in a single task.

    This function opens the file once and reads multiple planes, which is more
    efficient than opening the file for each plane separately. This significantly
    reduces overhead for large datasets.

    Args:
        filepath: Path to the CZI file (local path or URL).
        chunk_indices: List of index tuples for planes to read.
        read_dims: List of dimension names in reading order.
        read_starts: Starting indices for each dimension.
        stack_idx: Index of the stack to read (same as scene index in CZIs).
        squeeze_grayscale: If True, squeeze the trailing dimension for grayscale images.
        plane_shape: Shape of a single plane.
        dtype: Data type of the array.
        zoom: Downscale factor for 2D plane reads [0.01 - 1.0].
        readertype: The pylibCZIrw reader type (Standard for local files, Curl for URLs).

    Returns:
        3D numpy array of shape (num_planes, Y, X) or (num_planes, Y, X, A).
    """
    num_planes = len(chunk_indices)
    chunk_array = np.empty((num_planes,) + plane_shape, dtype=dtype)

    with pyczi.open_czi(filepath, readertype) as czidoc:
        for i, combo in enumerate(chunk_indices):
            # Build plane dict with actual coordinate values
            plane = {name: start + offset for name, start, offset in zip(read_dims, read_starts, combo)}

            # Read the plane
            if stack_idx is None:
                img2d = czidoc.read(plane=plane, zoom=zoom)
            else:
                img2d = czidoc.read(plane=plane, scene=stack_idx, zoom=zoom)

            if squeeze_grayscale:
                img2d = img2d[..., 0]

            chunk_array[i] = img2d

    return chunk_array


# ---------------------------------------------------------------------------
# Spatial Y/X tiling helpers
#
# Gigapixel CZI files can have single 2D planes that are tens of GB when
# uncompressed (e.g. 93,555 × 138,996 uint16 ≈ 24 GB per channel). Reading
# such planes as one dask chunk defeats lazy access because the very first
# read allocates the whole plane in RAM. The helpers below break a plane into
# a grid of spatial tiles using pylibCZIrw's ROI-based read
# (``CziReader.read(roi=(x, y, w, h), ...)``), so each dask task only fetches
# the pixels of a single tile.
#
# The tiled path is only used when a single plane exceeds ``chunk_memory_limit``
# in ``read_stacks``; smaller planes keep the fast whole-plane path.
# ---------------------------------------------------------------------------


def _read_tile_delayed(
    filepath: str,
    plane: dict[str, int],
    stack_idx: int | None,
    roi: tuple[int, int, int, int],
    squeeze_grayscale: bool,
    zoom: float = 1.0,
    readertype: pyczi.ReaderFileInputTypes = pyczi.ReaderFileInputTypes.Standard,
) -> np.ndarray:
    """Read a single 2D spatial tile from a CZI file via a ROI.

    Called lazily by dask when a specific spatial chunk is requested. The ROI
    is expressed in the CZI's own coordinate system, so for scene-based files
    the caller must add the scene's ``stack_rect.x/y`` offset before passing
    the tuple in. ``pylibCZIrw.CziReader.read`` accepts ``roi`` as a plain
    ``(x, y, w, h)`` tuple; using a tuple avoids importing/constructing
    ``pyczi.Rectangle`` here.

    Args:
        filepath: Path to the CZI file (local path or URL).
        plane: Dictionary mapping non-spatial dimension names (T/C/Z/...) to
            their coordinate values for this tile.
        stack_idx: Scene index to read from, or ``None`` when the CZI has no
            explicit scenes (the ROI is then interpreted in the total
            bounding rectangle's coordinate space).
        roi: ``(x, y, w, h)`` region of interest in the CZI's coordinate
            system. ``w`` and ``h`` are the tile's actual size (edge tiles
            may be smaller than the nominal tile size).
        squeeze_grayscale: If True, squeeze the trailing pixel-type axis
            (A=1) for grayscale data. RGB (A=3) is preserved.
        zoom: Downscale factor for the 2D read [0.01 - 1.0].
        readertype: The pylibCZIrw reader type (Standard for local files,
            Curl for URLs).

    Returns:
        2D numpy array of shape ``(h, w)`` for grayscale or ``(h, w, A)`` for
        RGB. Dimensions match the ROI's ``h`` and ``w``.
    """
    # Each dask worker opens/closes the file independently. pylibCZIrw is
    # thread-safe for read operations and releases the GIL during I/O.
    with pyczi.open_czi(filepath, readertype) as czidoc:
        if stack_idx is None:
            # Scene-less CZI: ROI is in the total bounding rectangle's space.
            img2d = czidoc.read(plane=plane, roi=roi, zoom=zoom)
        else:
            img2d = czidoc.read(plane=plane, scene=stack_idx, roi=roi, zoom=zoom)

        if squeeze_grayscale:
            img2d = img2d[..., 0]
    return img2d


def _compute_tile_extent(
    spatial_y: int,
    spatial_x: int,
    dtype_nbytes: int,
    components: int,
    tile_size: int,
    memory_limit: int,
) -> tuple[int, int]:
    """Choose ``(tile_h, tile_w)`` so a single tile fits under ``memory_limit``.

    Starts from a square tile of ``tile_size`` (clamped to the actual plane
    extent) and iteratively halves the larger dimension until the tile's
    estimated memory footprint (``tile_h * tile_w * dtype_nbytes * components``)
    is at or below ``memory_limit``. The halving guarantees termination; the
    smallest allowed extent is one pixel.

    Args:
        spatial_y: Full plane height in pixels.
        spatial_x: Full plane width in pixels.
        dtype_nbytes: Bytes per pixel for a single component (e.g. 2 for
            ``uint16``).
        components: Number of pixel components (1 for grayscale, 3 for RGB).
        tile_size: Requested nominal tile edge in pixels. Ignored if it
            exceeds the plane extent.
        memory_limit: Maximum allowed uncompressed bytes for one tile.

    Returns:
        ``(tile_h, tile_w)`` in pixels.
    """
    # Clamp the requested tile size so we never claim a larger tile than the
    # plane itself; this also lets small planes use a single-tile layout.
    tile_h = max(1, min(tile_size, spatial_y))
    tile_w = max(1, min(tile_size, spatial_x))

    def _bytes(h: int, w: int) -> int:
        return h * w * dtype_nbytes * components

    # Halve the larger dimension first to keep tiles reasonably square.
    while _bytes(tile_h, tile_w) > memory_limit and (tile_h > 1 or tile_w > 1):
        if tile_w >= tile_h and tile_w > 1:
            tile_w = max(1, tile_w // 2)
        elif tile_h > 1:
            tile_h = max(1, tile_h // 2)
        else:  # pragma: no cover - both are 1
            break
    return tile_h, tile_w


def _plan_tile_grid(
    rect_extent_layer0: int,
    tile_extent_zoomed: int,
    zoom: float,
) -> tuple[int, list[int], list[int]]:
    """Lay out the per-tile sizes along one axis.

    The tiled reader must agree with the outer code on the plane's total
    zoomed size and on each tile's zoomed extent, otherwise the dask array's
    real shape and its declared coordinate arrays disagree. This helper is
    the single source of truth: given the axis extent in layer-0 coords, the
    requested tile size in *zoomed* coords, and the zoom factor, it returns
    the total zoomed extent (as the sum of per-tile zoomed sizes) alongside
    the layer-0 offsets and zoomed sizes of every non-empty tile.

    ``sum(zoomed_sizes)`` can differ from ``int(rect_extent_layer0 * zoom)``:
    sum-of-truncations is not always equal to truncation-of-sum. Using the
    sum keeps every downstream consumer (coord arrays, xarray labels, napari
    layer scale) in agreement with the tile grid actually assembled.

    Args:
        rect_extent_layer0: Full axis extent in native (layer-0) pixels.
        tile_extent_zoomed: Requested tile edge in zoomed pixels.
        zoom: Zoom factor forwarded to libCZI. ``0 < zoom <= 1``.

    Returns:
        Tuple ``(spatial_zoomed, offsets_layer0, sizes_zoomed)`` where
        ``spatial_zoomed = sum(sizes_zoomed)`` is the true total zoomed
        extent of the tiled composition, and the two lists give one entry
        per emitted tile.
    """
    # Convert the requested zoomed tile size to layer-0 space. ``ceil``
    # ensures the zoomed tile is at least as large as requested.
    tile_extent_layer0 = max(1, math.ceil(tile_extent_zoomed / zoom))
    offsets_layer0: list[int] = []
    sizes_zoomed: list[int] = []
    pos = 0
    while pos < rect_extent_layer0:
        # Clip the trailing tile so it does not exceed the plane extent.
        size_layer0 = min(tile_extent_layer0, rect_extent_layer0 - pos)
        size_zoomed = int(size_layer0 * zoom)
        if size_zoomed > 0:
            # A zero-sized tile happens at extreme zooms where the last few
            # layer-0 pixels round down to zero; skip them so the resulting
            # dask array has no empty chunks.
            offsets_layer0.append(pos)
            sizes_zoomed.append(size_zoomed)
        pos += tile_extent_layer0
    spatial_zoomed = sum(sizes_zoomed)
    return spatial_zoomed, offsets_layer0, sizes_zoomed


def _build_tiled_plane_dask(
    filepath: str,
    plane: dict[str, int],
    stack_idx: int | None,
    origin_xy: tuple[int, int],
    row_offsets_layer0: list[int],
    row_sizes_zoomed: list[int],
    col_offsets_layer0: list[int],
    col_sizes_zoomed: list[int],
    tile_h_layer0: int,
    tile_w_layer0: int,
    rect_h_layer0: int,
    rect_w_layer0: int,
    squeeze_grayscale: bool,
    dtype: np.dtype,
    zoom: float,
    readertype: pyczi.ReaderFileInputTypes,
    has_rgb_component: bool,
    num_components: int | None,
) -> da.Array:
    """Compose a single 2D plane out of ROI-tile reads as one dask array.

    The tile layout is planned once by :func:`_plan_tile_grid` (called from
    the outer reader) so both the caller (for coordinate arrays) and this
    builder (for the dask graph) see the same per-row / per-column sizes.
    Passing the layout in avoids the classic sum-of-truncations vs
    truncation-of-sum mismatch, e.g. at zoom ``1/3`` the direct
    ``int(rect * zoom)`` can differ by 1–3 pixels from the tile-composed
    total.

    Coordinate spaces are the tricky part when ``zoom != 1.0``:

    - pylibCZIrw's ROI is always in **layer-0 (native) coordinates**.
    - The output array is downsampled by ``zoom``, so the returned shape
      per tile is ``int(roi_w * zoom) × int(roi_h * zoom)`` (libCZI
      truncates). Each declared tile shape is exactly what libCZI returns.

    ``da.block`` is intentionally not used here: it matches its nesting
    depth against the *last* N axes, which for RGB planes with a trailing
    ``A`` axis would concatenate along X and A instead of Y and X,
    silently corrupting the pixel layout.

    Args:
        filepath: Path to the CZI file.
        plane: Non-spatial coordinate dict (T/C/Z/...) shared by every tile.
        stack_idx: Scene index or ``None`` for scene-less files.
        origin_xy: ``(x, y)`` offset of the scene (or total bounding
            rectangle) in native coordinates. Added to each tile's local
            layer-0 offset to build the ROI.
        row_offsets_layer0: Starting Y offset in layer-0 coords for each
            row of tiles.
        row_sizes_zoomed: Corresponding zoomed height of each row (from
            :func:`_plan_tile_grid`).
        col_offsets_layer0: Starting X offset in layer-0 coords for each
            column of tiles.
        col_sizes_zoomed: Corresponding zoomed width of each column.
        tile_h_layer0: Nominal tile height in layer-0 pixels. Used to
            derive each tile's clipped layer-0 height.
        tile_w_layer0: Nominal tile width in layer-0 pixels.
        rect_h_layer0: Scene rectangle height in native pixels (for
            clipping the trailing edge tiles).
        rect_w_layer0: Scene rectangle width in native pixels.
        squeeze_grayscale: If True, grayscale planes drop the trailing A axis.
        dtype: NumPy dtype of the pixels.
        zoom: Downscale factor. Forwarded to each ROI read.
        readertype: pylibCZIrw reader type (Standard / Curl).
        has_rgb_component: True when the plane has an RGB pixel-type axis
            that must be preserved (A=3).
        num_components: Number of pixel components; only used when
            ``has_rgb_component`` is True.

    Returns:
        A single ``dask.array.Array`` whose shape is
        ``(sum(row_sizes_zoomed), sum(col_sizes_zoomed))`` (with an
        optional trailing ``A`` axis for RGB) and whose chunks are the
        individual tile reads. Nothing is read until the dask array is
        computed or indexed.
    """
    origin_x, origin_y = origin_xy

    grid: list[list[da.Array]] = []
    for row_index, y0_l0 in enumerate(row_offsets_layer0):
        h_l0 = min(tile_h_layer0, rect_h_layer0 - y0_l0)
        this_h = row_sizes_zoomed[row_index]
        row_tiles: list[da.Array] = []
        for col_index, x0_l0 in enumerate(col_offsets_layer0):
            w_l0 = min(tile_w_layer0, rect_w_layer0 - x0_l0)
            this_w = col_sizes_zoomed[col_index]
            # ROI is in layer-0 CZI coordinates (see pylibCZIrw docstring).
            # The scene origin adds any non-zero offset for scene-based files.
            roi = (origin_x + x0_l0, origin_y + y0_l0, w_l0, h_l0)
            tile_shape: tuple[int, ...]
            if has_rgb_component and num_components is not None:
                tile_shape = (this_h, this_w, num_components)
            else:
                tile_shape = (this_h, this_w)
            delayed_tile = dask_delayed(_read_tile_delayed)(
                filepath,
                plane,
                stack_idx,
                roi,
                squeeze_grayscale,
                zoom,
                readertype,
            )
            row_tiles.append(da.from_delayed(delayed_tile, shape=tile_shape, dtype=dtype))
        grid.append(row_tiles)

    # Concatenate columns along X first (axis 1), then rows along Y (axis 0).
    # For a single-tile grid, this collapses to that tile without any copy.
    if len(grid) == 1 and len(grid[0]) == 1:
        return grid[0][0]
    row_arrays = [(row_tiles[0] if len(row_tiles) == 1 else da.concatenate(row_tiles, axis=1)) for row_tiles in grid]
    return row_arrays[0] if len(row_arrays) == 1 else da.concatenate(row_arrays, axis=0)


def read_stacks(
    filepath: CziPath,
    use_dask: bool = False,
    use_xarray: bool = True,
    stack_scenes: bool = False,
    planes: dict[str, tuple[int, int]] | None = None,
    zoom: float = 1.0,
    adapt_metadata: bool = False,
    chunk_policy: str = "none",
    chunk_memory_limit: int = 256 * 1024 * 1024,
    lazy_read_strategy: LazyReadStrategy = "chunk",
    planes_per_chunk: int = 64,
    tile_size: int = 4096,
    scene_stack_tolerance: int = 0,
    metadata: czimd.CziMetadata | None = None,
) -> ReadStacksWithMetaReturn:
    """Read all 2D planes from a CZI file, grouped per stack.

    This function reads pixel data based on the total_bounding_box (derived from
    CZI subblocks, not XML metadata). It supports all CZI dimensions and
    returns arrays with a canonical dimension order.

    Dimension order is always: [V, R, I, H, M] + T + C + Z + Y + X [+ A]
      - Extra dims (V, R, I, H, M) appear first if present in the file
      - Core dims (T, C, Z) are always present (size=1 if missing in file)
      - B dimension is always removed (assumed size=1)
      - S (scene) is tracked separately; each scene is a separate array in the list
      - Spatial (Y, X) and optional pixel-type (A for RGB, size=3) are last

    Args:
        filepath: Path to the CZI file.
        use_dask: If True, return lazy dask arrays. A representative plane is
            read while constructing each scene to determine its dtype and
            component layout; the remaining pixel reads are deferred until
            computation. Defaults to False.
        use_xarray: If True, return xr.DataArray with labeled dimensions. If False,
            return plain np.ndarray (or dask.array if use_dask=True). Defaults to True.
        stack_scenes: If True and all scenes have the same shape, stack them
            into a single array with S as the first dimension. If shapes differ
            beyond ``scene_stack_tolerance``, returns a list (with a warning).
            Defaults to False.
        scene_stack_tolerance: Maximum allowed pixel difference in the Y or X
            dimension between scenes when ``stack_scenes=True``. Scenes within
            this tolerance are cropped to the smallest common Y/X shape before
            stacking. All non-spatial dims (T, C, Z, …) must still be identical.
            Defaults to 0 (strict equality, backward-compatible).
        planes: Optional dict specifying substack ranges (keys: S, T, C, Z).
            Values are (start, end) tuples, zero-based inclusive. Mirrors
            `read_6darray` semantics.
        zoom: Downscale factor for 2D plane reads [0.01 - 1.0]. Defaults to 1.0.
        adapt_metadata: If True, update metadata dimensions (SizeS, SizeT,
            SizeC, SizeZ) to match selected `planes`. Defaults to False.
        chunk_policy: How to rechunk per-stack Dask arrays before stacking.
            - 'none' (default): do not rechunk.
            - 'scene-shape': rechunk each per-scene array to its scene shape
                (useful when scenes are not stacked but you want predictable
                chunking per-scene).
            - 'stack-shape': rechunk each per-scene array to the stacked
                per-stack shape (the target shape used for stacking S axis).
                This can help avoid dask.concat/reshape incompatibilities.
        chunk_memory_limit: Maximum allowed bytes for a single chunk when
            rechunking with `stack-shape`. If a target rechunk would create
            a chunk larger than this limit, spatial chunk sizes (Y/X) are
            halved iteratively until the estimated chunk size is below
            `chunk_memory_limit`. Default is 256MB.
        lazy_read_strategy: Dask I/O strategy. ``"chunk"`` (default) groups
            multiple planes per task and opens the CZI once per group.
            ``"plane"`` creates one task and file open per plane for the
            finest-grained random access.
        planes_per_chunk: Maximum planes read by each chunk task when
            ``lazy_read_strategy="chunk"``. Defaults to 64.
        tile_size: Nominal spatial tile edge in pixels used for Y/X tiling of
            very large planes when ``use_dask=True``. Only activated when a
            single uncompressed 2D plane would exceed ``chunk_memory_limit``
            (default 256 MB). Small planes always keep the whole-plane path
            and pay no tiling overhead. When tiling is triggered, each dask
            chunk corresponds to one ROI read via
            ``pylibCZIrw.CziReader.read(roi=(x, y, w, h))`` and the T/C/Z
            grouping strategy is forced to ``"plane"`` so per-plane tile
            grids can be composed with ``dask.array.block``. The requested
            edge is halved iteratively so a single tile stays within
            ``chunk_memory_limit``. Defaults to 4096.
        metadata: Optional preloaded CZI metadata. Only the metadata objects
            mutated by this function are copied. Supplying it avoids repeatedly
            parsing or deeply copying metadata when reading many scenes or
            pyramid levels from the same CZI.

    Returns:
        Tuple of (arrays_or_list, dims, num_stacks, metadata):
            - If stack_scenes=False: List of arrays (one per scene).
            - If stack_scenes=True and shapes match: Single array with S dim.
            - If stack_scenes=True but shapes differ: List (with warning).
        Each array has shape ([V, R, I, H, M,] T, C, Z, Y, X [, A]). Missing
        core dims (T, C, Z) get size=1. Also returns the list of canonical
        dimension labels and the number of stacks returned.

    Example:
        >>> from czitools.read_tools import read_tools
        >>> # Lazy loading with xarray
        >>> arrays, dims, num_stacks, mdata = read_tools.read_stacks(
        ...     "path/to/file.czi", use_dask=True, use_xarray=True
        ... )
        >>> # Stack scenes if they have the same shape
        >>> stacked, dims, num_stacks, mdata = read_tools.read_stacks(
        ...     "path/to/file.czi", stack_scenes=True
        ... )
    """
    filepath = str(filepath)

    # check zoom factor for valid range
    zoom = misc._check_zoom(zoom=zoom)

    if lazy_read_strategy not in {"chunk", "plane"}:
        raise ValueError("lazy_read_strategy must be either 'chunk' or 'plane'.")
    if planes_per_chunk < 1:
        raise ValueError("planes_per_chunk must be at least 1.")
    if tile_size < 1:
        raise ValueError("tile_size must be at least 1.")

    # Determine reader type for URL or local file support
    readertype, is_url = misc.get_pyczi_readertype(filepath)
    if is_url:
        logger.info("read_stacks: Reading from URL - using Curl reader")

    # Validate/create planes using CziMetadata (same rules as read_6darray)
    if metadata is None:
        mdata = czimd.CziMetadata(filepath)
    else:
        mdata = copy.copy(metadata)
        mdata.scale = copy.copy(metadata.scale_required)
        mdata.scale.ratio = dict(metadata.scale_required.ratio or {})
        if adapt_metadata:
            mdata.image = copy.copy(metadata.image_required)
    image = mdata.image_required
    bbox = mdata.bbox_required
    scale = mdata.scale_required

    # update scaling for zoomed XY reads
    scale.X_sf = np.round(_as_float(scale.X) * (1 / zoom), 3)
    scale.Y_sf = np.round(_as_float(scale.Y) * (1 / zoom), 3)
    if scale.ratio is None:
        scale.ratio = {}
    scale.ratio["zx_sf"] = np.round(_as_float(scale.Z) / _as_float(scale.X_sf), 3)

    # Normalize planes without mutating caller-provided dict.
    planes_input = planes
    if planes_input:
        planes = dict(planes_input)
        bbox_total = bbox.total_bounding_box or {}
        for k in ["S", "T", "C", "Z"]:
            if k in planes.keys() and k in bbox_total.keys():
                if bbox_total[k][1] - 1 < planes[k][1]:
                    logger.info(
                        f"Planes indices (zero-based) for {planes[k]} are invalid. BBox for {[k]}: {bbox_total[k]}"
                    )
                    return [], [], 0, mdata
    else:
        planes = {}
        for dim, size_attr in [
            ("S", image.SizeS),
            ("T", image.SizeT),
            ("C", image.SizeC),
            ("Z", image.SizeZ),
        ]:
            planes[dim] = (0, size_attr - 1) if size_attr is not None else (0, 0)

    # Ensure all expected dims exist in local planes mapping.
    for k in ["S", "T", "C", "Z"]:
        if k not in planes.keys():
            if k == "S":
                planes[k] = (0, image.SizeS - 1) if image.SizeS is not None else (0, 0)
            elif k == "T":
                planes[k] = (0, image.SizeT - 1) if image.SizeT is not None else (0, 0)
            elif k == "C":
                planes[k] = (0, image.SizeC - 1) if image.SizeC is not None else (0, 0)
            elif k == "Z":
                planes[k] = (0, image.SizeZ - 1) if image.SizeZ is not None else (0, 0)

    stack_arrays: StackList = []
    stack_shapes: list[tuple[int, ...]] = []

    with pyczi.open_czi(filepath, readertype) as czidoc:
        total_bbox = czidoc.total_bounding_box_no_pyramid
        total_num_stacks = len(czidoc.scenes_bounding_rectangle)
        # If the CZI has no explicit scenes, by default return an empty
        # result (no stacks). If the caller explicitly requested
        # `stack_scenes=True`, treat the total_bounding_rectangle as an
        # implicit single stack and continue. This preserves backward
        # compatibility with the stack_scenes option while matching tests
        # that expect empty results for scene-less files.
        if total_num_stacks == 0:
            if not stack_scenes:
                logger.info("read_stacks: no explicit scenes found; returning empty result")
                return [], [], 0, mdata
            logger.debug("read_stacks: No scenes found — using total_bounding_rectangle as single stack")
            total_num_stacks = 1

        s_start = planes["S"][0]
        s_end = planes["S"][1]

        # Validate requested scene range against available scene count.
        if s_start < 0 or s_end < s_start or s_end >= total_num_stacks:
            logger.info(
                f"read_stacks: Invalid S planes range {planes['S']} for available scene count {total_num_stacks}"
            )
            return [], [], 0, mdata

        # Number of stacks to read/return after applying S subset.
        num_stacks = s_end - s_start + 1
        logger.info(
            f"read_stacks: num_stacks={num_stacks} (selected from total={total_num_stacks}), "
            f"total_bounding_box={total_bbox}"
        )

        # Build dimension info from total_bounding_box
        # dim_from_bbox: {dim_name: (start, size)}
        dim_from_bbox: dict[str, tuple[int, int]] = {}
        for dim in _PLANE_DIMS_READ:
            if dim in total_bbox:
                dim_from_bbox[dim] = total_bbox[dim]

        # Build canonical dimension order:
        # 1. Extra dims that are present (in _EXTRA_DIMS order)
        # 2. Core dims (always present, default size=1)
        canonical_dims: list[str] = []
        dim_sizes_map: dict[str, int] = {}
        dim_starts_map: dict[str, int] = {}

        # Add extra dims if present
        for dim in _EXTRA_DIMS:
            if dim in dim_from_bbox:
                start, size = dim_from_bbox[dim]
                canonical_dims.append(dim)
                dim_sizes_map[dim] = size
                dim_starts_map[dim] = start

        # Add core dims (always, default size=1 if missing)
        for dim in _CORE_DIMS:
            if dim in dim_from_bbox:
                start, size = dim_from_bbox[dim]
                dim_sizes_map[dim] = size
                dim_starts_map[dim] = start
            else:
                # Missing core dim -> size=1, start=0
                dim_sizes_map[dim] = 1
                dim_starts_map[dim] = 0
            canonical_dims.append(dim)

        # Override core dims (T, C, Z) with planes selection if provided
        for dim in ["T", "C", "Z"]:
            if dim in planes:
                pstart, pend = planes[dim]
                dim_sizes_map[dim] = pend - pstart + 1
                dim_starts_map[dim] = pstart

        # Track S separately, apply planes selection if provided
        dim_sizes_map["S"] = num_stacks
        dim_starts_map["S"] = s_start

        # Dims for reading planes (excludes S which is the scene loop)
        read_dims = canonical_dims  # already excludes S
        read_sizes = [dim_sizes_map[d] for d in read_dims]
        read_starts = [dim_starts_map[d] for d in read_dims]

        logger.info(f"read_stacks: canonical_dims={canonical_dims}, dim_sizes={dim_sizes_map}")
        if use_dask:
            logger.info("read_stacks: Using lazy dask arrays - data will be read on demand")

        all_dims: list[str] = []  # will be set in loop

        for stack_idx in range(num_stacks):
            # Map local loop index to actual scene index when S-range was provided
            if len(czidoc.scenes_bounding_rectangle) > 0:
                scene_index = dim_starts_map.get("S", 0) + stack_idx
                if scene_index < len(czidoc.scenes_bounding_rectangle):
                    stack_rect = czidoc.scenes_bounding_rectangle[scene_index]
                else:
                    # Out-of-range — fallback to total bounding rectangle
                    stack_rect = czidoc.total_bounding_rectangle
            else:
                scene_index = None
                stack_rect = czidoc.total_bounding_rectangle
            size_y, size_x = stack_rect.h, stack_rect.w
            logger.debug(f"read_stacks: Stack {stack_idx}: Y={size_y}, X={size_x}")

            # Peek at dtype and pixel-component layout using a 1x1 ROI. A
            # full-plane sample here would allocate the entire plane just to
            # inspect ``sample.dtype`` and ``sample.shape[2]``; that costs
            # ~24 GB per level on gigapixel files. The 1x1 ROI is a few bytes
            # and gives us the same dtype/RGB information. Spatial extents
            # come from ``stack_rect`` (which we already have) scaled by the
            # zoom factor — matching what pylibCZIrw returns from a full read.
            sample_plane = {name: start for name, start in zip(read_dims, read_starts)}
            probe_roi = (int(stack_rect.x), int(stack_rect.y), 1, 1)
            if scene_index is not None and len(czidoc.scenes_bounding_rectangle) > scene_index:
                sample = czidoc.read(plane=sample_plane, scene=scene_index, roi=probe_roi, zoom=zoom)
            else:
                sample = czidoc.read(plane=sample_plane, roi=probe_roi, zoom=zoom)
            dtype = sample.dtype

            # sample.shape is (1, 1) for grayscale or (1, 1, C) for RGB after
            # the 1x1 ROI, so pixel-type detection stays identical.
            has_pixel_type = sample.ndim == 3
            if has_pixel_type:
                num_components = int(sample.shape[2])
                squeeze_grayscale = num_components == 1
            else:
                num_components = None
                squeeze_grayscale = False

            # Full-plane spatial dims come from the scene rectangle, scaled by
            # zoom. Use ``int()`` truncation to match libCZI/pylibCZIrw's
            # convention (``round()`` overshoots by one for zooms like 0.037037
            # and produces a shape mismatch when the actual read comes back).
            # ``max(1, ...)`` guards against tiny scenes at extreme zooms
            # rounding down to zero, which would produce empty arrays.
            spatial_y = max(1, int(float(stack_rect.h) * zoom))
            spatial_x = max(1, int(float(stack_rect.w) * zoom))

            # Decide up-front whether the dask branch will need spatial tiling.
            # When it does the actual composed shape is the sum of per-tile
            # zoomed sizes, which can differ from the direct ``int(rect*zoom)``
            # estimate by a few pixels due to sum-of-truncations vs
            # truncation-of-sum. Recompute here so array_shape, coords, and
            # the dask graph all agree.
            dtype_nbytes = int(np.dtype(dtype).itemsize)
            plane_components = num_components if (has_pixel_type and not squeeze_grayscale) else 1
            plane_bytes = spatial_y * spatial_x * dtype_nbytes * (plane_components or 1)
            use_spatial_tiling = use_dask and plane_bytes > chunk_memory_limit

            tile_h_zoomed = tile_w_zoomed = 0
            tile_h_layer0 = tile_w_layer0 = 0
            row_offsets_layer0: list[int] = []
            row_sizes_zoomed: list[int] = []
            col_offsets_layer0: list[int] = []
            col_sizes_zoomed: list[int] = []
            if use_spatial_tiling:
                tile_h_zoomed, tile_w_zoomed = _compute_tile_extent(
                    spatial_y=spatial_y,
                    spatial_x=spatial_x,
                    dtype_nbytes=dtype_nbytes,
                    components=plane_components or 1,
                    tile_size=tile_size,
                    memory_limit=chunk_memory_limit,
                )
                # Plan Y and X grids once; both the dask builder and the
                # coordinate arrays below use the resulting per-tile sizes.
                spatial_y, row_offsets_layer0, row_sizes_zoomed = _plan_tile_grid(
                    rect_extent_layer0=int(stack_rect.h),
                    tile_extent_zoomed=tile_h_zoomed,
                    zoom=zoom,
                )
                spatial_x, col_offsets_layer0, col_sizes_zoomed = _plan_tile_grid(
                    rect_extent_layer0=int(stack_rect.w),
                    tile_extent_zoomed=tile_w_zoomed,
                    zoom=zoom,
                )
                tile_h_layer0 = max(1, math.ceil(tile_h_zoomed / zoom))
                tile_w_layer0 = max(1, math.ceil(tile_w_zoomed / zoom))
                logger.info(
                    "read_stacks: Stack %d plane is %.2f MB (> %.2f MB limit); "
                    "using spatial tiling %dx%d (%d tiles per plane, tile %dx%d zoomed).",
                    stack_idx,
                    plane_bytes / (1024 * 1024),
                    chunk_memory_limit / (1024 * 1024),
                    len(row_sizes_zoomed),
                    len(col_sizes_zoomed),
                    len(row_sizes_zoomed) * len(col_sizes_zoomed),
                    tile_h_zoomed,
                    tile_w_zoomed,
                )

            # Build final dimension list and shape (excluding S, since we loop over scenes)
            # Shape: (*read_sizes, Y, X [, A])
            if has_pixel_type and not squeeze_grayscale:
                array_shape = tuple(read_sizes) + (spatial_y, spatial_x, num_components)
                plane_shape = (spatial_y, spatial_x, num_components)
                all_dims = read_dims + ["Y", "X", "A"]
            else:
                array_shape = tuple(read_sizes) + (spatial_y, spatial_x)
                plane_shape = (spatial_y, spatial_x)
                all_dims = read_dims + ["Y", "X"]

            stack_shapes.append(array_shape)

            # Ensure all_dims is set (used later when returning dims). It is
            # safe to set it per-stack because canonical read dims are the same
            # for all stacks; this avoids returning an empty `all_dims` when the
            # file has no scenes.
            all_dims = all_dims or read_dims + (
                ["Y", "X", "A"] if has_pixel_type and not squeeze_grayscale else ["Y", "X"]
            )

            if use_dask:
                total_planes = int(np.prod(read_sizes)) if read_sizes else 1

                # ROI origin: for scene-based files the ROI is expressed in the
                # CZI's coordinate system, so use the scene's top-left corner.
                # For scene-less files fall back to the total bounding rectangle
                # (its (x, y) is 0 by default but may differ on legacy files).
                if scene_index is not None and len(czidoc.scenes_bounding_rectangle) > scene_index:
                    origin_xy = (int(stack_rect.x), int(stack_rect.y))
                else:
                    origin_xy = (int(stack_rect.x), int(stack_rect.y))

                # Tiling forces per-plane task construction so we can compose a
                # ``da.block`` grid at the leaves. The multi-plane "chunk"
                # strategy is incompatible with per-plane spatial grids because
                # each task returns a flat contiguous buffer.
                effective_strategy: LazyReadStrategy = "plane" if use_spatial_tiling else lazy_read_strategy
                if use_spatial_tiling and lazy_read_strategy == "chunk":
                    logger.info(
                        "read_stacks: spatial tiling active — overriding "
                        "lazy_read_strategy='chunk' with 'plane' for this stack."
                    )

                if effective_strategy == "chunk":
                    logger.info(
                        f"read_stacks: Grouping {total_planes} planes into tasks "
                        f"of at most {planes_per_chunk} planes"
                    )
                    ranges = [range(size) for size in read_sizes]
                    index_iterator = itertools.product(*ranges)
                    delayed_chunks: list[da.Array] = []

                    while True:
                        chunk_indices = list(itertools.islice(index_iterator, planes_per_chunk))
                        if not chunk_indices:
                            break

                        delayed_chunk = dask_delayed(_read_plane_chunk)(
                            filepath,
                            chunk_indices,
                            read_dims,
                            read_starts,
                            scene_index,
                            squeeze_grayscale,
                            plane_shape,
                            dtype,
                            zoom,
                            readertype,
                        )
                        delayed_chunks.append(
                            da.from_delayed(
                                delayed_chunk,
                                shape=(len(chunk_indices),) + plane_shape,
                                dtype=dtype,
                            )
                        )

                    flattened = (
                        delayed_chunks[0] if len(delayed_chunks) == 1 else da.concatenate(delayed_chunks, axis=0)
                    )
                    stack = flattened.reshape(array_shape)
                else:
                    if not use_spatial_tiling:
                        logger.info(f"read_stacks: Creating {total_planes} fine-grained plane tasks")

                    # Capture whether the pixel-type axis must be preserved as
                    # the trailing dim in every tile.
                    has_rgb_component = has_pixel_type and not squeeze_grayscale

                    def build_dask_stack(
                        dims_remaining: list[int],
                        current_indices: list[int],
                    ) -> da.Array:
                        """Recursively stack per-plane dask arrays.

                        Leaves are either a single whole-plane delayed read
                        or, when ``use_spatial_tiling`` is True, a tile grid
                        composed by :func:`_build_tiled_plane_dask`. The
                        interior nodes stack per-dimension along a new axis,
                        matching the canonical dim order.
                        """
                        if not dims_remaining:
                            plane = {
                                name: start + index
                                for name, start, index in zip(
                                    read_dims,
                                    read_starts,
                                    current_indices,
                                )
                            }
                            if use_spatial_tiling:
                                # Each leaf is already a chunked dask array of
                                # shape plane_shape — nothing is read yet.
                                # The tile builder works in layer-0 coords
                                # because pylibCZIrw ROIs are native and the
                                # output is downsampled internally by libCZI.
                                # Pre-planned row/col layouts guarantee the
                                # dask array shape matches spatial_y/spatial_x
                                # exactly (see _plan_tile_grid).
                                return _build_tiled_plane_dask(
                                    filepath=filepath,
                                    plane=plane,
                                    stack_idx=scene_index,
                                    origin_xy=origin_xy,
                                    row_offsets_layer0=row_offsets_layer0,
                                    row_sizes_zoomed=row_sizes_zoomed,
                                    col_offsets_layer0=col_offsets_layer0,
                                    col_sizes_zoomed=col_sizes_zoomed,
                                    tile_h_layer0=tile_h_layer0,
                                    tile_w_layer0=tile_w_layer0,
                                    rect_h_layer0=int(stack_rect.h),
                                    rect_w_layer0=int(stack_rect.w),
                                    squeeze_grayscale=squeeze_grayscale,
                                    dtype=dtype,
                                    zoom=zoom,
                                    readertype=readertype,
                                    has_rgb_component=has_rgb_component,
                                    num_components=num_components,
                                )
                            delayed_read = _read_plane_delayed(
                                filepath,
                                plane,
                                scene_index,
                                squeeze_grayscale,
                                zoom,
                                readertype,
                            )
                            return da.from_delayed(
                                delayed_read,
                                shape=plane_shape,
                                dtype=dtype,
                            )

                        dim_size = dims_remaining[0]
                        return da.stack(
                            [
                                build_dask_stack(
                                    dims_remaining[1:],
                                    current_indices + [index],
                                )
                                for index in range(dim_size)
                            ],
                            axis=0,
                        )

                    stack = build_dask_stack(read_sizes, [])

                stack_chunks = getattr(stack, "chunks", None)
                logger.debug(f"read_stacks: Stack {stack_idx} -> array shape={stack.shape}, chunks={stack_chunks}")

            else:
                # Eager loading - read all planes immediately
                stack = np.empty(array_shape, dtype=dtype)

                # Build all index combinations for the read dimensions
                ranges = [range(s) for s in read_sizes]
                total_planes = int(np.prod(read_sizes)) if read_sizes else 1

                for idx, combo in enumerate(itertools.product(*ranges)):
                    # Build plane dict with actual coordinate values
                    plane = {name: start + offset for name, start, offset in zip(read_dims, read_starts, combo)}
                    # If no explicit scenes exist, omit scene param
                    if scene_index is not None and len(czidoc.scenes_bounding_rectangle) > scene_index:
                        img2d = czidoc.read(plane=plane, scene=scene_index, zoom=zoom)
                    else:
                        img2d = czidoc.read(plane=plane, zoom=zoom)

                    # Squeeze grayscale (A=1) but keep RGB (A=3) / RGBA (A=4)
                    if squeeze_grayscale:
                        img2d = img2d[..., 0]

                    # Store in the correct position
                    stack[combo] = img2d

                logger.debug(f"read_stacks: Stack {stack_idx} -> np.ndarray shape={stack.shape}")

            # check if stack is an BGR image and convert to RGB
            contains_rgb = any((mdata.isRGB or {}).values())
            if contains_rgb and stack.shape[-1] == 3:
                # image has BGR values and need to be converted to RGB
                stack = stack[..., ::-1]

            if use_xarray:
                # Build coordinate arrays for each dimension
                coords = {}
                for dim in read_dims:
                    start = dim_starts_map[dim]
                    size = dim_sizes_map[dim]
                    coords[dim] = np.arange(start, start + size)
                coords["Y"] = np.arange(spatial_y)
                coords["X"] = np.arange(spatial_x)
                if has_pixel_type and not squeeze_grayscale:
                    coords["A"] = np.arange(num_components)

                xr_da = xr.DataArray(
                    stack,
                    dims=all_dims,
                    coords=coords,
                    attrs={
                        "stack": stack_idx,
                        "filepath": filepath,
                        "axes": "".join(all_dims),
                        "subset_planes": planes,
                    },
                )

                # Assign coordinate values based on metadata scaling and channel names
                spatial_coords = {
                    ax: np.arange(xr_da.sizes[ax]) * _get_axis_coord_step(mdata.scale, ax, zoom) for ax in "ZYX"
                }
                xr_da = xr_da.assign_coords(
                    C=_channel_names_or_default(mdata, xr_da.sizes["C"]), **cast(Any, spatial_coords)
                )

                stack_arrays.append(xr_da)
            else:
                stack_arrays.append(stack)

    # Optionally stack scenes if requested and all shapes match
    if stack_scenes:
        # If no stacks were collected, just return the list (nothing to stack).
        if not stack_shapes:
            logger.info("read_stacks: stack_scenes requested but no stacks were found; returning list")
            return stack_arrays, all_dims, num_stacks, mdata

        unique_shapes = set(stack_shapes)

        # Crop scenes to a common Y/X shape when they differ within tolerance.
        if len(unique_shapes) > 1 and scene_stack_tolerance > 0:
            y_idx = all_dims.index("Y")
            x_idx = all_dims.index("X")
            all_y = [s[y_idx] for s in stack_shapes]
            all_x = [s[x_idx] for s in stack_shapes]
            non_yx = [tuple(v for i, v in enumerate(s) if i not in (y_idx, x_idx)) for s in stack_shapes]
            if (
                len(set(non_yx)) == 1
                and max(all_y) - min(all_y) <= scene_stack_tolerance
                and max(all_x) - min(all_x) <= scene_stack_tolerance
            ):
                min_y, min_x = min(all_y), min(all_x)
                logger.info(
                    "read_stacks: Y/X sizes differ by at most %d px; cropping all "
                    "stacks to Y=%d, X=%d for stacking (scene_stack_tolerance=%d).",
                    max(max(all_y) - min(all_y), max(all_x) - min(all_x)),
                    min_y,
                    min_x,
                    scene_stack_tolerance,
                )
                cropped: list[Any] = []
                for arr in stack_arrays:
                    if isinstance(arr, xr.DataArray):
                        arr = arr.isel(Y=slice(0, min_y), X=slice(0, min_x))
                    else:
                        sel: list[Any] = [slice(None)] * len(all_dims)
                        sel[y_idx] = slice(0, min_y)
                        sel[x_idx] = slice(0, min_x)
                        arr = arr[tuple(sel)]
                    cropped.append(arr)
                stack_arrays = cropped
                stack_shapes = [tuple(int(v) for v in arr.shape) for arr in stack_arrays]
                unique_shapes = set(stack_shapes)

        if len(unique_shapes) == 1:
            logger.info(f"read_stacks: Stacking {num_stacks} stacks (all shapes equal: {stack_shapes[0]})")
            stacked_dims = ["S"] + all_dims

            if use_xarray:
                # Stack xr.DataArrays along new S dimension
                # Prepare arrays according to chunk_policy
                prepared = []
                target_shape = stack_shapes[0]
                for arr in stack_arrays:
                    if isinstance(arr, xr.DataArray) and hasattr(arr.data, "chunks"):
                        if chunk_policy == "scene-shape":
                            # chunk by the scene's own shape (arr.shape)
                            chunk_map = {dim: size for dim, size in zip(arr.dims, arr.shape)}
                            arr = arr.chunk(chunk_map)
                        elif chunk_policy == "stack-shape":
                            # Estimate bytes per element
                            dtype_nbytes = int(np.dtype(arr.dtype).itemsize)
                            # Build initial chunk_map equal to target_shape
                            chunk_map = {dim: size for dim, size in zip(arr.dims, target_shape)}
                            # Estimate chunk bytes: product of chunk dims * dtype size
                            elems = 1
                            for d in arr.dims:
                                elems *= chunk_map[d]
                            est_bytes = elems * dtype_nbytes
                            # If estimated bytes exceed limit, reduce spatial chunks (Y/X)
                            if est_bytes > chunk_memory_limit:
                                # Identify spatial dims (heuristic: last two dims are Y, X)
                                spatial_dims = list(arr.dims[-2:]) if len(arr.dims) >= 2 else []
                                # Copy sizes to mutable list
                                spatial_sizes = [chunk_map[d] for d in spatial_dims]
                                # Iteratively halve spatial sizes until under limit
                                while est_bytes > chunk_memory_limit and any(s > 1 for s in spatial_sizes):
                                    for i, s in enumerate(spatial_sizes):
                                        if s > 1:
                                            spatial_sizes[i] = max(1, s // 2)
                                    # update chunk_map and est_bytes
                                    for d, s in zip(spatial_dims, spatial_sizes):
                                        chunk_map[d] = s
                                    elems = 1
                                    for d in arr.dims:
                                        elems *= chunk_map[d]
                                    est_bytes = elems * dtype_nbytes
                                logger.warning(
                                    "read_stacks: target stack-shape chunk exceeded chunk_memory_limit; "
                                    "downscaled spatial chunking to avoid huge single chunk"
                                )
                            arr = arr.chunk(chunk_map)
                    prepared.append(arr)

                stacked = xr.concat(prepared, dim="S")
                stacked = stacked.assign_coords(S=np.arange(s_start, s_start + num_stacks))
                stacked.attrs["filepath"] = filepath
                mdata.array6d_size = stacked.shape
                if adapt_metadata:
                    image.SizeS = planes["S"][1] - planes["S"][0] + 1 if "S" in planes else image.SizeS
                    image.SizeT = planes["T"][1] - planes["T"][0] + 1 if "T" in planes else image.SizeT
                    image.SizeC = planes["C"][1] - planes["C"][0] + 1 if "C" in planes else image.SizeC
                    image.SizeZ = planes["Z"][1] - planes["Z"][0] + 1 if "Z" in planes else image.SizeZ
                return stacked, stacked_dims, num_stacks, mdata
            else:
                # Stack arrays (numpy or dask). Rechunk dask arrays to
                if use_dask:
                    prepared = []
                    target_shape = stack_shapes[0]
                    for a in stack_arrays:
                        if isinstance(a, da.Array):
                            if chunk_policy == "scene-shape":
                                a = cast(Any, a).rechunk(a.shape)
                            elif chunk_policy == "stack-shape":
                                # Estimate memory per chunk and optionally downscale
                                dtype_nbytes = int(np.dtype(a.dtype).itemsize)
                                elems = 1
                                for s in target_shape:
                                    elems *= s
                                est_bytes = elems * dtype_nbytes
                                if est_bytes > chunk_memory_limit:
                                    # Reduce spatial axes (last two) progressively
                                    spatial = list(target_shape[-2:]) if len(target_shape) >= 2 else []
                                    spatial_sizes = spatial.copy()
                                    while est_bytes > chunk_memory_limit and any(s > 1 for s in spatial_sizes):
                                        for i, s in enumerate(spatial_sizes):
                                            if s > 1:
                                                spatial_sizes[i] = max(1, s // 2)
                                        elems = 1
                                        for i, dim_size in enumerate(target_shape):
                                            if i >= len(target_shape) - 2:
                                                elems *= spatial_sizes[i - (len(target_shape) - 2)]
                                            else:
                                                elems *= dim_size
                                        est_bytes = elems * dtype_nbytes
                                    # Build rechunk tuple: use target_shape but replace last two dims
                                    rechunk_tuple = list(target_shape)
                                    if len(rechunk_tuple) >= 2:
                                        rechunk_tuple[-2] = spatial_sizes[0]
                                        rechunk_tuple[-1] = spatial_sizes[1]
                                    a = cast(Any, a).rechunk(tuple(rechunk_tuple))
                                else:
                                    a = cast(Any, a).rechunk(target_shape)
                        prepared.append(a)
                    stacked = da.stack(prepared, axis=0)
                else:
                    stacked = np.stack([np.asarray(a) for a in stack_arrays], axis=0)
                mdata.array6d_size = stacked.shape
                if adapt_metadata:
                    image.SizeS = planes["S"][1] - planes["S"][0] + 1 if "S" in planes else image.SizeS
                    image.SizeT = planes["T"][1] - planes["T"][0] + 1 if "T" in planes else image.SizeT
                    image.SizeC = planes["C"][1] - planes["C"][0] + 1 if "C" in planes else image.SizeC
                    image.SizeZ = planes["Z"][1] - planes["Z"][0] + 1 if "Z" in planes else image.SizeZ
                return stacked, stacked_dims, num_stacks, mdata
        else:
            logger.warning(f"read_stacks: Cannot stack stacks - shapes differ: {unique_shapes}")

    if stack_arrays:
        mdata.array6d_size = tuple(int(s) for s in stack_arrays[0].shape)
    if adapt_metadata:
        image.SizeS = planes["S"][1] - planes["S"][0] + 1 if "S" in planes else image.SizeS
        image.SizeT = planes["T"][1] - planes["T"][0] + 1 if "T" in planes else image.SizeT
        image.SizeC = planes["C"][1] - planes["C"][0] + 1 if "C" in planes else image.SizeC
        image.SizeZ = planes["Z"][1] - planes["Z"][0] + 1 if "Z" in planes else image.SizeZ

    return stack_arrays, all_dims, num_stacks, mdata


def read_stacks_list(
    filepath: CziPath,
    use_dask: bool = False,
    use_xarray: bool = True,
    planes: dict[str, tuple[int, int]] | None = None,
    zoom: float = 1.0,
    adapt_metadata: bool = False,
    chunk_policy: str = "none",
    chunk_memory_limit: int = 256 * 1024 * 1024,
    lazy_read_strategy: LazyReadStrategy = "chunk",
    planes_per_chunk: int = 64,
    tile_size: int = 4096,
) -> tuple[StackList, list[str], int, czimd.CziMetadata]:
    """Read stacks and always return a list (one element per scene).

    This is a typed convenience wrapper around ``read_stacks(..., stack_scenes=False)``.
    Use this function when you want a stable list return contract for static typing.
    """
    result, dims, num_stacks, mdata = read_stacks(
        filepath=filepath,
        use_dask=use_dask,
        use_xarray=use_xarray,
        stack_scenes=False,
        planes=planes,
        zoom=zoom,
        adapt_metadata=adapt_metadata,
        chunk_policy=chunk_policy,
        chunk_memory_limit=chunk_memory_limit,
        lazy_read_strategy=lazy_read_strategy,
        planes_per_chunk=planes_per_chunk,
        tile_size=tile_size,
    )

    if not isinstance(result, list):
        raise ValueError("read_stacks_list expected a list result.")
    return result, dims, num_stacks, mdata


def read_stacks_stacked(
    filepath: CziPath,
    use_dask: bool = False,
    use_xarray: bool = True,
    planes: dict[str, tuple[int, int]] | None = None,
    zoom: float = 1.0,
    adapt_metadata: bool = False,
    chunk_policy: str = "none",
    chunk_memory_limit: int = 256 * 1024 * 1024,
    lazy_read_strategy: LazyReadStrategy = "chunk",
    planes_per_chunk: int = 64,
    tile_size: int = 4096,
) -> tuple[StackArray, list[str], int, czimd.CziMetadata]:
    """Read stacks and require a single stacked output with an S dimension.

    This is a typed convenience wrapper around ``read_stacks(..., stack_scenes=True)``.
    It raises ``ValueError`` when scenes cannot be stacked into a single output
    (for example, when scene shapes differ).
    """
    result, dims, num_stacks, mdata = read_stacks(
        filepath=filepath,
        use_dask=use_dask,
        use_xarray=use_xarray,
        stack_scenes=True,
        planes=planes,
        zoom=zoom,
        adapt_metadata=adapt_metadata,
        chunk_policy=chunk_policy,
        chunk_memory_limit=chunk_memory_limit,
        lazy_read_strategy=lazy_read_strategy,
        planes_per_chunk=planes_per_chunk,
        tile_size=tile_size,
    )

    if isinstance(result, list):
        raise ValueError(
            "read_stacks_stacked requires stackable scenes, but read_stacks returned a list. "
            "Use read_stacks_list for per-scene outputs."
        )

    return result, dims, num_stacks, mdata
