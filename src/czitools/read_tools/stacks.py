"""Read CZI scenes as regular or irregular eager/lazy stacks."""

import itertools
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
            into a single array with S as the first dimension. If shapes differ,
            returns a list (with a warning). Defaults to False.
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

    # Determine reader type for URL or local file support
    readertype, is_url = misc.get_pyczi_readertype(filepath)
    if is_url:
        logger.info("read_stacks: Reading from URL - using Curl reader")

    # Validate/create planes using CziMetadata (same rules as read_6darray)
    mdata = czimd.CziMetadata(filepath)
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

            # Sample read to get dtype and actual spatial shape
            sample_plane = {name: start for name, start in zip(read_dims, read_starts)}
            # If scenes present, read from the mapped scene_index; otherwise read total
            if scene_index is not None and len(czidoc.scenes_bounding_rectangle) > scene_index:
                sample = czidoc.read(plane=sample_plane, scene=scene_index, zoom=zoom)
            else:
                sample = czidoc.read(plane=sample_plane, zoom=zoom)
            dtype = sample.dtype

            # Handle pixel type dimension (A) - grayscale (A=1) or RGB (A=3)
            has_pixel_type = sample.ndim == 3
            if has_pixel_type:
                spatial_y, spatial_x, num_components = sample.shape
                squeeze_grayscale = num_components == 1  # squeeze if grayscale
            else:
                spatial_y, spatial_x = sample.shape
                num_components = None
                squeeze_grayscale = False

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

                if lazy_read_strategy == "chunk":
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
                        delayed_chunks[0]
                        if len(delayed_chunks) == 1
                        else da.concatenate(delayed_chunks, axis=0)
                    )
                    stack = flattened.reshape(array_shape)
                else:
                    logger.info(
                        f"read_stacks: Creating {total_planes} fine-grained plane tasks"
                    )

                    def build_dask_stack(
                        dims_remaining: list[int],
                        current_indices: list[int],
                    ) -> da.Array:
                        """Recursively stack delayed single-plane reads."""
                        if not dims_remaining:
                            plane = {
                                name: start + index
                                for name, start, index in zip(
                                    read_dims,
                                    read_starts,
                                    current_indices,
                                )
                            }
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
    )

    if isinstance(result, list):
        raise ValueError(
            "read_stacks_stacked requires stackable scenes, but read_stacks returned a list. "
            "Use read_stacks_list for per-scene outputs."
        )

    return result, dims, num_stacks, mdata
