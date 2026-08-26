"""Read regular CZI data as an STCZYX(A) array."""

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
    Array6D,
    CziPath,
    _as_float,
    _as_int,
    _channel_names_or_default,
    _get_axis_coord_step,
    logger,
)

try:
    import progressbar

    HAS_PROGRESSBAR = True
except ImportError:
    HAS_PROGRESSBAR = False


def _spatial_region(
    mdata: czimd.CziMetadata,
    scene: int,
    zoom: float,
) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    """Return the layer-0 ROI and zoomed YX shape for one scene."""
    bbox = mdata.bbox_required
    rect = None

    if mdata.has_scenes:
        scene_rects = bbox.scenes_bounding_rect_no_pyramid or bbox.scenes_bounding_rect or {}
        rect = scene_rects.get(scene)
    else:
        rect = bbox.total_rect_no_pyramid or bbox.total_rect

    if rect is None:
        raise ValueError("Cannot determine the CZI plane shape from its bounding rectangles.")

    roi = (int(rect.x), int(rect.y), int(rect.w), int(rect.h))
    shape = (max(1, int(rect.h * zoom)), max(1, int(rect.w * zoom)))
    return roi, shape


def _read_plane(
    filepath: str,
    plane: dict[str, int],
    scene: int | None,
    roi: tuple[int, int, int, int],
    zoom: float,
    readertype: pyczi.ReaderFileInputTypes,
    squeeze_grayscale: bool,
) -> np.ndarray:
    """Read one plane; Dask schedules this function only during computation."""
    with pyczi.open_czi(filepath, readertype) as czidoc:
        if scene is None:
            image2d = czidoc.read(plane=plane, roi=roi, zoom=zoom)
        else:
            image2d = czidoc.read(plane=plane, scene=scene, roi=roi, zoom=zoom)

    if squeeze_grayscale:
        image2d = image2d[..., 0]
    return image2d


def _build_lazy_array(
    filepath: str,
    sizes: tuple[int, int, int, int],
    starts: tuple[int, int, int, int],
    plane_shape: tuple[int, ...],
    dtype: np.dtype,
    has_scenes: bool,
    spatial_rois: dict[int | None, tuple[int, int, int, int]],
    zoom: float,
    readertype: pyczi.ReaderFileInputTypes,
    squeeze_grayscale: bool,
) -> da.Array:
    """Build an STCZ plane graph without reading pixel data."""
    dimension_names = ("S", "T", "C", "Z")

    def build_axis(axis: int, indices: tuple[int, ...]) -> da.Array:
        if axis == len(sizes):
            actual = tuple(start + index for start, index in zip(starts, indices))
            plane = {name: value for name, value in zip(dimension_names[1:], actual[1:])}
            scene = actual[0] if has_scenes else None
            delayed_plane = dask_delayed(_read_plane)(
                filepath,
                plane,
                scene,
                spatial_rois[scene],
                zoom,
                readertype,
                squeeze_grayscale,
            )
            return da.from_delayed(delayed_plane, shape=plane_shape, dtype=dtype)

        return da.stack(
            [build_axis(axis + 1, indices + (index,)) for index in range(sizes[axis])],
            axis=0,
        )

    return build_axis(0, ())


def read_6darray(
    filepath: CziPath,
    use_dask: bool = False,
    chunk_zyx: bool = False,
    planes: dict[str, tuple[int, int]] | None = None,
    zoom: float = 1.0,
    use_xarray: bool = True,
    adapt_metadata: bool = False,
) -> tuple[Array6D | None, czimd.CziMetadata]:
    """Read a CZI image file as an STCZYX(A) array.

    Only scenes with equal size and CZIs with consistent pixel types can be
    represented by this regular array API.

    Args:
        filepath: Path to the CZI image file.
        use_dask: If True, return a genuinely lazy Dask-backed array. Pixel
            planes are not read until the array is computed or indexed.
        chunk_zyx: If True and use_dask=True, rechunk so each chunk contains a
            complete ZYX block for one S/T/C position.
        planes: Substack ranges for S, T, Z, and C as inclusive zero-based
            (start, end) tuples.
        zoom: Downscale factor for images in the range 0.01 through 1.0.
        use_xarray: If True, wrap the NumPy or Dask array in xr.DataArray.
        adapt_metadata: If True, update metadata dimensions to the selected
            S/T/C/Z output sizes.

    Returns:
        A tuple of the image array and CZI metadata. The array is None when the
        CZI cannot be represented as a regular 6D array.

    Notes:
        Pixel reads are constrained to each selected scene's full-resolution,
        non-pyramid bounding rectangle. This keeps regular array shapes aligned
        with the layer-0 image when coarse pyramid coverage is rounded outward.
    """
    filepath = str(filepath)
    zoom = misc._check_zoom(zoom=zoom)
    dims = ("S", "T", "C", "Z", "Y", "X", "A")

    mdata = czimd.CziMetadata(filepath)
    image = mdata.image_required
    bbox = mdata.bbox_required
    scale = mdata.scale_required

    if not mdata.consistent_pixeltypes:
        logger.info("Detected PixelTypes are not consistent. Cannot create array6d")
        return None, mdata

    scale.X_sf = np.round(_as_float(scale.X) * (1 / zoom), 3)
    scale.Y_sf = np.round(_as_float(scale.Y) * (1 / zoom), 3)
    if scale.ratio is None:
        scale.ratio = {}
    scale.ratio["zx_sf"] = np.round(_as_float(scale.Z) / _as_float(scale.X_sf), 3)

    if planes:
        selected_planes = dict(planes)
        bbox_total = bbox.total_bounding_box or {}
        for dim in ["S", "T", "C", "Z"]:
            if dim in selected_planes and dim in bbox_total:
                if bbox_total[dim][1] - 1 < selected_planes[dim][1]:
                    logger.info(
                        f"Planes indices (zero-based) for {selected_planes[dim]} are invalid. "
                        f"BBox for {[dim]}: {bbox_total[dim]}"
                    )
                    return None, mdata
    else:
        selected_planes = {}
        for dim, size_attr in [
            ("S", image.SizeS),
            ("T", image.SizeT),
            ("C", image.SizeC),
            ("Z", image.SizeZ),
        ]:
            selected_planes[dim] = (0, size_attr - 1) if size_attr is not None else (0, 0)

    for dim, size_attr in [
        ("S", image.SizeS),
        ("T", image.SizeT),
        ("C", image.SizeC),
        ("Z", image.SizeZ),
    ]:
        if dim not in selected_planes:
            selected_planes[dim] = (0, size_attr - 1) if size_attr is not None else (0, 0)

    if not mdata.scene_shape_is_consistent:
        one_scene_selected = selected_planes["S"][0] == selected_planes["S"][1]
        if not one_scene_selected:
            logger.warning("Scenes have inconsistent shape. Cannot read 6D array")
            return None, mdata

    size_s = _as_int(misc._check_dimsize(image.SizeS, set2value=1), 1)
    size_t = _as_int(misc._check_dimsize(image.SizeT, set2value=1), 1)
    size_c = _as_int(misc._check_dimsize(image.SizeC, set2value=1), 1)
    size_z = _as_int(misc._check_dimsize(image.SizeZ, set2value=1), 1)
    starts = {"S": 0, "T": 0, "C": 0, "Z": 0}

    for dim, size_attr in [
        ("S", image.SizeS),
        ("T", image.SizeT),
        ("C", image.SizeC),
        ("Z", image.SizeZ),
    ]:
        if size_attr is None:
            continue
        start, end = selected_planes[dim]
        selected_size = end - start + 1
        starts[dim] = start
        if dim == "S":
            size_s = selected_size
            image.SizeS = selected_size
        elif dim == "T":
            size_t = selected_size
            image.SizeT = selected_size
        elif dim == "C":
            size_c = selected_size
            image.SizeC = selected_size
        else:
            size_z = selected_size
            image.SizeZ = selected_size

    contains_rgb = any((mdata.isRGB or {}).values())
    squeeze_grayscale = not contains_rgb
    if squeeze_grayscale:
        dims = ("S", "T", "C", "Z", "Y", "X")

    if mdata.is_url:
        logger.info("Reading pixel data via network from link location.")

    total_planes = size_s * size_t * size_c * size_z
    logger.info(
        f"read_6darray: {'Scheduling' if use_dask else 'Reading'} {total_planes} planes "
        f"(S={size_s}, T={size_t}, C={size_c}, Z={size_z})"
    )
    if total_planes == 0:
        logger.warning("No planes were selected for read_6darray.")
        return None, mdata

    selected_scenes = range(starts["S"], starts["S"] + size_s) if mdata.has_scenes else (0,)
    spatial_regions = {scene: _spatial_region(mdata, scene, zoom) for scene in selected_scenes}
    scene_for_shape = starts["S"] if mdata.has_scenes else 0
    _, (size_y, size_x) = spatial_regions[scene_for_shape]
    spatial_rois = {(scene if mdata.has_scenes else None): region[0] for scene, region in spatial_regions.items()}
    plane_shape = (size_y, size_x, 3) if contains_rgb else (size_y, size_x)
    sizes = (size_s, size_t, size_c, size_z)
    start_values = (starts["S"], starts["T"], starts["C"], starts["Z"])
    dtype = np.dtype((mdata.npdtype_list or [np.uint16])[0])

    if use_dask:
        array6d: Array6D = _build_lazy_array(
            filepath,
            sizes,
            start_values,
            plane_shape,
            dtype,
            mdata.has_scenes,
            spatial_rois,
            zoom,
            mdata.pyczi_readertype,
            squeeze_grayscale,
        )
    else:
        full_shape = sizes + plane_shape
        eager_array = np.empty(full_shape, dtype=dtype)
        plane_iterator: Any = itertools.product(
            enumerate(range(starts["S"], starts["S"] + size_s)),
            enumerate(range(starts["T"], starts["T"] + size_t)),
            enumerate(range(starts["C"], starts["C"] + size_c)),
            enumerate(range(starts["Z"], starts["Z"] + size_z)),
        )
        if HAS_PROGRESSBAR:
            widgets = [
                progressbar.Percentage(),
                " ",
                progressbar.Bar(),
                " ",
                progressbar.ETA(),
                " ",
                progressbar.SimpleProgress(),
            ]
            plane_iterator = progressbar.progressbar(
                plane_iterator,
                widgets=widgets,
                max_value=total_planes,
                term_width=80,
            )

        with pyczi.open_czi(filepath, mdata.pyczi_readertype) as czidoc:
            for s, t, c, z in plane_iterator:
                plane = {"T": t[1], "Z": z[1], "C": c[1]}
                scene = s[1] if mdata.has_scenes else None
                roi = spatial_rois[scene]
                if mdata.has_scenes:
                    image2d = czidoc.read(plane=plane, scene=s[1], roi=roi, zoom=zoom)
                else:
                    image2d = czidoc.read(plane=plane, roi=roi, zoom=zoom)
                if squeeze_grayscale:
                    image2d = image2d[..., 0]
                eager_array[s[0], t[0], c[0], z[0], ...] = image2d
        array6d = eager_array

    if contains_rgb and array6d.shape[-1] == 3:
        array6d = array6d[..., ::-1]

    if use_dask and chunk_zyx and isinstance(array6d, da.Array):
        chunks = (1, 1, 1, size_z, size_y, size_x)
        if contains_rgb:
            chunks += (3,)
        array6d = cast(Any, array6d).rechunk(chunks=chunks)

    mdata.array6d_size = array6d.shape

    if use_xarray:
        coords = {dim: range(array6d.shape[index]) for index, dim in enumerate(dims)}
        array6d = xr.DataArray(array6d, dims=dims, coords=coords)
        spatial_coords = {
            axis: np.arange(array6d.sizes[axis]) * _get_axis_coord_step(mdata.scale, axis, zoom) for axis in "ZYX"
        }
        array6d = array6d.assign_coords(
            C=_channel_names_or_default(mdata, array6d.sizes["C"]),
            **cast(Any, spatial_coords),
        )
        array6d.attrs = {
            "description": "6D image data from CZI file",
            "source": mdata.filepath,
            "axes": "".join(dims),
            "subset_planes": selected_planes,
        }

    if adapt_metadata:
        image.SizeS = selected_planes["S"][1] - selected_planes["S"][0] + 1
        image.SizeT = selected_planes["T"][1] - selected_planes["T"][0] + 1
        image.SizeC = selected_planes["C"][1] - selected_planes["C"][0] + 1
        image.SizeZ = selected_planes["Z"][1] - selected_planes["Z"][0] + 1

    return array6d, mdata
