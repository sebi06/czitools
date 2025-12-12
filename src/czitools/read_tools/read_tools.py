# -*- coding: utf-8 -*-

#################################################################
# File        : read_tools.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from typing import Dict, Tuple, Optional, Union, List
import gc
import warnings
import itertools
from pylibCZIrw import czi as pyczi
from aicspylibczi import CziFile
from czitools.metadata_tools import czi_metadata as czimd
from czitools.utils import misc
import numpy as np
from pathlib import Path
import dask.array as da
import dask
import dask.delayed
import os
from tqdm import tqdm
from tqdm.contrib.itertools import product
import tempfile
import shutil
from czitools.utils import logging_tools
from czitools.metadata_tools.helper import AttachmentType
import xarray as xr

# from memory_profiler import profile

logger = logging_tools.set_logging()

# Canonical dimension order for read_scenes
# Extra dimensions come first, then core dimensions
_EXTRA_DIMS = ["V", "R", "I", "H", "M"]  # optional extra dims (prepended if present)
_CORE_DIMS = ["T", "C", "Z"]  # always present (default size=1); S handled separately
# B is intentionally excluded - always assumed to be 1 and removed
# Plane dimensions for reading (excludes S which is handled via scene parameter)
_PLANE_DIMS_READ = ["T", "Z", "C", "V", "R", "I", "H", "M"]  # B excluded


# code for which memory has to be monitored
# instantiating the decorator
# @profile
def read_6darray(
    filepath: Union[str, os.PathLike[str]],
    use_dask: bool = False,
    chunk_zyx: bool = False,
    planes: Optional[Dict[str, Tuple[int, int]]] = None,
    zoom: float = 1.0,
    use_xarray: bool = True,
    adapt_metadata: bool = False,
) -> Tuple[Optional[Union[np.ndarray, da.Array, xr.DataArray]], czimd.CziMetadata]:
    """Read a CZI image file as 6D array.

    Important: Currently supported are only scenes with equal size and CZIs
    with consistent pixel types. The output array has always the dimension
    order: STCZYX (or STCZYXA for RGB images).

    Args:
        filepath: Path to the CZI image file.
        use_dask: If True, store image data as dask array. Note that this
                  still reads all data eagerly - for true lazy loading use
                  `read_6darray_lazy()` instead. Defaults to False.
        chunk_zyx: If True and use_dask=True, rechunk the dask array so each
                   chunk contains the full Z-stack. Defaults to False.
        planes: Dictionary specifying substack ranges. Allowed keys are S, T, Z, C.
                Values are tuples of (start, end) indices (zero-based, inclusive).
                Example: {"Z": (0, 2)} returns 3 z-planes with indices 0, 1, 2.
                Example: {"Z": (5, 5)} returns a single z-plane with index 5.
                Defaults to None (read all planes).
        zoom: Downscale factor for images [0.01 - 1.0]. Defaults to 1.0.
        use_xarray: If True, return an xr.DataArray with labeled dimensions.
                    If False, return a plain numpy or dask array. Defaults to True.
        adapt_metadata: If True, update metadata dimensions (SizeS, SizeT, SizeC, SizeZ)
                        to match the output array. Defaults to False.

    Returns:
        Tuple of (array6d, metadata) where array6d is a 6D numpy array, dask array,
        or xr.DataArray (depending on options), and metadata is CziMetadata.
        Returns (None, metadata) if the file cannot be read as a 6D array.
    """

    if isinstance(filepath, Path):
        # convert to string
        filepath = str(filepath)

    # check zoom factor for valid range
    zoom = misc.check_zoom(zoom=zoom)

    # define used dimensions
    dims = ("S", "T", "C", "Z", "Y", "X", "A")

    # get the complete metadata_tools at once as one big class
    mdata = czimd.CziMetadata(filepath)

    if not mdata.consistent_pixeltypes:
        logger.info("Detected PixelTypes are not consistent. Cannot create array6d")
        return None, mdata

    # update scaling
    mdata.scale.X_sf = np.round(mdata.scale.X * (1 / zoom), 3)
    mdata.scale.Y_sf = np.round(mdata.scale.Y * (1 / zoom), 3)
    mdata.scale.ratio["zx_sf"] = np.round(mdata.scale.Z / mdata.scale.X_sf, 3)

    # Normalize planes argument without mutating caller-provided dict.
    # Treat falsy/missing `planes` the same as before (i.e., fill defaults).
    planes_input = planes
    if planes_input:
        planes = dict(planes_input)
        # Validate provided ranges against total_bounding_box
        for k in ["S", "T", "C", "Z"]:
            if k in planes.keys() and k in mdata.bbox.total_bounding_box.keys():
                if mdata.bbox.total_bounding_box[k][1] - 1 < planes[k][1]:
                    logger.info(
                        f"Planes indices (zero-based) for {planes[k]} are invalid. BBox for {[k]}: {mdata.bbox.total_bounding_box[k]}"
                    )
                    return None, mdata
    else:
        planes = {}
        for dim, size_attr in [
            ("S", mdata.image.SizeS),
            ("T", mdata.image.SizeT),
            ("C", mdata.image.SizeC),
            ("Z", mdata.image.SizeZ),
        ]:
            planes[dim] = (0, size_attr - 1) if size_attr is not None else (0, 0)

    # Ensure all expected dims exist in the local `planes` copy
    for k in ["S", "T", "C", "Z"]:
        if k not in planes.keys():
            if k == "S":
                planes[k] = (0, mdata.image.SizeS - 1) if mdata.image.SizeS is not None else (0, 0)
            elif k == "T":
                planes[k] = (0, mdata.image.SizeT - 1) if mdata.image.SizeT is not None else (0, 0)
            elif k == "C":
                planes[k] = (0, mdata.image.SizeC - 1) if mdata.image.SizeC is not None else (0, 0)
            elif k == "Z":
                planes[k] = (0, mdata.image.SizeZ - 1) if mdata.image.SizeZ is not None else (0, 0)

    if mdata.consistent_pixeltypes:
        # use pixel type from first channel
        use_pixeltype = mdata.npdtype_list[0]

    # Check if scene shapes are consistent across the CZI file
    if not mdata.scene_shape_is_consistent:
        # By default, assume scenes are not valid to read due to inconsistent shapes
        scenes_valid_to_read = False

        # Check if scene dimension ("S") is specified in the planes parameter
        if planes is not None and "S" in planes.keys():
            # Check if only a single scene is selected (start and end indices are the same)
            # This allows reading even when scenes have inconsistent shapes, as long as
            # we're only reading one scene at a time
            if planes["S"][1] - planes["S"][0] == 0:
                # Single scene selection is valid even with inconsistent scene shapes
                scenes_valid_to_read = True

        if not scenes_valid_to_read:
            logger.warning("Scenes have inconsistent shape. Cannot read 6D array")
            return None, mdata

    # open the CZI document to read the
    with pyczi.open_czi(filepath, mdata.pyczi_readertype) as czidoc:

        # check if dimensions are None (because they do not exist for that image)
        size_c = misc.check_dimsize(mdata.image.SizeC, set2value=1)
        size_z = misc.check_dimsize(mdata.image.SizeZ, set2value=1)
        size_t = misc.check_dimsize(mdata.image.SizeT, set2value=1)
        size_s = misc.check_dimsize(mdata.image.SizeS, set2value=1)

        s_start = 0
        s_end = size_s
        t_start = 0
        t_end = size_t
        c_start = 0
        c_end = size_c
        z_start = 0
        z_end = size_z

        # check for additional arguments to create substacks
        if mdata.image.SizeS is not None and "S" in planes.keys():
            size_s = planes["S"][1] - planes["S"][0] + 1
            mdata.image.SizeS = size_s
            s_start = planes["S"][0]
            s_end = planes["S"][1] + 1

        if mdata.image.SizeT is not None and "T" in planes.keys():
            size_t = planes["T"][1] - planes["T"][0] + 1
            mdata.image.SizeT = size_t
            t_start = planes["T"][0]
            t_end = planes["T"][1] + 1

        if mdata.image.SizeZ is not None and "Z" in planes.keys():
            size_z = planes["Z"][1] - planes["Z"][0] + 1
            mdata.image.SizeZ = size_z
            z_start = planes["Z"][0]
            z_end = planes["Z"][1] + 1

        if mdata.image.SizeC is not None and "C" in planes.keys():
            size_c = planes["C"][1] - planes["C"][0] + 1
            mdata.image.SizeC = size_c
            c_start = planes["C"][0]
            c_end = planes["C"][1] + 1

        # Check if A dimension can be removed (grayscale vs RGB)
        contains_rgb = any(mdata.isRGB.values())
        remove_adim = not contains_rgb

        if mdata.is_url:
            logger.info("Reading pixel data via network from link location.")

        total_planes = size_s * size_t * size_c * size_z
        logger.info(f"read_6darray: Reading {total_planes} planes (S={size_s}, T={size_t}, C={size_c}, Z={size_z})")

        planecount = 0

        # read array for the scene 2Dplane-by-2Dplane
        for s, t, c, z in product(
            enumerate(range(s_start, s_end)),
            enumerate(range(t_start, t_end)),
            enumerate(range(c_start, c_end)),
            enumerate(range(z_start, z_end)),
            desc="Reading 2D planes",
            unit=" 2Dplanes",
        ):
            planecount += 1

            # read a 2D image plane from the CZI
            if mdata.has_scenes:
                image2d = czidoc.read(plane={"T": t[1], "Z": z[1], "C": c[1]}, scene=s[1], zoom=zoom)
            else:
                image2d = czidoc.read(plane={"T": t[1], "Z": z[1], "C": c[1]}, zoom=zoom)

            if planecount == 1:
                # Allocate array based on the actual 2D plane size (accounts for zoom)
                size_y, size_x = image2d.shape[0], image2d.shape[1]
                array_shape = (
                    size_s,
                    size_t,
                    size_c,
                    size_z,
                    size_y,
                    size_x,
                    3 if contains_rgb else 1,
                )

                if use_dask:
                    array6d = da.empty(array_shape, dtype=use_pixeltype, chunks=array_shape)
                else:
                    array6d = np.empty(array_shape, dtype=use_pixeltype)

            # insert 2D image plane into the array6d
            array6d[s[0], t[0], c[0], z[0], ...] = image2d

        # Remove the A dimension for grayscale images
        if remove_adim:
            array6d = np.squeeze(array6d, axis=-1)
            dims = ("S", "T", "C", "Z", "Y", "X")

        # Rechunk if requested (only applies to dask arrays)
        if use_dask and chunk_zyx:
            if remove_adim:
                array6d = array6d.rechunk(chunks=(1, 1, 1, size_z, size_y, size_x))
            else:
                array6d = array6d.rechunk(chunks=(1, 1, 1, size_z, size_y, size_x, 3))

    # update metadata_tools
    mdata.array6d_size = array6d.shape

    if use_xarray:

        # create xarray from array6d
        coords = {}
        for index, dim in enumerate(dims):
            # Create a range for each dimension
            coords[dim] = range(array6d.shape[index])

        # Create the xarray.DataArray
        array6d = xr.DataArray(array6d, dims=dims, coords=coords)

        # Set attributes for the DataArray
        array6d.attrs = {
            "description": "6D image data from CZI file",
            "source": mdata.filepath,
            "axes": "".join(dims),
            "subset_planes": planes,
            # "metadata": mdata,  # Include metadata if it's a dictionary or serializable
        }

    # adapt metadata for STCZ
    if adapt_metadata:

        mdata.image.SizeS = planes["S"][1] - planes["S"][0] + 1 if "S" in planes else mdata.image.SizeS
        mdata.image.SizeT = planes["T"][1] - planes["T"][0] + 1 if "T" in planes else mdata.image.SizeT
        mdata.image.SizeC = planes["C"][1] - planes["C"][0] + 1 if "C" in planes else mdata.image.SizeC
        mdata.image.SizeZ = planes["Z"][1] - planes["Z"][0] + 1 if "Z" in planes else mdata.image.SizeZ

    return array6d, mdata


# code for which memory has to be monitored
# instantiating the decorator
# @profile
def read_6darray_lazy(
    filepath: Union[str, os.PathLike[str]],
    chunk_zyx: bool = False,
    planes: Optional[Dict[str, Tuple[int, int]]] = None,
    zoom: float = 1.0,
    use_xarray: bool = False,
) -> Tuple[Optional[Union[da.Array, xr.DataArray]], czimd.CziMetadata]:
    """Read a CZI image file as 6D dask array with delayed plane reading.

    Important: Currently supported are only scenes with equal size and CZIs
    with consistent pixel types. The output array has always the dimension
    order: STCZYX(A).

    Note: This function creates lazy dask arrays. The actual pixel data is only
    read when the array is computed (e.g., via .compute() or indexing).

    Args:
        filepath: Path to the CZI image file.
        chunk_zyx: If True, rechunk the dask array so each chunk contains
                   the full Z-stack for a single S, T, C position. Defaults to False.
        planes: Dictionary specifying substack ranges. Allowed keys are S, T, Z, C.
                Values are tuples of (start, end) indices (zero-based, inclusive).
                Example: {"Z": (0, 2)} returns 3 z-planes with indices 0, 1, 2.
                Example: {"Z": (5, 5)} returns a single z-plane with index 5.
                Defaults to None (read all planes).
        zoom: Downscale factor for images [0.01 - 1.0]. Defaults to 1.0.
        use_xarray: If True, return an xr.DataArray with labeled dimensions.
                    If False, return a plain dask array. Defaults to False.

    Returns:
        Tuple of (array6d, metadata) where array6d is a 6D dask array or
        xr.DataArray (if use_xarray=True), and metadata is CziMetadata.
        Returns (None, metadata) if the file cannot be read as a 6D array.
    """

    if isinstance(filepath, Path):
        # convert to string
        filepath = str(filepath)

    # get the complete metadata_tools at once as one big class
    mdata = czimd.CziMetadata(filepath)

    if not mdata.consistent_pixeltypes:
        logger.info("Detected PixelTypes are not consistent. Cannot create array6d")
        return None, mdata

    # check zoom factor for valid range
    zoom = misc.check_zoom(zoom=zoom)

    # update scaling
    mdata.scale.X_sf = np.round(mdata.scale.X * (1 / zoom), 3)
    mdata.scale.Y_sf = np.round(mdata.scale.Y * (1 / zoom), 3)
    mdata.scale.ratio["zx_sf"] = np.round(mdata.scale.Z / mdata.scale.X_sf, 3)

    # check planes
    if planes:
        for k in ["S", "T", "C", "Z"]:
            if k in planes.keys() and k in mdata.bbox.total_bounding_box.keys():
                if mdata.bbox.total_bounding_box[k][1] - 1 < planes[k][1]:
                    logger.info(
                        f"Planes indices (zero-based) for {planes[k]} are invalid. BBox for {[k]}: {mdata.bbox.total_bounding_box[k]}"
                    )
                    return None, mdata

    if not mdata.scene_shape_is_consistent and "S" not in (planes or {}):
        logger.info("Scenes have inconsistent shape. Cannot read 6D array")
        return None, mdata

    # open the CZI document to read
    with pyczi.open_czi(filepath, mdata.pyczi_readertype) as czidoc:
        if mdata.image.SizeS is not None:
            # use the size of the 1st scenes_bounding_rectangle
            size_x = czidoc.scenes_bounding_rectangle[0].w
            size_y = czidoc.scenes_bounding_rectangle[0].h
        else:
            # use the size of the total_bounding_rectangle
            size_x = czidoc.total_bounding_rectangle.w
            size_y = czidoc.total_bounding_rectangle.h

        # check if dimensions are None (because they do not exist for that image)
        size_c = misc.check_dimsize(mdata.image.SizeC, set2value=1)
        size_z = misc.check_dimsize(mdata.image.SizeZ, set2value=1)
        size_t = misc.check_dimsize(mdata.image.SizeT, set2value=1)
        size_s = misc.check_dimsize(mdata.image.SizeS, set2value=1)

        s_start, s_end = 0, size_s
        t_start, t_end = 0, size_t
        c_start, c_end = 0, size_c
        z_start, z_end = 0, size_z

        # check for additional arguments to create substacks
        if planes and mdata.image.SizeS is not None and "S" in planes.keys():
            size_s = planes["S"][1] - planes["S"][0] + 1
            mdata.image.SizeS = size_s
            s_start = planes["S"][0]
            s_end = planes["S"][1] + 1

        if planes and mdata.image.SizeT is not None and "T" in planes.keys():
            size_t = planes["T"][1] - planes["T"][0] + 1
            mdata.image.SizeT = size_t
            t_start = planes["T"][0]
            t_end = planes["T"][1] + 1

        if planes and mdata.image.SizeZ is not None and "Z" in planes.keys():
            size_z = planes["Z"][1] - planes["Z"][0] + 1
            mdata.image.SizeZ = size_z
            z_start = planes["Z"][0]
            z_end = planes["Z"][1] + 1

        if planes and mdata.image.SizeC is not None and "C" in planes.keys():
            size_c = planes["C"][1] - planes["C"][0] + 1
            mdata.image.SizeC = size_c
            c_start = planes["C"][0]
            c_end = planes["C"][1] + 1

        # check if the A Dimension can be removed because image is grayscale
        contains_rgb = any(mdata.isRGB.values())
        remove_adim = not contains_rgb

        # Determine shape of 2D planes
        # A dimension is either 1 (grayscale) or 3 (RGB)
        if contains_rgb:
            shape2d = (size_y, size_x, 3)
            num_components = 3
        else:
            shape2d = (size_y, size_x)
            num_components = 1

        # Build dask array structure with delayed plane reads
        # Note: No actual data is read here - only the task graph is constructed
        total_planes = size_s * size_t * size_c * size_z
        logger.info(f"read_6darray_lazy: Building dask graph for {total_planes} planes")
        if mdata.is_url:
            logger.info("read_6darray_lazy: Reading from URL - using Curl reader")

        img = []
        for s in range(s_start, s_end):
            time_stack = []

            for t in range(t_start, t_end):
                ch_stack = []

                for c in range(c_start, c_end):
                    z_stack = []

                    for z in range(z_start, z_end):
                        # Use filepath-based delayed read (file opened inside delayed fn)
                        z_slice = da.from_delayed(
                            _read_2dplane_delayed(
                                filepath,
                                s=s,
                                t=t,
                                c=c,
                                z=z,
                                has_scenes=mdata.has_scenes,
                                zoom=zoom,
                                remove_adim=remove_adim,
                                readertype=mdata.pyczi_readertype,
                            ),
                            shape=shape2d,
                            dtype=mdata.npdtype_list[0],
                        )
                        z_stack.append(z_slice)

                    # Stack z-planes into a Z dimension
                    z_stack = da.stack(z_stack, axis=0)
                    ch_stack.append(z_stack)

                time_stack.append(ch_stack)

            img.append(time_stack)

        # Create final STCZYX dask array
        array6d = da.stack(img, axis=0)

        # Rechunk if requested
        if chunk_zyx:
            if remove_adim:
                array6d = array6d.rechunk(chunks=(1, 1, 1, size_z, size_y, size_x))
            else:
                array6d = array6d.rechunk(chunks=(1, 1, 1, size_z, size_y, size_x, num_components))

    # Update metadata with array shape
    mdata.array6d_size = array6d.shape

    # Convert to xarray if requested
    if use_xarray:
        dims = ("S", "T", "C", "Z", "Y", "X") if remove_adim else ("S", "T", "C", "Z", "Y", "X", "A")
        coords = {
            "S": range(size_s),
            "T": range(size_t),
            "C": range(size_c),
            "Z": range(size_z),
            "Y": range(size_y),
            "X": range(size_x),
        }
        if not remove_adim:
            coords["A"] = range(num_components)

        array6d = xr.DataArray(
            array6d,
            dims=dims,
            coords=coords,
            attrs={
                "description": "6D image data from CZI file (lazy dask array)",
                "source": mdata.filepath,
                "axes": "".join(dims),
                "subset_planes": planes,
            },
        )

    return array6d, mdata


@dask.delayed
def _read_2dplane_delayed(
    filepath: str,
    s: int = 0,
    t: int = 0,
    c: int = 0,
    z: int = 0,
    has_scenes: bool = True,
    zoom: float = 1.0,
    remove_adim: bool = True,
    readertype: pyczi.ReaderFileInputTypes = pyczi.ReaderFileInputTypes.Standard,
) -> np.ndarray:
    """Dask delayed function to read a 2D plane from a CZI image.

    This function opens the CZI file, reads a single plane, and closes the file.
    It is designed to be called lazily by dask when the data is actually needed.
    Supports both local files and URLs.

    Args:
        filepath: Path to the CZI image file (local path or URL).
        s: Scene index. Defaults to 0.
        t: Time index. Defaults to 0.
        c: Channel index. Defaults to 0.
        z: Z-Plane index. Defaults to 0.
        has_scenes: If True, use scene parameter when reading. Defaults to True.
        zoom: Downscale factor [0.01 - 1.0]. Defaults to 1.0.
        remove_adim: If True, squeeze the trailing A dimension for grayscale images.
                     Defaults to True.
        readertype: The pylibCZIrw reader type (Standard for local files, Curl for URLs).

    Returns:
        2D numpy array with shape (Y, X) or (Y, X, A) depending on remove_adim.
    """
    with pyczi.open_czi(filepath, readertype) as czidoc:
        if has_scenes:
            image2d = czidoc.read(plane={"T": t, "C": c, "Z": z}, scene=s, zoom=zoom)
        else:
            image2d = czidoc.read(plane={"T": t, "C": c, "Z": z}, zoom=zoom)

    # Remove trailing A dimension for grayscale images
    if remove_adim:
        return image2d[..., 0]
    return image2d


# Keep old function name for backward compatibility (deprecated)
@dask.delayed
def read_2dplane(
    czidoc: pyczi.CziReader,
    s: int = 0,
    t: int = 0,
    c: int = 0,
    z: int = 0,
    has_scenes: bool = True,
    zoom: float = 1.0,
    remove_adim: bool = True,
) -> np.ndarray:
    """Dask delayed function to read a 2D plane from a CZI image.

    .. deprecated::
        This function is deprecated because it requires an open file handle
        which may be closed before the delayed function executes.
        Use `_read_2dplane_delayed` with a filepath instead.

    Args:
        czidoc: CziReader object (must remain open until compute is called).
        s: Scene index. Defaults to 0.
        t: Time index. Defaults to 0.
        c: Channel index. Defaults to 0.
        z: Z-Plane index. Defaults to 0.
        has_scenes: If True, use scene parameter when reading. Defaults to True.
        zoom: Downscale factor [0.01 - 1.0]. Defaults to 1.0.
        remove_adim: If True, squeeze the trailing A dimension. Defaults to True.

    Returns:
        2D numpy array with shape (Y, X) or (Y, X, A).
    """
    warnings.warn(
        "read_2dplane is deprecated. Use _read_2dplane_delayed with filepath instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if has_scenes:
        image2d = czidoc.read(plane={"T": t, "C": c, "Z": z}, scene=s, zoom=zoom)
    else:
        image2d = czidoc.read(plane={"T": t, "C": c, "Z": z}, zoom=zoom)

    if remove_adim:
        return image2d[..., 0]
    return image2d


def read_attachments(
    czi_filepath: Union[str, os.PathLike],
    attachment_type: AttachmentType = AttachmentType.SlidePreview,
    copy: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Read attachment images from a CZI image as numpy array

    Args:
        czi_filepath (str): FilePath of the CZI image file
        attachment_type (str, optional): Type of the attachment to be read. Defaults to "SlidePreview".
        copy (bool, optional): Option to copy the attachments as CZI image directly. Defaults to True.

    Raises:
        Exception: If specified attachment is not found and exception is raised.

    Returns:
        Tuple[nd,array, [Optional[str]]: Tuple containing the 2d image array and optionally the location of the copied image.
    """

    try:
        import czifile

        if attachment_type not in AttachmentType:
            # if attachment_type not in ["SlidePreview", "Label", "]:
            raise Exception(f"{attachment_type} is not supported. Valid types are: SlidePreview, Label or Prescan.")

        att = czimd.CziAttachments(czi_filepath)

        if attachment_type == AttachmentType.Label and not att.has_label:
            # if attachment_type == "Label" and not att.has_label:
            return np.array([]), None

        if attachment_type == AttachmentType.SlidePreview and not att.has_preview:
            # if attachment_type == "SlidePreview" and not att.has_preview:
            return np.array([]), None

        if attachment_type == AttachmentType.Prescan and not att.has_prescan:
            # if attachment_type == "SlidePreview" and not att.has_preview:
            return np.array([]), None

        # create CZI-object using czifile library
        with czifile.CziFile(czi_filepath) as cz:
            with tempfile.TemporaryDirectory() as tmpdirname:
                # save attachments to temporary directory
                cz.save_attachments(directory=tmpdirname)

                # iterate over attachments
                for att in cz.attachments():
                    if att.attachment_entry.name == attachment_type.name:
                        # get the full path of the attachment image
                        full_path = Path(tmpdirname) / att.attachment_entry.filename

                        if copy:
                            # create path to store the attachment image
                            att_path = str(czi_filepath)[:-4] + "_" + att.attachment_entry.name + ".czi"

                            # copy the file
                            dest = shutil.copyfile(full_path, att_path)

                        # open the CZI document to read array
                        with pyczi.open_czi(str(full_path)) as czidoc:
                            img2d = czidoc.read()

                        if copy:
                            return img2d, dest
                        if not copy:
                            return img2d

    except ImportError:  # as e:
        logger.warning("Package czifile not found. Cannot extract information about attached images.")

        return None, None


def read_tiles(filepath: Union[str, os.PathLike[str]], scene: int, tile: int, **kwargs) -> Tuple[np.ndarray, List]:
    """
    Reads a specific tile from a CZI file.
    Parameters:
    -----------
    filepath : Union[str, os.PathLike[str]]
        Path to the CZI file.
    scene : int
        The scene index to read from the CZI file.
    tile : int
        The tile index to read from the CZI file.
    **kwargs : dict
        Additional keyword arguments to specify substacks. Valid arguments are:
        - 'T': Time dimension
        - 'Z': Z-dimension (depth)
        - 'C': Channel dimension
    Returns:
    --------
    Tuple[np.ndarray, List]
        A tuple containing:
        - tile_stack (np.ndarray): The image data of the specified tile.
        - size (List): A list of tuples representing the dimensions and their sizes.
    Raises:
    -------
    ValueError
        If an invalid keyword argument is provided in **kwargs.
    Notes:
    ------
    This function uses the `aicspylibczi` library to read CZI files. If the CZI file
    is a mosaic, the M-dimension is handled accordingly.
    """

    if isinstance(filepath, Path):
        # convert to string
        filepath = str(filepath)

    valid_args = ["T", "Z", "C"]

    # check for invalid arguments to specify substacks
    for k, v in kwargs.items():
        if k not in valid_args:
            raise ValueError(f"Invalid keyword argument: {k}")

    # read CZI using aicspylibczi: : https://pypi.org/project/aicspylibczi/
    # Suppress ResourceWarning as we explicitly clean up the CziFile object in finally block
    # aicspylibczi.CziFile doesn't provide a close() method, so Python may warn about unclosed file
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ResourceWarning, message=".*CziFile.*")
        warnings.filterwarnings("ignore", category=ResourceWarning, message=".*BufferedReader.*")

        czi = CziFile(filepath)

        try:
            # show the values
            logger.info(f"Reading File: {filepath} Scene: {scene} - Tile {tile}")
            logger.info(f"Dimensions Shape: {czi.get_dims_shape()}")

            # in case the CZI is a mosaic file and has the M-dimension
            if czi.is_mosaic():
                tile_stack, size = czi.read_image(S=scene, M=tile, **kwargs)

                # remove the M-Dimension from the array and size
                tile_stack = np.squeeze(tile_stack, axis=czi.dims.find("M"))
                size.remove(("M", 1))

            # in case the CZI is not a mosaic file and has no M-dimension
            elif not czi.is_mosaic():
                logger.warning("CZI file is not a mosaic. No M-Dimension found.")
                tile_stack, size = czi.read_image(S=scene, **kwargs)

            return tile_stack, size
        finally:
            # Explicitly delete the CziFile object to close underlying file handle
            # aicspylibczi.CziFile doesn't provide a close() method, so we rely on deletion
            # and explicit garbage collection to ensure the file handle is released
            del czi
            gc.collect()


def _read_plane_delayed(
    filepath: str,
    plane: Dict[str, int],
    scene_idx: int,
    squeeze_grayscale: bool,
    readertype: pyczi.ReaderFileInputTypes = pyczi.ReaderFileInputTypes.Standard,
) -> np.ndarray:
    """Delayed function to read a single 2D plane from a CZI file.

    This function is called lazily by dask when the data is actually needed.
    Supports both local files and URLs.

    Args:
        filepath: Path to the CZI file (local path or URL).
        plane: Dictionary mapping dimension names to coordinate values.
        scene_idx: Index of the scene to read.
        squeeze_grayscale: If True, squeeze the trailing dimension for grayscale images.
        readertype: The pylibCZIrw reader type (Standard for local files, Curl for URLs).

    Returns:
        2D numpy array of the plane data.
    """
    with pyczi.open_czi(filepath, readertype) as czidoc:
        img2d = czidoc.read(plane=plane, scene=scene_idx)
        if squeeze_grayscale:
            img2d = img2d[..., 0]
    return img2d


def read_scenes(
    filepath: Union[str, os.PathLike[str]],
    use_dask: bool = False,
    use_xarray: bool = True,
    stack_scenes: bool = False,
) -> Tuple[
    Union[List[xr.DataArray], List[np.ndarray], xr.DataArray, np.ndarray, List[da.Array], da.Array], List[str], int
]:
    """Read all 2D planes from a CZI file, grouped by scene.

    This function reads pixel data based on the total_bounding_box (derived from
    CZI subblocks, not XML metadata). It supports all CZI dimensions and returns
    arrays with a canonical dimension order.

    Dimension order is always: [V, R, I, H, M] + T + C + Z + Y + X [+ A]
      - Extra dims (V, R, I, H, M) appear first if present in the file
      - Core dims (T, C, Z) are always present (size=1 if missing in file)
      - B dimension is always removed (assumed size=1)
      - S (scene) is tracked separately; each scene is a separate array in the list
      - Spatial (Y, X) and optional pixel-type (A for RGB, size=3) are last

    Args:
        filepath: Path to the CZI file.
        use_dask: If True, return lazy dask arrays. Data is only read when
                  actually accessed (e.g., via .compute() or indexing).
                  Defaults to False.
        use_xarray: If True, return xr.DataArray with labeled dimensions.
                    If False, return plain np.ndarray (or dask.array if use_dask=True).
                    Defaults to True.
        stack_scenes: If True and all scenes have the same shape, stack them
                      into a single array with S as the first dimension.
                      If shapes differ, returns a list (with a warning).
                      Defaults to False.

    Returns:
        Tuple of:
            - If stack_scenes=False: List of arrays (one per scene).
            - If stack_scenes=True and shapes match: Single array with S dim.
            - If stack_scenes=True but shapes differ: List (with warning).
            Each array has shape ([V, R, I, H, M,] T, C, Z, Y, X [, A]).
            Missing core dims (T, C, Z) get size=1.
            - List of dimension labels in canonical order.
            - Number of scenes.

    Example:
        >>> from czitools.read_tools import read_tools
        >>> # Lazy loading with xarray
        >>> arrays, dims, num_scenes = read_tools.read_scenes(
        ...     "path/to/file.czi", use_dask=True, use_xarray=True
        ... )
        >>> # Stack scenes if they have the same shape
        >>> stacked, dims, num_scenes = read_tools.read_scenes(
        ...     "path/to/file.czi", stack_scenes=True
        ... )
    """
    if isinstance(filepath, Path):
        filepath = str(filepath)

    # Determine reader type for URL or local file support
    readertype, is_url = misc.get_pyczi_readertype(filepath)
    if is_url:
        logger.info("read_scenes: Reading from URL - using Curl reader")

    scene_arrays: List[Union[xr.DataArray, np.ndarray, da.Array]] = []
    scene_shapes: List[Tuple[int, ...]] = []

    with pyczi.open_czi(filepath, readertype) as czidoc:
        total_bbox = czidoc.total_bounding_box
        num_scenes = len(czidoc.scenes_bounding_rectangle)
        logger.info(f"read_scenes: num_scenes={num_scenes}, total_bounding_box={total_bbox}")

        # Build dimension info from total_bounding_box
        # dim_from_bbox: {dim_name: (start, size)}
        dim_from_bbox: Dict[str, Tuple[int, int]] = {}
        for dim in _PLANE_DIMS_READ:
            if dim in total_bbox:
                dim_from_bbox[dim] = total_bbox[dim]

        # Build canonical dimension order:
        # 1. Extra dims that are present (in _EXTRA_DIMS order)
        # 2. Core dims (always present, default size=1)
        canonical_dims: List[str] = []
        dim_sizes_map: Dict[str, int] = {}
        dim_starts_map: Dict[str, int] = {}

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

        # Track S separately
        dim_sizes_map["S"] = num_scenes
        dim_starts_map["S"] = 0

        # Dims for reading planes (excludes S which is the scene loop)
        read_dims = canonical_dims  # already excludes S
        read_sizes = [dim_sizes_map[d] for d in read_dims]
        read_starts = [dim_starts_map[d] for d in read_dims]

        logger.info(f"read_scenes: canonical_dims={canonical_dims}, dim_sizes={dim_sizes_map}")
        if use_dask:
            logger.info("read_scenes: Using lazy dask arrays - data will be read on demand")

        all_dims: List[str] = []  # will be set in loop

        for scene_idx in range(num_scenes):
            scene_rect = czidoc.scenes_bounding_rectangle[scene_idx]
            size_y, size_x = scene_rect.h, scene_rect.w
            logger.debug(f"read_scenes: Scene {scene_idx}: Y={size_y}, X={size_x}")

            # Sample read to get dtype and actual spatial shape
            sample_plane = {name: start for name, start in zip(read_dims, read_starts)}
            sample = czidoc.read(plane=sample_plane, scene=scene_idx)
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

            scene_shapes.append(array_shape)

            if use_dask:
                # Build lazy dask array from delayed reads
                total_planes = int(np.prod(read_sizes)) if read_sizes else 1

                # Build nested list structure for da.block
                def build_dask_block(dims_remaining, current_indices):
                    """Recursively build the nested list structure for dask.block."""
                    if not dims_remaining:
                        # Base case: create a delayed array for this plane
                        plane = {name: start + idx for name, start, idx in zip(read_dims, read_starts, current_indices)}
                        delayed_read = dask.delayed(_read_plane_delayed)(
                            filepath, plane, scene_idx, squeeze_grayscale, readertype
                        )
                        return da.from_delayed(delayed_read, shape=plane_shape, dtype=dtype)
                    else:
                        # Recursive case: build list for this dimension
                        dim_size = dims_remaining[0]
                        return [build_dask_block(dims_remaining[1:], current_indices + [i]) for i in range(dim_size)]

                if read_sizes:
                    nested_blocks = build_dask_block(read_sizes, [])
                    stack = da.block(nested_blocks)
                    # da.block can collapse singleton dimensions, so reshape to ensure correct shape
                    if stack.shape != array_shape:
                        stack = stack.reshape(array_shape)
                else:
                    # No read dimensions, just one plane
                    plane = {name: start for name, start in zip(read_dims, read_starts)}
                    delayed_read = dask.delayed(_read_plane_delayed)(
                        filepath, plane, scene_idx, squeeze_grayscale, readertype
                    )
                    stack = da.from_delayed(delayed_read, shape=plane_shape, dtype=dtype)

                logger.debug(f"read_scenes: Scene {scene_idx} -> dask.array shape={stack.shape}, chunks={stack.chunks}")

            else:
                # Eager loading - read all planes immediately
                stack = np.empty(array_shape, dtype=dtype)

                # Build all index combinations for the read dimensions
                ranges = [range(s) for s in read_sizes]
                total_planes = int(np.prod(read_sizes)) if read_sizes else 1

                for idx, combo in enumerate(itertools.product(*ranges)):
                    # Build plane dict with actual coordinate values
                    plane = {name: start + offset for name, start, offset in zip(read_dims, read_starts, combo)}
                    img2d = czidoc.read(plane=plane, scene=scene_idx)

                    # Squeeze grayscale (A=1) but keep RGB (A=3) / RGBA (A=4)
                    if squeeze_grayscale:
                        img2d = img2d[..., 0]

                    # Store in the correct position
                    stack[combo] = img2d

                logger.debug(f"read_scenes: Scene {scene_idx} -> np.ndarray shape={stack.shape}")

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
                    attrs={"scene": scene_idx, "filepath": filepath},
                )
                scene_arrays.append(xr_da)
            else:
                scene_arrays.append(stack)

    # Optionally stack scenes if requested and all shapes match
    if stack_scenes:
        unique_shapes = set(scene_shapes)
        if len(unique_shapes) == 1:
            logger.info(f"read_scenes: Stacking {num_scenes} scenes (all shapes equal: {scene_shapes[0]})")
            stacked_dims = ["S"] + all_dims

            if use_xarray:
                # Stack xr.DataArrays along new S dimension
                stacked = xr.concat(scene_arrays, dim="S")
                stacked = stacked.assign_coords(S=np.arange(num_scenes))
                stacked.attrs["filepath"] = filepath
                return stacked, stacked_dims, num_scenes
            else:
                # Stack arrays (numpy or dask)
                if use_dask:
                    stacked = da.stack(scene_arrays, axis=0)
                else:
                    stacked = np.stack(scene_arrays, axis=0)
                return stacked, stacked_dims, num_scenes
        else:
            logger.warning(f"read_scenes: Cannot stack scenes - shapes differ: {unique_shapes}")

    return scene_arrays, all_dims, num_scenes
