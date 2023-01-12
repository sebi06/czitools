# -*- coding: utf-8 -*-

#################################################################
# File        : pylibczirw_tools.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Type, Any, Union, Mapping
from pylibCZIrw import czi as pyczi
from czitools import pylibczirw_metadata as czimd
from czitools import misc
import numpy as np
from pathlib import Path, PurePath, PureWindowsPath, PurePosixPath
from tqdm.contrib.itertools import product
import dask
import dask.array as da
import os


def read_6darray(filepath: Union[str, os.PathLike[str]],
                 output_order: str = "STCZYX",
                 output_dask: bool = False,
                 chunks_auto: bool = False,
                 remove_adim: bool = True,
                 **kwargs: int) -> Tuple[Optional[Union[np.ndarray, da.Array]], czimd.CziMetadata, str]:
    """Read a CZI image file as 6D numpy or dask array.
    Important: Currently supported are only scenes with equal size and CZIs with consistent pixel types.

    Args:
        filepath (str | Path): filepath for the CZI image
        output_order (str, optional): Order of dimensions for the output array. Defaults to "STCZYX".
        output_dask (bool, optional): If True the output will be a dask array. Defaults to False.
        chunks_auto (bool, optional): Use a chunk size create automatically if True and otherwise use the array shape. Default to False.
        remove_adim (bool, optional): If true the dimension for the pixel type will be removed. Defaults to True.
        kwargs (int, optional): Allowed kwargs are S, T, Z, C and values must be >=0 (zero-based).
                                Will be used to read only a substack from the array

    Returns:
        Tuple[array6d, mdata, dim_string]: output as 6D numpy or dask array, metadata and dimstring
    """

    if isinstance(filepath, Path):
        # convert to string
        filepath = str(filepath)

    # get the complete metadata at once as one big class
    mdata = czimd.CziMetadata(filepath)

    if not mdata.consistent_pixeltypes:
        print("Detected PixelTypes ar not consistent. Cannot create array6d")
        return None, mdata, ""

    if mdata.consistent_pixeltypes:
        # use pixel type from first channel
        use_pixeltype = mdata.npdtype[0]

    valid_order = ["STCZYX", "STZCYX"]

    if output_order not in valid_order:
        print("Invalid dimension order 6D:", output_order)
        return None, mdata, ""

    if not mdata.scene_shape_is_consistent:
        print("Scenes have inconsistent shape. Cannot read 6D array")
        return None, mdata, ""

    # open the CZI document to read the
    with pyczi.open_czi(filepath) as czidoc:

        if mdata.image.SizeS is not None:
            # get size for a single scene using the 1st
            # works only if scene shape is consistent

            # TODO Open question whether to use XML metadata or bounding boxes

            # use the size of the 1st scenes_bounding_rectangle
            size_x = czidoc.scenes_bounding_rectangle[0].w
            size_y = czidoc.scenes_bounding_rectangle[0].h

        if mdata.image.SizeS is None:
            # use the size of the total_bounding_rectangle
            size_x = czidoc.total_bounding_rectangle.w
            size_y = czidoc.total_bounding_rectangle.h

        # check if dimensions are None (because they do not exist for that image)
        size_c = misc.check_dimsize(mdata.image.SizeC, set2value=1)
        size_z = misc.check_dimsize(mdata.image.SizeZ, set2value=1)
        size_t = misc.check_dimsize(mdata.image.SizeT, set2value=1)
        size_s = misc.check_dimsize(mdata.image.SizeS, set2value=1)

        # check for additional **kwargs to create substacks
        if kwargs is not None and mdata.image.SizeS is not None and "S" in kwargs:
            size_s = kwargs["S"] + 1
            mdata.image.SizeS = 1

        if kwargs is not None and "T" in kwargs:
            size_t = kwargs["T"] + 1
            mdata.image.SizeT = 1

        if kwargs is not None and "Z" in kwargs:
            size_z = kwargs["Z"] + 1
            mdata.image.SizeZ = 1

        if kwargs is not None and "C" in kwargs:
            size_c = kwargs["C"] + 1
            mdata.image.SizeC = 1

        # assume default dimorder = STZCYX(A)
        shape = [size_s, size_t, size_c, size_z, size_y, size_x, 3 if mdata.isRGB else 1]

        if not output_dask:
            array6d = np.empty(shape, dtype=use_pixeltype)
        if output_dask:
            if chunks_auto:
                array6d = da.empty(shape, dtype=use_pixeltype, chunks="auto")
            if not chunks_auto:
                array6d = da.empty(shape, dtype=use_pixeltype, chunks=shape)

        # read array for the scene
        for s, t, c, z in product(range(size_s),
                                  range(size_t),
                                  range(size_c),
                                  range(size_z)):

            # read a 2D image plane from the CZI
            if mdata.image.SizeS is None:
                image2d = czidoc.read(plane={'T': t, 'Z': z, 'C': c})
            else:
                image2d = czidoc.read(plane={'T': t, 'Z': z, 'C': c}, scene=s)

            # # check if the image2d is really not too big
            # if (mdata.bbox.total_bounding_box["X"][1] > mdata.image.SizeX or
            #         mdata.bbox.total_bounding_box["Y"][1] > mdata.image.SizeY):

            #     image2d = image2d[..., 0:mdata.image.SizeY, 0:mdata.image.SizeX, :]

            # insert 2D image plane into the array6d
            array6d[s, t, c, z, ...] = image2d

        # change the dimension order if needed
        if output_order == "STZCYX":
            dim_string = "STZCYXA"
            array6d = np.swapaxes(array6d, 2, 3)

        if output_order == "STCZYX":
            dim_string = "STCZYXA"

        if remove_adim:
            dim_string = dim_string.replace("A", "")
            array6d = np.squeeze(array6d, axis=-1)

    return array6d, mdata, dim_string


def read_5darray(filepath: Union[str, os.PathLike[str]],
                 scene: int = 0,
                 output_order: str = "TCZYX",
                 output_dask: bool = False,
                 chunks_auto: bool = False,
                 remove_adim: bool = True,
                 **kwargs: int) -> Tuple[Optional[Union[np.ndarray, da.Array]], czimd.CziMetadata, str]:
    """Read a CZI image file as 5D numpy or dask array.
    Important: Currently supported are only scenes with equal size and CZIs with consistent pixel types.

    Args:
        filepath (str | Path): filepath for the CZI image
        scene (int, optional): Index of scene to be read as 5D stack.
        output_order (str, optional): Order of dimensions for the output array. Defaults to "TCZYX".
        output_dask (bool, optional): If True the output will be a dask array. Defaults to False.
        chunks_auto (bool, optional): Use a chunk size create automatically if True and otherwise use the array shape. Default to False.
        remove_adim (bool, optional): If true the dimension for the pixel type will be removed. Defaults to True.
        kwargs (int, optional): Allowed kwargs are S, T, Z, C and values must be >=0 (zero-based).
                                Will be used to read only a substack from the array

    Returns:
        Tuple[np.ndarray, czimd.Metadata, dim_order6d]: output as 6D numpy or dask array and metadata
    """

    if isinstance(filepath, Path):
        # convert to string
        filepath = str(filepath)

    # get the complete metadata at once as one big class
    mdata = czimd.CziMetadata(filepath)

    if not mdata.consistent_pixeltypes:
        print("Detected PixelTypes ar not consistent. Cannot create array6d")
        return None, mdata, ""
    if mdata.consistent_pixeltypes:
        # use pixel type from first channel
        use_pixeltype = mdata.npdtype[0]

    valid_order = ["TCZYX", "TZCYX"]

    if output_order not in valid_order:
        print("Invalid dimension order:", output_order)
        return None, mdata, ""

    # open the CZI document to read the
    with pyczi.open_czi(filepath) as czidoc:

        if mdata.image.SizeS is not None:

            # use the size of the 1st scenes_bounding_rectangle
            size_x = czidoc.scenes_bounding_rectangle[scene].w
            size_y = czidoc.scenes_bounding_rectangle[scene].h

        elif mdata.image.SizeS is None:

            # use the size of the total_bounding_rectangle
            size_x = czidoc.total_bounding_rectangle.w
            size_y = czidoc.total_bounding_rectangle.h

        # check if dimensions are None (because they do not exist for that image)
        size_c = misc.check_dimsize(mdata.image.SizeC, set2value=1)
        size_z = misc.check_dimsize(mdata.image.SizeZ, set2value=1)
        size_t = misc.check_dimsize(mdata.image.SizeT, set2value=1)

        shape = [size_t, size_z, size_c, size_y, size_x, 3 if mdata.isRGB else 1]

        if not output_dask:
            array5d = np.empty(shape, dtype=use_pixeltype)

        elif output_dask:
            if chunks_auto:
                array5d = da.empty(shape, dtype=use_pixeltype, chunks="auto")
            if not chunks_auto:
                array5d = da.empty(shape, dtype=use_pixeltype, chunks=shape)

        # read array for the scene
        for t, z, c in product(range(size_t),
                               range(size_z),
                               range(size_c)):

            # read a 2D image plane from the CZI
            if mdata.image.SizeS is None:
                image2d = czidoc.read(plane={'T': t, 'Z': z, 'C': c})
            else:
                image2d = czidoc.read(plane={'T': t, 'Z': z, 'C': c}, scene=scene)

            # insert 2D image plane into the array
            array5d[t, z, c, ...] = image2d

            # change the dimension order if needed
        if output_order == "TZCYX":
            dim_string = "TZCYXA"

        if output_order == "TCZYX":
            dim_string = "TCZYXA"
            array5d = np.swapaxes(array5d, 1, 2)

        if remove_adim:
            dim_string = dim_string.replace("A", "")
            array5d = np.squeeze(array5d, axis=-1)

    return array5d, mdata, dim_string


def read_mdarray_lazy(filepath: Union[str, os.PathLike[str]],
                      remove_adim: bool = True) -> Tuple[Optional[da.Array], czimd.CziMetadata, str]:

    def read_5d(filepath: Union[str, os.PathLike[str]],
                sizes: Tuple[int, int, int, int, int],
                s: int,
                mdata: czimd.CziMetadata,
                remove_adim: bool = True) -> np.ndarray:

        if isinstance(filepath, Path):
            # convert to string
            filepath = str(filepath)

        array5d = da.empty([sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], 3 if mdata.isRGB else 1],
                           dtype=mdata.npdtype)

        # open the CZI document to read the
        with pyczi.open_czi(filepath) as czidoc5d:

            # read array for the scene
            for t, z, c in product(range(sizes[0]),
                                   range(sizes[1]),
                                   range(sizes[2])):

                if mdata.image.SizeS is None:
                    image2d = czidoc5d.read()
                else:
                    image2d = czidoc5d.read(
                        plane={'T': t, 'Z': z, 'C': c}, scene=s)

                # # check if the image2d is really not too big
                # if mdata.pyczi_dims["X"][1] > mdata.image.SizeX or mdata.pyczi_dims["Y"][1] > mdata.image.SizeY:
                #     image2d = image2d[..., 0:mdata.image.SizeY, 0:mdata.image.SizeX, :]

                array5d[t, z, c, ...] = image2d

        if remove_adim:
            array5d = np.squeeze(array5d, axis=-1)

        return array5d

    if isinstance(filepath, Path):
        # convert to string
        filepath = str(filepath)

    # get the metadata
    mdata = czimd.CziMetadata(filepath)

    if not mdata.consistent_pixeltypes:
        print("Detected PixelTypes ar not consistent. Cannot create array6d")
        return None, mdata, ""
    if mdata.consistent_pixeltypes:
        # use pixel type from first channel
        use_pixeltype = mdata.npdtype[0]

    # open the CZI document to read the
    with pyczi.open_czi(filepath) as czidoc:

        if mdata.image.SizeS is not None:
            # use the size of the 1st scenes_bounding_rectangle
            size_x = czidoc.scenes_bounding_rectangle[0].w
            size_y = czidoc.scenes_bounding_rectangle[0].h

        if mdata.image.SizeS is None:
            # use the size of the total_bounding_rectangle
            size_x = czidoc.total_bounding_rectangle.w
            size_y = czidoc.total_bounding_rectangle.h

    # check if dimensions are None (because they do not exist for that image)
    size_c = misc.check_dimsize(mdata.image.SizeC, set2value=1)
    size_z = misc.check_dimsize(mdata.image.SizeZ, set2value=1)
    size_t = misc.check_dimsize(mdata.image.SizeT, set2value=1)
    size_s = misc.check_dimsize(mdata.image.SizeS, set2value=1)

    sizes = (size_t, size_z, size_c, size_y, size_x)

    # define the required shape
    if remove_adim:
        sp = [size_t, size_z, size_c, size_y, size_x]
    if not remove_adim:
        sp = [size_t, size_z, size_c, size_y, size_x, 3 if mdata.isRGB else 1]

    # create dask stack of lazy image readers
    lazy_process_image = dask.delayed(read_5d)  # lazy reader
    lazy_arrays = [lazy_process_image(
        filepath, sizes, s, mdata, remove_adim) for s in range(size_s)]

    dask_arrays = [da.from_delayed(
        lazy_array, shape=sp, dtype=use_pixeltype) for lazy_array in lazy_arrays]

    # Stack into one large dask.array
    array_md = da.stack(dask_arrays, axis=0)

    # define the dimension order to be STZCYXA
    dim_string = "STZCYXA"

    if remove_adim:
        dim_string = dim_string.replace("A", "")

    return array_md, mdata, dim_string
