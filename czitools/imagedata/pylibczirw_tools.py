# -*- coding: utf-8 -*-

#################################################################
# File        : pylibczirw_tools.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from pylibCZIrw import czi as pyczi
from czitools.metadata import pylibczirw_metadata as czimd
from czitools.utils import misc
import numpy as np
from typing import List, Dict, Tuple, Optional, Type, Any, Union
from tqdm.contrib.itertools import product
import dask
import dask.array as da


def read_6darray(filename: str,
                 dimorder: str = "STCZYX",
                 output_dask: bool = False,
                 chunks_auto: bool = False,
                 remove_Adim: bool = True) -> Tuple[Union[np.ndarray, da.Array], str]:
    """Read a CZI image file as 6D numpy or dask array.
    Important: Currently supported are only scenes with equal size.

    Args:
        filename (str): filepath for the CZI image
        dimorder (str, optional): Order of dimensions for the output array. Defaults to "STCZYX".
        output_dask (bool, optional): If True the output will be a dask array. Defaults to False.
        chunks_auto (bool, optional): Use a chunk size create automatically if True and otherwise use the array shape. Default to False.
        remove_Adim (bool, optional): If true the dimension for the pixel type will be removed. Defaults to True.

    Returns:
        Tuple[np.ndarray, str]: output as 6D numpy or dask array
    """

    valid_dimorders = ["STCZYX", "STZCYX"]

    if dimorder not in valid_dimorders:
        print("Invalid dimension order:", dimorder)
        return np.array([]), ""

    # get the complete metadata at once as one big class
    mdata = czimd.CziMetadata(filename)

    if not mdata.scene_shape_is_consistent:
        print("Scenes have inconsistent shape. Cannot read 6D array")
        return np.array([]), ""

    # open the CZI document to read the
    with pyczi.open_czi(filename) as czidoc:

        if mdata.image.SizeS is not None:
            # get size for a single scene using the 1st
            # works only if scene shape is consistent

            # TODO Open question whether to use XML metadata or bounding boxes

            # use the size of the 1st scenes_bounding_rectangle
            size_x = czidoc.scenes_bounding_rectangle[0].w
            size_y = czidoc.scenes_bounding_rectangle[0].h

        if mdata.image.SizeS is None:

            # use the size from the metadata
            #size_x = mdata.image.SizeX
            #size_y = mdata.image.SizeY

            # use the size of the total_bounding_rectangle
            size_x = czidoc.total_bounding_rectangle.w
            size_y = czidoc.total_bounding_rectangle.h

        # check if dimensions are None (because they do not exist for that image)
        size_c = misc.check_dimsize(mdata.image.SizeC, set2value=1)
        size_z = misc.check_dimsize(mdata.image.SizeZ, set2value=1)
        size_t = misc.check_dimsize(mdata.image.SizeT, set2value=1)
        size_s = misc.check_dimsize(mdata.image.SizeS, set2value=1)

        if dimorder == "STZCYX":

            # define the dimension order to be STZCYXA
            dimstring = "STZCYXA"

            shape = [size_s, size_t, size_z, size_c, size_y, size_x, 3 if mdata.isRGB else 1]

            if not output_dask:
                array6d = np.empty(shape, dtype=mdata.npdtype)
            if output_dask:
                if chunks_auto:
                    array6d = da.empty(shape, dtype=mdata.npdtype, chunks="auto")
                if not chunks_auto:
                    array6d = da.empty(shape, dtype=mdata.npdtype, chunks=shape)

            # read array for the scene
            for s, t, z, c in product(range(size_s),
                                      range(size_t),
                                      range(size_z),
                                      range(size_c)):

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
                array6d[s, t, z, c, ...] = image2d

        if dimorder == "STCZYX":

            # define the dimension order to be STCZYXA
            dimstring = "STCZYXA"

            shape = [size_s, size_t, size_c, size_z, size_y, size_x, 3 if mdata.isRGB else 1]

            if not output_dask:
                array6d = np.empty(shape, dtype=mdata.npdtype)
            if output_dask:
                if chunks_auto:
                    array6d = da.empty(shape, dtype=mdata.npdtype, chunks="auto")
                if not chunks_auto:
                    array6d = da.empty(shape, dtype=mdata.npdtype, chunks=shape)

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
                #        mdata.bbox.total_bounding_box["Y"][1] > mdata.image.SizeY):

                #     image2d = image2d[..., 0:mdata.image.SizeY, 0:mdata.image.SizeX, :]

                # insert 2D image plane into the array6d
                array6d[s, t, c, z, ...] = image2d

        if remove_Adim:
            dimstring = dimstring.replace("A", "")
            array6d = np.squeeze(array6d, axis=-1)

    return array6d, dimstring


def read_5darray(filename: str,
                 scene: int = 0,
                 dimorder: str = "TCZYX",
                 output_dask: bool = False,
                 chunks_auto: bool = False,
                 remove_Adim: bool = True) -> Tuple[Union[np.ndarray, da.Array], str]:
    """Read a CZI image file as 5D numpy or dask array.
    Important: Currently supported are only scenes with equal size.

    Args:
        filename (str): filepath for the CZI image
        dimorder (str, optional): Order of dimensions for the output array. Defaults to "TCZYX".
        output_dask (bool, optional): If True the output will be a dask array. Defaults to False.
        chunks_auto (bool, optional): Use a chunk size create automatically if True and otherwise use the array shape. Default to False.
        remove_Adim (bool, optional): If true the dimension for the pixel type will be removed. Defaults to True.

    Returns:
        Tuple[np.ndarray, str]: output as 5D numpy or dask array
    """

    valid_dimorders = ["TCZYX", "TZCYX"]

    if dimorder not in valid_dimorders:
        print("Invalid dimension order:", dimorder)
        return np.array([]), ""

    # get the complete metadata at once as one big class
    mdata = czimd.CziMetadata(filename)

    # open the CZI document to read the
    with pyczi.open_czi(filename) as czidoc:

        if mdata.image.SizeS is not None:

            # use the size of the 1st scenes_bounding_rectangle
            size_x = czidoc.scenes_bounding_rectangle[scene].w
            size_y = czidoc.scenes_bounding_rectangle[scene].h

        if mdata.image.SizeS is None:

            # use the size of the total_bounding_rectangle
            size_x = czidoc.total_bounding_rectangle.w
            size_y = czidoc.total_bounding_rectangle.h

        # check if dimensions are None (because they do not exist for that image)
        size_c = misc.check_dimsize(mdata.image.SizeC, set2value=1)
        size_z = misc.check_dimsize(mdata.image.SizeZ, set2value=1)
        size_t = misc.check_dimsize(mdata.image.SizeT, set2value=1)

        if dimorder == "TZCYX":

            # define the dimension order to be TZCYXA
            dimstring = "TZCYXA"

            shape = [size_t, size_z, size_c, size_y, size_x, 3 if mdata.isRGB else 1]

            if not output_dask:
                array5d = np.empty(shape, dtype=mdata.npdtype)
            if output_dask:
                if chunks_auto:
                    array5d = da.empty(shape, dtype=mdata.npdtype, chunks="auto")
                if not chunks_auto:
                    array5d = da.empty(shape, dtype=mdata.npdtype, chunks=shape)

            # read array for the scene
            for t, z, c in product(range(size_t),
                                   range(size_z),
                                   range(size_c)):

                # read a 2D image plane from the CZI
                if mdata.image.SizeS is None:
                    image2d = czidoc.read(plane={'T': t, 'Z': z, 'C': c})
                else:
                    image2d = czidoc.read(plane={'T': t, 'Z': z, 'C': c}, scene=scene)

                # insert 2D image plane into the array6d
                array5d[t, z, c, ...] = image2d

        if dimorder == "TCZYX":

            # define the dimension order to be TCZYXA
            dimstring = "TCZYXA"

            shape = [size_t, size_c, size_z, size_y, size_x, 3 if mdata.isRGB else 1]

            if not output_dask:
                array5d = np.empty(shape, dtype=mdata.npdtype)
            if output_dask:
                if chunks_auto:
                    array5d = da.empty(shape, dtype=mdata.npdtype, chunks="auto")
                if not chunks_auto:
                    array5d = da.empty(shape, dtype=mdata.npdtype, chunks=shape)

            # read array for the scene
            for t, c, z in product(range(size_t),
                                   range(size_c),
                                   range(size_z)):

                # read a 2D image plane from the CZI
                if mdata.image.SizeS is None:
                    image2d = czidoc.read(plane={'T': t, 'Z': z, 'C': c})
                else:
                    image2d = czidoc.read(plane={'T': t, 'Z': z, 'C': c}, scene=scene)

                # insert 2D image plane into the array6d
                array5d[t, c, z, ...] = image2d

        if remove_Adim:
            dimstring = dimstring.replace("A", "")
            array5d = np.squeeze(array5d, axis=-1)

    return array5d, dimstring


###### EXPERIMENTAL #####
def read_mdarray_lazy(filename: str, remove_Adim: bool = True) -> Tuple[da.Array, str]:

    def read_5d(filename: str,
                sizes: Tuple[int, int, int, int, int],
                s: int,
                mdata: czimd.CziMetadata,
                remove_Adim: bool = True) -> np.ndarray:

        array_md = da.empty([sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], 3 if mdata.isRGB else 1],
                            dtype=mdata.npdtype)

        # open the CZI document to read the
        with pyczi.open_czi(filename) as czidoc:

            # read array for the scene
            for t, z, c in product(range(sizes[0]),
                                   range(sizes[1]),
                                   range(sizes[2])):

                if mdata.image.SizeS is None:
                    image2d = czidoc.read()
                else:
                    image2d = czidoc.read(
                        plane={'T': t, 'Z': z, 'C': c}, scene=s)

                # # check if the image2d is really not too big
                # if mdata.pyczi_dims["X"][1] > mdata.image.SizeX or mdata.pyczi_dims["Y"][1] > mdata.image.SizeY:
                #     image2d = image2d[..., 0:mdata.image.SizeY, 0:mdata.image.SizeX, :]

                array_md[t, z, c, ...] = image2d

        if remove_Adim:
            array_md = np.squeeze(array_md, axis=-1)

        return array_md

    # get the metadata
    mdata = czimd.CziMetadata(filename)

    # open the CZI document to read the
    with pyczi.open_czi(filename) as czidoc:

        if mdata.image.SizeS is not None:

            # use the size of the 1st scenes_bounding_rectangle
            sizeX = czidoc.scenes_bounding_rectangle[0].w
            sizeY = czidoc.scenes_bounding_rectangle[0].h

        if mdata.image.SizeS is None:

            # use the size of the total_bounding_rectangle
            sizeX = czidoc.total_bounding_rectangle.w
            sizeY = czidoc.total_bounding_rectangle.h

    # if mdata.image.SizeS is not None:
    #     # get size for a single scene using the 1st
    #     # works only if scene shape is consistent
    #     sizeX = mdata.bbox.scenes_total_rect[0].w
    #     sizeY = mdata.bbox.scenes_total_rect[0].h

    # if mdata.image.SizeS is None:
    #     sizeX = mdata.bbox.total_bounding_rectangle.w
    #     sizeY = mdata.bbox.total_bounding_rectangle.h

    # check if dimensions are None (because they do not exist for that image)
    sizeC = misc.check_dimsize(mdata.image.SizeC, set2value=1)
    sizeZ = misc.check_dimsize(mdata.image.SizeZ, set2value=1)
    sizeT = misc.check_dimsize(mdata.image.SizeT, set2value=1)
    sizeS = misc.check_dimsize(mdata.image.SizeS, set2value=1)

    sizes = (sizeT, sizeZ, sizeC, sizeY, sizeX)

    # define the required shape
    if remove_Adim:
        sp = [sizeT, sizeZ, sizeC, sizeY, sizeX]
    if not remove_Adim:
        sp = [sizeT, sizeZ, sizeC, sizeY, sizeX, 3 if mdata.isRGB else 1]

    # create dask stack of lazy image readers
    lazy_process_image = dask.delayed(read_5d)  # lazy reader
    lazy_arrays = [lazy_process_image(
        filename, sizes, s, mdata, remove_Adim) for s in range(sizeS)]

    dask_arrays = [da.from_delayed(
        lazy_array, shape=sp, dtype=mdata.npdtype) for lazy_array in lazy_arrays]

    # Stack into one large dask.array
    array_md = da.stack(dask_arrays, axis=0)

    # define the dimension order to be STZCYXA
    dimstring = "STZCYXA"

    if remove_Adim:
        dimstring = dimstring.replace("A", "")

    return array_md, dimstring
