# -*- coding: utf-8 -*-

#################################################################
# File        : pylibczirw_tools.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from typing import List, Dict, Tuple, Optional, Type, Any, Union, Mapping
from pylibCZIrw import czi as pyczi
from czitools import pylibczirw_metadata as czimd
from czitools import misc
import numpy as np
from pathlib import Path
import dask
import dask.array as da
import os
from tqdm import trange


def read_6darray(filepath: Union[str, os.PathLike[str]],
                 output_order: str = "STCZYX",
                 **kwargs: int) -> Tuple[Optional[Union[np.ndarray, da.Array]], czimd.CziMetadata, str]:
    """Read a CZI image file as 6D numpy or dask array.
    Important: Currently supported are only scenes with equal size and CZIs with consistent pixel types.

    Args:
        filepath (str | Path): filepath for the CZI image
        output_order (str, optional): Order of dimensions for the output array. Defaults to "STCZYX".
        kwargs (int, optional): Allowed kwargs are S, T, Z, C and values must be >=0 (zero-based).
                                Will be used to read only a substack from the array

    Returns:
        Tuple[array6d, mdata, dim_string]: output as 6D dask array, metadata and dimstring
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

        if mdata.isRGB:
            shape2d = (size_y, size_x, 3)
        if not mdata.isRGB:
            shape2d = (size_y, size_x)

        remove_Adim = False if mdata.isRGB else True

        # initialise empty list to hold the dask arrays
        img = []

        for s in trange(size_s):

            time_stack = []
            for time in range(size_t):

                ch_stack = []
                for ch in trange(size_c):

                    z_stack = []
                    for z in range(size_z):

                        if mdata.image.SizeS is not None:

                            # z_slice = da.from_delayed(
                            #    da_delayed(czidoc.read)(plane={'C': ch, 'T': time, 'Z': z}, scene=s), shape=shape2d, dtype=mdata.npdtype[0]
                            # )

                            z_slice = da.from_delayed(read_plane(
                                czidoc, s=s, t=time, c=ch, z=z, has_scenes=True, remove_Adim=remove_Adim), shape=shape2d, dtype=mdata.npdtype[0])

                        if mdata.image.SizeS is None:

                            # z_slice = da.from_delayed(
                            #    da_delayed(czidoc.read)(plane={'C': ch, 'T': time, 'Z': z}), shape=shape2d, dtype=mdata.npdtype[0]
                            # )

                            z_slice = da.from_delayed(read_plane(
                                czidoc, s=s, t=time, c=ch, z=z, has_scenes=False, remove_Adim=remove_Adim), shape=shape2d, dtype=mdata.npdtype[0])

                        # create 2d array
                        z_slice = da.squeeze(z_slice)

                        # append to the z-stack
                        z_stack.append(z_slice)

                    # stack the array and create new dask array
                    z_stack = da.stack(z_stack, axis=0)

                    # create CZYX list of dask array
                    ch_stack.append(z_stack)

                # create TCZYX list of dask array
                time_stack.append(ch_stack)

            # create STCZYX list of dask array
            img.append(time_stack)

        # create final STCZYX dask arry
        array6d = da.stack(img, axis=0)

        # change the dimension order if needed
        if output_order == "STZCYX":
            dim_string = "STZCYXA"
            array6d = np.swapaxes(array6d, 2, 3)

        if output_order == "STCZYX":
            dim_string = "STCZYXA"

        if remove_Adim:
            dim_string = dim_string.replace("A", "")

    return array6d, mdata, dim_string


@dask.delayed
def read_plane(czidoc: pyczi.CziReader, s: int = 0, t: int = 0, c: int = 0, z: int = 0,
               has_scenes: bool = True,
               remove_Adim: bool = True):
    """Dask delayed function to read a 2d plane from a CZI image, which has the shape
       (Y, X, 1) or (Y, X, 3).
       If the image is a "grayscale image the last array dimension will be removed, when
       the option is ste to True.  

    Args:
        czidoc (pyczi.CziReader): Czireader objects
        s (int, optional): Scene index. Defaults to 0.
        t (int, optional): Time index. Defaults to 0.
        c (int, optional): Channel index. Defaults to 0.
        z (int, optional): Z-Plane index. Defaults to 0.
        has_scenes (bool, optional): Defines if the CZI actually contains scenes. Defaults to True.
        remove_Adim (bool, optional): Option to remove the last dimension of the 2D array. Defaults to True.

    Returns:
        dask.array: 6d dask.array with delayed reading for individual 2d planes
    """

    if has_scenes:
        image2d = czidoc.read(plane={"T": t, "C": c, "Z": z}, scene=s)
    if not has_scenes:
        image2d = czidoc.read(plane={"T": t, "C": c, "Z": z})

    if remove_Adim:
        return image2d[..., 0]
    if not remove_Adim:
        return image2d
