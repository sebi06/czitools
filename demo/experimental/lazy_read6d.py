from czitools import pylibczirw_metadata as czimd
from czitools import pylibczirw_tools
from czitools import napari_tools
from czitools import misc
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Type, Any, Union
from tqdm.contrib.itertools import product
import dask
import dask.array as da
from pylibCZIrw import czi as pyczi
import napari

@dask.delayed
def read2dplane_noscenes(czi, plane: Dict):

    image2d = czi.read(plane=plane)

    return image2d


@dask.delayed
def read2dplane(czi, plane: Dict, scene: int):

    image2d = czi.read(plane=plane,scene=scene)

    return image2d


def read_6darray_lazy(filepath: Union[str, os.PathLike[str]],
                      output_order: str = "STCZYX",
                      chunks_auto: bool = False,
                      remove_adim: bool = True,
                      **kwargs: int) -> Tuple[Optional[da.Array], czimd.CziMetadata, str]:
    """Read a CZI image file as dask array.
    Important: Currently supported are only scenes with equal size and CZIs with consistent pixel types.

    Args:
        filepath (str | Path): filepath for the CZI image
        output_order (str, optional): Order of dimensions for the output array. Defaults to "STCZYX".
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
                image2d = read2dplane_noscenes(czidoc, plane={'T': t, 'Z': z, 'C': c})
                #image2d = czidoc.read(plane={'T': t, 'Z': z, 'C': c})
            else:
                image2d = read2dplane(czidoc, {'T': t, 'Z': z, 'C': c}, scene=s)
                #image2d = czidoc.read(plane={'T': t, 'Z': z, 'C': c}, scene=s)

            # insert 2D image plane into the array6d
            array6d[s, t, c, z, ...] = image2d.compute()

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


# get the CZI filepath
defaultdir = Path(Path(__file__).resolve().parents[2]) / "data"
czifile = defaultdir / "WP96_4Pos_B4-10_DAPI.czi"
print("File exists:", os.path.exists(czifile))

array6d, mdata, dim_string6d = read_6darray_lazy(czifile, output_order="STCZYX",
                                                          chunks_auto=False,
                                                          remove_adim=True)

print("Done.")