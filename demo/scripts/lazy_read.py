from czitools.metadata import pylibczirw_metadata as czimd
from czitools.imagedata import pylibczirw_tools
from czitools.utils import napari_tools
from czitools.utils import misc
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Type, Any, Union
from tqdm.contrib.itertools import product
import dask
import dask.array as da
from pylibCZIrw import czi as pyczi
import napari


# get the CZI filepath
filepath = r"E:\testpictures\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=3_Z=4_CH=2.czi"
print("File exists:", os.path.exists(filepath))


def read_4d(filename: str,
            sizes: Tuple[int, int, int, int],
            s: int,
            t: int,
            mdata: czimd.CziMetadata,
            remove_Adim: bool = True) -> np.ndarray:

    array_4d = da.empty([sizes[0], sizes[1], sizes[2], sizes[3], 3 if mdata.isRGB else 1],
                        dtype=mdata.npdtype)

    # open the CZI document to read the
    with pyczi.open_czi(filename) as czidoc:

        # read array for the scene
        for z, c in product(range(sizes[0]),
                            range(sizes[1])):

            if mdata.image.SizeS is None:
                image2d = czidoc.read()
            else:
                image2d = czidoc.read(plane={"T": t, "Z": z, "C": c}, scene=s)

            # check if the image2d is really not too big
            if mdata.pyczi_dims["X"][1] > mdata.image.SizeX or mdata.pyczi_dims["Y"][1] > mdata.image.SizeY:
                image2d = image2d[..., 0:mdata.image.SizeY, 0:mdata.image.SizeX, :]

            array_4d[z, c, ...] = image2d

    if remove_Adim:
        array_4d = np.squeeze(array_4d, axis=-1)

    return array_4d


def read_lazy_ZCYX(filename: str,
                   s: int,
                   sizeT: int,
                   sizeZ: int,
                   sizeC: int,
                   sizeY: int,
                   sizeX: int,
                   dtype: np.dtype,
                   isRGB: bool,
                   remove_Adim: bool = True) -> Tuple[da.Array, str]:

    sizes = (sizeZ, sizeC, sizeY, sizeX)
    sp = [sizeZ, sizeC, sizeY, sizeX, 3 if isRGB else 1]

    # create dask stack of lazy image readers
    lazy_process_image = dask.delayed(read_4d)  # lazy reader
    lazy_arrays = [lazy_process_image(filename, sizes, s, t, mdata,
                                      remove_Adim=remove_Adim) for t in range(sizeT)]

    dask_arrays = [da.from_delayed(
        lazy_array, shape=sp, dtype=dtype) for lazy_array in lazy_arrays]

    # Stack into one large dask.array
    array_4d = da.stack(dask_arrays, axis=0)

    return array_4d


##########################################################

# get the complete metadata at once as one big class
mdata = czimd.CziMetadata(filepath)
remove_Adim = True

if mdata.image.SizeS is not None:
    # get size for a single scene using the 1st
    # works only if scene shape is consistent
    sizeX = mdata.bbox.scenes_bounding_rect[0].w
    sizeY = mdata.bbox.scenes_bounding_rect[0].h

if mdata.image.SizeS is None:
    sizeX = mdata.bbox.total_bounding_rectangle.w
    sizeY = mdata.bbox.total_bounding_rectangle.h

# check if dimensions are None (because they do not exist for that image)
sizeC = misc.check_dimsize(mdata.image.SizeC, set2value=1)
sizeZ = misc.check_dimsize(mdata.image.SizeZ, set2value=1)
sizeT = misc.check_dimsize(mdata.image.SizeT, set2value=1)
sizeS = misc.check_dimsize(mdata.image.SizeS, set2value=1)
sizes = (sizeT, sizeZ, sizeC, sizeY, sizeX)
if remove_Adim:
    sp = [sizeT, sizeZ, sizeC, sizeY, sizeX]
if not remove_Adim:
    sp = [sizeT, sizeZ, sizeC, sizeY, sizeX, 3 if mdata.isRGB else 1]

# loop over scenes

for s in range(sizeS):

    lazy_process_image = dask.delayed(read_lazy_ZCYX)
    lazy_arrays = [lazy_process_image(filepath, s,
                                      sizeT=sizeT,
                                      sizeZ=sizeZ,
                                      sizeC=sizeC,
                                      sizeY=sizeY,
                                      sizeX=sizeX,
                                      isRGB=mdata.isRGB,
                                      dtype=mdata.npdtype,
                                      remove_Adim=True) for s in range(sizeS)]

    dask_arrays = [da.from_delayed(lazy_array, shape=sp, dtype=mdata.npdtype)
                   for lazy_array in lazy_arrays]

    ls_STZCYX = da.stack(dask_arrays, axis=0)

# define the dimension order to be STZCYXA
dimstring = "STZCYXA"

if remove_Adim:
    dimstring = dimstring.replace("A", "")

# remove A dimension do display the array inside Napari
dim_order, dim_index, dim_valid = czimd.CziMetadata.get_dimorder(dimstring)

# show array inside napari viewer
viewer = napari.Viewer()
layers = napari_tools.show(viewer, ls_STZCYX, mdata,
                           dim_order=dim_order,
                           blending="additive",
                           contrast='napari_auto',
                           gamma=0.85,
                           add_mdtable=True,
                           name_sliders=True)

napari.run()
