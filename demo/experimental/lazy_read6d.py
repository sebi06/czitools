from czitools import pylibczirw_metadata as czimd
from pathlib import Path
from tqdm.contrib.itertools import product
import dask.array as da
from pylibCZIrw import czi as pyczi
import dask
from czitools import napari_tools
from czitools import misc
import napari
import os

@dask.delayed
def read_plane(filepath, has_scene, s: int, t: int, c: int, z: int):

    with pyczi.open_czi(filepath) as czidoc:

        if has_scene:
            image2d = czidoc.read(plane={"T": t, "C": c, "Z": z}, scene=s)
        if not has_scene:
            image2d = czidoc.read(plane={"T": t, "C": c, "Z": z})

    return image2d[..., 0]


def read6d(filepath):
    if isinstance(filepath, Path):
        # convert to string
        filepath = str(filepath)

    # get the complete metadata at once as one big class
    mdata = czimd.CziMetadata(filepath)

    if mdata.image.SizeS is not None:
        # use the size of the 1st scenes_bounding_rectangle
        size_x = mdata.bbox.scenes_bounding_rect[0].w
        size_y = mdata.bbox.scenes_bounding_rect[0].h

    if mdata.image.SizeS is None:
        # use the size of the total_bounding_rectangle
        size_x = mdata.bbox.total_rect.w
        size_y = mdata.bbox.total_rect.h

    # check if dimensions are None (because they do not exist for that image)
    size_c = misc.check_dimsize(mdata.image.SizeC, set2value=1)
    size_z = misc.check_dimsize(mdata.image.SizeZ, set2value=1)
    size_t = misc.check_dimsize(mdata.image.SizeT, set2value=1)
    size_s = misc.check_dimsize(mdata.image.SizeS, set2value=1)

    shape = (size_s, size_t, size_c, size_z, size_y, size_x)
    array6d = da.empty(shape, dtype='uint16', chunks=shape)

    # read array for the scene
    for s, t, c, z in product(range(size_s),
                              range(size_t),
                              range(size_c),
                              range(size_z)):

        # read a 2D image plane from the CZI
        if mdata.image.SizeS is None:
            array6d[s, t, c, z, :, :] = da.from_delayed(read_plane(filepath, False, s, t, c, z),
                                                        shape=(size_y, size_x),
                                                        dtype='uint16')
        else:
            array6d[s, t, c, z, :, :] = da.from_delayed(read_plane(filepath, True, s, t, c, z),
                                                        shape=(size_y, size_x),
                                                        dtype='uint16')

    return array6d, mdata, "STCZYX"


# get the CZI filepath
#filename = r"E:\testpictures\Testdata_Zeiss\LatticeLightSheet\LS_Mitosis_T=150-300.czi"
filename = r"E:\testpictures\Testdata_Zeiss\LatticeLightSheet\LS_Mitosis_T=150-220_subset_CH=1.czi"

from aicsimageio import AICSImage

aics_img = AICSImage(filename)
out = aics_img.dask_data

print("Done")
# array6d, mdata, dim_string6d = read6d(filename)
#
# # show array inside napari viewer
# viewer = napari.Viewer()
# layers = napari_tools.show(viewer, array6d, mdata,
#                            dim_string=dim_string6d,
#                            blending="additive",
#                            contrast='from_czi',
#                            gamma=0.85,
#                            add_mdtable=True,
#                            name_sliders=True)
#
# napari.run()
