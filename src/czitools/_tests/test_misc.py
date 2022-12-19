from czitools import pylibczirw_tools, misc
import os
from pathlib import Path
import dask.array as da

basedir = Path(__file__).resolve().parents[3]


def test_slicedim():

    # get the CZI filepath
    filepath = os.path.join(basedir, r"data/CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi")

    mdarray, mdata, dimstring = pylibczirw_tools.read_6darray(filepath,
                                                              output_dask=False,
                                                              chunks_auto=False,
                                                              output_order="STCZYX",
                                                              remove_adim=True)

    dim_array = misc.slicedim(mdarray, 0, 2)
    assert(dim_array.shape == (1, 3, 1, 5, 170, 240))

    dim_array = misc.slicedim(mdarray, 0, 1)
    assert(dim_array.shape == (1, 1, 2, 5, 170, 240))

    dim_array = misc.slicedim(mdarray, 2, 3)
    assert(dim_array.shape == (1, 3, 2, 1, 170, 240))

    # get the CZI filepath
    filepath = os.path.join(basedir, r"data/S=2_3x3_CH=2.czi")

    mdarray, mdata, dimstring = pylibczirw_tools.read_6darray(filepath,
                                                              output_dask=False,
                                                              chunks_auto=False,
                                                              output_order="STCZYX",
                                                              remove_adim=True)

    dim_array = misc.slicedim(mdarray, 0, 0)
    assert(dim_array.shape == (1, 1, 2, 1, 1792, 1792))

    dim_array = misc.slicedim(mdarray, 0, 1)
    assert(dim_array.shape == (2, 1, 2, 1, 1792, 1792))

    dim_array = misc.slicedim(mdarray, 0, 2)
    assert(dim_array.shape == (2, 1, 1, 1, 1792, 1792))

    dim_array = misc.slicedim(mdarray, 1, 2)
    assert(dim_array.shape == (2, 1, 1, 1, 1792, 1792))


test_slicedim()
