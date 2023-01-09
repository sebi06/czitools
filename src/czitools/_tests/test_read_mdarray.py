from czitools import pylibczirw_tools
from pathlib import Path
import dask.array as da
import numpy as np

basedir = Path(__file__).resolve().parents[3]


def test_read_mdarray_1():

    # get the CZI filepath
    filepath = basedir / "data" / "w96_A1+A2.czi"

    mdarray, mdata, dimstring = pylibczirw_tools.read_6darray(filepath,
                                                              output_order="STCZYX",
                                                              output_dask=False,
                                                              chunks_auto=False,
                                                              remove_adim=False)

    assert (dimstring == "STCZYXA")
    assert (mdarray.shape == (2, 1, 2, 1, 1416, 1960, 1))

    mdarray, mdata, dimstring = pylibczirw_tools.read_6darray(filepath,
                                                              output_order="STZCYX",
                                                              output_dask=False,
                                                              chunks_auto=False,
                                                              remove_adim=True)

    assert (dimstring == "STZCYX")
    assert (mdarray.shape == (2, 1, 1, 2, 1416, 1960))

    # get the CZI filepath
    filepath = basedir / "data" / "S=3_1Pos_2Mosaic_T=2=Z=3_CH=2_sm.czi"

    mdarray, mdata, dimstring = pylibczirw_tools.read_6darray(filepath,
                                                              output_order="STCZYX",
                                                              output_dask=False,
                                                              chunks_auto=False,
                                                              remove_adim=False)

    assert (mdarray is None)
    assert (mdata == mdata)
    assert (dimstring == "")


def test_read_mdarray_2():

    # get the CZI filepath
    filepath = basedir / r"data/S=2_3x3_CH=2.czi"

    mdarray, mdata, dimstring = pylibczirw_tools.read_6darray(filepath,
                                                              output_dask=False,
                                                              chunks_auto=False,
                                                              output_order="STZCYX",
                                                              remove_adim=False)

    assert (dimstring == "STZCYXA")
    assert (mdarray.shape == (2, 1, 1, 2, 1792, 1792, 1))

    mdarray, mdata, dimstring = pylibczirw_tools.read_6darray(filepath,
                                                              output_dask=False,
                                                              chunks_auto=False,
                                                              output_order="STCZYX",
                                                              remove_adim=True)

    assert (dimstring == "STCZYX")
    assert (mdarray.shape == (2, 1, 2, 1, 1792, 1792))


def test_read_mdarray_3():

    # get the CZI filepath
    filepath = basedir / r"data/FOV7_HV110_P0500510000.czi"

    mdarray, mdata, dimstring = pylibczirw_tools.read_6darray(filepath,
                                                              output_dask=False,
                                                              chunks_auto=False,
                                                              output_order="STZCYX",
                                                              remove_adim=False)

    assert (dimstring == "STZCYXA")
    assert (mdarray.shape == (1, 1, 1, 1, 512, 512, 1))


def test_read_mdarray_4():

    # get the CZI filepath
    filepath = basedir / r"data/newCZI_compressed.czi"

    mdarray, mdata, dimstring = pylibczirw_tools.read_6darray(filepath,
                                                              output_dask=True,
                                                              chunks_auto=False,
                                                              output_order="STZCYX",
                                                              remove_adim=True)

    assert (type(mdarray == da.array))
    assert (dimstring == "STZCYX")
    assert (mdarray.shape == (1, 1, 1, 1, 512, 512))


def test_read_mdarray_lazy_1():

    # get the CZI filepath
    filepath = basedir / r"data/w96_A1+A2.czi"

    mdarray, mdata, dimstring = pylibczirw_tools.read_mdarray_lazy(filepath, remove_adim=False)

    assert (dimstring == "STZCYXA")
    assert (mdarray.shape == (2, 1, 1, 2, 1416, 1960, 1))
    assert (mdarray.ndim == 7)
    assert (mdarray.chunksize == (1, 1, 1, 2, 1416, 1960, 1))

    mdarray, mdata, dimstring = pylibczirw_tools.read_mdarray_lazy(filepath, remove_adim=True)

    assert (dimstring == "STZCYX")
    assert (mdarray.shape == (2, 1, 1, 2, 1416, 1960))
    assert (mdarray.ndim == 6)
    assert (mdarray.chunksize == (1, 1, 1, 2, 1416, 1960))


def test_read_mdarray_substack():

    # get the CZI filepath
    filepath = basedir / r"data/w96_A1+A2.czi"

    # read only a specific scene from the CZI
    mdarray, mdata, dimstring = pylibczirw_tools.read_6darray(filepath,
                                                              output_order="STCZYX",
                                                              output_dask=False,
                                                              chunks_auto=False,
                                                              remove_adim=False,
                                                              S=0)

    assert (dimstring == "STCZYXA")
    assert (mdarray.shape == (1, 1, 2, 1, 1416, 1960, 1))
    assert (mdata.image.SizeS == 1)

    # get the CZI filepath
    filepath = basedir / r"data/CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi"

    # read only a specific scene from the CZI
    mdarray, mdata, dimstring = pylibczirw_tools.read_6darray(filepath,
                                                              output_order="STZCYX",
                                                              output_dask=True,
                                                              chunks_auto=False,
                                                              remove_adim=False,
                                                              S=0,
                                                              T=0,
                                                              Z=0)

    assert (dimstring == "STZCYXA")
    assert (mdarray.shape == (1, 1, 1, 2, 170, 240, 1))
    assert (mdata.image.SizeS is None)

    # read only a specific scene from the CZI
    mdarray, mdata, dimstring = pylibczirw_tools.read_6darray(filepath,
                                                              output_order="STZCYX",
                                                              output_dask=False,
                                                              chunks_auto=False,
                                                              remove_adim=True,
                                                              T=0)

    assert (dimstring == "STZCYX")
    assert (mdarray.shape == (1, 1, 5, 2, 170, 240))
    assert (mdata.image.SizeS is None)

    # read only a specific scene from the CZI
    mdarray, mdata, dimstring = pylibczirw_tools.read_6darray(filepath,
                                                              output_order="STZCYX",
                                                              output_dask=False,
                                                              chunks_auto=False,
                                                              remove_adim=True,
                                                              C=0)

    assert (dimstring == "STZCYX")
    assert (mdarray.shape == (1, 3, 5, 1, 170, 240))
    assert (mdata.image.SizeS is None)
