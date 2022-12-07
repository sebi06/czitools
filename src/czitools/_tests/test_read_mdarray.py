
from czitools import pylibczirw_tools
import os
from pathlib import Path

basedir = Path(__file__).resolve().parents[3]


def test_read_mdarray_1():

    # get the CZI filepath
    filepath = os.path.join(basedir, r"data/w96_A1+A2.czi")

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


def test_read_mdarray_2():

    # get the CZI filepath
    filepath = os.path.join(basedir, r"data/S=2_3x3_CH=2.czi")

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


def test_read_mdarray_lazy_1():

    # get the CZI filepath
    filepath = os.path.join(basedir, r"data/w96_A1+A2.czi")

    mdarray, dimstring = pylibczirw_tools.read_mdarray_lazy(filepath, remove_adim=False)

    assert (dimstring == "STZCYXA")
    assert (mdarray.shape == (2, 1, 1, 2, 1416, 1960, 1))
    assert (mdarray.ndim == 7)
    assert (mdarray.chunksize == (1, 1, 1, 2, 1416, 1960, 1))

    mdarray, dimstring = pylibczirw_tools.read_mdarray_lazy(filepath, remove_adim=True)

    assert (dimstring == "STZCYX")
    assert (mdarray.shape == (2, 1, 1, 2, 1416, 1960))
    assert (mdarray.ndim == 6)
    assert (mdarray.chunksize == (1, 1, 1, 2, 1416, 1960))


def test_read_mdarray_substack():

    # get the CZI filepath
    filepath = os.path.join(basedir, r"data/w96_A1+A2.czi")

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
    filepath = os.path.join(basedir, r"data/CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi")

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
