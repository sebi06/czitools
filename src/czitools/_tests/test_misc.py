from czitools import pylibczirw_tools, misc
import os
from pathlib import Path
import dask.array as da
import pandas as pd

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


def test_get_planetable():

    # get the CZI filepath
    filepath = os.path.join(basedir, r"data/WP96_4Pos_B4-10_DAPI.czi")

    isczi = False
    iscsv = False

    # check if the input is a csv or czi file
    if filepath.lower().endswith('.czi'):
        isczi = True
    if filepath.lower().endswith('.csv'):
        iscsv = True

    # separator of CSV file
    separator = ','

    # read the data from CSV file
    if iscsv:
        planetable = pd.read_csv(filepath, sep=separator)
    if isczi:
        # read the data from CZI file
        planetable, csvfile = misc.get_planetable(filepath,
                                                  norm_time=True,
                                                  savetable=True,
                                                  separator=",",
                                                  index=True)

        assert(csvfile == os.path.join(basedir, r"data/WP96_4Pos_B4-10_DAPI_planetable.csv"))

    planetable_filtered = misc.filter_planetable(planetable, s=0, t=0, z=0, c=0)

    assert(planetable_filtered["xstart"][0] == 148118)
    assert(planetable_filtered["xstart"][1] == 166242)
    assert(planetable_filtered["ystart"][0] == 78118)
    assert(planetable_filtered["ystart"][1] == 78118)


def test_check_dimsize():

    assert(misc.check_dimsize(0, set2value=1) == 0)
    assert (misc.check_dimsize(None, set2value=1) == 1)
    assert (misc.check_dimsize(-1, set2value=2) == -1)
    assert (misc.check_dimsize(None, set2value=3) == 3)



