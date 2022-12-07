from czitools import misc
import os
import pandas as pd
from pathlib import Path

basedir = Path(__file__).resolve().parents[3]


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

test_get_planetable()
