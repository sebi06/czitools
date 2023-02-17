from czitools import pylibczirw_tools, misc
from pathlib import Path
import dask.array as da
import pandas as pd
import pytest
from typing import List, Dict, Tuple, Optional, Type, Any, Union, Mapping

basedir = Path(__file__).resolve().parents[3]

@pytest.mark.parametrize(
    "czifile, dimindex, posdim, shape",
    [
        ("CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi", 0, 2, (1, 3, 1, 5, 170, 240)),
        ("CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi", 0, 1, (1, 1, 2, 5, 170, 240)),
        ("CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi", 2, 3, (1, 3, 2, 1, 170, 240)),
        ("S=2_3x3_CH=2.czi", 0, 0, (1, 1, 2, 1, 1792, 1792)),
        ("S=2_3x3_CH=2.czi", 0, 1, (2, 1, 2, 1, 1792, 1792)),
        ("S=2_3x3_CH=2.czi", 0, 2, (2, 1, 1, 1, 1792, 1792)),
        ("S=2_3x3_CH=2.czi", 1, 2, (2, 1, 1, 1, 1792, 1792))
    ]
)
def test_slicedim(czifile: str, dimindex: int, posdim: int, shape: Tuple[int]) -> None:

    # get the CZI filepath
    filepath = basedir / "data" / czifile

    mdarray, mdata, dimstring = pylibczirw_tools.read_6darray(filepath,
                                                              output_dask=False,
                                                              chunks_auto=False,
                                                              output_order="STCZYX",
                                                              remove_adim=True)

    dim_array = misc.slicedim(mdarray, dimindex, posdim)
    assert(dim_array.shape == shape)


@pytest.mark.parametrize(
    "czifile, csvfile, xstart, ystart",
    [
        ("WP96_4Pos_B4-10_DAPI.czi", "WP96_4Pos_B4-10_DAPI_planetable.csv", [148118, 166242], [78118, 78118])
    ]
)
def test_get_planetable(czifile: str, csvfile: str, xstart: List[int], ystart: List[int]) -> None:

    # get the CZI filepath
    filepath = (basedir / "data" / czifile).as_posix()

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

        assert(csvfile == (basedir / "data" / csvfile).as_posix())

        # remove the file
        Path.unlink(Path(csvfile))

    planetable_filtered = misc.filter_planetable(planetable, s=0, t=0, z=0, c=0)

    assert(planetable_filtered["xstart"][0] == xstart[0])
    assert(planetable_filtered["xstart"][1] == xstart[1])
    assert(planetable_filtered["ystart"][0] == ystart[0])
    assert(planetable_filtered["ystart"][1] == ystart[1])


@pytest.mark.parametrize(
    "entry, set2value, result",
    [
        (0, 1, 0),
        (None, 1, 1),
        (-1, 2, -1),
        (None, 3, 3)
    ]
)
def test_check_dimsize(entry: Optional[int], set2value: int, result: int) -> None:

    assert (misc.check_dimsize(entry, set2value=set2value) == result)



