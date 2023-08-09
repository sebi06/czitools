from czitools import read_tools, misc_tools
from pathlib import Path
import dask.array as da
import zarr
import pandas as pd
import pytest
import numpy as np
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
        ("S=2_3x3_CH=2.czi", 1, 2, (2, 1, 1, 1, 1792, 1792)),
    ],
)
def test_slicedim(czifile: str, dimindex: int, posdim: int, shape: Tuple[int]) -> None:
    # get the CZI filepath
    filepath = basedir / "data" / czifile

    mdarray, mdata, dimstring = read_tools.read_6darray(
        filepath, output_order="STCZYX"
    )

    dim_array = misc_tools.slicedim(mdarray, dimindex, posdim)
    assert dim_array.shape == shape


@pytest.mark.parametrize(
    "czifile, csvfile, xstart, ystart",
    [
        (
            "WP96_4Pos_B4-10_DAPI.czi",
            "WP96_4Pos_B4-10_DAPI_planetable.csv",
            [148118, 166242],
            [78118, 78118],
        )
    ],
)
def test_get_planetable(
    czifile: str, csvfile: str, xstart: List[int], ystart: List[int]
) -> None:
    # get the CZI filepath
    filepath = (basedir / "data" / czifile).as_posix()

    isczi = False
    iscsv = False

    # check if the input is a csv or czi file
    if filepath.lower().endswith(".czi"):
        isczi = True
    if filepath.lower().endswith(".csv"):
        iscsv = True

    # separator of CSV file
    separator = ","

    # read the data from CSV file
    if iscsv:
        planetable = pd.read_csv(filepath, sep=separator)
    if isczi:
        # read the data from CZI file
        planetable, csvfile = misc_tools.get_planetable(
            filepath, norm_time=True, savetable=True, separator=",", index=True
        )

        assert csvfile == (basedir / "data" / csvfile).as_posix()

        # remove the file
        Path.unlink(Path(csvfile))

    planetable_filtered = misc_tools.filter_planetable(planetable, s=0, t=0, z=0, c=0)

    assert planetable_filtered["xstart"][0] == xstart[0]
    assert planetable_filtered["xstart"][1] == xstart[1]
    assert planetable_filtered["ystart"][0] == ystart[0]
    assert planetable_filtered["ystart"][1] == ystart[1]


@pytest.mark.parametrize(
    "entry, set2value, result", [(0, 1, 0), (None, 1, 1), (-1, 2, -1), (None, 3, 3)]
)
def test_check_dimsize(entry: Optional[int], set2value: int, result: int) -> None:
    """
    This function checks the dimension size of an entry against a set value
    and compares it to a result.

    Parameters:
    entry (Optional[int]): The entry to be checked.
    set2value (int): The set value to compare the entry against.
    result (int): The expected result of the comparison.
    Returns: None.
    """

    assert misc_tools.check_dimsize(entry, set2value=set2value) == result


@pytest.mark.parametrize(
    "array, min_value, max_value, corr_min, corr_max",
    [
        (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], np.int16), 1, 9, 1.0, 1.0),
        (
            zarr.array(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], np.int16)),
            1,
            9,
            1.0,
            1.0,
        ),
        (
            da.from_array(np.array([[1, 2, 0], [4, -5, 6], [7, 8, 9]], np.int16)),
            -5,
            9,
            1.0,
            1.0,
        ),
    ],
)
def test_calc_scaling(
    array: Union[np.ndarray, da.Array, zarr.Array],
    min_value: int,
    max_value: int,
    corr_min: float,
    corr_max: float,
) -> None:
    minv, maxv = misc_tools.calc_scaling(array, corr_min=corr_min, corr_max=corr_max)

    assert min_value == minv
    assert max_value == maxv
