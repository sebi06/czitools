from czitools.read_tools import read_tools
from czitools.utils import misc
from pathlib import Path
import dask.array as da
import zarr
import pandas as pd
import pytest
import numpy as np
from typing import List, Tuple, Optional, Union, Dict
from unittest.mock import patch, Mock
from czitools.utils.misc import is_valid_czi_url

basedir = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "czifile, dimindex, posdim, shape",
    [
        ("CellDivision_T3_Z5_CH2_X240_Y170.czi", 0, 2, (1, 3, 1, 5, 170, 240)),
        ("CellDivision_T3_Z5_CH2_X240_Y170.czi", 0, 1, (1, 1, 2, 5, 170, 240)),
        ("CellDivision_T3_Z5_CH2_X240_Y170.czi", 2, 3, (1, 3, 2, 1, 170, 240)),
        ("S2_3x3_CH2.czi", 0, 0, (1, 1, 2, 1, 1792, 1792)),
        ("S2_3x3_CH2.czi", 0, 1, (2, 1, 2, 1, 1792, 1792)),
        ("S2_3x3_CH2.czi", 0, 2, (2, 1, 1, 1, 1792, 1792)),
        ("S2_3x3_CH2.czi", 1, 2, (2, 1, 1, 1, 1792, 1792)),
    ],
)
def test_slicedim(czifile: str, dimindex: int, posdim: int, shape: Tuple[int]) -> None:
    # get the CZI filepath
    filepath = basedir / "data" / czifile

    # mdarray, mdata, dimstring = read_tools.read_6darray(filepath, output_order="STCZYX")
    mdarray, mdata = read_tools.read_6darray(filepath)

    dim_array = misc.slicedim(mdarray, dimindex, posdim)
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
def test_get_planetable(czifile: str, csvfile: str, xstart: List[int], ystart: List[int]) -> None:
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
        planetable, savepath = misc.get_planetable(filepath, norm_time=True)

    planetable_filtered = misc.filter_planetable(planetable, planes={"scene":0, "time":0, "channel":0, "zplane":0})

    assert planetable_filtered["xstart"].values[0] == xstart[0]
    assert planetable_filtered["ystart"].values[0] == ystart[0]


@pytest.mark.parametrize("entry, set2value, result", [(0, 1, 0), (None, 1, 1), (-1, 2, -1), (None, 3, 3)])
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

    assert misc.check_dimsize(entry, set2value=set2value) == result


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
    minv, maxv = misc.calc_scaling(array, corr_min=corr_min, corr_max=corr_max)

    assert min_value == minv
    assert max_value == maxv


def test_norm_columns_min(df):
    result = misc.norm_columns(df, colname="Time [s]", mode="min")
    expected = pd.DataFrame({"Time [s]": [0, 1, 2, 3], "Value": [10, 20, 30, 40]})
    pd.testing.assert_frame_equal(result, expected)


def test_norm_columns_max(df):
    result = misc.norm_columns(df, colname="Time [s]", mode="max")
    expected = pd.DataFrame({"Time [s]": [-3, -2, -1, 0], "Value": [10, 20, 30, 40]})
    pd.testing.assert_frame_equal(result, expected)


def test_filter_planetable(planetable):

    # Test for scene index
    result = misc.filter_planetable(planetable, planes={"scene": 1})
    assert result["S"].eq(1).all(), "Scene index filter failed"

    # Test for time index
    result = misc.filter_planetable(planetable, planes={"time": 1})
    assert result["T"].eq(1).all(), "Time index filter failed"

    # Test for z-plane index
    result = misc.filter_planetable(planetable, planes={"zplane": 1})
    assert result["Z"].eq(1).all(), "Z-plane index filter failed"

    # Test for channel index
    result = misc.filter_planetable(planetable, planes={"channel": 5})
    assert result["C"].eq(5).all(), "Channel index filter failed"

    # Test for invalid indices
    result = misc.filter_planetable(planetable, planes={"scene": 2, "time": 2, "zplane": 2, "channel": 2})
    assert result.empty, "Invalid index filter failed"


@pytest.mark.parametrize(
    "input_dict, expected_dict",
    [
        ({"a": None, "b": 1, "c": [], "d": {}, "e": "test"}, {"b": 1, "e": "test"}),
        ({"a": [1, 2, 3], "b": {"c": [], "d": {}}}, {"a": [1, 2, 3]}),
        ({"a": None, "b": None}, {}),
        ({"a": [None, {}], "b": {"c": None}}, {"a": [None, {}]}),
        ({"a": [], "b": {}, "c": 0}, {"c": 0}),
    ],
)
def test_clean_dict(input_dict: Dict, expected_dict: Dict) -> None:
    result = misc.clean_dict(input_dict)
    assert result == expected_dict


@pytest.mark.parametrize(
    "czifile, planes, expected_columns",
    [
        (
            "WP96_4Pos_B4-10_DAPI.czi",
            {"scene": 0, "time": 0, "channel": 0, "zplane": 0},
            [
                "Subblock",
                "S",
                "M",
                "T",
                "C",
                "Z",
                "X[micron]",
                "Y[micron]",
                "Z[micron]",
                "Time[s]",
                "xstart",
                "ystart",
                "width",
                "height",
            ],
        ),
    ],
)
def test_get_planetable_columns(czifile: str, planes: dict[str, int], expected_columns: list) -> None:
    filepath = (basedir / "data" / czifile).as_posix()

    planetable, savepath = misc.get_planetable(filepath, planes=planes)
    assert list(planetable.columns) == expected_columns, "Column names do not match expected values."


@pytest.mark.parametrize(
    "czifile, planes, expected_shape",
    [
        (
            "WP96_4Pos_B4-10_DAPI.czi",
            {"scene": 0, "time": 0, "channel": 0, "zplane": 0},
            (1, 14),  # Example shape based on expected data
        ),
    ],
)
def test_get_planetable_shape(czifile: str, planes: dict[str, int], expected_shape: tuple) -> None:
    filepath = (basedir / "data" / czifile).as_posix()

    planetable, savepath = misc.get_planetable(filepath, planes=planes)
    assert planetable.shape == expected_shape, "DataFrame shape does not match expected values."


@pytest.mark.parametrize(
    "czifile, planes, norm_time, expected_min_time",
    [
        (
            "WP96_4Pos_B4-10_DAPI.czi",
            {"scene": 0, "time": 0, "channel": 0, "zplane": 0},
            True,
            0.0,  # Example normalized minimum time
        ),
    ],
)
def test_get_planetable_normalized_time(czifile: str, planes: dict[str, int], norm_time: bool, expected_min_time: float) -> None:
    filepath = (basedir / "data" / czifile).as_posix()

    planetable, savepath = misc.get_planetable(filepath, norm_time=norm_time, planes=planes)
    assert planetable["Time[s]"].min() == expected_min_time, "Normalized time does not match expected value."


@pytest.mark.parametrize(
    "czifile, planes, save_table, expected_saved_path",
    [
        (
            "WP96_4Pos_B4-10_DAPI.czi",
            {"scene": 0, "time": 0, "channel": 0, "zplane": 0},
            True,
            "WP96_4Pos_B4-10_DAPI_planetable.csv",  # Example saved file name
        ),
    ],
)
def test_get_planetable_save_table(czifile: str, planes: dict[str, int], save_table: bool, expected_saved_path: str) -> None:
    filepath = (basedir / "data" / czifile).as_posix()

    _, saved_path = misc.get_planetable(filepath, save_table=save_table, planes=planes)
    assert saved_path.endswith(expected_saved_path), "Saved file path does not match expected value."
