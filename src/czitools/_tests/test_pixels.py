import pytest
from czitools.utils import pixels
import numpy as np


@pytest.mark.parametrize(
    "pixeltype, expected_dtype, expected_maxvalue",
    [
        ("gray16", np.dtype(np.uint16), 65535),
        ("Gray16", np.dtype(np.uint16), 65535),
        ("gray8", np.dtype(np.uint8), 255),
        ("Gray8", np.dtype(np.uint8), 255),
        ("bgr48", np.dtype(np.uint16), 65535),
        ("Bgr48", np.dtype(np.uint16), 65535),
        ("bgr24", np.dtype(np.uint8), 255),
        ("Bgr24", np.dtype(np.uint8), 255),
        ("bgr96float", np.dtype(np.uint16), 65535),
        ("Bgr96Float", np.dtype(np.uint16), 65535),
        ("unknown", None, None),
    ],
)
def test_get_dtype_fromstring(pixeltype, expected_dtype, expected_maxvalue):
    dtype, maxvalue = pixels.get_dtype_fromstring(pixeltype)
    assert dtype == expected_dtype
    assert maxvalue == expected_maxvalue


@pytest.mark.parametrize(
    "pixeltypes, expected_is_rgb, expected_is_consistent",
    [
        (
            {1: "Bgr24", 2: "Bgr24", 3: "Bgr24"},
            {1: True, 2: True, 3: True},
            True,
        ),
        (
            {1: "Bgr24", 2: "Gray8", 3: "Bgr24"},
            {1: True, 2: False, 3: True},
            False,
        ),
        (
            {1: "Gray8", 2: "Gray8", 3: "Gray8"},
            {1: False, 2: False, 3: False},
            True,
        ),
        (
            {1: "Bgr24", 2: "Bgr48", 3: "Bgr96Float"},
            {1: True, 2: True, 3: True},
            False,
        ),
        (
            {1: "Gray8", 2: "Gray16", 3: "Gray8"},
            {1: False, 2: False, 3: False},
            False,
        ),
    ],
)
def test_check_if_rgb(pixeltypes, expected_is_rgb, expected_is_consistent):
    is_rgb, is_consistent = pixels.check_if_rgb(pixeltypes)
    assert is_rgb == expected_is_rgb
    assert is_consistent == expected_is_consistent


@pytest.mark.parametrize(
    "dim_string, expected_dims_dict, expected_dimindex_list, expected_numvalid_dims",
    [
        (
            "RIMHVBS",
            {
                "R": 0,
                "I": 1,
                "M": 2,
                "H": 3,
                "V": 4,
                "B": 5,
                "S": 6,
                "T": -1,
                "C": -1,
                "Z": -1,
                "Y": -1,
                "X": -1,
                "A": -1,
            },
            [0, 1, 2, 3, 4, 5, 6, -1, -1, -1, -1, -1, -1],
            7,
        ),
        (
            "TXYZ",
            {
                "R": -1,
                "I": -1,
                "M": -1,
                "H": -1,
                "V": -1,
                "B": -1,
                "S": -1,
                "T": 0,
                "C": -1,
                "Z": 3,
                "Y": 2,
                "X": 1,
                "A": -1,
            },
            [-1, -1, -1, -1, -1, -1, -1, 0, -1, 3, 2, 1, -1],
            4,
        ),
        (
            "ABC",
            {
                "R": -1,
                "I": -1,
                "M": -1,
                "H": -1,
                "V": -1,
                "B": 1,
                "S": -1,
                "T": -1,
                "C": 2,
                "Z": -1,
                "Y": -1,
                "X": -1,
                "A": 0,
            },
            [-1, -1, -1, -1, -1, 1, -1, -1, 2, -1, -1, -1, 0],
            3,
        ),
        (
            "",
            {
                "R": -1,
                "I": -1,
                "M": -1,
                "H": -1,
                "V": -1,
                "B": -1,
                "S": -1,
                "T": -1,
                "C": -1,
                "Z": -1,
                "Y": -1,
                "X": -1,
                "A": -1,
            },
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            0,
        ),
    ],
)
def test_get_dimorder(
    dim_string, expected_dims_dict, expected_dimindex_list, expected_numvalid_dims
):
    dims_dict, dimindex_list, numvalid_dims = pixels.get_dimorder(dim_string)
    assert dims_dict == expected_dims_dict
    assert dimindex_list == expected_dimindex_list
    assert numvalid_dims == expected_numvalid_dims
