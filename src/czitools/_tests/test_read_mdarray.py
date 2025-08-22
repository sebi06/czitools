from czitools.read_tools import read_tools
from pathlib import Path
import dask.array as da
import numpy as np
import pytest
from pylibCZIrw import czi as pyczi
from typing import List, Dict, Tuple, Optional, Type, Any, Union, Mapping


basedir = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "czifile, shape, use_dask, chunk_zyx",
    [
        ("w96_A1+A2.czi", (2, 1, 2, 1, 1416, 1960), False, False),
        ("w96_A1+A2.czi", (2, 1, 2, 1, 1416, 1960), True, False),
        ("w96_A1+A2.czi", (2, 1, 2, 1, 1416, 1960), True, True),
        ("S3_1Pos_2Mosaic_T1_Z1_CH1.czi", AttributeError, False, False),
        ("S2_3x3_CH2.czi", (2, 1, 2, 1, 1792, 1792), False, False),
        ("S2_3x3_CH2.czi", (2, 1, 2, 1, 1792, 1792), True, False),
        ("S2_3x3_CH2.czi", (2, 1, 2, 1, 1792, 1792), True, True),
        (
            "FOV7_HV110_P0500510000.czi",
            (1, 1, 1, 1, 512, 512),
            False,
            False,
        ),
        (
            "FOV7_HV110_P0500510000.czi",
            (1, 1, 1, 1, 512, 512),
            False,
            True,
        ),
        (
            "FOV7_HV110_P0500510000.czi",
            (1, 1, 1, 1, 512, 512),
            True,
            True,
        ),
        ("newCZI_compressed.czi", (1, 1, 1, 1, 512, 512), False, False),
        ("newCZI_compressed.czi", (1, 1, 1, 1, 512, 512), False, True),
        ("newCZI_compressed.czi", (1, 1, 1, 1, 512, 512), True, True),
    ],
)
def test_read_mdarray(czifile: str, shape: Optional[Tuple[int]], use_dask: bool, chunk_zyx: bool) -> None:
    """
    Test the reading of a multidimensional array from a CZI file.
    Parameters:
    czifile (str): The name of the CZI file to read.
    shape (Optional[Tuple[int]]): The expected shape of the multidimensional array.
                                  If an exception type is provided, the test will check
                                  if the exception is raised.
    use_dask (bool): Whether to use Dask for reading the array.
    chunk_zyx (bool): Whether to chunk the array along the Z, Y, and X dimensions.
    Returns:
    None
    """

    # get the CZI filepath
    filepath = basedir / "data" / czifile

    mdarray, mdata = read_tools.read_6darray(filepath, use_dask=use_dask, chunk_zyx=chunk_zyx)

    if type(shape) == type and issubclass(shape, Exception):
        with pytest.raises(shape):
            mdarray.shape
    else:
        assert mdarray.shape == shape


@pytest.mark.parametrize(
    "czifile, shape, size_s, planes",
    [
        (
            "w96_A1+A2.czi",
            (1, 1, 2, 1, 1416, 1960),
            1,
            {"S": (0, 0)},
        ),
        (
            "CellDivision_T3_Z5_CH2_X240_Y170.czi",
            (1, 1, 2, 1, 170, 240),
            None,
            {"S": (0, 0), "T": (0, 0), "C": (0, 1), "Z": (0, 0)},
        ),
        (
            "CellDivision_T3_Z5_CH2_X240_Y170.czi",
            (1, 1, 2, 5, 170, 240),
            None,
            {"T": (0, 0)},
        ),
        (
            "CellDivision_T3_Z5_CH2_X240_Y170.czi",
            (1, 3, 1, 5, 170, 240),
            None,
            {"C": (0, 0)},
        ),
    ],
)
def test_read_mdarray_substack(
    czifile: str,
    shape: Tuple[int],
    size_s: Optional[int],
    planes: Dict[str, tuple[int, int]],
) -> None:
    # get the CZI filepath
    filepath = basedir / "data" / czifile

    # read only a specific scene from the CZI
    mdarray, mdata = read_tools.read_6darray(filepath, planes=planes, adapt_metadata=False)

    assert mdarray.shape == shape
    assert mdata.image.SizeS == size_s


@pytest.mark.parametrize(
    "czifile, scene, has_scenes",
    [
        ("CellDivision_T3_Z5_CH2_X240_Y170.czi", 0, False),
        ("S2_3x3_CH2.czi", 0, True),
    ],
)
def test_readczi_scenes(czifile: str, scene: int, has_scenes: bool) -> None:
    # get the CZI filepath
    filepath = basedir / "data" / czifile

    # open the CZI document to read the
    with pyczi.open_czi(str(filepath)) as czidoc:
        # read without providing a scene index
        image2d_1 = czidoc.read(plane={"T": 0, "Z": 0, "C": 0})
        image2d_2 = czidoc.read(plane={"T": 0, "Z": 0, "C": 0, "S": scene})

        assert np.array_equal(image2d_1, image2d_2) is True

        if has_scenes:
            image2d_3 = czidoc.read(plane={"T": 0, "Z": 0, "C": 0, "S": scene}, scene=scene)
            image2d_4 = czidoc.read(plane={"T": 0, "Z": 0, "C": 0}, scene=scene)

            assert np.array_equal(image2d_3, image2d_4) is True


@pytest.mark.parametrize(
    "czifile, planes2read, sub_planes_expected, shape",
    [
        (
            "CellDivision_T3_Z5_CH2_X240_Y170.czi",
            None,
            {"S": (0, 0), "T": (0, 2), "C": (0, 1), "Z": (0, 4)},
            (1, 3, 2, 5, 170, 240),
        ),
        (
            "CellDivision_T3_Z5_CH2_X240_Y170.czi",
            {"S": (0, 0), "T": (0, 2), "C": (0, 1)},
            {"S": (0, 0), "T": (0, 2), "C": (0, 1), "Z": (0, 4)},
            (1, 3, 2, 5, 170, 240),
        ),
        (
            "CellDivision_T3_Z5_CH2_X240_Y170.czi",
            {"T": (0, 1), "C": (1, 1), "Z": (3, 4)},
            {"S": (0, 0), "T": (0, 1), "C": (1, 1), "Z": (3, 4)},
            (1, 2, 1, 2, 170, 240),
        ),
        (
            "CellDivision_T3_Z5_CH2_X240_Y170.czi",
            {"T": (0, 1), "Z": (3, 4)},
            {"S": (0, 0), "T": (0, 1), "C": (0, 1), "Z": (3, 4)},
            (1, 2, 2, 2, 170, 240),
        ),
    ],
)
def test_readczi_planes(
    czifile: str,
    planes2read: Dict[str, tuple[int, int]],
    sub_planes_expected: Dict[str, tuple[int, int]],
    shape: tuple[int, int, int, int, int, int],
) -> None:
    # get the CZI filepath
    filepath = basedir / "data" / czifile

    # read with planes parameter and check output planes
    mdarray, mdata = read_tools.read_6darray(filepath, planes=planes2read, adapt_metadata=True)

    assert mdarray.attrs["subset_planes"] == sub_planes_expected
    assert mdarray.shape == shape
    assert mdata.image.SizeS == shape[0]
    assert mdata.image.SizeT == shape[1]
    assert mdata.image.SizeC == shape[2]
    assert mdata.image.SizeZ == shape[3]
