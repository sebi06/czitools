from czitools import read_tools
from pathlib import Path
import dask.array as da
import numpy as np
import pytest
from pylibCZIrw import czi as pyczi
from typing import List, Dict, Tuple, Optional, Type, Any, Union, Mapping


basedir = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "czifile, dimorder, dimstring, shape, use_dask",
    [
        ("w96_A1+A2.czi", "STCZYX", "STCZYX", (2, 1, 2, 1, 1416, 1960), False),
        ("w96_A1+A2.czi", "STZCYX", "STZCYX", (2, 1, 1, 2, 1416, 1960), False),
        ("S=3_1Pos_2Mosaic_T=2=Z=3_CH=2_sm.czi", "STCZYX", "", AttributeError, False),
        ("S=2_3x3_CH=2.czi", "STZCYX", "STZCYX", (2, 1, 1, 2, 1792, 1792), False),
        ("S=2_3x3_CH=2.czi", "STCZYX", "STCZYX", (2, 1, 2, 1, 1792, 1792), False),
        (
            "FOV7_HV110_P0500510000.czi",
            "STZCYX",
            "STZCYX",
            (1, 1, 1, 1, 512, 512),
            False,
        ),
        ("newCZI_compressed.czi", "STZCYX", "STZCYX", (1, 1, 1, 1, 512, 512), False),
        ("w96_A1+A2.czi", "STCZYX", "STCZYX", (2, 1, 2, 1, 1416, 1960), True),
        ("w96_A1+A2.czi", "STZCYX", "STZCYX", (2, 1, 1, 2, 1416, 1960), True),
        ("S=3_1Pos_2Mosaic_T=2=Z=3_CH=2_sm.czi", "STCZYX", "", AttributeError, True),
        ("S=2_3x3_CH=2.czi", "STZCYX", "STZCYX", (2, 1, 1, 2, 1792, 1792), True),
        ("S=2_3x3_CH=2.czi", "STCZYX", "STCZYX", (2, 1, 2, 1, 1792, 1792), True),
        (
            "FOV7_HV110_P0500510000.czi",
            "STZCYX",
            "STZCYX",
            (1, 1, 1, 1, 512, 512),
            True,
        ),
        ("newCZI_compressed.czi", "STZCYX", "STZCYX", (1, 1, 1, 1, 512, 512), True),
    ],
)
def test_read_mdarray_1(
    czifile: str,
    dimorder: str,
    dimstring: str,
    shape: Optional[Tuple[int]],
    use_dask: bool,
) -> None:
    # get the CZI filepath
    filepath = basedir / "data" / czifile

    mdarray, mdata, ds = read_tools.read_6darray(
        filepath, use_dask=use_dask, output_order=dimorder
    )

    assert ds == dimstring

    if type(shape) == type and issubclass(shape, Exception):
        with pytest.raises(shape):
            mdarray.shape
    else:
        assert mdarray.shape == shape


@pytest.mark.parametrize(
    "czifile, dimorder, dimstring, shape, chunk_zyx",
    [
        ("w96_A1+A2.czi", "STCZYX", "STCZYX", (2, 1, 2, 1, 1416, 1960), False),
        ("w96_A1+A2.czi", "STZCYX", "STZCYX", (2, 1, 1, 2, 1416, 1960), False),
        ("S=3_1Pos_2Mosaic_T=2=Z=3_CH=2_sm.czi", "STCZYX", "", AttributeError, False),
        ("S=2_3x3_CH=2.czi", "STZCYX", "STZCYX", (2, 1, 1, 2, 1792, 1792), False),
        ("S=2_3x3_CH=2.czi", "STCZYX", "STCZYX", (2, 1, 2, 1, 1792, 1792), False),
        (
            "FOV7_HV110_P0500510000.czi",
            "STZCYX",
            "STZCYX",
            (1, 1, 1, 1, 512, 512),
            False,
        ),
        ("newCZI_compressed.czi", "STZCYX", "STZCYX", (1, 1, 1, 1, 512, 512), False),
        ("w96_A1+A2.czi", "STCZYX", "STCZYX", (2, 1, 2, 1, 1416, 1960), True),
        ("w96_A1+A2.czi", "STZCYX", "STZCYX", (2, 1, 1, 2, 1416, 1960), True),
        ("S=3_1Pos_2Mosaic_T=2=Z=3_CH=2_sm.czi", "STCZYX", "", AttributeError, True),
        ("S=2_3x3_CH=2.czi", "STZCYX", "STZCYX", (2, 1, 1, 2, 1792, 1792), True),
        ("S=2_3x3_CH=2.czi", "STCZYX", "STCZYX", (2, 1, 2, 1, 1792, 1792), True),
        (
            "FOV7_HV110_P0500510000.czi",
            "STZCYX",
            "STZCYX",
            (1, 1, 1, 1, 512, 512),
            True,
        ),
        ("newCZI_compressed.czi", "STZCYX", "STZCYX", (1, 1, 1, 1, 512, 512), True),
    ],
)
def test_read_mdarray_2(
    czifile: str,
    dimorder: str,
    dimstring: str,
    shape: Optional[Tuple[int]],
    chunk_zyx: bool,
) -> None:
    # get the CZI filepath
    filepath = basedir / "data" / czifile

    mdarray, mdata, ds = read_tools.read_6darray_lazy(
        filepath, chunk_zyx=chunk_zyx, output_order=dimorder
    )

    assert ds == dimstring

    if type(shape) == type and issubclass(shape, Exception):
        with pytest.raises(shape):
            mdarray.shape
    else:
        assert mdarray.shape == shape


@pytest.mark.parametrize(
    "czifile, dimorder, dimstring, shape, size_s, planes",
    [
        (
            "w96_A1+A2.czi",
            "STCZYX",
            "STCZYX",
            (1, 1, 2, 1, 1416, 1960),
            1,
            {"S": (0, 0)},
        ),
        (
            "CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi",
            "STZCYX",
            "STZCYX",
            (1, 1, 1, 2, 170, 240),
            None,
            {"S": (0, 0), "T": (0, 0), "C": (0, 1), "Z": (0, 0)},
        ),
        (
            "CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi",
            "STZCYX",
            "STZCYX",
            (1, 1, 5, 2, 170, 240),
            None,
            {"T": (0, 0)},
        ),
        (
            "CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi",
            "STZCYX",
            "STZCYX",
            (1, 3, 5, 1, 170, 240),
            None,
            {"C": (0, 0)},
        ),
    ],
)
def test_read_mdarray_substack(
    czifile: str,
    dimorder: str,
    dimstring: str,
    shape: Tuple[int],
    size_s: Optional[int],
    planes: Dict[str, tuple[int, int]],
) -> None:
    # get the CZI filepath
    filepath = basedir / "data" / czifile

    # read only a specific scene from the CZI
    mdarray, mdata, ds = read_tools.read_6darray(
        filepath, output_order=dimorder, planes=planes
    )

    assert ds == dimstring
    assert mdarray.shape == shape
    assert mdata.image.SizeS == size_s


@pytest.mark.parametrize(
    "czifile, scene, has_scenes",
    [
        ("CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi", 0, False),
        ("S=2_3x3_CH=2.czi", 0, True),
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
            image2d_3 = czidoc.read(
                plane={"T": 0, "Z": 0, "C": 0, "S": scene}, scene=scene
            )
            image2d_4 = czidoc.read(plane={"T": 0, "Z": 0, "C": 0}, scene=scene)

            # TODO: This has to be checked why this gives different results

            image2d_5 = czidoc.read(plane={"T": 0, "Z": 0, "C": 0, "S": scene})

            assert np.array_equal(image2d_3, image2d_4) is True
            # assert (np.array_equal(image2d_3, image2d_5) is True)
            # assert (np.array_equal(image2d_4, image2d_5) is True)
