from czitools.metadata_tools import czi_metadata as czimd
from pathlib import Path
from box import BoxList
import pytest
from typing import List, Dict

basedir = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "czifile, results",
    [
        ("CellDivision_T3_Z5_CH2_X240_Y170.czi", {'X': 0.091, 'X_sf': 0.091, 'Y': 0.091, 'Y_sf': 0.091, 'Z': 0.32, 'ratio': {'xy': 1.0, 'zx': 3.516, 'zx_sf': 3.516}, 'unit': 'micron', 'zoom': 1.0}),
        ("Al2O3_SE_020_sp.czi", {'X': 0.028, 'X_sf': 0.028, 'Y': 0.028, 'Y_sf': 0.028, 'Z': 1.0, 'ratio': {'xy': 1.0, 'zx': 35.714, 'zx_sf': 35.714}, 'unit': 'micron', 'zoom': 1.0}),
        ("w96_A1+A2.czi", {'X': 0.457, 'X_sf': 0.457, 'Y': 0.457, 'Y_sf': 0.457, 'Z': 1.0, 'ratio': {'xy': 1.0, 'zx': 2.188, 'zx_sf': 2.188}, 'unit': 'micron', 'zoom': 1.0}),
        ("Airyscan.czi", {'X': 0.044, 'X_sf': 0.044, 'Y': 0.044, 'Y_sf': 0.044, 'Z': 1.0, 'ratio': {'xy': 1.0, 'zx': 22.727, 'zx_sf': 22.727}, 'unit': 'micron', 'zoom': 1.0}),
        ("newCZI_zloc.czi", {'X': 1.0, 'X_sf': 1.0, 'Y': 1.0, 'Y_sf': 1.0, 'Z': 1.0, 'ratio': {'xy': 1.0, 'zx': 1.0, 'zx_sf': 1.0}, 'unit': 'micron', 'zoom': 1.0}),
        ("FOV7_HV110_P0500510000.czi", {'X': 1.0, 'X_sf': 1.0, 'Y': 1.0, 'Y_sf': 1.0, 'Z': 1.0, 'ratio': {'xy': 1.0, 'zx': 1.0, 'zx_sf': 1.0}, 'unit': 'micron', 'zoom': 1.0}),
        ("Tumor_HE_RGB.czi", {'X': 0.22, 'X_sf': 0.22, 'Y': 0.22, 'Y_sf': 0.22, 'Z': 1.0, 'ratio': {'xy': 1.0, 'zx': 4.545, 'zx_sf': 4.545}, 'unit': 'micron', 'zoom': 1.0})
    ]
)
def test_scaling1(czifile: str, results: Dict) -> None:

    # get the filepath and the metadata_tools
    filepath = basedir / "data" / czifile
    czi_scaling = czimd.CziScaling(filepath)
    out = czi_scaling.__dict__

    del out['czisource']

    assert (out == results)


@pytest.mark.parametrize(
    "dist, expected",
    [
        (
            BoxList([{'Value': 1.0, 'Id': 'X'},
                     {'Value': 2.0, 'Id': 'Y'},
                     {'Value': 3.0, 'Id': 'Z'}]),
            [1000000.0, 2000000.0, 3000000.0]
        ),

        (
            BoxList([{'Value': 1.0, 'Id': 'X'},
                     {'Value': None, 'Id': 'Y'},
                     {'Value': 3.0, 'Id': None}]),
            [1000000.0, 1.0, 3000000.0]
        ),

        (
            BoxList([{'Value': 1.0, 'Id': 'X'},
                    {'Value': None, 'Id': 'Y'},
                    {'Value': -0.0, 'Id': None}]),
            [1000000.0, 1.0, 1.0]
        ),

        (
            BoxList([{'Value': 1.0, 'Id': 'X'},
                    {'Valuuue': None, 'Id': 'Y'},
                    {'Value': -0.0, 'Id': None}]),
            [1000000.0, 1.0, 1.0]
        )
    ]
)
def test_safe_get_scale(dist: BoxList, expected: List[float]) -> None:

    assert(czimd.CziScaling.safe_get_scale(dist, 0) == expected[0])
    assert(czimd.CziScaling.safe_get_scale(dist, 1) == expected[1])
    assert(czimd.CziScaling.safe_get_scale(dist, 2) == expected[2])


@pytest.mark.parametrize(
    "czifile, x, y, z, ratio",
    [
        ("DAPI_GFP.czi", 1.0, 1.0, 1.0, {'xy': 1.0, 'zx': 1.0, 'zx_sf': 1.0})
    ]
)
def test_scaling2(czifile: str, x: float, y: float, z: float, ratio: Dict[str, float]) -> None:

    # get the CZI filepath
    filepath = basedir / "data" / czifile
    md = czimd.CziMetadata(filepath)

    assert(md.scale.X == x)
    assert(md.scale.Y == y)
    assert(md.scale.Z == z)
    assert(md.scale.ratio == ratio)

    scaling = czimd.CziScaling(filepath)
    assert(scaling.X == x)
    assert(scaling.Y == y)
    assert(scaling.Z == z)
    assert(scaling.ratio == ratio)
