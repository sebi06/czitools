from czitools import metadata_tools as czimd
from pathlib import Path
from box import Box, BoxList
import pytest
from typing import List, Dict, Tuple, Optional, Type, Any, Union, Mapping

basedir = Path(__file__).resolve().parents[3]

@pytest.mark.parametrize(
    "czifile, results",
    [
        ("CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi", {'X': 0.09057667415221031, 'Y': 0.09057667415221031, 'Z': 0.32, 'ratio': {'xy': 1.0, 'zx': 3.533}, 'unit': 'micron'}),
        ("Al2O3_SE_020_sp.czi", {'X': 0.02791, 'Y': 0.02791, 'Z': 1.0, 'ratio': {'xy': 1.0, 'zx': 35.829}, 'unit': 'micron'}),
        ("w96_A1+A2.czi", {'X': 0.45715068250381635, 'Y': 0.45715068250381635, 'Z': 1.0, 'ratio': {'xy': 1.0, 'zx': 2.187}, 'unit': 'micron'}),
        ("Airyscan.czi", {'X': 0.04392940851660524, 'Y': 0.04392940851660524, 'Z': 1.0, 'ratio': {'xy': 1.0, 'zx': 22.764}, 'unit': 'micron'}),
        ("newCZI_zloc.czi", {'X': 1.0, 'Y': 1.0, 'Z': 1.0, 'ratio': {'xy': 1.0, 'zx': 1.0}, 'unit': 'micron'}),
        ("FOV7_HV110_P0500510000.czi", {'X': 1.0, 'Y': 1.0, 'Z': 1.0, 'ratio': {'xy': 1.0, 'zx': 1.0}, 'unit': 'micron'}),
        ("Tumor_HE_RGB.czi", {'X': 0.21999999999999997, 'Y': 0.21999999999999997, 'Z': 1.0, 'ratio': {'xy': 1.0, 'zx': 4.545}, 'unit': 'micron'})
    ]
)
def test_scaling1(czifile: str, results: Dict) -> None:

    # get the filepath and the metadata
    filepath = basedir / "data" / czifile
    czi_scaling = czimd.CziScaling(filepath)
    out = czi_scaling.__dict__

    del out['czisource']

    assert(out == results)


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
        ("DAPI_GFP.czi", 1.0, 1.0, 1.0, {'xy': 1.0, 'zx': 1.0})
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
