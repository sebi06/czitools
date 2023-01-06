
from czitools import pylibczirw_metadata as czimd
from pathlib import Path
from box import Box, BoxList

basedir = Path(__file__).resolve().parents[3]

# get the CZI filepath
filepath = basedir / r"data/CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi"


def test_scaling1():

    to_test = {0: r"data/CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi",
            1: r"data/Al2O3_SE_020_sp.czi",
            2: r"data/w96_A1+A2.czi",
            3: r"data/Airyscan.czi",
            4: r"data/newCZI_zloc.czi",
            5: r"data/FOV7_HV110_P0500510000.czi",
            6: r"data/Tumor_HE_RGB.czi"
            }

    results = {0: {'X': 0.09057667415221031, 'Y': 0.09057667415221031, 'Z': 0.32, 'ratio': {'xy': 1.0, 'zx': 3.533}, 'unit': 'micron'},
               1: {'X': 0.02791, 'Y': 0.02791, 'Z': 1.0, 'ratio': {'xy': 1.0, 'zx': 35.829}, 'unit': 'micron'},
               2: {'X': 0.45715068250381635, 'Y': 0.45715068250381635, 'Z': 1.0, 'ratio': {'xy': 1.0, 'zx': 2.187}, 'unit': 'micron'},
               3: {'X': 0.04392940851660524, 'Y': 0.04392940851660524, 'Z': 1.0, 'ratio': {'xy': 1.0, 'zx': 22.764}, 'unit': 'micron'},
               4: {'X': 1.0, 'Y': 1.0, 'Z': 1.0, 'ratio': {'xy': 1.0, 'zx': 1.0}, 'unit': 'micron'},
               5: {'X': 1.0, 'Y': 1.0, 'Z': 1.0, 'ratio': {'xy': 1.0, 'zx': 1.0}, 'unit': 'micron'},
               6: {'X': 0.21999999999999997, 'Y': 0.21999999999999997, 'Z': 1.0, 'ratio': {'xy': 1.0, 'zx': 4.545}, 'unit': 'micron'},
               }

    for t in range(len(to_test)):

        # get the filepath and the metadata
        filepath = basedir / to_test[t]
        czi_scaling = czimd.CziScaling(filepath)
        out = czi_scaling.__dict__
        
        del out['filepath']

        assert(out == results[t])

def test_safe_get_scale():

    dist = BoxList([{'Value': 1.0, 'Id': 'X'},
                     {'Value': 2.0, 'Id': 'Y'},
                     {'Value': 3.0, 'Id': 'Z'}])

    assert(czimd.CziScaling.safe_get_scale(dist, 0) == 1000000.0)
    assert(czimd.CziScaling.safe_get_scale(dist, 1) == 2000000.0)
    assert(czimd.CziScaling.safe_get_scale(dist, 2) == 3000000.0)

    dist = BoxList([{'Value': 1.0, 'Id': 'X'},
                    {'Value': None, 'Id': 'Y'},
                    {'Value': 3.0, 'Id': None}])

    assert(czimd.CziScaling.safe_get_scale(dist, 0) == 1000000.0)
    assert(czimd.CziScaling.safe_get_scale(dist, 1) == 1.0)
    assert(czimd.CziScaling.safe_get_scale(dist, 2) == 3000000.0)

    dist = BoxList([{'Value': 1.0, 'Id': 'X'},
                    {'Value': None, 'Id': 'Y'},
                    {'Value': -0.0, 'Id': None}])

    assert(czimd.CziScaling.safe_get_scale(dist, 0) == 1000000.0)
    assert(czimd.CziScaling.safe_get_scale(dist, 1) == 1.0)
    assert(czimd.CziScaling.safe_get_scale(dist, 2) == 1.0)

    dist = BoxList([{'Value': 1.0, 'Id': 'X'},
                    {'Valuuue': None, 'Id': 'Y'},
                    {'Value': -0.0, 'Id': None}])

    assert(czimd.CziScaling.safe_get_scale(dist, 0) == 1000000.0)
    assert(czimd.CziScaling.safe_get_scale(dist, 1) == 1.0)
    assert(czimd.CziScaling.safe_get_scale(dist, 2) == 1.0)


def test_scaling2():

    # get the CZI filepath
    filepath = basedir / r"data/DAPI_GFP.czi"
    md = czimd.CziMetadata(filepath)

    assert(md.scale.X == 1.0)
    assert(md.scale.Y == 1.0)
    assert(md.scale.Z == 1.0)
    assert(md.scale.ratio == {'xy': 1.0, 'zx': 1.0})

    scaling = czimd.CziScaling(filepath)
    assert(scaling.X == 1.0)
    assert(scaling.Y == 1.0)
    assert(scaling.Z == 1.0)
    assert(scaling.ratio == {'xy': 1.0, 'zx': 1.0})
