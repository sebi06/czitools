
from czitools import pylibczirw_metadata as czimd
import os
from pathlib import Path

basedir = Path(__file__).resolve().parents[3]

# get the CZI filepath
filepath = os.path.join(
    basedir, r"data/CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi")


def test_scaling():

    czi_scaling = czimd.CziScaling(filepath)

    print("Scaling - Unit: ", czi_scaling.Unit)
    print("Scaling - X: ", czi_scaling.X)
    print("Scaling - Y: ", czi_scaling.Y)
    print("Scaling - Z: ", czi_scaling.Z)
    print("Scaling - Ration: ", czi_scaling.ratio)

    assert (czi_scaling.Unit == "micron")
    assert (czi_scaling.X == 0.09057667415221031)
    assert (czi_scaling.Y == 0.09057667415221031)
    assert (czi_scaling.Z == 0.32)
    assert (czi_scaling.ratio["xy"] == 1.0)
    assert (czi_scaling.ratio["zx"] == 3.533)
