
from czitools import pylibczirw_metadata as czimd
from pathlib import Path

basedir = Path(__file__).resolve().parents[3]

# get the CZI filepath
filepath = basedir / r"data/CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi"


def test_dimensions():

    czi_dimensions = czimd.CziDimensions(filepath)
    print("SizeS: ", czi_dimensions.SizeS)
    print("SizeT: ", czi_dimensions.SizeT)
    print("SizeZ: ", czi_dimensions.SizeZ)
    print("SizeC: ", czi_dimensions.SizeC)
    print("SizeY: ", czi_dimensions.SizeY)
    print("SizeX: ", czi_dimensions.SizeX)

    assert (czi_dimensions.SizeS is None)
    assert (czi_dimensions.SizeT == 3)
    assert (czi_dimensions.SizeZ == 5)
    assert (czi_dimensions.SizeC == 2)
    assert (czi_dimensions.SizeY == 170)
    assert (czi_dimensions.SizeX == 240)
