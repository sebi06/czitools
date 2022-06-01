from pylibCZIrw import czi as pyczi
from czimetadata_tools import pylibczirw_metadata as czimd
from czimetadata_tools import misc
import os
from pathlib import Path

basedir = Path(__file__).resolve().parents[1]

# get the CZI filepath
filepath = os.path.join(
    basedir, r"data/CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi")


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
