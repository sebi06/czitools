from czitools.metadata_tools import czi_metadata as czimd
from pathlib import Path
import pytest
from typing import List, Any

basedir = Path(__file__).resolve().parents[3]

@pytest.mark.parametrize(
    "czifile, dimension",
    [
        ("CellDivision_T3_Z5_CH2_X240_Y170.czi", [None, 3, 5, 2, 170, 240])
    ]
)
def test_dimensions(czifile: str, dimension: List[Any]) -> None:

    filepath = basedir / "data" / czifile

    czi_dimensions = czimd.CziDimensions(filepath)
    print("SizeS: ", czi_dimensions.SizeS)
    print("SizeT: ", czi_dimensions.SizeT)
    print("SizeZ: ", czi_dimensions.SizeZ)
    print("SizeC: ", czi_dimensions.SizeC)
    print("SizeY: ", czi_dimensions.SizeY)
    print("SizeX: ", czi_dimensions.SizeX)

    assert (czi_dimensions.SizeS == dimension[0])
    assert (czi_dimensions.SizeT == dimension[1])
    assert (czi_dimensions.SizeZ == dimension[2])
    assert (czi_dimensions.SizeC == dimension[3])
    assert (czi_dimensions.SizeY == dimension[4])
    assert (czi_dimensions.SizeX == dimension[5])
