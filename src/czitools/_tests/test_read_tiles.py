from czitools.read_tools import read_tools
from pathlib import Path
import pytest
from typing import Dict, Tuple, Optional

basedir = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "czifile, scene, tile, substacks, shape",
    [
        ("S2_3x3_CH2.czi", 0, 0, {"C": 0}, (1, 1, 1, 1, 1, 640, 640)),
        ("S2_3x3_CH2.czi", 0, 0, {}, (1, 1, 1, 2, 1, 640, 640)),
        ("CellDivision_T3_Z5_CH2_X240_Y170.czi", 0, 0, {}, (1, 3, 2, 5, 170, 240)),
        ("CellDivision_T3_Z5_CH2_X240_Y170.czi", 0, 0, {"C": 0}, (1, 3, 1, 5, 170, 240)),
        ("CellDivision_T3_Z5_CH2_X240_Y170.czi", 0, 0, {"C": 0, "Z": 1}, (1, 3, 1, 1, 170, 240)),
        ("CellDivision_T3_Z5_CH2_X240_Y170.czi", 0, 0, {"T": 0, "C": 0, "Z": 1}, (1, 1, 1, 1, 170, 240)),
        ("S3_1Pos_2Mosaic_T2_Z3_CH2_sm.czi", 0, 0, {}, (1, 1, 2, 2, 3, 64, 64)),
        ("S3_1Pos_2Mosaic_T2_Z3_CH2_sm.czi", 1, 0, {"C": 0}, (1, 1, 2, 1, 3, 64, 64)),
        ("2_tileregions.czi", 1, 0, {}, (1, 1, 300, 400))
    ]
)
def test_read_tiles(
    czifile: str, scene: int, tile: int, substacks: Optional[Dict[str, int]], shape: Tuple[int]
) -> None:
    # get the CZI filepath
    filepath = basedir / "data" / czifile

    tile_stack, size = read_tools.read_tiles(filepath, scene=scene, tile=tile, **substacks)

    assert tile_stack.shape == shape
