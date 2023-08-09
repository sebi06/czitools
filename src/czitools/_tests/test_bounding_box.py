from czitools import metadata_tools as czimd
from pathlib import Path
from pylibCZIrw import czi
import pytest
from typing import List, Dict, Tuple, Optional, Type, Any, Union, Mapping

basedir = Path(__file__).resolve().parents[3] / "data"


@pytest.mark.parametrize(
    "czifile, rect, tbox, trect",
    [
        ("w96_A1+A2.czi",
         [czi.Rectangle(x=0, y=0, w=1960, h=1416),
          czi.Rectangle(x=19758, y=24, w=1960, h=1416)],
         {'T': (0, 1),
          'Z': (0, 1),
          'C': (0, 2),
          'B': (0, 1),
          'X': (0, 21718),
          'Y': (0, 1440)},
         czi.Rectangle(x=0, y=0, w=21718, h=1440)
         ),

        ("newCZI_zloc.czi",
         [czi.Rectangle(x=0, y=0, w=8192, h=512)],
         {'T': (0, 1),
          'Z': (0, 1),
          'C': (0, 1),
          'X': (0, 8192),
          'Y': (0, 512)},
         czi.Rectangle(x=0, y=0, w=8192, h=512)
         )
    ]
)
def test_bounding_box(czifile: str,
                      rect: czi.Rectangle,
                      tbox: Dict[str, Tuple[int, int]],
                      trect: czi.Rectangle) -> None:
    filepath = basedir / czifile

    # get the bounding box
    czi_bbox = czimd.CziBoundingBox(filepath)

    for r in range(len(rect)):
        assert (czi_bbox.scenes_bounding_rect[r] == rect[r])

    assert (czi_bbox.total_bounding_box == tbox)
    assert (czi_bbox.total_rect == trect)
