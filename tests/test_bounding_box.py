from czimetadata_tools import pylibczirw_metadata as czimd
import os
from pathlib import Path
from pylibCZIrw import czi

basedir = Path(__file__).resolve().parents[1]

# get the CZI filepath
filepath = os.path.join(basedir, r"data/w96_A1+A2.czi")


def test_scaling():

    czi_bbox = czimd.CziBoundingBox(filepath)

    print("BBox - all_scenes: ", czi_bbox.all_scenes)
    print("BBox - total_bounding_box: ", czi_bbox.total_bounding_box)
    print("BBox - total_rect: ", czi_bbox.total_rect)

    assert (czi_bbox.all_scenes[0] == czi.Rectangle(x=0, y=0, w=1960, h=1416))
    assert (czi_bbox.all_scenes[1] == czi.Rectangle(x=19758, y=24, w=1960, h=1416))
    assert (czi_bbox.total_bounding_box == {'T': (0, 1),
                                            'Z': (0, 1),
                                            'C': (0, 2),
                                            'B': (0, 1),
                                            'X': (0, 21718),
                                            'Y': (0, 1440)})
    assert (czi_bbox.total_rect == czi.Rectangle(x=0, y=0, w=21718, h=1440))
