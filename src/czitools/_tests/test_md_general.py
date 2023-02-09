
from czitools import pylibczirw_metadata as czimd
from pylibCZIrw import czi as pyczi
from pathlib import Path
import numpy as np
import pytest
from typing import List, Dict, Tuple, Optional, Type, Any, Union, Mapping


basedir = Path(__file__).resolve().parents[3]


def test_pixeltypes_1() -> None:

    # get the CZI filepath
    filepath = basedir / r"data/Tumor_HE_RGB.czi"
    md = czimd.CziMetadata(filepath)

    print("PixelTypes: ", md.pixeltypes)
    print("NumPy dtype: ", md.npdtype)
    print("Max. Pixel Value: ", md.maxvalue)

    assert (md.pixeltypes == {0: 'Bgr24'})
    assert (md.isRGB is True)


@pytest.mark.parametrize(
    "pts, dts, mvs",
    [
        ("gray16", np.dtype(np.uint16), 65535),
        ("Gray16", np.dtype(np.uint16), 65535),
        ("gray8", np.dtype(np.uint8), 255),
        ("Gray8", np.dtype(np.uint8), 255),
        ("bgr48", np.dtype(np.uint16), 65535),
        ("Bgr48", np.dtype(np.uint16), 65535),
        ("bgr24", np.dtype(np.uint8), 255),
        ("Bgr24", np.dtype(np.uint8), 255),
        ("bgr96float", np.dtype(np.uint16), 65535),
        ("Bgr96Float", np.dtype(np.uint16), 65535),
        ("abc", None, None),
        (None, None, None)
    ]
)
def test_pixeltypes_2(pts: str, dts: np.dtype, mvs: int) -> None:

    out = czimd.CziMetadata.get_dtype_fromstring(pts)
    assert (out[0] == dts)
    assert (out[1] == mvs)


def test_dimorder():

    # get the CZI filepath
    filepath = basedir / r"data/S=2_3x3_CH=2.czi"
    md = czimd.CziMetadata(filepath)

    assert (md.aics_dim_order == {'R': -1, 'I': -1, 'M': 5, 'H': 0, 'V': -1,
                                  'B': -1, 'S': 1, 'T': 2, 'C': 3, 'Z': 4, 'Y': 6, 'X': 7, 'A': -1})
    assert (md.aics_dim_index == [-1, -1, 5, 0, -1, -1, 1, 2, 3, 4, 6, 7, -1])
    assert (md.aics_dim_valid == 8)


@pytest.mark.parametrize(
    "czifile, shape",
    [
        ("S=3_1Pos_2Mosaic_T=2=Z=3_CH=2_sm.czi", False),
        ("CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi", True),
        ("WP96_4Pos_B4-10_DAPI.czi", True)
    ]
)
def test_scene_shape(czifile: str, shape: bool) -> None:

    filepath = basedir / "data" / czifile

    assert (Path.exists(filepath) is True)

    # get the complete metadata at once as one big class
    md = czimd.CziMetadata(filepath)

    assert (md.scene_shape_is_consistent == shape)


def test_reading_czi_fresh():

    filepath = basedir / r"data/A01_segSD.czi"

    # get the complete metadata at once as one big class
    mdata = czimd.CziMetadata(filepath)

    assert (mdata.sample.well_array_names == [])
    assert (mdata.sample.well_indices == [])
    assert (mdata.sample.well_position_names == [])
    assert (mdata.sample.well_colID == [])
    assert (mdata.sample.well_rowID == [])
    assert (mdata.sample.well_counter == {})
    assert (mdata.sample.scene_stageX == [])
    assert (mdata.sample.scene_stageY == [])
    assert (mdata.sample.image_stageX is None)
    assert (mdata.sample.image_stageY is None)
