
from czitools import metadata_tools as czimd
from pathlib import Path
import numpy as np
import pytest
from typing import List, Dict, Tuple, Optional, Type, Any, Union, Mapping

basedir = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "czifile",
    [
        "LLS7_small.czi",
        "FOV7_HV110_P0500510000.czi",
        "newCZI_compressed.czi",
        "Airyscan.czi",
        "CellDivision_T3_Z5_CH2_X240_Y170.czi",
        "2_tileregions.czi",
        "S2_3x3_CH2.czi",
        "S3_1Pos_2Mosaic_T2_Z3_CH2_sm.czi",
        "test_z16_ch3.czi",
        "Tumor_HE_RGB.czi",
    ]
)
def test_read_metadata_local(czifile: str) -> None:

    filepath = basedir / "data" / czifile
    md = czimd.CziMetadata(filepath)

    assert (isinstance(md, czimd.CziMetadata) is True)


@pytest.mark.parametrize(
    "link",
    [
        "https://raw.githubusercontent.com/zeiss-microscopy/OAD/master/jupyter_notebooks/pylibCZIrw/images/S%3D2_CH%3D2_well_A1%2BA2_zstd.czi",
        "https://github.com/sebi06/czitools/raw/main/data/CellDivision_T10_Z15_CH2_DCV_small.czi"
        #"https://www.dropbox.com/scl/fi/lazndscc5etck38k1vz8e/S-2_3x3_T-1_Z-1_CH-2.czi?rlkey=60apu65t2dza2zor15gq4sw15&dl=1"
    ]
)
def test_read_metadata_links(link: str) -> None:

    md = czimd.CziMetadata(link)

    assert (isinstance(md, czimd.CziMetadata) is True)


@pytest.mark.parametrize(
    "czifile, px, rgb, maxvalue",
    [
        ("LLS7_small.czi", {0: 'Gray16', 1: 'Gray16'}, False, [65535, 65535]),
        ("Tumor_HE_RGB.czi", {0: 'Bgr24'}, True, [255])
    ]
)
def test_pixeltypes_1(czifile: str, px: Dict, rgb: bool, maxvalue: int) -> None:

    # get the CZI filepath
    filepath = basedir / "data" / czifile
    md = czimd.CziMetadata(filepath)

    assert (md.pixeltypes == px)
    assert (md.maxvalue == maxvalue)
    assert (md.isRGB is rgb)


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
    filepath = basedir / r"data/S2_3x3_CH2.czi"
    md = czimd.CziMetadata(filepath)

    assert (md.aics_dim_order == {'R': -1, 'I': -1, 'M': 5, 'H': 0, 'V': -1,
                                  'B': -1, 'S': 1, 'T': 2, 'C': 3, 'Z': 4, 'Y': 6, 'X': 7, 'A': -1})
    assert (md.aics_dim_index == [-1, -1, 5, 0, -1, -1, 1, 2, 3, 4, 6, 7, -1])
    assert (md.aics_dim_valid == 8)


@pytest.mark.parametrize(
    "czifile, shape",
    [
        ("S3_1Pos_2Mosaic_T2_Z3_CH2_sm.czi", False),
        ("CellDivision_T3_Z5_CH2_X240_Y170.czi", True),
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
