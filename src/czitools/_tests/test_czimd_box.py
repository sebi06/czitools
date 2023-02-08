from czitools import pylibczirw_metadata as czimd
from pathlib import Path, PurePath
import pytest
from typing import List, Dict, Tuple, Optional, Type, Any, Union, Mapping


# adapt to your needs
defaultdir = Path(__file__).resolve().parents[3] / "data"


@pytest.mark.parametrize(
    "attribute",
    [
        "has_customattr",
        "has_exp",
        "has_disp",
        "has_hardware",
        "has_scale",
        "has_instrument",
        "has_microscopes",
        "has_detectors",
        "has_objectives",
        "has_tubelenses",
        "has_disp",
        "has_channels",
        "has_info",
        "has_app",
        "has_doc",
        "has_image",
        "has_scenes",
        "has_dims"
    ]
)
def test_general_attr(attribute: str) -> None:

    filepath = defaultdir / "CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi"

    czibox = czimd.get_czimd_box(filepath)

    assert (hasattr(czibox, attribute))


@pytest.mark.parametrize(
    "czifile, expected",
    [
        ("CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi", True),
        ("Al2O3_SE_020_sp.czi", True),
        ("w96_A1+A2.czi", True),
        ("Airyscan.czi", True),
        ("newCZI_zloc.czi", True),
        ("does_not_exist.czi", False),
        ("FOV7_HV110_P0500510000.czi", True),
        ("Tumor_HE_RGB.czi", True),
        ("WP96_4Pos_B4-10_DAPI.czi", True),
        ("S=3_1Pos_2Mosaic_T=2=Z=3_CH=2_sm.czi", True)
    ]
)
def test_box(czifile: List[str], expected: bool) -> None:

    filepath = defaultdir / czifile

    try:
        czibox = czimd.get_czimd_box(filepath)
        ok = True
    except Exception:
        ok = False

    assert (ok == expected)
