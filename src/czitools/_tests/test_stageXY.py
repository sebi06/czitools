from czitools import metadata_tools as czimd
from pathlib import Path
import pytest
from typing import List, Dict, Tuple, Optional, Type, Any, Union, Mapping

basedir = Path(__file__).resolve().parents[3]

@pytest.mark.parametrize(
    "czifile, results",
    [
        ("WellD6_S1.czi", {"X": [49500.0], "Y": [35500.0]}),
        ("WellD6-7_S2.czi", {"X": [49500.0, 58500.0], "Y": [35500.0, 35500.0]}),
        ("CellDivision_T3_Z5_CH2_X240_Y170.czi", {"X": 16977.153, "Y": 18621.489}),
        ("Al2O3_SE_020_sp.czi", {"X": 0.0, "Y": 0.0}),
        ("1_tileregion.czi", {"X": [47729.178], "Y": [35660.098]}),
        ("2_tileregions.czi", {"X": [48757.762, 47729.178], "Y": [35127.654, 35660.098]}),
        ("w96_A1+A2.czi", {"X": [4261.294, 13293.666], "Y": [8361.406, 8372.441]}),
        ("Airyscan.czi", {"X": 0.0, "Y": 0.0}),
        ("newCZI_zloc.czi", {"X": [], "Y": []}),
        ("FOV7_HV110_P0500510000.czi", {"X": 0.0, "Y": 0.0}),
        ("Tumor_HE_RGB.czi", {"X": 0.0, "Y": 0.0}),
        ("S3_1Pos_2Mosaic_T2_Z3_CH2_sm.czi", {"X": [38397.296, 37980.0, 38679.412], "Y": [12724.718, 13020.0, 13260.298]}),
        ("DAPI_GFP.czi", {"X": 0.0, "Y": 0.0})
    ]
)
def test_stage_xy(czifile: str, results: Dict) -> None:

    # get the CZI filepath
    filepath = basedir / "data" / czifile

    # read the metadata
    md = czimd.CziMetadata(filepath)

    if md.image.SizeS is not None:
        assert (md.sample.scene_stageX == results["X"])
        assert (md.sample.scene_stageY == results["Y"])
    if md.image.SizeS is None:
        assert (md.sample.image_stageX == results["X"])
        assert (md.sample.image_stageY == results["Y"])
