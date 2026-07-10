"""Tests for scene/well sample metadata extraction."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from czitools.metadata_tools.sample import CziSampleInfo


BASEDIR = Path(__file__).resolve().parents[3]


def _empty_sample(*, verbose: bool) -> CziSampleInfo:
    """Create an extraction target without requiring a complete CZI metadata tree."""

    sample = object.__new__(CziSampleInfo)
    sample.czisource = "unused"
    sample.well_array_names = []
    sample.well_indices = []
    sample.well_position_names = []
    sample.well_colID = []
    sample.well_rowID = []
    sample.well_counter = {}
    sample.well_scene_indices = {}
    sample.well_total_number = None
    sample.scene_count = 0
    sample.well_unique_number = 0
    sample.field_centerX = []
    sample.field_centerY = []
    sample.well_region_ids = []
    sample.image_stageX = None
    sample.image_stageY = None
    sample.multipos_per_well = False
    sample.verbose = verbose
    return sample


@pytest.mark.parametrize("verbose", [False, True])
def test_missing_scene_fields_stay_aligned(verbose: bool) -> None:
    sample = _empty_sample(verbose=verbose)
    scenes = [
        SimpleNamespace(
            ArrayName="A1",
            Index=0,
            Name="P1",
            Shape=SimpleNamespace(ColumnIndex=1, RowIndex=1),
            CenterPosition="10.5,20.5",
            RegionId=12345678901234567890,
        ),
        SimpleNamespace(
            ArrayName="A1",
            Index=None,
            Name=None,
            Shape=None,
            CenterPosition=None,
            RegionId=None,
        ),
        SimpleNamespace(
            ArrayName=None,
            Index="invalid",
            Name=None,
            Shape=SimpleNamespace(Name="A2", ColumnIndex=None, RowIndex=None),
            CenterPosition="malformed",
            RegionId="region-3",
        ),
    ]

    for scene in scenes:
        sample.get_well_info(scene)
    sample._finalize_scene_info()

    per_scene = [
        sample.well_array_names,
        sample.well_indices,
        sample.well_position_names,
        sample.well_colID,
        sample.well_rowID,
        sample.field_centerX,
        sample.field_centerY,
        sample.well_region_ids,
    ]
    assert all(len(values) == 3 for values in per_scene)
    assert sample.field_centerX == [10.5, None, None]
    assert sample.field_centerY == [20.5, None, None]
    assert sample.scene_stageX == [10.5, 0.0, 0.0]
    assert sample.scene_stageY == [20.5, 0.0, 0.0]
    assert sample.well_region_ids == ["12345678901234567890", None, "region-3"]
    assert sample.scene_count == 3
    assert sample.well_total_number == 3
    assert sample.well_unique_number == 2
    assert sample.well_counter == {"A1": 2, "A2": 1}
    assert sample.multipos_per_well is True


def test_included_wellplate_region_ids_and_counts() -> None:
    sample = CziSampleInfo(str(BASEDIR / "data" / "WP96_4Pos_B4-10_DAPI.czi"))

    assert sample.scene_count == 28
    assert sample.well_total_number == 28
    assert sample.well_unique_number == 7
    assert sample.multipos_per_well is True
    assert len(sample.well_region_ids) == sample.scene_count
    assert sample.well_region_ids[0] == "637232309317131710"
    assert all(region_id is not None for region_id in sample.well_region_ids)
    assert len(sample.field_centerX) == len(sample.field_centerY) == sample.scene_count
