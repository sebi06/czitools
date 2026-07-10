"""Tests for the explicit CZI HCS metadata model."""

from dataclasses import FrozenInstanceError
from pathlib import Path

from box import Box
import pytest

from czitools.metadata_tools.czi_metadata import CziMetadata
from czitools.metadata_tools.hcs import build_hcs_metadata, normalize_well_name


BASEDIR = Path(__file__).resolve().parents[3]


def _metadata_box(scenes: list[dict], *, rows: int = 8, columns: int = 12) -> Box:
    return Box(
        {
            "filepath": "/example/plate.czi",
            "ImageDocument": {
                "Metadata": {
                    "Experiment": {
                        "SampleHolder": {
                            "Template": {
                                "Name": "Multichamber 96",
                                "ShapeRows": rows,
                                "ShapeColumns": columns,
                            }
                        }
                    },
                    "Information": {
                        "Image": {"Dimensions": {"S": {"Scenes": {"Scene": scenes}}}}
                    },
                }
            },
        }
    )


def _scene(
    index: int,
    name: str,
    row: int,
    column: int,
    region_id: str | None,
    *,
    center: str | None = "10.0,20.0",
) -> dict:
    return {
        "Index": index,
        "Name": f"P{index + 1}",
        "ArrayName": name,
        "RegionId": region_id,
        "CenterPosition": center,
        "Shape": {"Name": name, "RowIndex": row, "ColumnIndex": column},
    }


@pytest.mark.parametrize(
    "source, expected",
    [("A01", ("A1", 0, 0)), ("B4", ("B4", 1, 3)), ("AA12", ("AA12", 26, 11))],
)
def test_normalize_well_name(source: str, expected: tuple[str, int, int]) -> None:
    assert normalize_well_name(source) == expected


def test_build_hcs_model_and_source_scoped_ids() -> None:
    result = build_hcs_metadata(
        _metadata_box(
            [
                _scene(0, "B4", 2, 4, "region-1"),
                _scene(1, "B4", 2, 4, None, center="malformed"),
                _scene(2, "B5", 2, 5, "region-3"),
            ]
        )
    )

    assert result.detected is True
    plate = result.plate
    assert plate is not None
    assert plate.schema_version == "1.0"
    assert plate.name == "Multichamber 96"
    assert (plate.declared_rows, plate.declared_columns) == (8, 12)
    assert plate.observed_row_indices == (1,)
    assert plate.observed_column_indices == (3, 4)

    well = plate.get_well("b04")
    assert well.canonical_name == "B4"
    assert well.canonical_path == "B/4"
    assert (well.source_row_index, well.source_column_index) == (2, 4)
    assert (well.row_index, well.column_index) == (1, 3)
    assert [field.field_index for field in well.fields] == [0, 1]
    assert [field.scene_index for field in well.fields] == [0, 1]
    assert well.fields[0].id == "field:region-1"
    assert well.fields[1].id == "scene:1"
    assert well.fields[1].scene_center_x is None
    assert well.fields[1].position_unit == "micrometer"
    with pytest.raises(FrozenInstanceError):
        well.fields[0].scene_index = 99  # type: ignore[misc]


@pytest.mark.parametrize(
    "scenes, reason_fragment",
    [
        ([_scene(0, "B4", 2, 4, "same"), _scene(1, "B4", 2, 4, "same")], "duplicated"),
        ([_scene(0, "B4", 2, 5, "region")], "conflicts"),
        ([_scene(0, "Scene 1", 2, 4, "region")], "Unsupported well name"),
        ([_scene(0, "B4", 0, 4, "region")], "lacks positive"),
    ],
)
def test_ambiguous_scene_metadata_is_rejected(scenes: list[dict], reason_fragment: str) -> None:
    result = build_hcs_metadata(_metadata_box(scenes))
    assert result.detected is False
    assert result.plate is None
    assert reason_fragment in result.reason


def test_czi_metadata_exposes_hcs_for_included_wellplate() -> None:
    metadata = CziMetadata(str(BASEDIR / "data" / "WP96_4Pos_B4-10_DAPI.czi"))

    assert metadata.hcs_status.detected is True
    assert metadata.hcs is metadata.hcs_status.plate
    assert metadata.hcs is not None
    assert len(metadata.hcs.wells) == 7
    assert sum(len(well.fields) for well in metadata.hcs.wells) == 28
    assert metadata.hcs.get_well("B4").fields[0].region_id == "637232309317131710"


def test_multiscene_non_plate_has_explanatory_status() -> None:
    metadata = CziMetadata(str(BASEDIR / "data" / "S3_1Pos_2Mosaic_T2_Z3_CH2_sm.czi"))

    assert metadata.has_scenes is True
    assert metadata.hcs is None
    assert metadata.hcs_status.detected is False
    assert metadata.hcs_status.reason
