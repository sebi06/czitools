"""Tests for the explicit CZI HCS metadata model."""

from dataclasses import FrozenInstanceError
from pathlib import Path

from box import Box
import pandas as pd
import pytest

from czitools.metadata_tools.czi_metadata import CziMetadata
from czitools.metadata_tools.hcs import (
    build_hcs_metadata,
    enrich_hcs_with_planetable,
    normalize_well_name,
    resolve_field,
    resolve_well,
    well_relative_field_positions,
)

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
                    "Information": {"Image": {"Dimensions": {"S": {"Scenes": {"Scene": scenes}}}}},
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


# ---------------------------------------------------------------------------
# Stage 3 - resolvers
# ---------------------------------------------------------------------------


def _plate():
    result = build_hcs_metadata(
        _metadata_box(
            [
                _scene(0, "B4", 2, 4, "region-1"),
                _scene(1, "B4", 2, 4, "region-2"),
                _scene(2, "B5", 2, 5, "region-3"),
            ]
        )
    )
    assert result.plate is not None
    return result.plate


@pytest.mark.parametrize("well_name", ["B4", "b04", "B/4"])
def test_resolve_well_accepts_name_variants(well_name: str) -> None:
    well = resolve_well(_plate(), well_name)
    assert well.canonical_name == "B4"


def test_resolve_field_by_index_and_region_id() -> None:
    plate = _plate()
    assert resolve_field(plate, "B4", 1).scene_index == 1
    assert resolve_field(plate, "B4", "region-1").field_index == 0
    assert resolve_field(plate, "B5").scene_index == 2  # default field 0


def test_resolve_field_rejects_bad_input() -> None:
    plate = _plate()
    with pytest.raises(IndexError):
        resolve_field(plate, "B4", 5)
    with pytest.raises(KeyError):
        resolve_field(plate, "B4", "missing-region")
    with pytest.raises(TypeError):
        resolve_field(plate, "B4", True)  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        resolve_well(plate, "Z9")


# ---------------------------------------------------------------------------
# Stage 2 - planetable position enrichment
# ---------------------------------------------------------------------------


def _planetable_frame() -> pd.DataFrame:
    # Scene 0: two subblocks with a small X spread (no conflict) and a large
    # Z spread (conflict). Scene 1: single subblock. Scene 2 is absent.
    return pd.DataFrame(
        {
            "S": [0, 0, 1],
            "M": [0, 1, 0],
            "T": [0, 0, 0],
            "C": [0, 0, 0],
            "Z": [0, 0, 0],
            "X[micron]": [100.0, 100.4, 200.0],
            "Y[micron]": [50.0, 50.0, 60.0],
            "Z[micron]": [10.0, 25.0, 5.0],
        }
    )


def test_enrich_hcs_with_planetable_aggregates_positions(monkeypatch) -> None:
    plate = _plate()

    def fake_get_planetable(filepath, *args, **kwargs):
        return _planetable_frame(), None

    monkeypatch.setattr("czitools.utils.planetable.get_planetable", fake_get_planetable)

    enriched = enrich_hcs_with_planetable(plate, "irrelevant.czi", position_tolerance=1.0)

    # Original plate must be untouched (immutability).
    assert plate.wells[0].fields[0].stage_x is None

    b4 = enriched.get_well("B4")
    field0 = b4.fields[0]
    assert field0.stage_x == pytest.approx(100.2)  # median of 100.0, 100.4
    assert field0.stage_x_range == (100.0, 100.4)
    assert field0.subblock_count == 2
    assert field0.stage_source and "StageXPosition" in field0.stage_source
    # X spread 0.4 <= 1.0 tolerance, but Z spread 15.0 > 1.0 => conflict.
    assert field0.position_conflict is True
    assert field0.acquisition_z == pytest.approx(17.5)
    assert field0.acquisition_z_range == (10.0, 25.0)

    # Scene 1 has a single subblock -> no conflict.
    field1 = b4.fields[1]
    assert field1.stage_x == pytest.approx(200.0)
    assert field1.position_conflict is False

    # Scene 2 absent from the planetable -> left unenriched.
    b5_field = enriched.get_well("B5").fields[0]
    assert b5_field.stage_x is None
    assert b5_field.subblock_count is None


def test_enrich_returns_plate_unchanged_when_planetable_empty(monkeypatch) -> None:
    plate = _plate()

    def empty_get_planetable(filepath, *args, **kwargs):
        return pd.DataFrame(), None

    monkeypatch.setattr("czitools.utils.planetable.get_planetable", empty_get_planetable)

    enriched = enrich_hcs_with_planetable(plate, "http://example.com/remote.czi")
    assert enriched is plate


def test_well_relative_field_positions() -> None:
    plate = build_hcs_metadata(
        _metadata_box(
            [
                _scene(0, "B4", 2, 4, "r0", center="0.0,0.0"),
                _scene(1, "B4", 2, 4, "r1", center="10.0,20.0"),
            ]
        )
    ).plate
    assert plate is not None
    offsets = well_relative_field_positions(plate.get_well("B4"))
    assert offsets == {0: (-5.0, -10.0), 1: (5.0, 10.0)}


def test_well_relative_field_positions_unavailable_when_center_missing() -> None:
    plate = build_hcs_metadata(
        _metadata_box(
            [
                _scene(0, "B4", 2, 4, "r0", center=None),
                _scene(1, "B4", 2, 4, "r1", center="10.0,20.0"),
            ]
        )
    ).plate
    assert plate is not None
    assert well_relative_field_positions(plate.get_well("B4")) is None
