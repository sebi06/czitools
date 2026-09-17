"""Tests for the portable CZI HCS metadata document."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from czitools.metadata_tools import CziHcsDocument, extract_hcs_document

BASEDIR = Path(__file__).resolve().parents[3]
PLATE_CZI = BASEDIR / "data" / "WP96_4Pos_B4-10_DAPI.czi"
NON_PLATE_CZI = BASEDIR / "data" / "S3_1Pos_2Mosaic_T2_Z3_CH2_sm.czi"
FIXED_TIMESTAMP = datetime(2020, 1, 1, tzinfo=timezone.utc)


def test_hcs_document_json_round_trip(tmp_path: Path) -> None:
    """The bundled plate remains equal after a JSON file round-trip."""
    document = extract_hcs_document(PLATE_CZI, generated_at=FIXED_TIMESTAMP)

    assert document.hcs.detected is True
    assert document.hcs.stored_plate is not None
    assert len(document.hcs.stored_plate.wells) == 7
    assert document.hcs.stored_field_count == 28
    assert document.hcs.declared_field_count == 28
    assert document.hcs.stored_scene_indices == tuple(range(28))

    output = document.write_json(tmp_path / "plate.hcs.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    restored = CziHcsDocument.read_json(output)

    assert payload["schema_version"] == "1.0"
    assert payload["provenance"]["extractor"] == "czitools"
    assert payload["image"]["axes"][0]["code"] == "S"
    assert restored == document


def test_hcs_document_is_reproducible_with_fixed_timestamp() -> None:
    """A fixed generation timestamp yields identical documents."""
    first = extract_hcs_document(PLATE_CZI, generated_at=FIXED_TIMESTAMP)
    second = extract_hcs_document(PLATE_CZI, generated_at=FIXED_TIMESTAMP)

    assert first == second
    assert first.provenance.generated_at_utc == "2020-01-01T00:00:00+00:00"


def test_hcs_document_rejects_bad_types() -> None:
    """Strict deserialization rejects values that violate the field types."""
    payload = extract_hcs_document(PLATE_CZI).to_dict()
    payload["hcs"]["detected"] = "not_a_bool"

    with pytest.raises(ValueError):
        CziHcsDocument.from_dict(payload)


def test_hcs_document_rejects_unsupported_schema_version() -> None:
    """An unknown schema version is rejected with a clear error."""
    payload = extract_hcs_document(PLATE_CZI).to_dict()
    payload["schema_version"] = "9.9"

    with pytest.raises(ValueError):
        CziHcsDocument.from_dict(payload)


def test_hcs_document_analytics_rows() -> None:
    """Well, field, and channel flatteners retain their expected cardinality."""
    document = extract_hcs_document(PLATE_CZI)

    well_rows = list(document.iter_well_rows())
    field_rows = list(document.iter_field_rows())
    channel_rows = list(document.iter_channel_rows())

    assert len(well_rows) == 7
    assert len(field_rows) == 28
    assert well_rows[0]["well_name"] == "B4"
    assert field_rows[0]["scene_index"] == 0
    assert len(channel_rows) == 1


def test_hcs_document_redacts_user_and_excludes_none() -> None:
    """Privacy and compact-output options affect only their intended fields."""
    document = extract_hcs_document(PLATE_CZI, redact_user=True)
    payload = document.to_dict(exclude_none=True)

    assert document.acquisition.user_name is None
    assert "user_name" not in payload["acquisition"]
    assert payload["hcs"]["stored_field_count"] == 28


def test_non_hcs_file_serializes_without_plate() -> None:
    """A multi-scene, non-plate CZI still produces a valid document."""
    document = extract_hcs_document(NON_PLATE_CZI)

    assert document.hcs.detected is False
    assert document.hcs.stored_plate is None
    assert list(document.iter_well_rows()) == []
    assert json.loads(document.to_json())["hcs"]["detected"] is False
