"""Tests for the portable CZI HCS metadata document."""

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from czitools.metadata_tools import CziHcsDocument, CziMetadata, extract_hcs_document
from czitools.metadata_tools.hcs import filter_hcs_by_scene_indices

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
    """Compact output is for presentation, not validated round-trip storage."""
    document = extract_hcs_document(PLATE_CZI, redact_user=True)
    payload = document.to_dict(exclude_none=True)

    assert document.acquisition.user_name is None
    assert "user_name" not in payload["acquisition"]
    assert payload["hcs"]["stored_field_count"] == 28
    with pytest.raises(ValueError, match="user_name"):
        CziHcsDocument.from_dict(payload)


def test_non_hcs_file_serializes_without_plate() -> None:
    """A multi-scene, non-plate CZI still produces a valid document."""
    document = extract_hcs_document(NON_PLATE_CZI)

    assert document.hcs.detected is False
    assert document.hcs.stored_plate is None
    assert list(document.iter_well_rows()) == []
    assert json.loads(document.to_json())["hcs"]["detected"] is False


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("hcs", "stored_scene_indices"), [], "stored scene indices"),
        (("hcs", "stored_scene_indices"), [0, 0], "unique"),
        (("hcs", "stored_field_count"), 999, "field counts"),
        (("hcs", "stored_plate", "wells", 0, "canonical_path"), "Z/99", "canonical path"),
        (("hcs", "stored_plate", "wells", 0, "source_row_index"), 1, "source indices"),
        (("hcs", "stored_plate", "declared_rows"), 1, "declared rows"),
        (("hcs", "stored_plate", "declared_columns"), 1, "declared columns"),
        (("hcs", "stored_plate", "observed_row_indices"), [], "observed row"),
        (("hcs", "stored_plate", "wells", 0, "fields", 0, "id"), "other-field", "declared field"),
    ],
)
def test_hcs_document_rejects_inconsistent_hierarchy(path, value, message) -> None:
    payload = extract_hcs_document(PLATE_CZI).to_dict()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        CziHcsDocument.from_dict(payload)


@pytest.mark.parametrize("scene_indices", [(0, 3, 5), ()])
def test_hcs_document_preserves_sparse_and_empty_stored_views(scene_indices) -> None:
    document = extract_hcs_document(PLATE_CZI)
    assert document.hcs.declared_plate is not None
    stored = filter_hcs_by_scene_indices(document.hcs.declared_plate, scene_indices)
    hcs = replace(
        document.hcs,
        stored_plate=stored,
        stored_scene_indices=scene_indices,
        stored_field_count=len(scene_indices),
    )
    document = replace(document, hcs=hcs)

    assert CziHcsDocument.from_dict(document.to_dict()) == document


def test_hcs_document_preserves_explicit_metadata_dictionaries() -> None:
    payload = extract_hcs_document(PLATE_CZI).to_dict()
    payload["extensions"]["example:analysis"] = {"method": "test", "optional": None}
    payload["sample"]["carrier"] = {"vendor_specific": {"rows": 8}}
    payload["sample"]["specimen"] = {"custom": [1, "value", None]}

    assert CziHcsDocument.from_dict(payload).to_dict() == payload


def test_hcs_document_rejects_non_string_keys_without_losing_metadata() -> None:
    payload = extract_hcs_document(PLATE_CZI).to_dict()
    payload["extensions"]["custom"] = {1: "first", "1": "second"}

    with pytest.raises(ValueError, match="JSON object keys must be strings"):
        CziHcsDocument.from_dict(payload)


def test_hcs_document_filters_stored_view_when_metadata_filtering_is_disabled() -> None:
    metadata = CziMetadata(PLATE_CZI, filter_hcs_to_stored_scenes=False)
    metadata.stored_scene_indices = (0, 3)

    document = CziHcsDocument.from_metadata(metadata)

    assert document.hcs.declared_field_count == 28
    assert document.hcs.stored_field_count == 2
    assert CziHcsDocument.from_dict(document.to_dict()) == document


def test_hcs_document_reader_keeps_json_strict(tmp_path: Path) -> None:
    output = tmp_path / "commented.json"
    output.write_text('{\n// illustration only\n"schema_version": "1.0"\n}', encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        CziHcsDocument.read_json(output)


@pytest.mark.parametrize("suffix", [".jsonc", ".JSONC"])
def test_hcs_document_reads_jsonc_without_changing_string_values(tmp_path: Path, suffix: str) -> None:
    payload = extract_hcs_document(PLATE_CZI).to_dict()
    payload["extensions"]["notes"] = 'https://example.org/a // literal /* text */ "quoted" \\path'
    payload["extensions"]["jsonc_list"] = [1, 2]
    text = json.dumps(payload)
    text = text.replace('"jsonc_list": [1, 2]', '"jsonc_list": [1, 2,]')
    text = "/* header */\n{// root comment\n" + text[1:-1] + ", /* final comma */\n}"
    source = tmp_path / f"plate{suffix}"
    source.write_text(text, encoding="utf-8")

    document = CziHcsDocument.read_json(source)

    assert document.to_dict() == payload
    output = document.write_json(tmp_path / "plain.json")
    assert json.loads(output.read_text(encoding="utf-8")) == payload


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ('{"schema_version": "1.0" /* unfinished', ValueError),
        ('{"schema_version": "1.0", "source": }', ValueError),
        ('[/* not an object */ 1, 2,]', TypeError),
    ],
)
def test_hcs_document_rejects_malformed_or_non_object_jsonc(tmp_path: Path, text: str, error) -> None:
    source = tmp_path / "invalid.jsonc"
    source.write_text(text, encoding="utf-8")

    with pytest.raises(error):
        CziHcsDocument.read_json(source)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("unexpected", "metadata", "Unexpected keyword argument"),
        ("hcs", {"detected": "true"}, "boolean"),
        ("schema_version", "9.9", "Unsupported"),
    ],
)
def test_hcs_document_jsonc_keeps_model_validation(tmp_path: Path, key: str, value, message: str) -> None:
    payload = extract_hcs_document(PLATE_CZI).to_dict()
    if key == "hcs":
        payload[key].update(value)
    else:
        payload[key] = value
    source = tmp_path / "invalid-metadata.jsonc"
    source.write_text("// valid JSONC, invalid metadata\n" + json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        CziHcsDocument.read_json(source)
