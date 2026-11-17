"""Tests for the EXPERIMENTAL CZI HCS JSON Schema generator."""

import json
from pathlib import Path

import pytest

from czitools.metadata_tools import CziHcsDocument, extract_hcs_document
from czitools.metadata_tools.experimental.hcs_schema import (
    EXPERIMENTAL_SCHEMA_ID,
    JSON_SCHEMA_DIALECT,
    generate_hcs_json_schema,
    write_hcs_json_schema,
)

BASEDIR = Path(__file__).resolve().parents[3]
PLATE_CZI = BASEDIR / "data" / "WP96_4Pos_B4-10_DAPI.czi"
NON_PLATE_CZI = BASEDIR / "data" / "S3_1Pos_2Mosaic_T2_Z3_CH2_sm.czi"


def test_generate_schema_warns_and_has_expected_shape() -> None:
    """Generation warns about experimental status and returns a JSON Schema."""
    with pytest.warns(UserWarning, match="experimental"):
        schema = generate_hcs_json_schema()

    assert schema["$schema"] == JSON_SCHEMA_DIALECT
    assert schema["$id"] == EXPERIMENTAL_SCHEMA_ID
    assert schema["type"] == "object"
    assert schema["x-czitools-status"] == "experimental"
    assert "non-plate files" in schema["description"]
    assert "$defs" in schema
    assert "schema_version" in schema["properties"]


def test_generate_schema_can_suppress_warning() -> None:
    """The warning can be silenced for programmatic use."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        schema = generate_hcs_json_schema(warn=False)

    assert schema["type"] == "object"


def test_write_schema_creates_parseable_file(tmp_path: Path) -> None:
    """Writing produces a UTF-8 JSON file that parses back to the schema."""
    output = write_hcs_json_schema(tmp_path / "sub" / "czi-hcs.schema.json", warn=False)

    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["x-czitools-schema-version"]


def test_bundled_document_validates_against_schema() -> None:
    """The extracted sample document conforms to the generated schema."""
    jsonschema = pytest.importorskip("jsonschema")

    schema = generate_hcs_json_schema(warn=False)
    document = extract_hcs_document(PLATE_CZI).to_dict()

    jsonschema.validate(instance=document, schema=schema)


def test_non_plate_document_validates_against_schema() -> None:
    """The schema covers ordinary CZI files without well-plate metadata."""
    jsonschema = pytest.importorskip("jsonschema")

    schema = generate_hcs_json_schema(warn=False)
    document = extract_hcs_document(NON_PLATE_CZI).to_dict()

    assert document["hcs"]["detected"] is False
    assert document["hcs"]["declared_plate"] is None
    assert document["hcs"]["stored_plate"] is None
    jsonschema.validate(instance=document, schema=schema)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), "9.9"),
        (("hcs", "stored_plate", "schema_version"), "9.9"),
        (("unexpected",), {"important": "do not discard"}),
        (("hcs", "stored_plate", "wells", 0, "unexpected"), "metadata"),
        (("hcs", "detected"), "true"),
        (("hcs", "stored_field_count"), "28"),
        (("source", "size_bytes"), -1),
        (("hcs", "stored_field_count"), -1),
        (("hcs", "stored_scene_indices"), [-1]),
        (("hcs", "stored_plate", "declared_rows"), 0),
        (("hcs", "stored_plate", "wells", 0, "fields", 0, "scene_index"), -1),
        (("hcs", "detected"), False),
        (("hcs", "declared_plate"), None),
        (("hcs", "stored_plate"), None),
    ],
)
def test_schema_and_reader_reject_invalid_contract_values(path, value) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    payload = extract_hcs_document(PLATE_CZI).to_dict()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(generate_hcs_json_schema(warn=False)).validate(payload)
    with pytest.raises(ValueError):
        CziHcsDocument.from_dict(payload)


def test_committed_schema_matches_generator_and_validates_dialect() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = generate_hcs_json_schema(warn=False)
    jsonschema.Draft202012Validator.check_schema(schema)
    committed = json.loads(
        (BASEDIR / "docs" / "schemas" / "czi-hcs-1.0.experimental.schema.json").read_text(encoding="utf-8")
    )

    assert committed == schema


def test_compact_output_is_not_a_schema_valid_document() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    payload = extract_hcs_document(PLATE_CZI).to_dict(exclude_none=True)

    with pytest.raises(jsonschema.ValidationError, match="required"):
        jsonschema.Draft202012Validator(generate_hcs_json_schema(warn=False)).validate(payload)
