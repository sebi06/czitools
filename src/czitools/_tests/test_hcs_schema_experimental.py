"""Tests for the EXPERIMENTAL CZI HCS JSON Schema generator."""

import json
from pathlib import Path

import pytest

from czitools.metadata_tools import extract_hcs_document
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
