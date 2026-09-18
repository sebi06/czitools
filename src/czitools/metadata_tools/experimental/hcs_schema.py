"""EXPERIMENTAL: JSON Schema generation for CZI metadata documents.

This module derives a JSON Schema (Draft 2020-12) from the ``CziHcsDocument``
dataclass graph using pydantic. It is intended as an early, unstable draft so
the shape of a future formal contract can be reviewed and iterated on. Despite
the model's historical HCS-oriented name, the schema describes both well-plate
and non-well-plate CZI files. For non-plate files, ``hcs.detected`` is false and
the declared and stored plate fields are null.

Stability warning:
    The emitted schema is EXPERIMENTAL. Its structure, field descriptions, and
    ``$id`` are expected to change, and such changes will NOT necessarily be
    accompanied by a bump of the document's domain ``schema_version``. Do not
    treat the generated schema as a stable, published contract yet.

Example:
    ```python
    from czitools.metadata_tools.experimental.hcs_schema import (
        generate_hcs_json_schema,
    )

    schema = generate_hcs_json_schema()
    ```
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from czitools.metadata_tools.hcs_document import (
    CZI_HCS_DOCUMENT_SCHEMA_VERSION,
    CziHcsDocument,
)

# JSON Schema dialect emitted by pydantic v2.
JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

# Placeholder identifier; not yet a resolvable, published URL.
EXPERIMENTAL_SCHEMA_ID = (
    "https://czitools.example/schemas/experimental/" f"czi-hcs-{CZI_HCS_DOCUMENT_SCHEMA_VERSION}.schema.json"
)

_EXPERIMENTAL_MESSAGE = (
    "The CZI HCS JSON Schema is experimental and its structure may change "
    "without notice. Do not rely on it as a stable contract."
)


def _warn_experimental() -> None:
    warnings.warn(_EXPERIMENTAL_MESSAGE, category=UserWarning, stacklevel=3)


def generate_hcs_json_schema(*, warn: bool = True) -> dict[str, Any]:
    """Generate an experimental JSON Schema for ``CziHcsDocument``.

    Args:
        warn (bool): Emit a ``UserWarning`` about the experimental status.
            Defaults to True.

    Returns:
        dict[str, Any]: A JSON Schema (Draft 2020-12) describing the CZI HCS
            metadata document, annotated with experimental markers.
    """
    if warn:
        _warn_experimental()

    schema = TypeAdapter(CziHcsDocument).json_schema()
    schema["$schema"] = JSON_SCHEMA_DIALECT
    schema["$id"] = EXPERIMENTAL_SCHEMA_ID
    schema["title"] = "CZI Metadata Document with Optional HCS Data (EXPERIMENTAL)"
    schema["description"] = (
        "Experimental metadata description for any CZI file. Well-plate/HCS "
        "content is optional; non-plate files use hcs.detected=false with "
        "null declared_plate and stored_plate values."
    )
    schema["$comment"] = (
        "EXPERIMENTAL schema for czitools CZI metadata documents, including "
        "both well-plate and non-well-plate files (domain "
        f"schema_version {CZI_HCS_DOCUMENT_SCHEMA_VERSION}). The structure is "
        "unstable and may change without a version bump."
    )
    schema["x-czitools-schema-version"] = CZI_HCS_DOCUMENT_SCHEMA_VERSION
    schema["x-czitools-status"] = "experimental"
    return schema


def write_hcs_json_schema(
    path: str | os.PathLike[str],
    *,
    indent: int | None = 2,
    warn: bool = True,
) -> Path:
    """Write the experimental JSON Schema to a UTF-8 file.

    Args:
        path (str | os.PathLike[str]): Destination file path.
        indent (int | None): JSON indentation. Defaults to 2.
        warn (bool): Emit a ``UserWarning`` about the experimental status.
            Defaults to True.

    Returns:
        Path: The path that was written.
    """
    schema = generate_hcs_json_schema(warn=warn)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(schema, indent=indent, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output
