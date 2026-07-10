"""Explicit high-content-screening metadata derived from CZI scene XML.

The value models in this module are OME/NGFF-oriented, but they are not a
serializer and do not claim complete compliance with either specification.
Source row and column indices are retained as written by CZI (normally
1-based), while normalized indices are always 0-based.

Schema version: ``1.0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Optional

from box import Box, BoxList


HCS_SCHEMA_VERSION = "1.0"
_WELL_NAME = re.compile(r"^([A-Za-z]+)0*([1-9][0-9]*)$")


@dataclass(frozen=True)
class CziField:
    """One CZI scene interpreted as a field of view within a well."""

    id: str
    field_index: int
    scene_index: int
    region_id: Optional[str]
    position_name: Optional[str]
    scene_center_x: Optional[float]
    scene_center_y: Optional[float]
    position_unit: str = "micrometer"
    position_source: str = "Scene.CenterPosition"
    acquisition_z: Optional[float] = None


@dataclass(frozen=True)
class CziWell:
    """A well with both original CZI and normalized 0-based indices."""

    id: str
    canonical_name: str
    canonical_path: str
    source_name: str
    source_row_index: int
    source_column_index: int
    row_index: int
    column_index: int
    external_id: Optional[str]
    fields: tuple[CziField, ...]


@dataclass(frozen=True)
class CziPlate:
    """An immutable plate view built from CZI scene and template metadata."""

    schema_version: str
    id: str
    name: Optional[str]
    declared_rows: Optional[int]
    declared_columns: Optional[int]
    naming_convention: str
    barcode: Optional[str]
    external_id: Optional[str]
    observed_row_indices: tuple[int, ...]
    observed_column_indices: tuple[int, ...]
    wells: tuple[CziWell, ...]

    def get_well(self, name: str) -> CziWell:
        """Return a well by normalized name, raising for absent/duplicate names."""

        canonical_name, _, _ = normalize_well_name(name)
        matches = [well for well in self.wells if well.canonical_name == canonical_name]
        if len(matches) != 1:
            raise KeyError(f"Expected exactly one well named {canonical_name!r}; found {len(matches)}.")
        return matches[0]


@dataclass(frozen=True)
class CziHcsResult:
    """Result of attempting to interpret CZI scene metadata as an HCS plate."""

    detected: bool
    reason: str
    plate: Optional[CziPlate] = None


def normalize_well_name(name: str) -> tuple[str, int, int]:
    """Normalize a well name and return ``(name, row, column)`` with 0-based indices."""

    match = _WELL_NAME.fullmatch(name.strip())
    if match is None:
        raise ValueError(f"Unsupported well name {name!r}; expected a row label and positive column number.")
    row_label, column_text = match.groups()
    row_label = row_label.upper()
    row_index = 0
    for character in row_label:
        row_index = row_index * 26 + (ord(character) - ord("A") + 1)
    row_index -= 1
    column_index = int(column_text) - 1
    return f"{row_label}{column_index + 1}", row_index, column_index


def build_hcs_metadata(czi_box: Box) -> CziHcsResult:
    """Build an HCS plate from scene XML, or return a precise rejection reason."""

    scenes = _get_scenes(czi_box)
    if not scenes:
        return CziHcsResult(False, "No scene XML is available.")

    parsed: list[dict[str, Any]] = []
    seen_scene_indices: set[int] = set()
    seen_region_ids: set[str] = set()
    well_coordinates: dict[str, tuple[int, int]] = {}

    for fallback_scene_index, scene in enumerate(scenes):
        scene_index = _optional_int(getattr(scene, "Index", None))
        if scene_index is None:
            scene_index = fallback_scene_index
        if scene_index < 0 or scene_index in seen_scene_indices:
            return CziHcsResult(False, f"Scene index {scene_index} is negative or duplicated.")
        seen_scene_indices.add(scene_index)

        shape = getattr(scene, "Shape", None)
        source_name = getattr(scene, "ArrayName", None)
        if source_name is None and shape is not None:
            source_name = getattr(shape, "Name", None)
        if not source_name:
            return CziHcsResult(False, f"Scene {scene_index} has no usable well name.")

        try:
            canonical_name, name_row, name_column = normalize_well_name(str(source_name))
        except ValueError as error:
            return CziHcsResult(False, f"Scene {scene_index}: {error}")

        source_row = _optional_int(getattr(shape, "RowIndex", None)) if shape is not None else None
        source_column = _optional_int(getattr(shape, "ColumnIndex", None)) if shape is not None else None
        if source_row is None or source_column is None or source_row < 1 or source_column < 1:
            return CziHcsResult(False, f"Scene {scene_index} lacks positive CZI well row/column indices.")
        row_index, column_index = source_row - 1, source_column - 1
        if (row_index, column_index) != (name_row, name_column):
            return CziHcsResult(
                False,
                f"Scene {scene_index} well name {canonical_name!r} conflicts with "
                f"CZI indices ({source_row}, {source_column}).",
            )

        previous = well_coordinates.setdefault(canonical_name, (source_row, source_column))
        if previous != (source_row, source_column):
            return CziHcsResult(False, f"Well {canonical_name!r} has inconsistent source indices.")

        region_value = getattr(scene, "RegionId", None)
        region_id = None if region_value is None else str(region_value)
        if region_id is not None:
            if region_id in seen_region_ids:
                return CziHcsResult(False, f"RegionId {region_id!r} is duplicated.")
            seen_region_ids.add(region_id)

        center_x, center_y = _parse_center(getattr(scene, "CenterPosition", None))
        parsed.append(
            {
                "canonical_name": canonical_name,
                "source_name": str(source_name),
                "source_row": source_row,
                "source_column": source_column,
                "row_index": row_index,
                "column_index": column_index,
                "scene_index": scene_index,
                "region_id": region_id,
                "position_name": _optional_str(getattr(scene, "Name", None)),
                "center_x": center_x,
                "center_y": center_y,
            }
        )

    fields_by_well: dict[str, list[dict[str, Any]]] = {}
    for item in sorted(parsed, key=lambda value: value["scene_index"]):
        fields_by_well.setdefault(item["canonical_name"], []).append(item)

    wells: list[CziWell] = []
    for canonical_name, field_items in sorted(
        fields_by_well.items(), key=lambda item: (item[1][0]["row_index"], item[1][0]["column_index"])
    ):
        first = field_items[0]
        fields = tuple(
            CziField(
                id=f"field:{item['region_id']}" if item["region_id"] else f"scene:{item['scene_index']}",
                field_index=field_index,
                scene_index=item["scene_index"],
                region_id=item["region_id"],
                position_name=item["position_name"],
                scene_center_x=item["center_x"],
                scene_center_y=item["center_y"],
            )
            for field_index, item in enumerate(field_items)
        )
        wells.append(
            CziWell(
                id=f"well:{canonical_name}",
                canonical_name=canonical_name,
                canonical_path=f"{_row_label(first['row_index'])}/{first['column_index'] + 1}",
                source_name=first["source_name"],
                source_row_index=first["source_row"],
                source_column_index=first["source_column"],
                row_index=first["row_index"],
                column_index=first["column_index"],
                external_id=None,
                fields=fields,
            )
        )

    template = _find_plate_template(czi_box)
    declared_rows = _optional_int(getattr(template, "ShapeRows", None)) if template else None
    declared_columns = _optional_int(getattr(template, "ShapeColumns", None)) if template else None
    if declared_rows is not None and any(well.row_index >= declared_rows for well in wells):
        return CziHcsResult(False, "An observed well row lies outside the declared plate template.")
    if declared_columns is not None and any(well.column_index >= declared_columns for well in wells):
        return CziHcsResult(False, "An observed well column lies outside the declared plate template.")

    plate_name = _optional_str(getattr(template, "Name", None)) if template else None
    source_path = getattr(czi_box, "filepath", None)
    source_name = Path(str(source_path)).name if source_path else "czi"
    plate = CziPlate(
        schema_version=HCS_SCHEMA_VERSION,
        id=f"plate:{source_name}",
        name=plate_name,
        declared_rows=declared_rows,
        declared_columns=declared_columns,
        naming_convention="row-letter-column-number",
        barcode=None,
        external_id=None,
        observed_row_indices=tuple(sorted({well.row_index for well in wells})),
        observed_column_indices=tuple(sorted({well.column_index for well in wells})),
        wells=tuple(wells),
    )
    return CziHcsResult(True, "Complete and consistent CZI well metadata was found.", plate)


def _get_scenes(czi_box: Box) -> list[Box]:
    try:
        value = czi_box.ImageDocument.Metadata.Information.Image.Dimensions.S.Scenes.Scene
    except AttributeError:
        return []
    if isinstance(value, Box):
        return [value]
    if isinstance(value, (BoxList, list)):
        return list(value)
    return []


def _find_plate_template(value: Any) -> Optional[Box]:
    """Find a template carrying declared plate dimensions without assuming XML depth."""

    if isinstance(value, (Box, dict)):
        if value.get("ShapeRows") is not None and value.get("ShapeColumns") is not None:
            return value if isinstance(value, Box) else Box(value)
        for child in value.values():
            found = _find_plate_template(child)
            if found is not None:
                return found
    elif isinstance(value, (BoxList, list)):
        for child in value:
            found = _find_plate_template(child)
            if found is not None:
                return found
    return None


def _parse_center(value: Any) -> tuple[Optional[float], Optional[float]]:
    if value is None:
        return None, None
    try:
        x_value, y_value = str(value).split(",", maxsplit=1)
        return float(x_value), float(y_value)
    except (TypeError, ValueError):
        return None, None


def _optional_int(value: Any) -> Optional[int]:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def _row_label(row_index: int) -> str:
    value = row_index + 1
    label = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        label = chr(ord("A") + remainder) + label
    return label
