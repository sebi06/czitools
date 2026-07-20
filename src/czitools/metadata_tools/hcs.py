"""Explicit high-content-screening metadata derived from CZI scene XML.

The value models in this module are OME/NGFF-oriented, but they are not a
serializer and do not claim complete compliance with either specification.
Source row and column indices are retained as written by CZI (normally
1-based), while normalized indices are always 0-based.

Schema version: ``1.0``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import math
import os
import re
from statistics import median
from typing import Any, Optional, Union

from box import Box, BoxList

HCS_SCHEMA_VERSION = "1.0"
_WELL_NAME = re.compile(r"^([A-Za-z]+)0*([1-9][0-9]*)$")


@dataclass(frozen=True)
class CziField:
    """One CZI scene interpreted as a field of view within a well.

    Coordinate provenance is explicit and kept separate by source. The
    ``scene_center_*`` values come from ``Scene.CenterPosition`` (a single
    physical scene center declared in the scene XML). The ``stage_*`` and
    ``acquisition_z`` values are aggregated per-subblock measurements from the
    planetable and are only populated after :func:`enrich_hcs_with_planetable`
    runs. The two sources are never merged.

    Attributes:
        id (str): Deterministic, source-scoped identifier.
        field_index (int): Well-local 0-based field index.
        scene_index (int): Global 0-based CZI scene index.
        region_id (Optional[str]): Source-scoped CZI ``RegionId`` when present.
        position_name (Optional[str]): Scene ``Name`` when present.
        scene_center_x (Optional[float]): Scene-center X in ``position_unit``.
        scene_center_y (Optional[float]): Scene-center Y in ``position_unit``.
        position_unit (str): Unit of the scene-center coordinates.
        position_source (str): Provenance of the scene-center coordinates.
        acquisition_z (Optional[float]): Representative focus Z from subblocks.
        stage_x (Optional[float]): Representative subblock stage X.
        stage_y (Optional[float]): Representative subblock stage Y.
        stage_x_range (Optional[tuple]): ``(min, max)`` of subblock stage X.
        stage_y_range (Optional[tuple]): ``(min, max)`` of subblock stage Y.
        acquisition_z_range (Optional[tuple]): ``(min, max)`` of focus Z.
        stage_unit (str): Unit of the stage/focus coordinates.
        stage_source (Optional[str]): Provenance of the stage/focus coordinates.
        subblock_count (Optional[int]): Number of aggregated subblocks.
        position_conflict (bool): True when any aggregated range exceeds the
            configured tolerance, signalling that a single representative value
            may be misleading.
    """

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
    stage_x: Optional[float] = None
    stage_y: Optional[float] = None
    stage_x_range: Optional[tuple[float, float]] = None
    stage_y_range: Optional[tuple[float, float]] = None
    acquisition_z_range: Optional[tuple[float, float]] = None
    stage_unit: str = "micrometer"
    stage_source: Optional[str] = None
    subblock_count: Optional[int] = None
    position_conflict: bool = False


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


# ---------------------------------------------------------------------------
# Stage 2 - planetable position enrichment (lazy / opt-in)
# ---------------------------------------------------------------------------


def enrich_hcs_with_planetable(
    plate: CziPlate,
    filepath: Union[str, os.PathLike[str]],
    position_tolerance: float = 1.0,
) -> CziPlate:
    """Return a new plate whose fields carry aggregated subblock positions.

    The base plate is built from scene XML only. This function is opt-in
    because it scans subblock metadata (via the planetable) and is therefore
    unavailable for URL sources. Fields are matched to planetable rows by the
    ``S`` (scene index) column so enrichment never relies on row order.

    All models are immutable, so this returns a **new** plate; the input is not
    modified. For every field with matching subblocks the representative value
    is the median across all ``M/T/C/Z`` subblocks of that scene, and the full
    ``(min, max)`` range is preserved. ``position_conflict`` is set when any
    range exceeds ``position_tolerance`` (in micrometers), signalling that a
    single representative value may hide real variation.

    Args:
        plate (CziPlate): Plate built by :func:`build_hcs_metadata`.
        filepath (Union[str, os.PathLike[str]]): Path to the CZI image file.
        position_tolerance (float): Range (in micrometers) above which a field
            is flagged as having conflicting subblock positions. Defaults to 1.0.

    Returns:
        CziPlate: A new plate with enriched fields, or the original plate
            unchanged when no planetable positions are available.
    """

    # Lazy import keeps the heavy subblock/pandas dependency off the base
    # metadata path and avoids any import cycle.
    from czitools.utils.planetable import get_planetable

    planetable, _ = get_planetable(filepath)
    if planetable is None or planetable.empty:
        # URL sources and files without subblock positions cannot be enriched.
        return plate

    groups = {int(scene): sub for scene, sub in planetable.groupby("S")}

    new_wells: list[CziWell] = []
    for well in plate.wells:
        new_fields: list[CziField] = []
        for field_value in well.fields:
            group = groups.get(field_value.scene_index)
            if group is None or group.empty:
                new_fields.append(field_value)
                continue

            stage_x, range_x = _aggregate_positions(group["X[micron]"])
            stage_y, range_y = _aggregate_positions(group["Y[micron]"])
            stage_z, range_z = _aggregate_positions(group["Z[micron]"])
            conflict = (
                _exceeds_tolerance(range_x, position_tolerance)
                or _exceeds_tolerance(range_y, position_tolerance)
                or _exceeds_tolerance(range_z, position_tolerance)
            )
            new_fields.append(
                replace(
                    field_value,
                    stage_x=stage_x,
                    stage_y=stage_y,
                    stage_x_range=range_x,
                    stage_y_range=range_y,
                    acquisition_z=stage_z,
                    acquisition_z_range=range_z,
                    stage_unit="micrometer",
                    stage_source="planetable(StageXPosition,StageYPosition,FocusPosition)",
                    subblock_count=int(len(group)),
                    position_conflict=conflict,
                )
            )
        new_wells.append(replace(well, fields=tuple(new_fields)))

    return replace(plate, wells=tuple(new_wells))


def _aggregate_positions(values: Any) -> tuple[Optional[float], Optional[tuple[float, float]]]:
    """Return ``(median, (min, max))`` of numeric values, ignoring NaN/None."""

    numeric = [float(value) for value in values if value is not None and not _is_nan(value)]
    if not numeric:
        return None, None
    return float(median(numeric)), (min(numeric), max(numeric))


def _exceeds_tolerance(value_range: Optional[tuple[float, float]], tolerance: float) -> bool:
    if value_range is None:
        return False
    low, high = value_range
    return (high - low) > tolerance


def _is_nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def well_relative_field_positions(well: CziWell) -> Optional[dict[int, tuple[float, float]]]:
    """Return each field's XY offset from the well's field centroid.

    The origin is the centroid (arithmetic mean) of the well's field
    scene-center positions (``Scene.CenterPosition``, micrometers, same axis
    orientation as the source). Because a trustworthy plate/well origin is not
    generally available from CZI metadata, this uses the well's own fields as a
    well-relative reference frame. It returns ``None`` (an explicit "unavailable"
    result) when any field lacks a scene center, since no origin can be derived.

    Args:
        well (CziWell): The well whose field offsets are requested.

    Returns:
        Optional[dict[int, tuple[float, float]]]: Mapping of ``field_index`` to
            ``(dx, dy)`` offsets in micrometers, or ``None`` when unavailable.
    """

    centers = [(field_value.scene_center_x, field_value.scene_center_y) for field_value in well.fields]
    if not centers or any(x is None or y is None for x, y in centers):
        return None

    mean_x = sum(float(x) for x, _ in centers) / len(centers)
    mean_y = sum(float(y) for _, y in centers) / len(centers)
    return {
        field_value.field_index: (
            float(field_value.scene_center_x) - mean_x,
            float(field_value.scene_center_y) - mean_y,
        )
        for field_value in well.fields
    }


def well_absolute_field_positions(
    well: CziWell,
    source: str = "stage",
) -> Optional[dict[int, tuple[float, float]]]:
    """Return each field's absolute XY position in the source coordinate frame.

    Unlike :func:`well_relative_field_positions`, which expresses each field as an
    offset from the well's field centroid, this returns the raw absolute
    coordinates (micrometers, same axis orientation as the source) without any
    origin subtraction. Two provenance-separated sources are supported and are
    never mixed:

    - ``"stage"`` uses the planetable-derived ``stage_x`` / ``stage_y`` values,
      which are only populated after :func:`enrich_hcs_with_planetable` runs.
    - ``"scene_center"`` uses ``Scene.CenterPosition`` (``scene_center_x`` /
      ``scene_center_y``), available directly from scene XML.

    It returns ``None`` (an explicit "unavailable" result) when any field lacks a
    coordinate for the requested source, so a partial mapping is never returned.

    Args:
        well (CziWell): The well whose absolute field positions are requested.
        source (str): Coordinate source to use, either ``"stage"`` (default) or
            ``"scene_center"``.

    Returns:
        Optional[dict[int, tuple[float, float]]]: Mapping of ``field_index`` to
            absolute ``(x, y)`` coordinates in micrometers, or ``None`` when the
            requested source is unavailable for any field.

    Raises:
        ValueError: If ``source`` is not ``"stage"`` or ``"scene_center"``.

    Notes:
        ``"stage"`` positions only exist after :func:`enrich_hcs_with_planetable`
        has run; calling this with ``source="stage"`` on a plate built from scene
        XML alone returns ``None``. Before enrichment (or for URL sources that
        cannot be enriched), use ``source="scene_center"`` for absolute values.
    """

    if source == "stage":
        get_x, get_y = (lambda f: f.stage_x), (lambda f: f.stage_y)
    elif source == "scene_center":
        get_x, get_y = (lambda f: f.scene_center_x), (lambda f: f.scene_center_y)
    else:
        raise ValueError(f"Unsupported source {source!r}; expected 'stage' or 'scene_center'.")

    positions = [(get_x(field_value), get_y(field_value)) for field_value in well.fields]
    if not positions or any(x is None or y is None for x, y in positions):
        return None

    return {
        field_value.field_index: (float(get_x(field_value)), float(get_y(field_value))) for field_value in well.fields
    }


# ---------------------------------------------------------------------------
# Stage 3 - pure resolvers keyed by plate/well/field
# ---------------------------------------------------------------------------


def resolve_well(plate: CziPlate, well: str) -> CziWell:
    """Return the well matching ``well`` after name normalization.

    Args:
        plate (CziPlate): The plate to resolve against.
        well (str): A well name such as ``"B4"``, ``"b04"`` or ``"B/4"``.

    Returns:
        CziWell: The matching well.

    Raises:
        KeyError: If no well or more than one well matches the name.
        ValueError: If the name cannot be parsed as a well name.
    """

    return plate.get_well(_normalize_well_key(well))


def resolve_field(plate: CziPlate, well: str, field: Union[int, str] = 0) -> CziField:
    """Resolve one field within a well by local index or by ``RegionId``.

    ``field`` is interpreted as a well-local 0-based field index when an ``int``
    is given, and as a ``RegionId`` string otherwise.

    Args:
        plate (CziPlate): The plate to resolve against.
        well (str): A well name (see :func:`resolve_well`).
        field (Union[int, str]): Well-local field index or a ``RegionId``.

    Returns:
        CziField: The resolved field.

    Raises:
        IndexError: If an integer index is out of range.
        KeyError: If a ``RegionId`` does not match exactly one field.
        TypeError: If ``field`` is a bool.
    """

    target_well = resolve_well(plate, well)

    # bool is a subtype of int; reject it explicitly to avoid silent surprises.
    if isinstance(field, bool):
        raise TypeError("field must be an int index or a RegionId string, not bool.")

    if isinstance(field, int):
        if field < 0 or field >= len(target_well.fields):
            raise IndexError(
                f"Field index {field} is out of range for well "
                f"{target_well.canonical_name!r} with {len(target_well.fields)} field(s)."
            )
        return target_well.fields[field]

    matches = [item for item in target_well.fields if item.region_id == field]
    if len(matches) != 1:
        raise KeyError(
            f"Expected exactly one field with RegionId {field!r} in well "
            f"{target_well.canonical_name!r}; found {len(matches)}."
        )
    return matches[0]


def _normalize_well_key(well: str) -> str:
    """Accept ``"B4"``, ``"b04"`` and NGFF-style ``"B/4"`` well identifiers."""

    return well.replace("/", "") if isinstance(well, str) else well
