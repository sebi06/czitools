# -*- coding: utf-8 -*-
"""Canonical HCS layout resolver for OME-Zarr export (Stage 5.5).

Both export backends consume a single normalized layout produced here. The
resolver prefers the explicit Stage 1 model (:attr:`CziMetadata.hcs`) and falls
back to the legacy :class:`~czitools.metadata_tools.sample.CziSampleInfo`
heuristics only when they yield a complete, unambiguous well mapping. Ambiguous
metadata raises :class:`ValueError` instead of manufacturing a plate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from czitools.metadata_tools.czi_metadata import CziMetadata
from czitools.export_tools.plate import extract_well_coordinates


@dataclass(frozen=True)
class HcsWellLayout:
    """A single well in the normalized export layout."""

    row: str
    column: str
    path: str
    well_id: str
    fields: Tuple[Tuple[int, int], ...]  # (field_index, scene_index)


@dataclass(frozen=True)
class HcsLayout:
    """Normalized plate layout shared by both export backends."""

    row_names: List[str]
    col_names: List[str]
    wells: List[HcsWellLayout]
    field_count: int
    source: str  # "hcs" or "sample"


def _pad_columns(columns: set[int], pad: bool) -> dict[int, str]:
    """Return a mapping of 1-based column number to its (optionally padded) label."""
    if not columns:
        return {}
    if pad:
        width = max(2, len(str(max(columns))))
        return {c: str(c).zfill(width) for c in columns}
    return {c: str(c) for c in columns}


def resolve_hcs_layout(mdata: CziMetadata, pad_columns: bool = True) -> HcsLayout:
    """Resolve a normalized HCS layout from CZI metadata.

    Args:
        mdata (CziMetadata): Metadata for the CZI file.
        pad_columns (bool): Zero-pad column labels (e.g. ``"04"``). Defaults to True.

    Returns:
        HcsLayout: Normalized rows, columns, and per-well field->scene mapping.

    Raises:
        ValueError: If neither the Stage 1 model nor a complete ``CziSampleInfo``
            mapping is available.
    """
    if mdata.hcs is not None:
        return _from_hcs_model(mdata, pad_columns)
    return _from_sample_info(mdata, pad_columns)


def _from_hcs_model(mdata: CziMetadata, pad_columns: bool) -> HcsLayout:
    plate = mdata.hcs
    assert plate is not None

    columns = {well.column_index + 1 for well in plate.wells}
    col_label = _pad_columns(columns, pad_columns)
    row_names = sorted({_row_letter(well.canonical_name) for well in plate.wells})
    col_names = [col_label[c] for c in sorted(columns)]

    wells: List[HcsWellLayout] = []
    field_count = 0
    for well in sorted(plate.wells, key=lambda w: (w.row_index, w.column_index)):
        row = _row_letter(well.canonical_name)
        column = col_label[well.column_index + 1]
        fields = tuple((f.field_index, f.scene_index) for f in well.fields)
        field_count = max(field_count, len(fields))
        wells.append(
            HcsWellLayout(
                row=row,
                column=column,
                path=f"{row}/{column}",
                well_id=well.canonical_name,
                fields=fields,
            )
        )

    return HcsLayout(row_names, col_names, wells, field_count, source="hcs")


def _from_sample_info(mdata: CziMetadata, pad_columns: bool) -> HcsLayout:
    sample = mdata.sample
    if sample is None or not sample.well_counter or not sample.well_scene_indices:
        reason = mdata.hcs_status.reason if mdata.hcs_status is not None else "no plate metadata"
        raise ValueError(
            "Cannot resolve an HCS plate layout: the Stage 1 HCS model is unavailable "
            f"({reason}) and CziSampleInfo does not provide an unambiguous well mapping."
        )

    row_names, col_names, _ = extract_well_coordinates(sample.well_counter, pad_columns=pad_columns)
    col_label = _pad_columns({int(c) for c in col_names}, pad_columns)

    wells: List[HcsWellLayout] = []
    field_count = 0
    for well_id in sorted(sample.well_scene_indices.keys(), key=_well_sort_key):
        scene_indices = sample.well_scene_indices[well_id]
        row = _row_letter(well_id)
        column_number = _column_number(well_id)
        column = col_label[column_number]
        fields = tuple((field_index, scene_index) for field_index, scene_index in enumerate(scene_indices))
        field_count = max(field_count, len(fields))
        wells.append(
            HcsWellLayout(
                row=row,
                column=column,
                path=f"{row}/{column}",
                well_id=well_id,
                fields=fields,
            )
        )

    wells.sort(key=lambda w: (w.row, _column_number(w.well_id)))
    return HcsLayout(row_names, col_names, wells, field_count, source="sample")


def _row_letter(well_name: str) -> str:
    return "".join(filter(str.isalpha, well_name)).upper()


def _column_number(well_name: str) -> int:
    digits = "".join(filter(str.isdigit, well_name))
    if not digits:
        raise ValueError(f"Well name {well_name!r} has no column number.")
    return int(digits)


def _well_sort_key(well_name: str) -> tuple[str, int]:
    return (_row_letter(well_name), _column_number(well_name))
