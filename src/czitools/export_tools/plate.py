# -*- coding: utf-8 -*-
"""HCS plate helpers for the OME-Zarr export tools.

Vendored (with light edits) from ``czi_omezarr_utils.hcs`` in the
``omezarr_playground`` repository as part of czitools Stage 5.

Contents:
  - ``extract_well_coordinates`` — row/column/well-path extraction from a well counter
  - ``PlateConfiguration`` / ``PlateType`` — standard microplate formats
  - ``define_plate`` / ``define_plate_by_well_count`` — ngff-zarr ``Plate`` builders
  - ``convert_hcs_omezarr2ozx`` — zip an HCS OME-Zarr directory into a single ``.ozx`` file

Note:
    These plate-format helpers are export-specific and intentionally distinct from
    the canonical read-side model in :mod:`czitools.metadata_tools.hcs`. The Stage 1
    model (``CziMetadata.hcs``) is the preferred source; see
    :mod:`czitools.export_tools.resolver`.
"""

import shutil
import logging
from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Dict, Optional, Union

from ngff_zarr.v04.zarr_metadata import Plate, PlateColumn, PlateRow, PlateWell
from ngff_zarr import write_store_to_zip

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Well coordinate helpers
# ---------------------------------------------------------------------------


def extract_well_coordinates(
    well_counter: dict,
    pad_columns: bool = True,
) -> tuple[list[str], list[str], list[str]]:
    """Extract unique row and column names from a well counter dictionary.

    Args:
        well_counter (dict): Dictionary with well positions as keys (e.g. ``{'B4': 4}``).
        pad_columns (bool): If ``True`` (default), zero-pad column numbers to at least
            2 digits (e.g. ``"04"``). The pad width grows automatically when the
            maximum column number requires more digits. If ``False``, columns are
            returned as-is.

    Returns:
        tuple[list[str], list[str], list[str]]: ``(row_names, col_names, well_paths)``
            where ``well_paths`` is the cartesian ``"row/column"`` grid.
    """
    rows: set[str] = set()
    cols: set[str] = set()

    for well in well_counter.keys():
        rows.add("".join(filter(str.isalpha, well)))
        cols.add("".join(filter(str.isdigit, well)))

    row_names = sorted(rows)
    sorted_cols = sorted(cols, key=int)
    if pad_columns:
        pad_width = max(2, len(str(max(int(c) for c in cols))))
        col_names = [c.zfill(pad_width) for c in sorted_cols]
    else:
        col_names = sorted_cols
    well_paths = [f"{row}/{col}" for row in row_names for col in col_names]

    return row_names, col_names, well_paths


# ---------------------------------------------------------------------------
# Standard plate format definitions
# ---------------------------------------------------------------------------


@dataclass
class PlateConfiguration:
    """Configuration for standard microplate formats."""

    rows: int
    columns: int
    name: str

    @property
    def total_wells(self) -> int:
        return self.rows * self.columns

    @property
    def row_labels(self) -> list[str]:
        """Generate row labels (A, B, C, ...)."""
        return [chr(ord("A") + i) for i in range(self.rows)]

    @property
    def column_labels(self) -> list[str]:
        """Generate column labels (1, 2, 3, ...)."""
        return [str(i) for i in range(1, self.columns + 1)]


@unique
class PlateType(Enum):
    """Standard microplate formats with their configurations."""

    PLATE_6 = PlateConfiguration(2, 3, "6-Well Plate")
    PLATE_24 = PlateConfiguration(4, 6, "24-Well Plate")
    PLATE_48 = PlateConfiguration(6, 8, "48-Well Plate")
    PLATE_96 = PlateConfiguration(8, 12, "96-Well Plate")
    PLATE_384 = PlateConfiguration(16, 24, "384-Well Plate")
    PLATE_1536 = PlateConfiguration(32, 48, "1536-Well Plate")


PLATE_FORMATS: Dict[int, PlateConfiguration] = {
    6: PlateType.PLATE_6.value,
    24: PlateType.PLATE_24.value,
    48: PlateType.PLATE_48.value,
    96: PlateType.PLATE_96.value,
    384: PlateType.PLATE_384.value,
    1536: PlateType.PLATE_1536.value,
}


def define_plate(plate_type: PlateType, field_count: int = 1) -> Plate:
    """Create a ``Plate`` metadata object for any standard plate format.

    Args:
        plate_type (PlateType): Plate format enum value.
        field_count (int): Number of fields per well. Defaults to 1.

    Returns:
        Plate: The ngff-zarr plate metadata object.
    """
    config = plate_type.value
    columns = [PlateColumn(name=label) for label in config.column_labels]
    rows = [PlateRow(name=label) for label in config.row_labels]
    wells = [
        PlateWell(path=f"{row.name}/{col.name}", rowIndex=row_idx, columnIndex=col_idx)
        for row_idx, row in enumerate(rows)
        for col_idx, col in enumerate(columns)
    ]
    return Plate(name=config.name, columns=columns, rows=rows, wells=wells, field_count=field_count)


def define_plate_by_well_count(well_count: int, field_count: int = 1) -> Plate:
    """Create a ``Plate`` metadata object by specifying the number of wells.

    Args:
        well_count (int): Number of wells (6, 24, 48, 96, 384, or 1536).
        field_count (int): Number of fields per well. Defaults to 1.

    Returns:
        Plate: The ngff-zarr plate metadata object.

    Raises:
        ValueError: If ``well_count`` is not a standard format.
    """
    if well_count not in PLATE_FORMATS:
        raise ValueError(f"Unsupported well count: {well_count}. Available formats: {list(PLATE_FORMATS.keys())}")
    config = PLATE_FORMATS[well_count]
    columns = [PlateColumn(name=label) for label in config.column_labels]
    rows = [PlateRow(name=label) for label in config.row_labels]
    wells = [
        PlateWell(path=f"{row.name}/{col.name}", rowIndex=row_idx, columnIndex=col_idx)
        for row_idx, row in enumerate(rows)
        for col_idx, col in enumerate(columns)
    ]
    return Plate(name=config.name, columns=columns, rows=rows, wells=wells, field_count=field_count)


# ---------------------------------------------------------------------------
# OZX conversion
# ---------------------------------------------------------------------------


def convert_hcs_omezarr2ozx(
    hcs_omezarr_path: Union[str, Path],
    remove_omezarr: bool = False,
    version: str = "0.5",
) -> Optional[Path]:
    """Convert an HCS OME-Zarr directory into a single-file ``.ozx`` archive.

    Args:
        hcs_omezarr_path (Union[str, Path]): Path to the input HCS OME-Zarr directory.
        remove_omezarr (bool): Remove the original directory after conversion.
        version (str): NGFF version string. Defaults to ``"0.5"``.

    Returns:
        Optional[Path]: Path to the created ``.ozx`` file, or ``None`` if the source
            directory does not exist.
    """
    hcs_omezarr_path = Path(hcs_omezarr_path)

    if not hcs_omezarr_path.exists():
        logger.info(f"OME-ZARR does not exist at {hcs_omezarr_path}. Cannot convert to OZX.")
        return None

    logger.info(f"Converting HCS OME-ZARR at {hcs_omezarr_path} to OZX format.")

    if hcs_omezarr_path.name.endswith(".ome.zarr"):
        ozx_name = hcs_omezarr_path.name.replace(".ome.zarr", ".ozx")
    else:
        ozx_name = hcs_omezarr_path.name + ".ozx"

    ozx_path = hcs_omezarr_path.parent / ozx_name

    write_store_to_zip(
        str(hcs_omezarr_path),
        str(ozx_path),
        version=version,
    )

    if remove_omezarr:
        logger.info(f"Removing original OME-ZARR directory at {hcs_omezarr_path}.")
        shutil.rmtree(hcs_omezarr_path, ignore_errors=False, onexc=None)

    return ozx_path
