"""Demonstrate Stage 1 plate detection and Plate -> Well -> Field traversal.

Run from the repository root:

    python demo/scripts/hcs_stage1_plate_model.py

The model is immutable and retains both the original 1-based CZI well indices
and normalized 0-based indices suitable for programmatic use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from czitools.metadata_tools import CziMetadata


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CZI = REPOSITORY_ROOT / "data" / "WP96_4Pos_B4-10_DAPI.czi"


def show_hcs_model(filepath: Path, selected_well: str) -> None:
    metadata = CziMetadata(filepath)

    print(f"File: {filepath}")
    print(f"HCS detected: {metadata.hcs_status.detected}")
    print(f"Detection reason: {metadata.hcs_status.reason}")
    if metadata.hcs is None:
        print("No Plate -> Well -> Field hierarchy was created.")
        return

    plate = metadata.hcs
    print("\nPlate:")
    print(f"  ID: {plate.id}")
    print(f"  name: {plate.name}")
    print(f"  model schema: {plate.schema_version}")
    print(f"  declared dimensions: {plate.declared_rows} rows x {plate.declared_columns} columns")
    print(f"  observed 0-based rows: {plate.observed_row_indices}")
    print(f"  observed 0-based columns: {plate.observed_column_indices}")
    print(f"  wells present in this CZI: {len(plate.wells)}")
    print(f"  fields present in this CZI: {sum(len(well.fields) for well in plate.wells)}")

    print("\nWells:")
    for well in plate.wells:
        print(
            f"  {well.canonical_name:>3}  path={well.canonical_path:<4} "
            f"CZI=({well.source_row_index}, {well.source_column_index}) "
            f"normalized=({well.row_index}, {well.column_index}) "
            f"fields={len(well.fields)}"
        )

    # get_well normalizes capitalization and zero padding, so B04 resolves to B4.
    well = plate.get_well(selected_well)
    print(f"\nFields in requested well {selected_well!r} -> {well.canonical_name}:")
    for field in well.fields:
        print(
            f"  local={field.field_index} scene={field.scene_index} id={field.id} "
            f"region={field.region_id} center=({field.scene_center_x}, {field.scene_center_y}) "
            f"{field.position_unit}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filepath", nargs="?", type=Path, default=DEFAULT_CZI)
    parser.add_argument("--well", default="B04", help="Well to inspect (default: B04, normalized to B4).")
    args = parser.parse_args()
    show_hcs_model(args.filepath, args.well)


if __name__ == "__main__":
    main()
