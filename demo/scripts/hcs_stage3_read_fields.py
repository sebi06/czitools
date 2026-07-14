"""Demonstrate Stage 3 convenience reads keyed by plate/well/field.

Run from the repository root:

    python demo/scripts/hcs_stage3_read_fields.py

The pure resolvers (`resolve_well`, `resolve_field`) turn well names and field
selectors into concrete scenes, and `read_field` / `read_well` reuse the
existing single-scene read path (`read_6darray` with planes={"S": (s, s)}).
A well name may be given as "B4", "b04" or NGFF-style "B/4"; a field may be a
well-local 0-based index (int) or a source-scoped RegionId (str).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from czitools.metadata_tools import CziMetadata
from czitools.metadata_tools.hcs import resolve_field, resolve_well
from czitools.read_tools import read_field, read_well

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CZI = REPOSITORY_ROOT / "data" / "WP96_4Pos_B4-10_DAPI.czi"


def show_reads(filepath: Path, selected_well: str) -> None:
    metadata = CziMetadata(filepath)

    print(f"File: {filepath}")
    print(f"HCS detected: {metadata.hcs_status.detected}")
    if metadata.hcs is None:
        print(f"No HCS plate available: {metadata.hcs_status.reason}")
        return

    plate = metadata.hcs

    # Pure resolution (no pixel reading). resolve_well normalizes the name.
    well = resolve_well(plate, selected_well)
    print(f"\nResolved well {selected_well!r} -> {well.canonical_name} with {len(well.fields)} field(s).")
    first_field = resolve_field(plate, selected_well, 0)
    print(f"  field 0 -> scene={first_field.scene_index} id={first_field.id} region={first_field.region_id}")

    # Read a single field (field 0) -> one scene, returned as an xarray.DataArray.
    print(f"\nread_field({well.canonical_name!r}, 0):")
    array, _ = read_field(filepath, selected_well, 0, use_xarray=True)
    if array is not None:
        print(f"  shape={array.shape} dims={getattr(array, 'dims', None)}")

    # Read a single field by its RegionId (equivalent to the local index above).
    if first_field.region_id is not None:
        print(f"\nread_field({well.canonical_name!r}, region_id={first_field.region_id!r}):")
        array_by_region, _ = read_field(filepath, selected_well, first_field.region_id)
        if array_by_region is not None:
            print(f"  shape={array_by_region.shape}")

    # Read all fields of the well as a list (fields may differ in shape).
    print(f"\nread_well({well.canonical_name!r}) -> list of per-field arrays:")
    arrays, _ = read_well(filepath, selected_well)
    if isinstance(arrays, list):
        for index, field_array in enumerate(arrays):
            print(f"  field {index}: shape={field_array.shape}")

    # Stack the fields along S (only allowed when all field shapes match).
    print(f"\nread_well({well.canonical_name!r}, stack=True) -> single stacked array:")
    try:
        stacked, _ = read_well(filepath, selected_well, stack=True)
        if stacked is not None:
            print(f"  stacked shape={stacked.shape} (S = number of fields)")
    except ValueError as error:
        print(f"  cannot stack: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filepath", nargs="?", type=Path, default=DEFAULT_CZI)
    parser.add_argument("--well", default="B04", help="Well to read (default: B04, normalized to B4).")
    args = parser.parse_args()
    show_reads(args.filepath, args.well)


if __name__ == "__main__":
    main()
