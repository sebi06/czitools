"""Demonstrate planetable position enrichment of the HCS model.

Run from the repository root:

    python demo/scripts/hcs_position_enrichment.py

Enrichment is opt-in: it scans subblock metadata (via the planetable) and is
unavailable for URL sources. It returns a NEW immutable plate whose fields carry
aggregated stage/focus positions. For every field with matching subblocks the
representative value is the median across all M/T/C/Z subblocks; the full
(min, max) range is preserved and ``position_conflict`` is set when a range
exceeds the tolerance. Scene-center positions (Scene.CenterPosition) are kept
separate from subblock stage positions and are never merged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from czitools.metadata_tools import CziMetadata
from czitools.metadata_tools.hcs import well_absolute_field_positions, well_relative_field_positions

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CZI = REPOSITORY_ROOT / "data" / "WP96_4Pos_B4-10_DAPI.czi"


def show_position_enrichment(filepath: Path, selected_well: str, tolerance: float) -> None:
    metadata = CziMetadata(filepath)

    print(f"File: {filepath}")
    print(f"HCS detected: {metadata.hcs_status.detected}")
    if metadata.hcs is None:
        print(f"No HCS plate available: {metadata.hcs_status.reason}")
        return

    # Before enrichment: only the scene-center (Scene.CenterPosition) is known.
    well_before = metadata.hcs.get_well(selected_well)
    print(f"\nBefore enrichment, well {well_before.canonical_name}:")
    for field in well_before.fields:
        print(
            f"  local={field.field_index} scene={field.scene_index} "
            f"scene_center=({field.scene_center_x}, {field.scene_center_y}) {field.position_unit} "
            f"stage_x={field.stage_x}"
        )

    # Opt-in enrichment. This scans subblocks and replaces metadata.hcs with an
    # enriched copy. The original plate object is not mutated.
    print(f"\nEnriching from the planetable (position_tolerance={tolerance} um)...")
    enriched = metadata.enrich_hcs_positions(position_tolerance=tolerance)
    if enriched is None:
        print("Enrichment returned no plate (e.g. URL source or no subblock positions).")
        return

    well = enriched.get_well(selected_well)
    print(f"\nAfter enrichment, well {well.canonical_name}:")
    for field in well.fields:
        print(
            f"  local={field.field_index} scene={field.scene_index} "
            f"subblocks={field.subblock_count} conflict={field.position_conflict}"
        )
        print(
            f"      stage_x={field.stage_x} range={field.stage_x_range}  "
            f"stage_y={field.stage_y} range={field.stage_y_range}"
        )
        print(
            f"      focus_z={field.acquisition_z} range={field.acquisition_z_range} "
            f"({field.stage_unit}, source={field.stage_source})"
        )

    # Well-relative offsets use the centroid of the well's field scene-centers as
    # origin, or return None when any scene-center is missing.
    offsets = well_relative_field_positions(well)
    print(f"\nWell-relative field offsets (um from field centroid) for {well.canonical_name}:")
    if offsets is None:
        print("  unavailable (a field is missing its scene-center position)")
    else:
        for field_index, (dx, dy) in offsets.items():
            print(f"  local={field_index}: dx={dx:.3f}, dy={dy:.3f}")

    # Absolute field positions are the raw coordinates (no origin subtraction).
    # "stage" positions come from the planetable enrichment above; "scene_center"
    # positions come directly from scene XML. Each source returns None when any
    # field lacks a coordinate for it.
    stage_positions = well_absolute_field_positions(well, source="stage")
    print(f"\nAbsolute field positions (um, source='stage') for {well.canonical_name}:")
    if stage_positions is None:
        print("  unavailable (a field is missing its stage position)")
    else:
        for field_index, (x, y) in stage_positions.items():
            print(f"  local={field_index}: x={x:.3f}, y={y:.3f}")

    scene_positions = well_absolute_field_positions(well, source="scene_center")
    print(f"\nAbsolute field positions (um, source='scene_center') for {well.canonical_name}:")
    if scene_positions is None:
        print("  unavailable (a field is missing its scene-center position)")
    else:
        for field_index, (x, y) in scene_positions.items():
            print(f"  local={field_index}: x={x:.3f}, y={y:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filepath", nargs="?", type=Path, default=DEFAULT_CZI)
    parser.add_argument("--well", default="B04", help="Well to inspect (default: B04, normalized to B4).")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0,
        help="Position range (um) above which a field is flagged as conflicting (default: 1.0).",
    )
    args = parser.parse_args()
    show_position_enrichment(args.filepath, args.well, args.tolerance)


if __name__ == "__main__":
    main()
