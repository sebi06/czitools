"""Demonstrate the sample/scene metadata improvements.

The preferred ``field_centerX/Y`` fields preserve missing-value information:
an explicit coordinate of ``0.0`` remains ``0.0``, while absent or malformed
``Scene.CenterPosition`` metadata becomes ``None``. The deprecated
``scene_stageX/Y`` compatibility properties convert ``None`` to ``0.0`` and
therefore cannot distinguish those two cases.

Run from the repository root:

    python demo/scripts/hcs_sample_metadata.py

Pass another CZI path as the first argument to inspect a different file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from czitools.metadata_tools import CziMetadata

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CZI = REPOSITORY_ROOT / "data" / "WP96_4Pos_B4-10_DAPI.czi"


def show_sample_metadata(filepath: Path) -> None:
    """Print the lossless and compatibility views of scene metadata."""

    metadata = CziMetadata(filepath)
    sample = metadata.sample
    if sample is None:
        print("No sample metadata object is available.")
        return

    print(f"File: {filepath}")
    print(f"Scene count: {sample.scene_count}")
    print(f"Unique well count: {sample.well_unique_number}")
    print(f"Multiple fields per well: {sample.multipos_per_well}")
    print(f"Deprecated well_total_number: {sample.well_total_number} (scene count)")

    # Stage 0 guarantees that these collections have one entry per scene.
    per_scene_lengths = {
        "well names": len(sample.well_array_names),
        "well indices": len(sample.well_indices),
        "position names": len(sample.well_position_names),
        "row indices": len(sample.well_rowID),
        "column indices": len(sample.well_colID),
        "field center X": len(sample.field_centerX),
        "field center Y": len(sample.field_centerY),
        "region IDs": len(sample.well_region_ids),
    }
    print("\nPer-scene collection lengths:")
    for label, length in per_scene_lengths.items():
        print(f"  {label:18}: {length}")

    if sample.scene_count:
        print("\nFirst scene:")
        print(f"  well: {sample.well_array_names[0] or '<missing>'}")
        print(f"  RegionId: {sample.well_region_ids[0]}")
        print(
            "  scene center (preserves missing values): "
            f"({sample.field_centerX[0]}, {sample.field_centerY[0]}) micrometers"
        )
        print("  legacy scene_stage view: " f"({sample.scene_stageX[0]}, {sample.scene_stageY[0]})")

    print("\nMissing-value behavior:")
    print("  field_centerX/Y keep a real 0.0 coordinate as 0.0.")
    print("  They use None when Scene.CenterPosition is absent or malformed.")
    print("  Legacy scene_stageX/Y convert None to 0.0, so that distinction is lost.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filepath", nargs="?", type=Path, default=DEFAULT_CZI)
    args = parser.parse_args()
    show_sample_metadata(args.filepath)


if __name__ == "__main__":
    main()
