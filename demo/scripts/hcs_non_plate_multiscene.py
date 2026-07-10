"""Show that failed HCS detection does not prevent normal CZI access.

Run metadata-only (fast):

    python demo/scripts/hcs_non_plate_multiscene.py

Also read one small STCZ subset of the pixel data:

    python demo/scripts/hcs_non_plate_multiscene.py --read-pixels
"""

from __future__ import annotations

import argparse
from pathlib import Path

from czitools.metadata_tools import CziMetadata
from czitools.read_tools.read_tools import read_6darray


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CZI = REPOSITORY_ROOT / "data" / "S3_1Pos_2Mosaic_T2_Z3_CH2_sm.czi"


def show_non_plate_access(filepath: Path, read_pixels: bool) -> None:
    metadata = CziMetadata(filepath)

    print(f"File: {filepath}")
    print(f"CZI has scenes: {metadata.has_scenes}")
    print(f"HCS detected: {metadata.hcs_status.detected}")
    print(f"HCS reason: {metadata.hcs_status.reason}")
    print(f"HCS model: {metadata.hcs}")

    # HCS detection is an additional interpretation. Standard metadata remains valid.
    image = metadata.image_required
    print("\nStandard metadata remains available:")
    print(f"  S/T/C/Z sizes: {image.SizeS}/{image.SizeT}/{image.SizeC}/{image.SizeZ}")
    print(f"  pixel types: {metadata.pixeltypes}")
    print(f"  scene shapes consistent: {metadata.scene_shape_is_consistent}")
    print(f"  sample scene count: {metadata.sample.scene_count if metadata.sample else None}")

    if not read_pixels:
        print("\nPixel reading skipped. Add --read-pixels to read scene/time/channel/Z index 0.")
        return

    pixels, read_metadata = read_6darray(
        filepath,
        planes={"S": (0, 0), "T": (0, 0), "C": (0, 0), "Z": (0, 0)},
        use_xarray=True,
        use_dask=False,
        adapt_metadata=True,
    )
    if pixels is None:
        print("\nThis file could not be represented by read_6darray; use read_stacks for irregular scenes.")
        return

    print("\nPixel subset read successfully:")
    print(f"  dimensions: {pixels.dims}")
    print(f"  shape: {pixels.shape}")
    print(f"  dtype: {pixels.dtype}")
    print(f"  HCS status is still informational: {read_metadata.hcs_status.reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filepath", nargs="?", type=Path, default=DEFAULT_CZI)
    parser.add_argument("--read-pixels", action="store_true")
    args = parser.parse_args()
    show_non_plate_access(args.filepath, args.read_pixels)


if __name__ == "__main__":
    main()
