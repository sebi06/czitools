"""Convert a CZI well-plate to OME-Zarr HCS format (czitools Stage 5).

Run from the repository root (requires the optional export dependencies:
``pip install "czitools[omezarr]"``)::

    python demo/scripts/omezarr_convert_hcs.py
    python demo/scripts/omezarr_convert_hcs.py --backend omezarr
    python demo/scripts/omezarr_convert_hcs.py --no-validate

The conversion routes through the canonical HCS layout resolver, which prefers
the Stage 1 model (``CziMetadata.hcs``) and falls back to ``CziSampleInfo`` when
that is unavailable but unambiguous.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from czitools.export_tools import (
    convert_czi2hcs_ngff,
    convert_czi2hcs_omezarr,
    validate_ome_zarr,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CZI = REPOSITORY_ROOT / "data" / "WP96_4Pos_B4-10_DAPI.czi"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filepath", nargs="?", type=Path, default=DEFAULT_CZI)
    parser.add_argument(
        "--backend",
        choices=["ngff", "omezarr"],
        default="ngff",
        help="Write backend: 'ngff' (OME-NGFF v0.5) or 'omezarr' (ome-zarr-py, v0.4).",
    )
    parser.add_argument(
        "--no-pad-columns",
        action="store_true",
        help="Do not zero-pad well column numbers (e.g. use 'B/4' instead of 'B/04').",
    )
    parser.add_argument("--no-validate", action="store_true", help="Skip OME-NGFF validation.")
    args = parser.parse_args()

    pad_columns = not args.no_pad_columns

    if args.backend == "ngff":
        output_path = convert_czi2hcs_ngff(args.filepath, overwrite=True, pad_columns=pad_columns)
    else:
        output_path = convert_czi2hcs_omezarr(args.filepath, overwrite=True, pad_columns=pad_columns)

    print(f"\nWrote HCS OME-Zarr plate: {output_path}")

    if not args.no_validate:
        ok = validate_ome_zarr(output_path)
        print(
            f"Validation ({'ngff-zarr v0.5' if args.backend == 'ngff' else 'ome-zarr-py v0.4'}): "
            f"{'passed' if ok else 'failed'}"
        )


if __name__ == "__main__":
    main()
