"""
validate_omezarr.py
===================
CLI wrapper around :func:`czitools.export_tools.validate_ome_zarr`.

Usage::

    python scripts/validate_omezarr.py <path_to_omezarr>
"""

import argparse
from pathlib import Path

from czitools.export_tools import validate_ome_zarr


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a local OME-ZARR file against the OME-NGFF v0.5 spec.")
    parser.add_argument("path", type=Path, help="Path to the OME-ZARR directory or archive.")
    args = parser.parse_args()

    valid = validate_ome_zarr(args.path)
    raise SystemExit(0 if valid else 1)


if __name__ == "__main__":
    main()
