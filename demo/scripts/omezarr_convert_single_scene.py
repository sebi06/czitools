"""Convert a single CZI scene to a standard (non-HCS) OME-Zarr image (Stage 5).

Run from the repository root (requires ``pip install "czitools[omezarr]"``)::

    python demo/scripts/omezarr_convert_single_scene.py
    python demo/scripts/omezarr_convert_single_scene.py --backend omezarr --scene 0

Uses ``read_6darray`` to read one scene, squeezes the scene dimension to a 5D
``(T, C, Z, Y, X)`` array, and writes it with the chosen backend.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr

from czitools.read_tools import read_tools
from czitools.export_tools import write_omezarr, write_omezarr_ngff, validate_ome_zarr

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CZI = REPOSITORY_ROOT / "data" / "CellDivision_T3_Z5_CH2_X240_Y170.czi"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filepath", nargs="?", type=Path, default=DEFAULT_CZI)
    parser.add_argument("--scene", type=int, default=0, help="Scene index to convert (default: 0).")
    parser.add_argument(
        "--backend",
        choices=["ngff", "omezarr"],
        default="ngff",
        help="Write backend: 'ngff' (OME-NGFF v0.5) or 'omezarr' (ome-zarr-py, v0.4).",
    )
    parser.add_argument("--no-validate", action="store_true", help="Skip OME-NGFF validation.")
    args = parser.parse_args()

    array, mdata = read_tools.read_6darray(str(args.filepath), planes={"S": (args.scene, args.scene)}, use_xarray=True)
    assert isinstance(array, xr.DataArray), "Expected an xarray DataArray"
    array = array.squeeze("S")  # 6D -> 5D (T, C, Z, Y, X)

    stem = args.filepath.stem
    if args.backend == "ngff":
        output_path = args.filepath.parent / f"{stem}_ngff.ome.zarr"
        write_omezarr_ngff(array, output_path, mdata, scale_factors=[2, 4], overwrite=True)
    else:
        output_path = args.filepath.parent / f"{stem}.ome.zarr"
        write_omezarr(array, zarr_path=output_path, metadata=mdata, overwrite=True)

    print(f"\nWrote OME-Zarr image: {output_path}")

    if not args.no_validate:
        ok = validate_ome_zarr(output_path)
        print(f"Validation: {'passed' if ok else 'failed'}")


if __name__ == "__main__":
    main()
