"""
Example script demonstrating CZI to OME-ZARR conversion.

This script provides examples for converting CZI (Carl Zeiss Image) files to OME-ZARR format
in two modes:
1. HCS (High Content Screening) format - for multi-well plate data
2. Standard OME-ZARR format - for single scene data

Supports two backend libraries:
- ome-zarr-py (OME_ZARR)
- ngff-zarr (NGFF_ZARR)
"""

from ome_zarr_utils import (
    convert_czi2hcs_omezarr,
    convert_czi2hcs_ngff,
    omezarr_package,
    write_omezarr,
    write_omezarr_ngff,
)
import ngff_zarr as nz
from pathlib import Path
from czitools.read_tools import read_tools
from typing import Optional


def main() -> None:
    """Main function to execute CZI to OME-ZARR conversion."""

    # ========== Configuration Parameters ==========
    # Toggle to display the result in napari viewer (requires napari installation)
    show_napari: bool = True

    # Mode selection: True for HCS (multi-well plate), False for standard OME-ZARR
    write_hcs: bool = False

    # Backend library selection: OME_ZARR (ome-zarr-py) or NGFF_ZARR (ngff-zarr)
    ome_package = omezarr_package.OME_ZARR

    # Scene ID for non-HCS format (ignored if write_hcs=True)
    scene_id: int = 0

    # ========== Input File Path ==========
    # Option 1: Use relative path to test data in repository
    # filepath: str = str(Path(__file__).parent.parent.parent / "data" / "WP96_4Pos_B4-10_DAPI.czi")
    filepath: str = str(Path(__file__).parent.parent.parent / "data" / "CellDivision_T10_Z15_CH2_DCV_small.czi")

    # Option 2: Use absolute path to external test data
    # filepath: str = r"F:\Testdata_Zeiss\CZI_Testfiles\testwell96.czi"

    # Validate input file exists
    if not Path(filepath).exists():
        raise FileNotFoundError(f"CZI file not found: {filepath}")

    # ========== HCS Format Conversion ==========
    if write_hcs:
        print(f"Converting CZI to HCS-ZARR format using {ome_package.name}...")

        if ome_package == omezarr_package.OME_ZARR:
            # Convert using ome-zarr-py backend
            zarr_output_path = convert_czi2hcs_omezarr(filepath, overwrite=True)

        elif ome_package == omezarr_package.NGFF_ZARR:
            # Convert using ngff-zarr backend
            zarr_output_path = convert_czi2hcs_ngff(filepath, overwrite=True)
        else:
            raise ValueError(f"Unsupported ome_package: {ome_package}")

        print(f"✓ Converted to OME-ZARR HCS format at: {zarr_output_path}")

        # Validate the HCS-ZARR file against OME-NGFF specification
        # This ensures proper metadata structure for multi-well plate data
        print("Validating created HCS-ZARR file against schema...")
        _ = nz.from_hcs_zarr(zarr_output_path, validate=True)
        print("✓ Validation successful - HCS metadata conforms to specification.")

    # ========== Standard OME-ZARR Conversion (Non-HCS) ==========
    if not write_hcs:
        print(f"Converting CZI scene {scene_id} to standard OME-ZARR format...")

        # Read the CZI file as a 6D array with dimension order STCZYX(A)
        # S=Scene, T=Time, C=Channel, Z=Z-stack, Y=Height, X=Width, A=Angle (optional)
        array, mdata = read_tools.read_6darray(filepath, planes={"S": (scene_id, scene_id)}, use_xarray=True)

        # Extract the specified scene (remove Scene dimension to get 5D array)
        # write_omezarr requires 5D array (TCZYX), not 6D (STCZYX)
        array = array.squeeze("S")  # Remove the Scene dimension
        print(f"Array Type: {type(array)}, Shape: {array.shape}, Dtype: {array.dtype}")

        if ome_package == omezarr_package.OME_ZARR:
            # Generate output path with .ome.zarr extension
            zarr_output_path: Path = Path(str(filepath)[:-4] + ".ome.zarr")

            # Write OME-ZARR using ome-zarr-py backend
            zarr_output_path: Optional[str] = write_omezarr(
                array, zarr_path=str(zarr_output_path), metadata=mdata, overwrite=True
            )
            print(f"✓ Written OME-ZARR using ome-zarr-py: {zarr_output_path}")

        elif ome_package == omezarr_package.NGFF_ZARR:
            # Generate output path with _ngff.ome.zarr extension
            zarr_output_path: Path = Path(str(filepath)[:-4] + "_ngff.ome.zarr")

            # Write OME-ZARR using ngff-zarr backend with multi-resolution pyramid
            # scale_factors=[2, 4] creates 3 resolution levels (1x, 2x, 4x downsampled)
            _ = write_omezarr_ngff(array, zarr_output_path, mdata, scale_factors=[2, 4], overwrite=True)
            print(f"✓ Written OME-ZARR using ngff-zarr: {zarr_output_path}")
        else:
            raise ValueError(f"Unsupported ome_package: {ome_package}")

    # ========== Optional: Visualize in napari ==========
    # Open the converted ZARR file in napari viewer for interactive visualization
    if show_napari:
        try:
            import napari

            print("Opening ZARR file in napari viewer...")
            viewer = napari.Viewer()
            viewer.open(zarr_output_path, plugin="napari-ome-zarr")
            napari.run()
        except ImportError:
            print("Warning: napari is not installed. Skipping visualization.")
            print("Install with: pip install napari[all] napari-ome-zarr")


if __name__ == "__main__":
    main()
