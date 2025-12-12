"""Demo: using read_stacks() from czitools.read_tools.

This script demonstrates the read_stacks() function which reads all 2D planes
from a CZI file, grouped per stack. It supports:
  - Lazy loading via dask arrays
  - xarray DataArrays with labeled dimensions
  - Optional stacking of scenes with matching shapes

Dimension order is always: [V, R, I, H, M] + T + C + Z + Y + X [+ A]
"""

import xarray as xr
from czitools.read_tools import read_tools
from czitools.metadata_tools.czi_metadata import CziMetadata
from czitools.utils.napari_tools import display_xarray_in_napari

# Test files
# filepath = r"F:\AzureDevOps\RMS_CAREamics_Container\_archive\calc_mean_testimage.czi"
# filepath = r"F:\Testdata_Zeiss\CZI_Testfiles\S=3_1Pos_2Mosaic_T=2=Z=3_CH=2.czi"
# filepath = r"F:\Testdata_Zeiss\CZI_Testfiles\WP96_4Pos_B4-10_DAPI.czi"
filepath = r"F:\Github\czitools\data\CellDivision_T10_Z15_CH2_DCV_small.czi"

# show resulting stack inside napari
show_napari = True

if __name__ == "__main__":
    # Read scenes using the new function from czitools.read_tools
    # Options:
    #   use_dask=True  -> lazy loading (data read only when accessed)
    #   use_xarray=True -> returns xr.DataArray with labeled dimensions
    #   stack_scenes=True -> combine scenes if they have the same shape

    mdata = CziMetadata(filepath)

    result, dims, num_stacks = read_tools.read_stacks(
        filepath,
        use_dask=True,
        use_xarray=True,
        stack_scenes=True,
    )

    print("\n=== Results ===")
    print(f"Number of stacks: {num_stacks}")
    print(f"Dimension order: {dims}")

    if isinstance(result, list):
        # List of per-scene arrays
        for idx, arr in enumerate(result):
            if isinstance(arr, xr.DataArray):
                print(f"Stack {idx}: dims={arr.dims}, shape={arr.shape}, dtype={arr.dtype}")
            else:
                print(f"Stack {idx}: shape={arr.shape}, dims={dims}, dtype={arr.dtype}")
    else:
        # Single stacked array
        if isinstance(result, xr.DataArray):
            print(f"Stacked: dims={result.dims}, shape={result.shape}, dtype={result.dtype}")
        else:
            print(f"Stacked: shape={result.shape}, dims={dims}, dtype={result.dtype}")

    # With use_dask=True, result is backed by dask - no data loaded yet
    print(f"\nArray shape (no data loaded): {result.shape}")

    # Load only a subset (triggers read for just those planes)
    if isinstance(result, xr.DataArray) and "T" in result.dims and "C" in result.dims:
        subset = result.sel(T=0, C=0).compute()
        print(f"Subset shape after .compute(): {subset.shape}")

    # Or load everything
    # full_data = result.compute()

    if show_napari:

        # get the planes
        subset_planes = result.attrs["subset_planes"]

        # Delegate Napari display to the utility function
        display_xarray_in_napari(result, mdata, subset_planes)
