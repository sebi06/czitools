"""Demo: using read_6darray_lazy() from czitools.read_tools.

This script demonstrates the read_6darray_lazy() function which reads a CZI file
as a 6D dask array with delayed plane reading. The actual pixel data is only
loaded when accessed (e.g., via .compute() or indexing).

Features:
  - Lazy loading via dask arrays (no data read until needed)
  - Optional xarray DataArray output with labeled dimensions
  - Optional Z-stack chunking for efficient processing
  - Substack selection via planes parameter

Dimension order is always: STCZYX (or STCZYXA for RGB images)
"""

from czitools.read_tools import read_tools

# Test file - same as read_mdstack.py
filepath = r"F:\Testdata_Zeiss\CZI_Testfiles\WP96_4Pos_B4-10_DAPI.czi"


if __name__ == "__main__":
    print("=" * 60)
    print("Demo: read_6darray_lazy()")
    print("=" * 60)

    # Basic lazy loading - returns dask array
    print("\n1. Basic lazy loading (dask array):")
    array6d, mdata = read_tools.read_6darray_lazy(
        filepath,
        use_xarray=False,  # Return plain dask array
    )

    print(f"   Array type: {type(array6d)}")
    print(f"   Shape: {array6d.shape}")
    print(f"   Dtype: {array6d.dtype}")
    print(f"   Chunks: {array6d.chunks}")
    print(f"   No data loaded yet - array is lazy!")

    # With xarray for labeled dimensions
    print("\n2. Lazy loading with xarray:")
    array6d_xr, mdata = read_tools.read_6darray_lazy(
        filepath,
        use_xarray=True,  # Return xr.DataArray with labeled dims
    )

    print(f"   Array type: {type(array6d_xr)}")
    print(f"   Dimensions: {array6d_xr.dims}")
    print(f"   Shape: {array6d_xr.shape}")
    print(f"   Coordinates: {list(array6d_xr.coords.keys())}")

    # With Z-stack chunking (useful for processing)
    print("\n3. Lazy loading with Z-stack chunking:")
    array6d_chunked, mdata = read_tools.read_6darray_lazy(
        filepath,
        chunk_zyx=True,  # Chunk so each Z-stack is one chunk
        use_xarray=True,
    )

    print(f"   Chunks: {array6d_chunked.chunks}")

    # Reading a substack (only specific planes)
    print("\n4. Reading a substack (first 2 scenes only):")
    array6d_sub, mdata = read_tools.read_6darray_lazy(
        filepath,
        planes={"S": (0, 1)},  # Only scenes 0 and 1
        use_xarray=True,
    )

    print(f"   Shape: {array6d_sub.shape}")
    print(f"   Dimensions: {array6d_sub.dims}")

    # Actually load some data
    print("\n5. Loading a subset of data:")
    # Select first scene, first timepoint, first channel
    subset = array6d_xr.isel(S=0, T=0, C=0)
    print(f"   Subset shape (before compute): {subset.shape}")

    # This triggers the actual read
    subset_loaded = subset.compute()
    print(f"   Subset shape (after compute): {subset_loaded.shape}")
    print(f"   Data loaded! Min={subset_loaded.values.min()}, Max={subset_loaded.values.max()}")

    # Show metadata info
    print("\n6. Metadata from CZI:")
    print(f"   Filepath: {mdata.filepath}")
    print(f"   Pixel type: {mdata.npdtype_list}")
    print(
        f"   Dimensions: S={mdata.image.SizeS}, T={mdata.image.SizeT}, " f"C={mdata.image.SizeC}, Z={mdata.image.SizeZ}"
    )
    print(f"   Scaling (µm): X={mdata.scale.X}, Y={mdata.scale.Y}, Z={mdata.scale.Z}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
