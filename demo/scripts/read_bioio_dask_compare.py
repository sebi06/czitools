"""Demo: comparing lazy CZI readers.

This script demonstrates two lazy-loading options for CZI files:
    - czitools.read_tools.read_stacks_stacked()
    - read_czi_bioio() based on BioImage

It supports:
  - Lazy loading via dask arrays
  - xarray DataArrays with labeled dimensions
  - Optional stacking of scenes with matching shapes

Dimension order is always: [V, R, I, H, M] + T + C + Z + Y + X [+ A]
"""

from czitools.read_tools import read_tools
from bioio import BioImage
import bioio_czi
from typing import Dict, Tuple, Optional, Any, Callable, cast
import numpy as np
from czitools.utils.ndv_tools import create_luts_ndv
import ndv


def read_czi_bioio(
    filepath,
    planes: Optional[Dict[str, Tuple[int, int]]] = None,
):
    """Read a CZI file using bioio library and return it as a dask-backed xarray.

    This function provides an alternative to czitools.read_tools.read_stacks() by
    using the bioio library. It loads a CZI file, optionally reconstructs mosaics,
    and returns the image data as an xarray DataArray with dask arrays for lazy
    loading. The output format is designed to match read_stacks() conventions:
    - Dimension naming follows CZI standard (using 'S' for scenes, not bioio's 'I')
    - 0-based integer indexing for all dimensions
    - Canonical dimension ordering: S, T, C, Z, Y, X, A (where applicable)

    Args:
        filepath (str or Path): Path to the CZI file to be read.
        planes (Optional[Dict[str, Tuple[int, int]]]): Dictionary specifying the
            range of planes to include for each dimension. Keys should be dimension
            names ('S', 'T', 'C', 'Z') and values should be tuples of (start, end)
            indices (inclusive, 0-based). If None, or if specific dimensions are
            not provided, defaults to the full range available in the file.
            Example: {"C": (0, 2), "Z": (5, 10)} selects channels 0-2 and z-planes 5-10.
            Defaults to None.

    Returns:
        tuple: A tuple containing three elements:
            - array (xarray.DataArray): The image data as a dask-backed xarray with
              dimensions in standard order (S, T, C, Z, Y, X, A where applicable).
              Attributes include: 'stack', 'filepath', 'axes', and 'subset_planes'.
            - dims (list[str]): List of dimension names present in the array, in
              canonical order.
            - num_stacks (int): Number of scenes/stacks in the CZI file.

    Notes:
        - This function uses bioio's get_xarray_dask_stack() for efficiency
        - The "I" (Image) dimension from bioio is renamed to "S" (Scene) for CZI standard
        - All data loading is lazy via dask - no actual pixel data is read until accessed
        - Coordinates are regenerated to ensure 0-based integer indexing

    Examples:
        >>> # Read entire CZI file
        >>> array, dims, num_stacks = read_czi_bioio("image.czi")
        >>>
        >>> # Read subset of channels and z-planes
        >>> array, dims, num_stacks = read_czi_bioio(
        ...     "image.czi",
        ...     planes={"C": (0, 2), "Z": (5, 10)}
        ... )
        >>>
        >>> # Access data (triggers actual reading)
        >>> first_channel = array.sel(C=0).compute()
    """

    # Load the CZI file using BioImage from bioio
    # - reconstruct_mosaic=True: Assembles tiled/mosaic acquisitions into single images
    # - use_aicspylibczi=False: Use bioio's native reader (faster for most cases)
    img = BioImage(filepath, reader=bioio_czi.Reader, reconstruct_mosaic=True, use_aicspylibczi=False)

    # Set default planes if not provided - use full range for each dimension
    if planes is None:
        planes = {
            "S": (0, len(img.scenes) - 1),
            "T": (0, img.dims.T - 1),
            "C": (0, img.dims.C - 1),
            "Z": (0, img.dims.Z - 1),
        }
    else:
        # Fill in missing dimension ranges with defaults
        if "S" not in planes:
            planes["S"] = (0, len(img.scenes) - 1)
        if "T" not in planes:
            planes["T"] = (0, img.dims.T - 1)
        if "C" not in planes:
            planes["C"] = (0, img.dims.C - 1)
        if "Z" not in planes:
            planes["Z"] = (0, img.dims.Z - 1)

    # Use bioio's efficient stacking method to combine scenes with matching shapes
    # - drop_non_matching_scenes=True: Only stack scenes that have identical dimensions
    # - Returns a dask-backed xarray with "I" dimension for stacked images
    array = img.get_xarray_dask_stack(drop_non_matching_scenes=True)

    # Rename "I" dimension to "S" to match CZI standard naming convention
    # bioio uses "I" (Image) for stacked scenes, but CZI standard uses "S" (Scene)
    if "I" in array.dims:
        array = array.rename({"I": "S"})

    # Get dimensions in canonical order for consistency with read_stacks()
    # Standard order: S, T, C, Z, Y, X, A (where A is for RGB/RGBA channels)
    dims_from_bioio = list(array.dims)
    standard_order = ["S", "T", "C", "Z", "Y", "X", "A"]
    dims = [d for d in standard_order if d in dims_from_bioio]

    # Rebuild coordinates to match read_stacks format (0-based integer indices)
    # This ensures consistent indexing regardless of the original file's coordinate system
    new_coords = {}
    for dim in array.dims:
        dim_name = str(dim)
        if dim_name in ["S", "T", "C", "Z"]:
            # For dimension-like axes: use 0-based range matching planes selection
            if dim_name in planes:
                start, end = planes[dim_name]
                new_coords[dim_name] = np.arange(start, end + 1)
            else:
                # If not in planes dict, use full size starting from 0
                new_coords[dim_name] = np.arange(array.sizes[dim_name])
        elif dim_name in ["Y", "X", "A"]:
            # For spatial and pixel-type dimensions: use 0-based range of actual size
            new_coords[dim_name] = np.arange(array.sizes[dim_name])

    # Apply the new coordinate system to the array
    array = array.assign_coords(new_coords)

    # Set attributes to match read_stacks format for compatibility
    array.attrs = {
        "stack": 0,  # Stack index (always 0 for single stacked result)
        "filepath": filepath,  # Original file path
        "axes": "".join(dims),  # Dimension order as string (e.g., "STCZYX")
        "subset_planes": planes,  # Record which planes were selected
    }

    # Return number of scenes in the original file
    num_stacks = len(img.scenes)

    return array, dims, num_stacks


# Test files
# filepath = r"F:\AzureDevOps\RMS_CAREamics_Container\_archive\calc_mean_testimage.czi"
# filepath = r"F:\Testdata_Zeiss\CZI_Testfiles\S=3_1Pos_2Mosaic_T=2=Z=3_CH=2.czi"
# filepath = r"F:\Testdata_Zeiss\CZI_Testfiles\96well_S=192_2pos_CH=3.czi"
# filepath = r"F:\Testdata_Zeiss\CZI_Testfiles\WP96_4Pos_B4-10_DAPI.czi"
# filepath = r"F:\Github\czitools\data\CellDivision_T10_Z15_CH2_DCV_small.czi"
# filepath = r"f:\Testdata_Zeiss\LLS7\LS_Mitosis_T=150-300_ZSTD.czi"
# filepath = r"F:\Testdata_Zeiss\CD7\Z-Stack_DCV\NeuroSpheres_DCV.czi",
filepath = r"F:\Testdata_Zeiss\LLS7\LS_Mitosis_T=150-300sm_ZSTD.czi"

# show resulting stack inside napari
show_ndv = True

# Select which method result should be shown in NDV.
# Options: "auto", "read_stacks_stacked", "read_czi_bioio"
# - auto: prefers read_stacks_stacked, falls back to bioio.
ndv_result_source = "auto"

# Profiling configuration
# - If True, both methods are profiled.
# - If False, only methods with the corresponding profile_* flag are run.
profile_all = True

# Individual method switches (used when profile_all=False)
profile_read_stacks = True
profile_read_bioio = True


def profile_method(name: str, runner: Callable[[], Any]) -> Dict[str, Any]:
    """Profile one method and return a normalized summary dictionary."""
    print(name)
    print("-" * 80)

    tracemalloc.start()
    start_time = time.perf_counter()

    payload = runner()

    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if isinstance(payload, tuple):
        result = payload[0]
        dims = payload[1] if len(payload) > 1 else None
        num_stacks = payload[2] if len(payload) > 2 else None
        mdata = payload[3] if len(payload) > 3 else None
    else:
        result = payload
        dims = getattr(result, "dims", None)
        num_stacks = None
        mdata = None

    elapsed = end_time - start_time
    print(f"  Time:          {elapsed:.4f} seconds")
    print(f"  Memory (peak): {peak / 1024**2:.2f} MB")
    print(f"  Shape:         {getattr(result, 'shape', 'N/A')}")
    print(f"  Dims:          {getattr(result, 'dims', dims if dims is not None else 'N/A')}")
    print(f"  Chunks:        {result.chunks if hasattr(result, 'chunks') else 'N/A'}")
    if num_stacks is not None:
        print(f"  Num stacks:    {num_stacks}")
    print()

    return {
        "name": name,
        "result": result,
        "dims": dims,
        "num_stacks": num_stacks,
        "mdata": mdata,
        "time": elapsed,
        "peak": peak,
    }


if __name__ == "__main__":
    import time
    import tracemalloc

    print(f"\n{'='*80}")
    print("Profiling CZI reading methods")
    print(f"File: {filepath}")
    print(f"{'='*80}\n")

    methods_to_run: Dict[str, bool]
    if profile_all:
        methods_to_run = {
            "read_stacks_stacked": True,
            "read_czi_bioio": True,
        }
    else:
        methods_to_run = {
            "read_stacks_stacked": profile_read_stacks,
            "read_czi_bioio": profile_read_bioio,
        }

    if not any(methods_to_run.values()):
        raise ValueError("No profiling method selected. Enable at least one method.")

    summaries: Dict[str, Dict[str, Any]] = {}

    if methods_to_run["read_stacks_stacked"]:
        summaries["read_stacks_stacked"] = profile_method(
            "Method 1: czitools.read_tools.read_stacks_stacked()",
            lambda: read_tools.read_stacks_stacked(
                filepath,
                use_dask=True,
                use_xarray=True,
                adapt_metadata=True,
            ),
        )

    if methods_to_run["read_czi_bioio"]:
        summaries["read_czi_bioio"] = profile_method(
            "Method 2: read_czi_bioio() using BioImage",
            lambda: read_czi_bioio(filepath),
        )

    # Comparison of selected methods
    if len(summaries) >= 2:
        print("Comparison")
        print("-" * 80)

        method_names = list(summaries.keys())
        time_pairs = [(k, summaries[k]["time"]) for k in method_names]
        time_pairs_sorted = sorted(time_pairs, key=lambda x: x[1])

        fastest_name, fastest_time = time_pairs_sorted[0]
        slowest_name, slowest_time = time_pairs_sorted[-1]
        speed_ratio = (slowest_time / fastest_time) if fastest_time > 0 else float("inf")
        faster_percent = (1.0 - (fastest_time / slowest_time)) * 100 if slowest_time > 0 else 0.0

        for method_name, method_time in time_pairs_sorted:
            print(f"  {method_name:20s}: {method_time:.4f} seconds")
        print(f"  Fastest method:      {fastest_name} ({faster_percent:.1f}% faster than {slowest_name})")
        print(f"  Speed ratio:         {speed_ratio:.2f}x")
        print()

        # Verify shape/dims compatibility against the first method
        print("Results verification")
        print("-" * 80)
        reference_key = method_names[0]
        reference_result = summaries[reference_key]["result"]
        reference_shape = getattr(reference_result, "shape", None)
        reference_dims = tuple(getattr(reference_result, "dims", []))

        for method_name in method_names[1:]:
            method_result = summaries[method_name]["result"]
            shape_match = getattr(method_result, "shape", None) == reference_shape
            dims_match = tuple(getattr(method_result, "dims", [])) == reference_dims
            print(f"  {reference_key} vs {method_name}")
            print(f"    Shapes match: {shape_match}")
            print(f"    Dims match:   {dims_match}")
        print()

    # Select result to display in NDV.
    valid_sources = {"auto", "read_stacks_stacked", "read_czi_bioio"}
    if ndv_result_source not in valid_sources:
        raise ValueError(
            f"Invalid ndv_result_source: {ndv_result_source}. "
            "Valid options are: 'auto', 'read_stacks_stacked', 'read_czi_bioio'."
        )

    if ndv_result_source == "auto":
        if "read_stacks_stacked" in summaries:
            selected_key = "read_stacks_stacked"
        else:
            selected_key = "read_czi_bioio"
    else:
        selected_key = ndv_result_source
        if selected_key not in summaries:
            fallback_key = "read_stacks_stacked" if "read_stacks_stacked" in summaries else "read_czi_bioio"
            print(f"Requested NDV source '{selected_key}' was not profiled. " f"Falling back to '{fallback_key}'.")
            selected_key = fallback_key

    result = summaries[selected_key]["result"]
    print(f"NDV source: {selected_key}")

    viewer_data = None

    if show_ndv:

        if isinstance(result, list):
            viewer_data = result[0] if len(result) > 0 else None
        elif result is not None:
            viewer_data = result

        # Prefer metadata returned by read_stacks_stacked (adapted to planes),
        # otherwise fall back to fresh metadata from filepath.
        selected_payload = summaries.get(selected_key, {})
        selected_mdata = selected_payload.get("mdata") if isinstance(selected_payload, dict) else None
        if selected_mdata is None and "read_stacks_stacked" in summaries:
            selected_mdata = summaries["read_stacks_stacked"].get("mdata")
        if selected_mdata is None:
            from czitools.metadata_tools.czi_metadata import CziMetadata

            selected_mdata = CziMetadata(filepath)

        luts: Any = create_luts_ndv(cast(Any, selected_mdata))
        viewer = ndv.imshow(viewer_data, channel_mode="composite", luts=luts)
