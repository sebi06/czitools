"""
Tests for read_stacks and read_6darray_lazy functions in czitools.read_tools.

These tests verify the lazy loading functionality and stack-based reading
capabilities of the czitools library.
"""

from czitools.read_tools import read_tools
from pathlib import Path
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from typing import List, Tuple, Optional, Union


basedir = Path(__file__).resolve().parents[3]


# ============================================================================
# Tests for read_6darray_lazy
# ============================================================================


@pytest.mark.parametrize(
    "czifile, shape, chunk_zyx",
    [
        ("CellDivision_T3_Z5_CH2_X240_Y170.czi", (1, 3, 2, 5, 170, 240), False),
        ("CellDivision_T3_Z5_CH2_X240_Y170.czi", (1, 3, 2, 5, 170, 240), True),
        ("w96_A1+A2.czi", (2, 1, 2, 1, 1416, 1960), False),
        ("S2_3x3_CH2.czi", (2, 1, 2, 1, 1792, 1792), False),
        ("FOV7_HV110_P0500510000.czi", (1, 1, 1, 1, 512, 512), False),
    ],
)
def test_read_6darray_lazy_basic(
    czifile: str, shape: Tuple[int, ...], chunk_zyx: bool
) -> None:
    """Test basic lazy loading with read_6darray_lazy."""
    filepath = basedir / "data" / czifile

    array6d, mdata = read_tools.read_6darray_lazy(filepath, chunk_zyx=chunk_zyx)

    # Verify it returns a dask array
    assert array6d is not None
    assert isinstance(array6d, da.Array)
    assert array6d.shape == shape

    # Verify metadata is returned
    assert mdata is not None
    assert mdata.filename == czifile


@pytest.mark.parametrize(
    "czifile, shape",
    [
        ("CellDivision_T3_Z5_CH2_X240_Y170.czi", (1, 3, 2, 5, 170, 240)),
        ("w96_A1+A2.czi", (2, 1, 2, 1, 1416, 1960)),
    ],
)
def test_read_6darray_lazy_xarray(czifile: str, shape: Tuple[int, ...]) -> None:
    """Test lazy loading with xarray output."""
    filepath = basedir / "data" / czifile

    array6d, mdata = read_tools.read_6darray_lazy(filepath, use_xarray=True)

    # Verify it returns an xarray DataArray
    assert array6d is not None
    assert isinstance(array6d, xr.DataArray)
    assert array6d.shape == shape

    # Verify dimension labels
    assert list(array6d.dims) == ["S", "T", "C", "Z", "Y", "X"]


@pytest.mark.parametrize(
    "czifile, planes, expected_shape",
    [
        (
            "CellDivision_T3_Z5_CH2_X240_Y170.czi",
            {"T": (0, 1), "Z": (0, 2)},
            (1, 2, 2, 3, 170, 240),
        ),
        (
            "CellDivision_T3_Z5_CH2_X240_Y170.czi",
            {"C": (0, 0)},
            (1, 3, 1, 5, 170, 240),
        ),
        (
            "CellDivision_T3_Z5_CH2_X240_Y170.czi",
            {"T": (1, 2), "C": (1, 1), "Z": (2, 4)},
            (1, 2, 1, 3, 170, 240),
        ),
    ],
)
def test_read_6darray_lazy_planes(
    czifile: str,
    planes: dict,
    expected_shape: Tuple[int, ...],
) -> None:
    """Test lazy loading with plane selection (substack)."""
    filepath = basedir / "data" / czifile

    array6d, mdata = read_tools.read_6darray_lazy(filepath, planes=planes)

    assert array6d is not None
    assert isinstance(array6d, da.Array)
    assert array6d.shape == expected_shape


def test_read_6darray_lazy_compute() -> None:
    """Test that lazy array can be computed to numpy."""
    filepath = basedir / "data" / "CellDivision_T3_Z5_CH2_X240_Y170.czi"

    array6d, mdata = read_tools.read_6darray_lazy(filepath)

    assert isinstance(array6d, da.Array)

    # Compute a small subset
    subset = array6d[0, 0, 0, 0, :10, :10].compute()

    assert isinstance(subset, np.ndarray)
    assert subset.shape == (10, 10)


@pytest.mark.parametrize(
    "czifile, zoom, expected_reduction",
    [
        ("CellDivision_T3_Z5_CH2_X240_Y170.czi", 0.5, 0.5),
    ],
)
def test_read_6darray_lazy_zoom(
    czifile: str,
    zoom: float,
    expected_reduction: float,
) -> None:
    """Test lazy loading with zoom/downscaling."""
    filepath = basedir / "data" / czifile

    # Read full resolution
    array_full, _ = read_tools.read_6darray_lazy(filepath, zoom=1.0)
    # Read zoomed
    array_zoom, mdata = read_tools.read_6darray_lazy(filepath, zoom=zoom)

    assert array_zoom is not None

    # Y and X dimensions should be scaled
    expected_y = int(array_full.shape[4] * expected_reduction)
    expected_x = int(array_full.shape[5] * expected_reduction)

    assert array_zoom.shape[4] == expected_y
    assert array_zoom.shape[5] == expected_x


# ============================================================================
# Tests for read_stacks
# ============================================================================


@pytest.mark.parametrize(
    "czifile, num_stacks_expected, use_dask",
    [
        # Files WITH scenes
        ("w96_A1+A2.czi", 2, False),
        ("w96_A1+A2.czi", 2, True),
        ("S2_3x3_CH2.czi", 2, False),
        ("S2_3x3_CH2.czi", 2, True),
        # File with many scenes (28 scenes)
        ("WP96_4Pos_B4-10_DAPI.czi", 28, False),
        ("WP96_4Pos_B4-10_DAPI.czi", 28, True),
    ],
)
def test_read_stacks_basic(
    czifile: str, num_stacks_expected: int, use_dask: bool
) -> None:
    """Test basic scene reading functionality for files with scenes."""
    filepath = basedir / "data" / czifile

    arrays, dims, num_stacks = read_tools.read_stacks(
        filepath, use_dask=use_dask, use_xarray=True
    )

    # Verify number of scenes
    assert num_stacks == num_stacks_expected

    # Verify we get a list of arrays (one per scene)
    assert isinstance(arrays, list)
    assert len(arrays) == num_stacks_expected

    # Verify dimension labels are returned
    assert isinstance(dims, list)
    assert len(dims) > 0

    # Verify array types
    for arr in arrays:
        assert isinstance(arr, xr.DataArray)
        if use_dask:
            assert isinstance(arr.data, da.Array)


@pytest.mark.parametrize(
    "czifile, use_xarray",
    [
        # Files WITH scenes
        ("w96_A1+A2.czi", True),
        ("w96_A1+A2.czi", False),
        ("S2_3x3_CH2.czi", True),
    ],
)
def test_read_stacks_xarray_option(czifile: str, use_xarray: bool) -> None:
    """Test read_stacks with and without xarray output."""
    filepath = basedir / "data" / czifile

    arrays, dims, num_stacks = read_tools.read_stacks(
        filepath, use_dask=False, use_xarray=use_xarray
    )

    assert len(arrays) > 0

    if use_xarray:
        assert isinstance(arrays[0], xr.DataArray)
    else:
        assert isinstance(arrays[0], np.ndarray)


@pytest.mark.parametrize(
    "czifile, expected_stacked",
    [
        # Multiple scenes with same shape - should stack
        ("w96_A1+A2.czi", True),
        ("S2_3x3_CH2.czi", True),
        # Many scenes (28) - should stack if shapes match
        ("WP96_4Pos_B4-10_DAPI.czi", True),
    ],
)
def test_read_stacks_stack(czifile: str, expected_stacked: bool) -> None:
    """Test read_stacks with stack_scenes option."""
    filepath = basedir / "data" / czifile

    result, dims, num_stacks = read_tools.read_stacks(
        filepath, use_xarray=True, stack_scenes=True
    )

    if expected_stacked and num_stacks > 1:
        # Should be a single stacked array, not a list
        assert isinstance(result, xr.DataArray)
        # First dimension should be S (scenes)
        assert "S" in result.dims or result.shape[0] == num_stacks
    else:
        # Single scene case - could be array or list with one element
        assert result is not None


def test_read_stacks_different_shapes() -> None:
    """Test read_stacks with stacks of different shapes (cannot stack)."""
    # This file has scenes with different shapes
    filepath = basedir / "data" / "S3_1Pos_2Mosaic_T2_Z3_CH2_sm.czi"

    # Without stacking - should return list of arrays
    arrays, dims, num_stacks = read_tools.read_stacks(
        filepath, use_xarray=True, stack_scenes=False
    )

    assert num_stacks > 0
    assert isinstance(arrays, list)
    assert len(arrays) == num_stacks

    # Verify each scene is an xarray
    for arr in arrays:
        assert isinstance(arr, xr.DataArray)


def test_read_stacks_different_shapes_stack_warning() -> None:
    """Test that stacking stacks with different shapes returns a list with warning."""
    # This file has scenes with different shapes - stacking should fall back to list
    filepath = basedir / "data" / "S3_1Pos_2Mosaic_T2_Z3_CH2_sm.czi"

    result, dims, num_stacks = read_tools.read_stacks(
        filepath, use_xarray=True, stack_scenes=True
    )

    # When shapes differ, should return a list (not stacked)
    assert num_stacks > 0
    # Result should be a list since shapes are different
    assert isinstance(result, list)


def test_read_stacks_dask_lazy() -> None:
    """Test that dask arrays from read_stacks are truly lazy."""
    filepath = basedir / "data" / "w96_A1+A2.czi"

    arrays, dims, num_stacks = read_tools.read_stacks(
        filepath, use_dask=True, use_xarray=True
    )

    assert len(arrays) > 0
    arr = arrays[0]

    # Verify it's lazy (dask array inside xarray)
    assert isinstance(arr.data, da.Array)

    # Compute a small subset to verify data can be loaded
    subset = arr.isel(Y=slice(0, 10), X=slice(0, 10)).compute()
    assert isinstance(subset.data, np.ndarray)


@pytest.mark.parametrize(
    "czifile",
    [
        "w96_A1+A2.czi",
        "S2_3x3_CH2.czi",
        "WP96_4Pos_B4-10_DAPI.czi",
    ],
)
def test_read_stacks_dimensions(czifile: str) -> None:
    """Test that read_stacks returns proper dimension labels."""
    filepath = basedir / "data" / czifile

    arrays, dims, num_stacks = read_tools.read_stacks(
        filepath, use_xarray=True, use_dask=False
    )

    # Core dimensions should always be present
    assert "T" in dims
    assert "C" in dims
    assert "Z" in dims
    assert "Y" in dims
    assert "X" in dims

    # Verify xarray dimension labels match
    arr = arrays[0]
    for dim in ["T", "C", "Z", "Y", "X"]:
        assert dim in arr.dims


# ============================================================================
# Tests for Path object support
# ============================================================================


def test_read_6darray_lazy_path_object() -> None:
    """Test that read_6darray_lazy accepts Path objects."""
    filepath = basedir / "data" / "CellDivision_T3_Z5_CH2_X240_Y170.czi"

    # Pass as Path object (not string)
    array6d, mdata = read_tools.read_6darray_lazy(filepath)

    assert array6d is not None
    assert mdata is not None


def test_read_stacks_path_object() -> None:
    """Test that read_stacks accepts Path objects."""
    filepath = basedir / "data" / "w96_A1+A2.czi"

    # Pass as Path object (not string)
    arrays, dims, num_stacks = read_tools.read_stacks(filepath)

    assert len(arrays) > 0
    assert num_stacks >= 1


# ============================================================================
# Tests for files without explicit scenes
# ============================================================================


def test_read_stacks_no_scenes_returns_empty() -> None:
    """Test that read_stacks returns empty list for files without scenes."""
    # CellDivision has no explicit scenes (scenes_bounding_rectangle is empty)
    filepath = basedir / "data" / "CellDivision_T3_Z5_CH2_X240_Y170.czi"

    arrays, dims, num_stacks = read_tools.read_stacks(filepath)

    # Files without scenes return empty results
    assert num_stacks == 0
    assert len(arrays) == 0
