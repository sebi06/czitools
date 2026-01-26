# -*- coding: utf-8 -*-

"""
Test thread-safe mode and Napari compatibility helpers.
"""

import os
import pytest
from pathlib import Path

# Test data directory
basedir = Path(__file__).resolve().parents[3]


def test_napari_safe_mode_environment_variable():
    """Test that environment variable correctly disables aicspylibczi."""
    # Save original state
    original_value = os.environ.get("CZITOOLS_DISABLE_AICSPYLIBCZI")

    try:
        # Test enabling safe mode
        os.environ["CZITOOLS_DISABLE_AICSPYLIBCZI"] = "1"

        # Import after setting env var
        from czitools.utils.napari_helpers import is_napari_safe_mode

        assert is_napari_safe_mode() is True

        # Test disabling safe mode
        os.environ["CZITOOLS_DISABLE_AICSPYLIBCZI"] = "0"
        assert is_napari_safe_mode() is False

    finally:
        # Restore original state
        if original_value is None:
            os.environ.pop("CZITOOLS_DISABLE_AICSPYLIBCZI", None)
        else:
            os.environ["CZITOOLS_DISABLE_AICSPYLIBCZI"] = original_value


def test_enable_napari_safe_mode():
    """Test the enable_napari_safe_mode helper function."""
    # Save original state
    original_value = os.environ.get("CZITOOLS_DISABLE_AICSPYLIBCZI")

    try:
        # Clear the env var first
        os.environ.pop("CZITOOLS_DISABLE_AICSPYLIBCZI", None)

        from czitools.utils.napari_helpers import enable_napari_safe_mode, is_napari_safe_mode

        # Should be disabled initially
        assert is_napari_safe_mode() is False

        # Enable safe mode
        enable_napari_safe_mode()

        # Should now be enabled
        assert is_napari_safe_mode() is True

    finally:
        # Restore original state
        if original_value is None:
            os.environ.pop("CZITOOLS_DISABLE_AICSPYLIBCZI", None)
        else:
            os.environ["CZITOOLS_DISABLE_AICSPYLIBCZI"] = original_value


def test_check_napari_compatibility():
    """Test the napari compatibility checker."""
    # Save original state
    original_value = os.environ.get("CZITOOLS_DISABLE_AICSPYLIBCZI")

    try:
        from czitools.utils.napari_helpers import check_napari_compatibility

        # Test with safe mode enabled
        os.environ["CZITOOLS_DISABLE_AICSPYLIBCZI"] = "1"
        compatible, message = check_napari_compatibility()
        assert compatible is True
        assert "compatible" in message.lower()

        # Test with safe mode disabled
        os.environ["CZITOOLS_DISABLE_AICSPYLIBCZI"] = "0"
        compatible, message = check_napari_compatibility()
        assert compatible is False
        assert "warning" in message.lower()

    finally:
        # Restore original state
        if original_value is None:
            os.environ.pop("CZITOOLS_DISABLE_AICSPYLIBCZI", None)
        else:
            os.environ["CZITOOLS_DISABLE_AICSPYLIBCZI"] = original_value


def test_get_recommended_read_params():
    """Test that recommended parameters are returned correctly."""
    from czitools.utils.napari_helpers import get_recommended_read_params

    params = get_recommended_read_params()

    assert params["use_dask"] is True
    assert params["use_xarray"] is True
    assert params["chunk_zyx"] is True


def test_read_tiles_with_safe_mode_raises_error():
    """Test that read_tiles raises an error when safe mode is enabled."""
    # Save original state
    original_value = os.environ.get("CZITOOLS_DISABLE_AICSPYLIBCZI")

    try:
        # Enable safe mode
        os.environ["CZITOOLS_DISABLE_AICSPYLIBCZI"] = "1"

        # Import after setting env var
        from czitools.read_tools import read_tools

        # Attempt to use read_tiles should raise RuntimeError
        with pytest.raises(RuntimeError, match="requires aicspylibczi"):
            read_tools.read_tiles("dummy.czi", 0, 0)

    finally:
        # Restore original state
        if original_value is None:
            os.environ.pop("CZITOOLS_DISABLE_AICSPYLIBCZI", None)
        else:
            os.environ["CZITOOLS_DISABLE_AICSPYLIBCZI"] = original_value


@pytest.mark.parametrize(
    "czifile",
    [
        "CellDivision_T3_Z5_CH2_X240_Y170.czi",
    ],
)
def test_read_6darray_works_in_safe_mode(czifile: str):
    """Test that read_6darray works correctly in safe mode."""
    # Save original state
    original_value = os.environ.get("CZITOOLS_DISABLE_AICSPYLIBCZI")

    try:
        # Enable safe mode
        os.environ["CZITOOLS_DISABLE_AICSPYLIBCZI"] = "1"

        # Import after setting env var
        from czitools.read_tools import read_tools

        filepath = basedir / "data" / czifile

        # This should work fine in safe mode
        array6d, mdata = read_tools.read_6darray(filepath, use_dask=True, use_xarray=True)

        assert array6d is not None
        assert mdata is not None
        assert mdata.image.SizeC == 2
        assert mdata.image.SizeZ == 5
        assert mdata.image.SizeT == 3

    finally:
        # Restore original state
        if original_value is None:
            os.environ.pop("CZITOOLS_DISABLE_AICSPYLIBCZI", None)
        else:
            os.environ["CZITOOLS_DISABLE_AICSPYLIBCZI"] = original_value


@pytest.mark.parametrize(
    "czifile",
    [
        "CellDivision_T3_Z5_CH2_X240_Y170.czi",
    ],
)
def test_metadata_works_in_safe_mode(czifile: str):
    """Test that CziMetadata works correctly in safe mode."""
    # Save original state
    original_value = os.environ.get("CZITOOLS_DISABLE_AICSPYLIBCZI")

    try:
        # Enable safe mode
        os.environ["CZITOOLS_DISABLE_AICSPYLIBCZI"] = "1"

        # Import after setting env var
        from czitools.metadata_tools.czi_metadata import CziMetadata

        filepath = basedir / "data" / czifile

        # This should work fine in safe mode
        mdata = CziMetadata(filepath)

        assert mdata is not None
        assert mdata.image.SizeC == 2
        assert mdata.image.SizeZ == 5
        assert mdata.image.SizeT == 3

        # aicspylibczi-specific fields should be None or default
        # but core metadata from pylibCZIrw should still work
        assert mdata.pyczi_dims is not None

    finally:
        # Restore original state
        if original_value is None:
            os.environ.pop("CZITOOLS_DISABLE_AICSPYLIBCZI", None)
        else:
            os.environ["CZITOOLS_DISABLE_AICSPYLIBCZI"] = original_value
