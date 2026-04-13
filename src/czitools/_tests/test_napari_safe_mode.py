# -*- coding: utf-8 -*-

"""
Test Napari helper utilities.
"""

import pytest
from pathlib import Path

# Test data directory
basedir = Path(__file__).resolve().parents[3]


def test_get_recommended_read_params():
    """Test that recommended parameters are returned correctly."""
    from czitools.utils.napari_helpers import _get_recommended_read_params as get_recommended_read_params

    params = get_recommended_read_params()

    assert params["use_dask"] is True
    assert params["use_xarray"] is True
    assert params["chunk_zyx"] is True


@pytest.mark.parametrize(
    "czifile",
    [
        "CellDivision_T3_Z5_CH2_X240_Y170.czi",
    ],
)
def test_read_6darray_with_recommended_params(czifile: str):
    """Test that read_6darray works correctly with recommended Napari params."""
    from czitools.read_tools import read_tools

    filepath = basedir / "data" / czifile

    array6d, mdata = read_tools.read_6darray(filepath, use_dask=True, use_xarray=True)

    assert array6d is not None
    assert mdata is not None
    assert mdata.image.SizeC == 2
    assert mdata.image.SizeZ == 5
    assert mdata.image.SizeT == 3
