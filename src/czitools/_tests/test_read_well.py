"""Integration tests for the HCS-aware convenience reads (Stage 3)."""

from pathlib import Path

import pytest
import xarray as xr

from czitools.read_tools.read_tools import read_field, read_well

BASEDIR = Path(__file__).resolve().parents[3]
WELLPLATE = BASEDIR / "data" / "WP96_4Pos_B4-10_DAPI.czi"
NON_PLATE = BASEDIR / "data" / "S3_1Pos_2Mosaic_T2_Z3_CH2_sm.czi"


def test_read_field_returns_single_scene() -> None:
    array, mdata = read_field(WELLPLATE, "B4", 0, use_xarray=True)

    assert array is not None
    assert isinstance(array, xr.DataArray)
    # Exactly one scene is read for a single field.
    assert array.sizes["S"] == 1
    assert mdata.hcs is not None


def test_read_field_by_region_id_matches_index() -> None:
    array_idx, _ = read_field(WELLPLATE, "B4", 0)
    # The first field of B4 has this RegionId (see test_hcs).
    array_region, _ = read_field(WELLPLATE, "B4", "637232309317131710")

    assert array_idx is not None and array_region is not None
    assert array_idx.shape == array_region.shape


def test_read_well_returns_list_of_fields() -> None:
    arrays, mdata = read_well(WELLPLATE, "B4")

    assert isinstance(arrays, list)
    assert mdata.hcs is not None
    # B4 has 4 fields in the included plate.
    assert len(arrays) == len(mdata.hcs.get_well("B4").fields)
    assert all(a is not None for a in arrays)


def test_read_well_stack_concatenates_along_s() -> None:
    arrays, _ = read_well(WELLPLATE, "B4")
    stacked, _ = read_well(WELLPLATE, "B4", stack=True)

    assert isinstance(arrays, list)
    assert stacked is not None
    assert stacked.sizes["S"] == len(arrays)


def test_read_well_selected_fields() -> None:
    arrays, _ = read_well(WELLPLATE, "B4", fields=[0, 2])

    assert isinstance(arrays, list)
    assert len(arrays) == 2


def test_read_field_without_hcs_raises() -> None:
    with pytest.raises(ValueError, match="no usable HCS plate"):
        read_field(NON_PLATE, "A1", 0)
