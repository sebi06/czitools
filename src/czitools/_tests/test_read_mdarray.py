from czitools import pylibczirw_tools
from pathlib import Path
import dask.array as da
import numpy as np
import pytest
from typing import List, Dict, Tuple, Optional, Type, Any, Union, Mapping


basedir = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "czifile, output_dask, remove_adim, dimorder, dimstring, shape, art",
    [
        ("w96_A1+A2.czi", False, False, "STCZYX", "STCZYXA", (2, 1, 2, 1, 1416, 1960, 1), np.ndarray),

        ("w96_A1+A2.czi", False, True, "STZCYX", "STZCYX", (2, 1, 1, 2, 1416, 1960), np.ndarray),

        ("S=3_1Pos_2Mosaic_T=2=Z=3_CH=2_sm.czi", False, False, "STCZYX", "",  AttributeError, None),

        ("S=2_3x3_CH=2.czi", False, False, "STZCYX", "STZCYXA", (2, 1, 1, 2, 1792, 1792, 1), np.ndarray),

        ("S=2_3x3_CH=2.czi", False, True, "STCZYX", "STCZYX", (2, 1, 2, 1, 1792, 1792), np.ndarray),

        ("FOV7_HV110_P0500510000.czi", False, False, "STZCYX",
         "STZCYXA", (1, 1, 1, 1, 512, 512, 1), np.ndarray),

        ("newCZI_compressed.czi", True, True, "STZCYX", "STZCYX", (1, 1, 1, 1, 512, 512), da.Array)
    ]
)
def test_read_mdarray_1(czifile: str,
                        output_dask: bool,
                        remove_adim: bool,
                        dimorder: str,
                        dimstring: str,
                        shape: Optional[Tuple[int]],
                        art: Optional[Union[np.ndarray, da.array]]) -> None:

    # get the CZI filepath
    filepath = basedir / "data" / czifile

    mdarray, mdata, ds = pylibczirw_tools.read_6darray(filepath,
                                                       output_order=dimorder,
                                                       output_dask=output_dask,
                                                       chunks_auto=False,
                                                       remove_adim=remove_adim)

    assert (ds == dimstring)

    if type(shape) == type and issubclass(shape, Exception):
        with pytest.raises(shape):
            mdarray.shape
    else:
        assert (mdarray.shape == shape)
        assert (type(mdarray) == art)


@pytest.mark.parametrize(
    "czifile, dimstring, remove_adim, shape, ndim, chunksize",
    [
        ("w96_A1+A2.czi", "STZCYXA", False, (2, 1, 1, 2, 1416, 1960, 1), 7, (1, 1, 1, 2, 1416, 1960, 1)),

        ("w96_A1+A2.czi", "STZCYX", True, (2, 1, 1, 2, 1416, 1960), 6, (1, 1, 1, 2, 1416, 1960))
    ]
)
def test_read_mdarray_lazy_1(czifile: str,
                             dimstring: str,
                             remove_adim: bool,
                             shape: Tuple[int],
                             ndim: int,
                             chunksize: Tuple[int]) -> None:

    # get the CZI filepath
    filepath = basedir / "data" / czifile

    mdarray, mdata, ds = pylibczirw_tools.read_mdarray_lazy(filepath, remove_adim=remove_adim)

    assert (ds == dimstring)
    assert (mdarray.shape == shape)
    assert (mdarray.ndim == ndim)
    assert (mdarray.chunksize == chunksize)


@pytest.mark.parametrize(
    "czifile, dimorder, output_dask, dimstring, shape, size_s, plane",
    [
        ("w96_A1+A2.czi", "STCZYX", False, "STCZYXA", (1, 1, 2, 1, 1416, 1960, 1), 1, {"S": 0}),

        ("CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi", "STZCYX", True,
         "STZCYXA", (1, 1, 1, 2, 170, 240, 1), None, {"S": 0, "T": 0, "Z": 0}),

        ("CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi", "STZCYX",
         True, "STZCYXA", (1, 1, 5, 2, 170, 240, 1), None, {"T": 0}),

        ("CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi", "STZCYX",
         True, "STZCYXA", (1, 3, 5, 1, 170, 240, 1), None, {"C": 0})
    ]
)
def test_read_mdarray_substack(czifile: str,
                               dimorder: str,
                               output_dask: bool,
                               dimstring: str,
                               shape: Tuple[int],
                               size_s: Optional[int],
                               plane: Dict[str, int]) -> None:

    # get the CZI filepath
    filepath = basedir / "data" / czifile

    # read only a specific scene from the CZI
    mdarray, mdata, ds = pylibczirw_tools.read_6darray(filepath,
                                                       output_order=dimorder,
                                                       output_dask=output_dask,
                                                       chunks_auto=False,
                                                       remove_adim=False,
                                                       **plane)

    assert (ds == dimstring)
    assert (mdarray.shape == shape)
    assert (mdata.image.SizeS == size_s)
