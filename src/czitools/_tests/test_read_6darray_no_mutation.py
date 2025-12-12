from pathlib import Path
from czitools.read_tools.read_tools import read_6darray
import copy


def test_read_6darray_does_not_mutate_planes():
    filepath = Path(__file__).resolve().parents[3] / "data" / "CellDivision_T3_Z5_CH2_X240_Y170.czi"
    planes = {"Z": (0, 4), "T": (0, 2)}
    before = copy.deepcopy(planes)

    # call with dask to avoid heavy eager reads
    arr, mdata = read_6darray(str(filepath), use_dask=True, planes=planes, use_xarray=False)

    assert planes == before
