# -*- coding: utf-8 -*-

#################################################################
# File        : write_omezarr.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from czitools.read_tools import read_tools
from czitools.metadata_tools import czi_metadata as czimd
from pathlib import Path
import ngff_zarr as nz
import zarr
import shutil

# this script required ngff-zarr[validate, dask-image] to be installed
print(f"ZARR Verion: {zarr.__version__} NGFF-ZARR Version: {nz.__version__}")

# open s simple dialog to select a CZI file
defaultdir = Path(Path(__file__).resolve().parents[2]) / "data"
filepath = defaultdir / "CellDivision_T3_Z5_CH2_X240_Y170.czi"
zarr_path = Path(str(filepath)[:-4] + ".ome.zarr")

# optional: check if path exists
remove = True
if zarr_path.exists() and remove:
    shutil.rmtree(zarr_path, ignore_errors=False, onerror=None)

# get the metadata_tools at once as one big class
mdata = czimd.CziMetadata(filepath)
print("Number of Scenes: ", mdata.image.SizeS)
scene_id = 0

# return a 6D array with dimension order STCZYX(A)
array, mdata = read_tools.read_6darray(filepath)
array = array[scene_id, ...]

# create NGFF image from the array
image = nz.to_ngff_image(
    array.data,
    dims=["t", "c", "z", "y", "x"],
    scale={"y": mdata.scale.Y, "x": mdata.scale.X, "z": mdata.scale.Z},
    name=mdata.filename,
)

print(f"NGFF Image: {image}")

# create multi-scaled, chunked data structure from the image
multiscales = nz.to_multiscales(image, [2, 4], method=nz.Methods.DASK_IMAGE_GAUSSIAN)
