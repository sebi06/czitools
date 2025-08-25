# -*- coding: utf-8 -*-

#################################################################
# File        : write_omezarr.py
# Author      : sebi06
#
# Requires: ome-zarr, ngff-zarr
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from czitools.read_tools import read_tools
from omezarr_write_tools import write_omezarr
from czitools.metadata_tools import czi_metadata as czimd
import ngff_zarr as nz
from pathlib import Path

# open s simple dialog to select a CZI file
defaultdir = Path(Path(__file__).resolve().parents[2]) / "data"
filepath = defaultdir / "CellDivision_T3_Z5_CH2_X240_Y170.czi"

# get the metadata_tools at once as one big class
mdata = czimd.CziMetadata(filepath)
print("Number of Scenes: ", mdata.image.SizeS)
scene_id = 0

# return a 6D array with dimension order STCZYX(A)
array, mdata = read_tools.read_6darray(filepath, use_dask=True)
array = array[scene_id, ...]

# Approch 1: Use ome-zarr-py to write OME-ZARR
zarr_path1 = Path(str(filepath)[:-4] + "_1.ome.zarr")

# write OME-ZARR using utility function
zarr_path1 = write_omezarr(
    array, zarr_path=str(zarr_path1), axes="tczyx", overwrite=True
)

print(f"Written OME-ZARR using ome-zarr.py: {zarr_path1}")

# Approch 2: Use ngff-zarr to create NGFF structure and write using ome-zarr-py
zarr_path2 = Path(str(filepath)[:-4] + "_2.ome.zarr")

# create NGFF image from the array
image = nz.to_ngff_image(
    array.data,
    dims=["t", "c", "z", "y", "x"],
    scale={"y": mdata.scale.Y, "x": mdata.scale.X, "z": mdata.scale.Z},
    name=mdata.filename,
)

# create multi-scaled, chunked data structure from the image
multiscales = nz.to_multiscales(image, [2, 4], method=nz.Methods.DASK_IMAGE_GAUSSIAN)

# write using ngff-zarr
nz.to_ngff_zarr(zarr_path2, multiscales)
print(f"NGFF Image: {image}")
print(f"Written OME-ZARR using ngff-zarr: {zarr_path2}")
