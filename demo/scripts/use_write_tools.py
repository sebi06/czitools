# -*- coding: utf-8 -*-

#################################################################
# File        : use_write_tools.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from czitools import read_tools, write_tools
from czitools import metadata_tools as czimd
import napari
from pathlib import Path

# open s simple dialog to select a CZI file
filepath = r"data/CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi"
zarr_path = Path(filepath[:-4] + ".ome.zarr")

# get the metadata at once as one big class
mdata = czimd.CziMetadata(filepath)
print("Number of Scenes: ", mdata.image.SizeS)
scene_id = 0

# return a array with dimension order STCZYX(A)
array, mdata, dim_string6d = read_tools.read_6darray(
    filepath, output_order="STCZYX", use_dask=True
)

array = array[scene_id, ...]

# write OME-ZARR using utility function
zarr_path = write_tools.write_omezarr(
    array, zarr_path=zarr_path, axes="tczyx", overwrite=True
)

if array is None:
    print("Empty array. Nothing to display in Napari")
else:
    # show OME-ZARR inside napari viewer
    viewer = napari.Viewer()
    viewer.open(zarr_path)

    napari.run()
