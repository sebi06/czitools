# -*- coding: utf-8 -*-

#################################################################
# File        : read_czi_aicsimageio.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

import napari
from czitools.metadata import pylibczirw_metadata as czimd
from aicsimageio import AICSImage
from czitools.utils import misc, napari_tools
import os
from pathlib import Path

# adapt to your needs
defaultdir = os.path.join(Path(__file__).resolve().parents[1], "data")

# open s simple dialog to select a CZI file
filepath = misc.openfile(directory=defaultdir,
                         title="Open CZI Image File",
                         ftypename="CZI Files",
                         extension="*.czi")
print(filepath)

# get the complete metadata at once as one big class
mdata = czimd.CziMetadata(filepath)

# test using AICSImageIO
aics_img = AICSImage(filepath)
print(aics_img.shape)
for k, v in aics_img.dims.items():
    print(k, v)

# get the stack as dask array
stack = misc.get_daskstack(aics_img)

mdata.aics_dimstring = "S" + aics_img.dims.order
dim_order, dim_index, dim_valid = czimd.CziMetadata.get_dimorder(mdata.aics_dimstring)

# start the napari viewer and show the image
viewer = napari.Viewer()
layers = napari_tools.show(viewer, stack, mdata,
                           dim_order=dim_order,
                           blending="additive",
                           contrast="napari_auto",
                           gamma=0.85,
                           add_mdtable=True,
                           name_sliders=True)

napari.run()
