# -*- coding: utf-8 -*-

#################################################################
# File        : test_aicsimageio.py
# Version     : 0.0.3
# Author      : sebi06
# Date        : 24.01.2022
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

import napari
from czimetadata_tools import pylibczirw_metadata as czimd
from aicsimageio import AICSImage
from czimetadata_tools import misc, napari_tools

filename = r".\testdata\w96_A1 + A2.czi"

# get the complete metadata at once as one big class
mdata = czimd.CziMetadata(filename)

# test using AICSImageIO
aics_img = AICSImage(filename)
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
