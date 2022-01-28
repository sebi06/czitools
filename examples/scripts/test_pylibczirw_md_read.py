# -*- coding: utf-8 -*-

#################################################################
# File        : test_pylibczirw_md_read.py
# Version     : 0.0.1
# Author      : sebi06
# Date        : 24.01.2022
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from czitools import pylibczirw_metadata as czimd
from czitools import pylibczirw_tools
from czitools import napari_tools
import napari
import numpy as np

filename = r'C:\Testdata_Zeiss\CZI_Testfiles\CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi'
#filename = r'C:\Testdata_Zeiss\CZI_Testfiles\testwell96.czi'
#filename = r"C:\Testdata_Zeiss\CZI_Testfiles\tobacco_z=10_tiles.czi"
#filename = r"C:\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=3_Z=4_CH=2.czi"

# get the complete metadata at once as one big class
mdata = czimd.CziMetadata(filename)

# return a 7d array with dimension order STZCYXA
mdarray, dimstring = pylibczirw_tools.read_mdarray(filename)
#mdarray, dimstring = pylibczirw_tools.read_mdarray_lazy(filename)

# remove A dimension do display the array inside Napari
dim_order, dim_index, dim_valid = czimd.CziMetadata.get_dimorder(dimstring)

# show array inside napari viewer
viewer = napari.Viewer()
layers = napari_tools.show(viewer, mdarray, mdata,
                           dim_order=dim_order,
                           blending="additive",
                           contrast='napari_auto',
                           gamma=0.85,
                           add_mdtable=True,
                           name_sliders=True)

napari.run()
