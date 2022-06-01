# -*- coding: utf-8 -*-

#################################################################
# File        : use_pylibczirw_md_read.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from czimetadata_tools import pylibczirw_metadata as czimd
from czimetadata_tools import pylibczirw_tools
from czimetadata_tools import napari_tools
import napari
import os
from pathlib import Path

basedir = Path(__file__).resolve().parents[2]
#filepath = os.path.join(basedir, r"testdata/WP96_4Pos_B4-10_DAPI.czi")
filepath = os.path.join(basedir, r"testdata/S=2_3x3_CH=2.czi")
#filepath = os.path.join(basedir, r"testdata/w96_A1+A2.czi")
#filepath = os.path.join(basedir, r"testdata/CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi")

# get the complete metadata at once as one big class
mdata = czimd.CziMetadata(filepath)

# return a 7d array with dimension order STZCYXA
mdarray, dimstring = pylibczirw_tools.read_mdarray(filepath)
#mdarray, dimstring = pylibczirw_tools.read_mdarray_lazy(filepath)

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
