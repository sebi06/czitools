# -*- coding: utf-8 -*-

#################################################################
# File        : use_pylibczirw_md_read.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from czitools import pylibczirw_metadata as czimd
from czitools import pylibczirw_tools
from czitools import napari_tools
from czitools import misc
import napari
import os
from pathlib import Path

# adapt to your needs
defaultdir = os.path.join(Path(__file__).resolve().parents[2], "data")

# open s simple dialog to select a CZI file
filepath = misc.openfile(directory=defaultdir,
                         title="Open CZI Image File",
                         ftypename="CZI Files",
                         extension="*.czi")
print(filepath)

# return a array with dimension order STZCYX(A)
array6d, mdata, dim_string6d = pylibczirw_tools.read_6darray(filepath,
                                                             output_order="STCZYX",
                                                             output_dask=False,
                                                             remove_Adim=True,
                                                             # T=0,
                                                             # Z=0
                                                             )

# show array inside napari viewer
viewer = napari.Viewer()
layers = napari_tools.show(viewer, array6d, mdata,
                           dim_string=dim_string6d,
                           blending="additive",
                           contrast='napari_auto',
                           gamma=0.85,
                           add_mdtable=True,
                           name_sliders=True)

napari.run()
