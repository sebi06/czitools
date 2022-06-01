# -*- coding: utf-8 -*-

#################################################################
# File        : read_czi_simple.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from pylibCZIrw import czi as pyczi
from czimetadata_tools import misc
import os
from pathlib import Path

# adapt to your needs
defaultdir = os.path.join(Path(__file__).resolve().parents[2], "testdata")

# open s simple dialog to select a CZI file
filepath = misc.openfile(directory=defaultdir,
                         title="Open CZI Image File",
                         ftypename="CZI Files",
                         extension="*.czi")
print(filepath)

zoom_level = [1.0, 0.5, 0.25, 0.125, 0.1]

# open the CZI document to read the
with pyczi.open_czi(filepath) as czidoc:

    # get the image dimensions as a dictionary, where the key identifies the dimension
    total_bounding_box = czidoc.total_bounding_box
    print(total_bounding_box)

    # get the total bounding box for all scenes
    total_bounding_rectangle = czidoc.total_bounding_rectangle
    print(total_bounding_rectangle)

    # get the bounding boxes for each individual scene
    scenes_bounding_rectangle = czidoc.scenes_bounding_rectangle
    print(scenes_bounding_rectangle)

    for zl in zoom_level:
        print("Trying to read with ZoomLevel:", zl)
        image2d = czidoc.read(plane={'T': 0, 'Z': 0, 'C': 0}, zoom=zl)
        print("Shape image2d", image2d.shape, "ZoomLevel", zl)
