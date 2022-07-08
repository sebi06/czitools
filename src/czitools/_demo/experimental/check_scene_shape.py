# -*- coding: utf-8 -*-

#################################################################
# File        : read_czi_simple.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from czitools.metadata import pylibczirw_metadata as czimd
from czitools.imagedata import pylibczirw_tools
from pylibCZIrw import czi as pyczi

filepath = r"C:\Users\m1srh\Documents\Github\czitools\data\S=3_1Pos_2Mosaic_T=2=Z=3_CH=2_sm.czi"
#filepath = r"C:\Users\m1srh\Documents\Github\czitools\data\CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi"
#filepath = r"C:\Users\m1srh\Documents\Github\czitools\data\WP96_4Pos_B4-10_DAPI.czi"
#filepath = r"D:\Testdata_Zeiss\CZI_Testfiles\DTScan_ID4.czi"

# get the complete metadata at once as one big class
mdata = czimd.CziMetadata(filepath)

scene_width = []
scene_height = []

shape_is_consistent = False

with pyczi.open_czi(filepath) as czidoc:

    if mdata.image.SizeS is not None:

        for s in range(mdata.image.SizeS):

            scene_width.append(czidoc.scenes_bounding_rectangle[s].w)
            scene_height.append(czidoc.scenes_bounding_rectangle[s].h)

        sw = scene_width.count(scene_width[0]) == len(scene_width)
        sh = scene_height.count(scene_height[0]) == len(scene_height)

        if sw == True and sh == True:
            shape_is_consistent = True

    else:

        shape_is_consistent = True

print("Shape Consistent", shape_is_consistent)

# return a array with dimension order STZCYX(A)
array6d, dimstring = pylibczirw_tools.read_6darray(filepath,
                                                   dimorder="STCZYX",
                                                   output_dask=False,
                                                   remove_Adim=True)

print("Array6d Shape", array6d.shape)
print("Dimension String", dimstring)

# return a array with dimension order STZCYX(A)
array5d, dimstring = pylibczirw_tools.read_5darray(filepath,
                                                   scene=1,
                                                   dimorder="TCZYX",
                                                   output_dask=False,
                                                   remove_Adim=True)
print("Array5d Shape", array5d.shape)
print("Dimension String", dimstring)
