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
#from czitools.utils import napari_tools
#from czitools.utils import misc

# open s simple dialog to select a CZI file
#filepath = r"C:\Testdata_Zeiss\CZI_Testfiles\DTScan_ID4.czi"
#filepath = r"C:\Testdata_Zeiss\CZI_Testfiles\OverViewScan.czi"
filepath = r"C:\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=3_Z=4_CH=2.czi"
#filepath = r"C:\Testdata_Zeiss\CZI_Testfiles\CellDivision_T=10_Z=15_CH=2_DCV_small.czi"

# get only specific metadata
czi_dimensions = czimd.CziDimensions(filepath)
print("SizeS: ", czi_dimensions.SizeS)
print("SizeT: ", czi_dimensions.SizeT)
print("SizeZ: ", czi_dimensions.SizeZ)
print("SizeC: ", czi_dimensions.SizeC)
print("SizeY: ", czi_dimensions.SizeY)
print("SizeX: ", czi_dimensions.SizeX)

# return a array with dimension order STZCYX(A)
array6d, dimstring = pylibczirw_tools.read_6darray(filepath,
                                                   dimorder="STCZYX",
                                                   output_dask=False,
                                                   remove_Adim=True)

print("Done.")
