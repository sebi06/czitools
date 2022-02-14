# -*- coding: utf-8 -*-

#################################################################
# File        : test_pylibczirw_metadata_class.py
# Version     : 0.0.4
# Author      : sebi06
# Date        : 14.02.2022
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from __future__ import annotations
from czitools import pylibczirw_metadata as czimd
from czitools import misc

# adapt to your needs
defaultdir = r"C:\Testdata_Zeiss\CZI_Testfiles"
#defaultdir = r"d:\Temp\test_czi_write"

# open s simple dialog to select a CZI file
filename = misc.openfile(directory=defaultdir,
                         title="Open CZI Image File",
                         ftypename="CZI Files",
                         extension="*.czi")
print(filename)

# get the complete metadata at once as one big class
mdata_sel = czimd.CziMetadata(filename)

# get only specific metadata
czi_dimensions = czimd.CziDimensions(filename)
print("SizeS: ", czi_dimensions.SizeS)
print("SizeT: ", czi_dimensions.SizeT)
print("SizeZ: ", czi_dimensions.SizeZ)
print("SizeC: ", czi_dimensions.SizeC)
print("SizeY: ", czi_dimensions.SizeY)
print("SizeX: ", czi_dimensions.SizeX)

# and get more info
czi_scaling = czimd.CziScaling(filename)
czi_channels = czimd.CziChannelInfo(filename)
czi_bbox = czimd.CziBoundingBox(filename)
czi_info = czimd.CziInfo(filename)
czi_objectives = czimd.CziObjectives(filename)
czi_detectors = czimd.CziDetector(filename)
czi_microscope = czimd.CziMicroscope(filename)
czi_sample = czimd.CziSampleInfo(filename)

# get selected metadata as a dictionary
mdata_sel_dict = czimd.obj2dict(mdata_sel)
for k,v in mdata_sel_dict.items():
    print(k, " : ", v)

# and convert to pd.DataFrame
df_md = misc.md2dataframe(mdata_sel_dict)
print(df_md)

# try to write XML to file
xmlfile = czimd.writexml(filename)
