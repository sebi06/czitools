# -*- coding: utf-8 -*-

#################################################################
# File        : use_pylibczirw_metadata_class.py
# Version     : 0.0.4
# Author      : sebi06
# Date        : 14.02.2022
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from __future__ import annotations
from czimetadata_tools import pylibczirw_metadata as czimd
from czimetadata_tools import misc
import os
from pathlib import Path

# adapt to your needs
defaultdir = os.path.join(Path(__file__).resolve().parents[2], "testdata")
#defaultdir = r"C:\Testdata_Zeiss\CZI_Testfiles"


# open s simple dialog to select a CZI file
filepath = misc.openfile(directory=defaultdir,
                         title="Open CZI Image File",
                         ftypename="CZI Files",
                         extension="*.czi")
print(filepath)

# get the complete metadata at once as one big class
mdata_sel = czimd.CziMetadata(filepath)

# get only specific metadata
czi_dimensions = czimd.CziDimensions(filepath)
print("SizeS: ", czi_dimensions.SizeS)
print("SizeT: ", czi_dimensions.SizeT)
print("SizeZ: ", czi_dimensions.SizeZ)
print("SizeC: ", czi_dimensions.SizeC)
print("SizeY: ", czi_dimensions.SizeY)
print("SizeX: ", czi_dimensions.SizeX)

# and get more info
czi_scaling = czimd.CziScaling(filepath)
czi_channels = czimd.CziChannelInfo(filepath)
czi_bbox = czimd.CziBoundingBox(filepath)
czi_info = czimd.CziInfo(filepath)
czi_objectives = czimd.CziObjectives(filepath)
czi_detectors = czimd.CziDetector(filepath)
czi_microscope = czimd.CziMicroscope(filepath)
czi_sample = czimd.CziSampleInfo(filepath)

# get selected metadata as a dictionary
mdata_sel_dict = czimd.obj2dict(mdata_sel)
for k, v in mdata_sel_dict.items():
    print(k, " : ", v)

# and convert to pd.DataFrame
df_md = misc.md2dataframe(mdata_sel_dict)
print(df_md)

# try to write XML to file
xmlfile = czimd.writexml(filepath)
