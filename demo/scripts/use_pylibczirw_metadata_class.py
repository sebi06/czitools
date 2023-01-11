# -*- coding: utf-8 -*-

#################################################################
# File        : use_pylibczirw_metadata_class.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from pylibCZIrw import czi as pyczi
from czitools import pylibczirw_metadata as czimd
from czitools import misc
from pathlib import Path

# adapt to your needs
defaultdir = Path(Path(__file__).resolve().parents[2]) / "data"

# open s simple dialog to select a CZI file
filepath = misc.openfile(directory=defaultdir,
                         title="Open CZI Image File",
                         ftypename="CZI Files",
                         extension="*.czi")
print(filepath)

# get the metadata at once as one big class
mdata = czimd.CziMetadata(filepath)

# get only specific metadata
czi_dimensions = czimd.CziDimensions(filepath)
print("SizeS: ", czi_dimensions.SizeS)
print("SizeT: ", czi_dimensions.SizeT)
print("SizeZ: ", czi_dimensions.SizeZ)
print("SizeC: ", czi_dimensions.SizeC)
print("SizeY: ", czi_dimensions.SizeY)
print("SizeX: ", czi_dimensions.SizeX)

# try to write XML to file
xmlfile = czimd.writexml(filepath)

# get info about the channels
czi_channels = czimd.CziChannelInfo(filepath)

# get the complete metadata from the CZI as one big object
czimd_complete = czimd.CziMetadataComplete(filepath)

# get an object containing only the dimension information
czi_dimensions = czimd.CziDimensions(filepath)
print("Number of Channels:", czi_dimensions.SizeC)

# get an object containing only the dimension information
czi_scale = czimd.CziScaling(filepath)

# get an object containing information about the sample
czi_sample = czimd.CziSampleInfo(filepath)

# get info about the objective, the microscope and the detectors
czi_objectives = czimd.CziObjectives(filepath)
czi_detectors = czimd.CziDetector(filepath)
czi_microscope = czimd.CziMicroscope(filepath)

# get info about the sample carrier
czi_sample = czimd.CziSampleInfo(filepath)

# get additional metainformation
czi_addmd = czimd.CziAddMetaData(filepath)

# get the complete data about the bounding boxes
czi_bbox = czimd.CziBoundingBox(filepath)

# show the total bounding box, box per scene, total rectangle and num. of channels
print(czi_bbox.total_bounding_box)
print(czi_bbox.scenes_bounding_rect)
print(czi_bbox.total_rect)
print(czi_bbox.total_bounding_box["C"])

# get selected metadata as a dictionary
mdata_dict = czimd.obj2dict(mdata)

# and convert to pd.DataFrame
df_md = misc.md2dataframe(mdata_dict)
print(df_md)
