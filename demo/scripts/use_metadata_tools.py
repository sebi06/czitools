# -*- coding: utf-8 -*-

#################################################################
# File        : use_metadata_tools.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from czitools.metadata_tools.czi_metadata import CziMetadata, writexml, get_metadata_as_object, obj2dict
from czitools.metadata_tools.dimension import CziDimensions
from czitools.metadata_tools.boundingbox import CziBoundingBox
from czitools.metadata_tools.channel import CziChannelInfo
from czitools.metadata_tools.scaling import CziScaling
from czitools.metadata_tools.sample import CziSampleInfo
from czitools.metadata_tools.objective import CziObjectives
from czitools.metadata_tools.microscope import CziMicroscope
from czitools.metadata_tools.add_metadata import CziAddMetaData
from czitools.metadata_tools.detector import CziDetector
from czitools.utils import misc
from pathlib import Path

# adapt to your needs
defaultdir = Path(Path(__file__).resolve().parents[2]) / "data"

# open s simple dialog to select a CZI file
filepath = misc.openfile(directory=str(defaultdir),
                         title="Open CZI Image File",
                         ftypename="CZI Files",
                         extension="*.czi")
print(filepath)

# get the metadata_tools at once as one big class
mdata = CziMetadata(filepath)

# get only specific metadata_tools
czi_dimensions = CziDimensions(filepath)
print("SizeS: ", czi_dimensions.SizeS)
print("SizeT: ", czi_dimensions.SizeT)
print("SizeZ: ", czi_dimensions.SizeZ)
print("SizeC: ", czi_dimensions.SizeC)
print("SizeY: ", czi_dimensions.SizeY)
print("SizeX: ", czi_dimensions.SizeX)

# try to write XML to file
xmlfile = writexml(filepath)

# get info about the channels
czi_channels = CziChannelInfo(filepath)

# get the complete metadata_tools from the CZI as one big object
czimd_complete = get_metadata_as_object(filepath)

# get an object containing only the dimension information
czi_dimensions = CziDimensions(filepath)
print("Number of Channels:", czi_dimensions.SizeC)

# get an object containing only the dimension information
czi_scale = CziScaling(filepath)

# get an object containing information about the sample
czi_sample = CziSampleInfo(filepath)

# get info about the objective, the microscope and the detectors
czi_objectives = CziObjectives(filepath)
czi_detectors = CziDetector(filepath)
czi_microscope = CziMicroscope(filepath)

# get info about the sample carrier
czi_sample = CziSampleInfo(filepath)

# get additional metainformation
czi_addmd = CziAddMetaData(filepath)

# get the complete data about the bounding boxes
czi_bbox = CziBoundingBox(filepath)

# show the total bounding box, box per scene, total rectangle and num. of channels
print(czi_bbox.total_bounding_box)
print(czi_bbox.scenes_bounding_rect)
print(czi_bbox.total_rect)
print(czi_bbox.total_bounding_box["C"])

# get selected metadata_tools as a dictionary
mdata_dict = obj2dict(mdata)

# and convert to pd.DataFrame
df_md = misc.md2dataframe(mdata_dict)
print(df_md)
