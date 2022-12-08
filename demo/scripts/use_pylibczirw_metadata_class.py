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

# get the metadata at once as one big class
mdata_sel = czimd.CziMetadata(filepath)

# get only specific metadata
czi_dimensions = czimd.CziDimensions(filepath)
print("SizeS: ", czi_dimensions.SizeS)
print("SizeT: ", czi_dimensions.SizeT)
print("SizeZ: ", czi_dimensions.SizeZ)
print("SizeC: ", czi_dimensions.SizeC)
print("SizeY: ", czi_dimensions.SizeY)
print("SizeX: ", czi_dimensions.SizeX)

# and get more info about various aspects of the CZI
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

# and convert to pd.DataFrame
df_md = misc.md2dataframe(mdata_sel_dict)
print(df_md)

# try to write XML to file
xmlfile = czimd.writexml(filepath)

# open a CZI file for reading
with pyczi.open_czi(filepath) as czidoc_r:

    # get the metadata as a dictionary
    metadata_parsed = czidoc_r.metadata

    # get a dictionary with the dimensions by parsing the dictionary
    dim_dict = czimd.CziDimensions.get_image_dimensions(metadata_parsed)
    # print(dim_dict)

# get the complete metadata from the CZI as one big object
czimd_complete = czimd.CziMetadataComplete(filepath)

# get an object containing only the dimension information
czi_dimensions = czimd.CziDimensions(filepath)
print("Number of Channels:", czi_dimensions.SizeC)

# get an object containing only the dimension information
czi_scale = czimd.CziScaling(filepath)

# get an object containing additional general information
czi_info = czimd.CziInfo(filepath)

# get an object containing information about the sample
czi_sample = czimd.CziSampleInfo(filepath)

# show those information as a dictionary
print(czi_dimensions.__dict__)
print(czi_scale.__dict__)
print(czi_sample.__dict__)

# get the complete data about the bounding boxes
czi_bbox = czimd.CziBoundingBox(filepath)

# dshow the total bounding box for all dimensions
print(czi_bbox.total_bounding_box)

# show the bbox for the scenes
print(czi_bbox.scenes_bounding_rect)

# show the total rectangle for all scenes
print(czi_bbox.total_rect)

# show just the number of channels
print(czi_bbox.total_bounding_box["C"])
