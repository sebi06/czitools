# -*- coding: utf-8 -*-

#################################################################
# File        : test_czimetadata_class.py
# Version     : 0.0.7
# Author      : sebi06
# Date        : 17.03.2022
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from __future__ import annotations
from czimetadata_tools import pylibczirw_metadata as czimd
from czimetadata_tools import pylibczirw_tools as czit
import napari
from aicspylibczi import CziFile
from czitools import misc, napari_tools

# adapt to your needs
defaultdir = r"C:\Testdata_Zeiss\CZI_Testfiles"

# open s simple dialog to select a CZI file
filename = misc.openfile(directory=defaultdir,
                         title="Open CZI Image File",
                         ftypename="CZI Files",
                         extension="*.czi")
print(filename)

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
czi_addmd = czimd.CziAddMetaData(filename)

# get the complete metadata at once as one big class
mdata = czimd.CziMetadata(filename)

# get selected metadata as a dictionary
mdata_dict = czimd.create_mdict_red(mdata)
for k, v in mdata_dict.items():
    print(k, " : ", v)

print("---------------------------------------------------")

# and convert to pd.DataFrame
df_md = misc.md2dataframe(mdata_dict)
print(df_md[:4])

print("---------------------------------------------------")

# write metadata as XML to disk
xmlfile = czimd.writexml(filename)

# get the planetable for the CZI file and save it (optional)
pt, csvfile = czimd.aics_get_planetable(filename,
                                        norm_time=True,
                                        savetable=True,
                                        separator=",",
                                        index=True)

print(pt[:5])

# get info from a specific scene
aicsczi = CziFile(filename)
scene = czimd.CziScene(aicsczi, 0)
print("Scene XY-Width-Height :", scene.xstart, scene.ystart, scene.width, scene.height)
print("Scene DimString :", scene.single_scene_dimstr)
print("Scene Shape :", scene.shape_single_scene)

# read pixel data
all_scenes, _ = czit.read(filename)

# show array inside napari viewer
viewer = napari.Viewer()
layers = napari_tools.show(viewer, all_scenes, mdata,
                           dim_order=mdata.aics_dim_order,
                           blending="additive",
                           contrast="napari_auto",
                           gamma=0.85,
                           add_mdtable=True,
                           name_sliders=True)

napari.run()
