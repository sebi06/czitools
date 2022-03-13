from __future__ import annotations
from czimetadata_tools import pylibczirw_metadata as czimd
from pylibCZIrw import czi as pyczi
from czimetadata_tools import misc

# adapt to your needs
defaultdir = r"D:\Testdata_Zeiss\CZI_Testfiles\Bugs"

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

# open the CZI document to read the
with pyczi.open_czi(filename) as czidoc:

    image2d = czidoc.read(plane={'T': 0, 'Z': 10, 'C': 0})
    print("Shape", image2d.shape)
