from czitools import pylibczirw_metadata as czimd
from pylibCZIrw import czi as pyczi

# filename = r'C:\Testdata_Zeiss\CZI_Testfiles\CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi'
# filename = r'C:\Testdata_Zeiss\CZI_Testfiles\testwell96.czi'
# filename = r"C:\Testdata_Zeiss\CZI_Testfiles\tobacco_z=10_tiles.czi"
filename = r"C:\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=3_Z=4_CH=2.czi"

with pyczi.open_czi(filename) as czidoc_r:
    metadata_parsed = czidoc_r.metadata
    dim_dict = czimd.CziDimensions.get_image_dimensions(metadata_parsed)

print(dim_dict)

czi_dimensions = czimd.CziDimensions(filename)
czi_scale = czimd.CziScaling(filename)

print(czi_dimensions.__dict__)
print(czi_scale.__dict__)

czi_bbox = czimd.CziBoundingBox(filename)
print(czi_bbox.all_scenes)
print(czi_bbox.total_rect)
print(czi_bbox.total_bounding_box)
print(type(czi_bbox.total_bounding_box["T"]))
