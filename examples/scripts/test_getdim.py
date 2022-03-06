from czimetadata_tools import pylibczirw_metadata as czimd
from pylibCZIrw import czi as pyczi

filename = r".\testdata\WP96_4Pos_B4-10_DAPI.czi"

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
