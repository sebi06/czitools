try:
    from czimetadata_tools import pylibczirw_metadata as czimd
except ImportError:
    import pylibczirw_metadata as czimd
from pylibCZIrw import czi as pyczi
import os
from pathlib import Path

basedir = Path(__file__).resolve().parents[2]
#filepath = os.path.join(basedir, r"testdata/WP96_4Pos_B4-10_DAPI.czi")
filepath = os.path.join(basedir, r"testdata/A01.czi")

with pyczi.open_czi(filepath) as czidoc_r:
    metadata_parsed = czidoc_r.metadata
    dim_dict = czimd.CziDimensions.get_image_dimensions(metadata_parsed)

print(dim_dict)

czimd_complete = czimd.CziMetadataComplete(filepath)

czi_dimensions = czimd.CziDimensions(filepath)
czi_scale = czimd.CziScaling(filepath)
czi_info = czimd.CziInfo(filepath)
czi_sample = czimd.CziSampleInfo(filepath)

print(czi_dimensions.__dict__)
print(czi_scale.__dict__)
print(czi_sample.__dict__)

czi_bbox = czimd.CziBoundingBox(filepath)
print(czi_bbox.all_scenes)
print(czi_bbox.total_rect)
print(czi_bbox.total_bounding_box)
print(type(czi_bbox.total_bounding_box["T"]))
