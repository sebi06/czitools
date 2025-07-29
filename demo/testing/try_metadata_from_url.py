from czitools.metadata_tools.czi_metadata import CziMetadata
from czitools.utils import misc


# Example usage
# url = r"https://github.com/sebi06/napari-czitools/raw/main/src/napari_czitools/sample_data/CellDivision_T3_Z6_CH1_X300_Y200_DCV_ZSTD.czi"
url = r"https://github.com/invalid/nonexistent/file.czi"
# url = r"https://www.dropbox.com/scl/fi/k2t0dqafsh8dn4n0jtfza/S3_1Pos_2Mosaic_T2_Z3_CH2_sm.czi?rlkey=xiui12mcr7evd0mt82newea1y&st=rnsq3hvi&dl=1"
is_valid, message = misc.is_valid_czi_url(url)
print(f"URL Validation Result: {is_valid}, Message: {message}")


# # What happens internally in czitools when reading from URL:
# print("Creating CziMetadata from URL...")
# mdata = CziMetadata(url)  # This partially works
# print(f"Metadata created successfully: {mdata}")
# print(f"Has bbox: {hasattr(mdata, 'bbox')}")
# print(f"bbox type: {type(mdata.bbox)}")
# print(f"total_bounding_box: {mdata.bbox.total_bounding_box}")
# print(f"total_bounding_box type: {type(mdata.bbox.total_bounding_box)}")

# # Leading to:
# k = "T"
# print(f"\nTesting access to key '{k}'...")
# if mdata.bbox.total_bounding_box is not None:
#     if k in mdata.bbox.total_bounding_box:  # This should work!
#         print(f"SUCCESS: {mdata.bbox.total_bounding_box[k]}")
#     else:
#         print(f"Key '{k}' not found in bounding box")
# else:
#     print("ERROR: total_bounding_box is None!")
