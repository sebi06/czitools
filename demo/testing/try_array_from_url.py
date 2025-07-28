from czitools.read_tools import read_tools

url = r"https://github.com/sebi06/napari-czitools/raw/main/src/napari_czitools/sample_data/CellDivision_T10_Z20_CH2_X600_Y500_DCV_ZSTD.czi"

# return an array with dimension order STCZYX(A)
array6d, mdata = read_tools.read_6darray(url)

print(f"Array shape: {array6d.shape}")
