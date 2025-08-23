from bioio import BioImage
import bioio_czi
from pathlib import Path
from matplotlib import pyplot as plt

# filename = "10x10.czi"
# filename = "CellDivision_T3_Z5_CH2_X240_Y170.czi"
# filename = "WP96_4Pos_B4-10_DAPI.czi"
filename = "w96_A1+A2.czi"
defaultdir = Path(Path(__file__).resolve().parents[2]) / "data"
filepath = defaultdir / filename

img = BioImage(filepath, reader=bioio_czi.Reader, reconstruct_mosaic=True, use_aicspylibczi=False)

print(f"File: {filepath}")
print(f"Shape: {img.shape}")
print(f"Scenes (all): {img.scenes}")
print(f"Dims: {img.dims}")
print(f"Dims Order: {img.dims.order}")
print(f"Resolution Levels: {img.resolution_levels}")

# # read using get_image_data
# fig1, ax1 = plt.subplots(1, len(img.scenes), figsize=(12, 6))
# fig1.suptitle("get_image_data")
# for i, scene in enumerate(img.scenes):
#     img.set_scene(i)
#     array5d_1 = img.get_image_data()
#     ax1[i].imshow(array5d_1[0, 0, 0, :, :], cmap="gray")
#     ax1[i].set_title(f"Scene {i}")

# read using get_stack
# array6d_2 = img.get_stack(drop_non_matching_scenes=True)
# print(f"Data shape: {array6d_2.shape}")

# fig3, ax3 = plt.subplots(1, len(img.scenes), figsize=(12, 6))
# fig3.suptitle("get_stack")
# for i, scene in enumerate(img.scenes):
#     print(f"Show Scene: {i}")
#     ax3[i].imshow(array6d_2[i, 0, 0, 0, :, :], cmap="gray")
#     ax3[i].set_title(f"Scene {i}")

# read using get_xarray_dask_stack
array6d_3 = img.get_xarray_dask_stack(drop_non_matching_scenes=True)
print(f"Data shape: {array6d_3.shape}")

fig3, ax3 = plt.subplots(1, len(img.scenes), figsize=(12, 6))
fig3.suptitle("get_xarray_dask_stack")
for i, scene in enumerate(img.scenes):
    print(f"Show Scene: {i}")
    ax3[i].imshow(array6d_3[i, 0, 0, 0, :, :], cmap="gray")
    ax3[i].set_title(f"Scene {i}")

# read using get_dask_stack
# array6d_4 = img.get_dask_stack(drop_non_matching_scenes=True)
# print(f"Data shape: {array6d_4.shape}")

# fig4, ax4 = plt.subplots(1, len(img.scenes), figsize=(12, 6))
# fig4.suptitle("get_dask_stack")
# for i, scene in enumerate(img.scenes):
#     ax4[i].imshow(array6d_4[i, 0, 0, 0, :, :], cmap="gray")
#     ax4[i].set_title(f"Scene {i}")

# # read using get_xarray_stack
# array6d_5 = img.get_xarray_stack(drop_non_matching_scenes=True)
# print(f"Data shape: {array6d_5.shape}")

# fig5, ax5 = plt.subplots(1, len(img.scenes), figsize=(12, 6))
# fig5.suptitle("get_xarray_stack")
# for i, scene in enumerate(img.scenes):
#     ax5[i].imshow(array6d_5[i, 0, 0, 0, :, :], cmap="gray")
#     ax5[i].set_title(f"Scene {i}")

plt.show()
