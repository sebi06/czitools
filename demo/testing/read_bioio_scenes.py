from bioio import BioImage
import bioio_czi
from pathlib import Path
from matplotlib import pyplot as plt

# filename = "10x10.czi"
# filename = "CellDivision_T3_Z5_CH2_X240_Y170.czi"
# filename = "WP96_4Pos_B4-10_DAPI.czi"
# filename = "w96_A1+A2.czi"
# filename = "F:\Github\czitools\data\S3_1Pos_2Mosaic_T2_Z3_CH2_sm.czi"
filename = r"/home/sebi06/Downloads/overview_2_scenes.czi"
defaultdir = Path(Path(__file__).resolve().parents[2]) / "data"
filepath = defaultdir / filename


aics = BioImage(filepath, reader=bioio_czi.Reader, reconstruct_mosaic=True, use_aicspylibczi=True)

print(f"File: {filepath}")
print(f"Shape: {aics.shape}")
print(f"Scenes (all): {aics.scenes}")
print(f"Dims: {aics.dims}")
print(f"Dims Order: {aics.dims.order}")
print(f"Resolution Levels: {aics.resolution_levels}")

fig, ax = plt.subplots(1, len(aics.scenes), figsize=(12, 6))
for i, scene in enumerate(aics.scenes):
    aics.set_scene(i)
    array5d_1 = aics.xarray_data
    print(f"AICS - Scene {i + 1}: {scene}")
    print(f"AICS - Array Shape: {array5d_1.shape}")
    print(f"AICS - Array Dims: {array5d_1.dims}")
    ax[i].imshow(array5d_1[0, 0, 0, :, :], cmap="gray")
    ax[i].set_title(f"AICS - Scene {i}")


czi = BioImage(filepath, reader=bioio_czi.Reader, reconstruct_mosaic=True, use_aicspylibczi=False)

print(f"File: {filepath}")
print(f"Shape: {czi.shape}")
print(f"Scenes: {czi.scenes}")
print(f"Dims: {czi.dims}")
print(f"Dims Order: {czi.dims.order}")
print(f"Resolution Levels: {czi.resolution_levels}")

fig, ax = plt.subplots(1, len(czi.scenes), figsize=(12, 6))
for i, scene in enumerate(czi.scenes):
    czi.set_scene(i)
    array5d_2 = czi.xarray_data
    print(f"PyLibCZI - Scene {i + 1}: {scene}")
    print(f"PyLibCZI - Array Shape: {array5d_2.shape}")
    print(f"PyLibCZI - Array Dims: {array5d_2.dims}")
    ax[i].imshow(array5d_2[0, 0, 0, :, :], cmap="gray")
    ax[i].set_title(f"PyLibCZI - Scene {i}")


plt.show()
