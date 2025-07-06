from bioio import BioImage
import bioio_czi

filename = r"/home/sebi06/github/czitools/data/w96_A1+A2.czi"

img = BioImage(filename, reader=bioio_czi.Reader)
array5d = img.xarray_data
print(f"Shape: {img.shape}")
print(f"Scenes: {img.scenes}")
print(f"Current Scene: {img.current_scene}")
print(f"Current Scene Index: {img.current_scene_index}")
print(f"Dims: {img.dims}")
print(f"Dims Order: {img.dims.order}")
print(f"Array5D Shape: {array5d.shape}")
print(f"Array5D Dims: {array5d.dims}")
print(f"Array5D Coords: {array5d.coords}")
