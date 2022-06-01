# czimetadata_tools

This repository provided a collection of tools Tools to simplify reading CZI (Carl Zeiss Image) pixel and metadata in Python.

## Reading the metadata

Please check [use_pylibczirw_metadata_class.py](examples/scripts/use_pylibczirw_metadata_class.py) for some examples.

```python
# get the metadata at once as one big class
mdata_sel = czimd.CziMetadata(filepath)

# get only specific metadata
czi_dimensions = czimd.CziDimensions(filepath)
print("SizeS: ", czi_dimensions.SizeS)
print("SizeT: ", czi_dimensions.SizeT)
print("SizeZ: ", czi_dimensions.SizeZ)
print("SizeC: ", czi_dimensions.SizeC)
print("SizeY: ", czi_dimensions.SizeY)
print("SizeX: ", czi_dimensions.SizeX)

# and get more info about various aspects of the CZI
czi_scaling = czimd.CziScaling(filepath)
czi_channels = czimd.CziChannelInfo(filepath)
czi_bbox = czimd.CziBoundingBox(filepath)
czi_info = czimd.CziInfo(filepath)
czi_objectives = czimd.CziObjectives(filepath)
czi_detectors = czimd.CziDetector(filepath)
czi_microscope = czimd.CziMicroscope(filepath)
czi_sample = czimd.CziSampleInfo(filepath)
```

## Reading CZI pixeldata

While the [pylibCZIrw](https://pypi.org/project/pylibCZIrw/) is focussing on reading individual planes it is also helpful to read CZI pixel data as a STZCYX(A) stack. Please check [use_pylibczirw_md_read.py](examples/scripts/use_pylibczirw_md_read.py) for some examples.

```python
# return a array with dimension order STZCYX(A)
mdarray, dimstring = pylibczirw_tools.read_mdarray(filepath, remove_Adim=True)

# remove A dimension do display the array inside Napari
dim_order, dim_index, dim_valid = czimd.CziMetadata.get_dimorder(dimstring)

# show array inside napari viewer
viewer = napari.Viewer()
layers = napari_tools.show(viewer, mdarray, mdata,
                           dim_order=dim_order,
                           blending="additive",
                           contrast='napari_auto',
                           gamma=0.85,
                           add_mdtable=True,
                           name_sliders=True)

napari.run()
```

![5D CZI inside Napari](docs/images/czi_napari1.png)