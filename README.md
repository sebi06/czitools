# czitools

[![PyPI](https://img.shields.io/pypi/v/czitools.svg?color=green)](https://pypi.org/project/czitools)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/czitools)](https://pypistats.org/packages/czitools)
[![License](https://img.shields.io/pypi/l/czitools.svg?color=green)](https://github.com/sebi06/czitools/raw/master/LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/czitools.svg?color=green)](https://python.org)
[![Development Status](https://img.shields.io/pypi/status/czitools.svg)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Alpha)

This repository provides a collection of tools to simplify reading CZI (Carl Zeiss Image) pixel and metadata in Python. In addition it also contains other useful utilities to visualize CZI images inside Napari (needs to be installed). It is also available as a [Python Package on PyPi](https://pypi.org/project/czitools/)

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

While the [pylibCZIrw](https://pypi.org/project/pylibCZIrw/) is focussing on reading individual planes it is also helpful to read CZI pixel data as a STZCYX(A) stack. Please check [use_pylibczirw_md_read.py](https://github.com/sebi06/czitools/raw/main/demo/scripts/use_pylibczirw_md_read.py) for some examples.

```python
# return a array with dimension order STZCYX(A)
array6d, mdata, dim_string6d = pylibczirw_tools.read_6darray(filepath,
                                                             output_order="STZCYX",
                                                             output_dask=False,
                                                             remove_adim=True
                                                             )

# show array inside napari viewer
viewer = napari.Viewer()
layers = napari_tools.show(viewer, array6d, mdata,
                           dim_string=dim_string6d,
                           blending="additive",
                           contrast='napari_auto',
                           gamma=0.85,
                           add_mdtable=True,
                           name_sliders=True)

napari.run()
```

![5D CZI inside Napari](https://github.com/sebi06/czitools/raw/main/images/czi_napari1.png)