# czitools

[![PyPI](https://img.shields.io/pypi/v/czitools.svg?color=green)](https://pypi.org/project/czitools)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/czitools)](https://pypistats.org/packages/czitools)
[![License](https://img.shields.io/pypi/l/czitools.svg?color=green)](https://github.com/sebi06/czitools/raw/master/LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/czitools.svg?color=green)](https://python.org)
[![Development Status](https://img.shields.io/pypi/status/czitools.svg)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Alpha)

This repository provides a collection of tools to simplify reading CZI (Carl Zeiss Image) pixel and metadata in Python. In addition it also contains other useful utilities to visualize CZI images inside Napari (needs to be installed). It is also available as a [Python Package on PyPi](https://pypi.org/project/czitools/)

## Reading the metadata

Please check [use_metadata_tools.py](https://github.com/sebi06/czitools/blob/main/demo/scripts/use_metadata_tools.py) for some examples.

```python
# get the metadata at once as one big class
mdata = czimd.CziMetadata(filepath)

# get only specific metadata
czi_dimensions = czimd.CziDimensions(filepath)
print("SizeS: ", czi_dimensions.SizeS)
print("SizeT: ", czi_dimensions.SizeT)
print("SizeZ: ", czi_dimensions.SizeZ)
print("SizeC: ", czi_dimensions.SizeC)
print("SizeY: ", czi_dimensions.SizeY)
print("SizeX: ", czi_dimensions.SizeX)

# try to write XML to file
xmlfile = czimd.writexml(filepath)

# get info about the channels
czi_channels = czimd.CziChannelInfo(filepath)

# get the complete metadata from the CZI as one big object
czimd_complete = czimd.get_metadata_as_object(filepath)

# get an object containing only the dimension information
czi_dimensions = czimd.CziDimensions(filepath)

# get an object containing only the dimension information
czi_scale = czimd.CziScaling(filepath)

# get an object containing information about the sample
czi_sample = czimd.CziSampleInfo(filepath)

# get info about the objective, the microscope and the detectors
czi_objectives = czimd.CziObjectives(filepath)
czi_detectors = czimd.CziDetector(filepath)
czi_microscope = czimd.CziMicroscope(filepath)

# get info about the sample carrier
czi_sample = czimd.CziSampleInfo(filepath)

# get additional metainformation
czi_addmd = czimd.CziAddMetaData(filepath)

# get the complete data about the bounding boxes
czi_bbox = czimd.CziBoundingBox(filepath)
```

## Reading CZI pixel data

While the [pylibCZIrw](https://pypi.org/project/pylibCZIrw/) is focussing on reading individual planes it is also helpful to read CZI pixel data as a STZCYX(A) stack. Please check [use_read_tools.py](https://github.com/sebi06/czitools/blob/main/demo/scripts/use_read_tools.py) for some examples.

```python
# return a dask or numpy array with dimension order STZCYX(A)
array6d, mdata, dim_string6d = read_tools.read_6darray(filepath,
                                                       output_order="STCZYX",
                                                       use_dask=True,
                                                       chunk_zyx=False,
                                                       # T=0,
                                                       # Z=0
                                                       # S=0
                                                       # C=0
                                                       )

if array6d is None:
    print("Empty array6d. Nothing to display in Napari")
else:

    # show array inside napari viewer
    viewer = napari.Viewer()
    layers = napari_tools.show(viewer, array6d, mdata,
                               dim_string=dim_string6d,
                               blending="additive",
                               contrast='from_czi',
                               gamma=0.85,
                               add_mdtable=True,
                               name_sliders=True)

    napari.run()
```

![5D CZI inside Napari](https://github.com/sebi06/czitools/raw/main/images/czi_napari1.png)

## Coloab Notebooks

### Read CZI metadata

The basic usage can be inferred from this sample notebook:&nbsp;
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/read_czi_metadata.ipynb)

### Read CZI pixeldata

The basic usage can be inferred from this sample notebook:&nbsp;
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/read_czi_pixeldata.ipynb)

### Write OME-ZARR from 5D CZI image data

The basic usage can be inferred from this sample notebook:&nbsp;
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/omezarr_from_czi_5d.ipynb)


### Write CZI using ZSTD compression

The basic usage can be inferred from this sample notebook:&nbsp;
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/save_with_ZSTD_compression.ipynb)

### Show planetable of a CZI image as surface

The basic usage can be inferred from this sample notebook:&nbsp;
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/show_czi_surface.ipynb)

## Remarks

The code to read multi-dimensional with delayed reading using Dask array was heavily inspired by input from: [Pradeep Rajasekhar](https://github.com/pr4deepr).
