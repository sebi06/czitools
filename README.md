# czitools

[![PyPI](https://img.shields.io/pypi/v/czitools.svg?color=green)](https://pypi.org/project/czitools)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/czitools)](https://pypistats.org/packages/czitools)
[![License](https://img.shields.io/pypi/l/czitools.svg?color=green)](https://github.com/sebi06/czitools/raw/master/LICENSE)
[![codecov](https://codecov.io/github/sebi06/czitools/graph/badge.svg?token=WK1KIMZARL)](https://codecov.io/github/sebi06/czitools)
[![Python Version](https://img.shields.io/pypi/pyversions/czitools.svg?color=green)](https://python.org)
[![Development Status](https://img.shields.io/pypi/status/czitools.svg)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Alpha)

This repository provides a collection of tools to simplify reading CZI (Carl Zeiss Image) pixel and metadata in Python. It is available as a [Python Package on PyPi](https://pypi.org/project/czitools/)

## Installation

To install czitools (core functionality) use:

```text
pip install czitools
```

To install the package with all optional dependencies use::

```text
pip install czitools[all]
```

### Local Installation

Local installation for developing etc.:

```text
pip install -e .
```

Local installation (full functionality):

```text
pip install -e ".[all]"
```

### Supported Operating Systems

Currently this only works on:

* Linux
* Windows

MacOS is not supported yet out of the box, but if one installs pylibCZIrw wheels for MacOS manually the package should work (not tested).

Thanks to the community for providing [MaxOS wheels for pylibCZIrw](https://pypi.scm.io/#/package/pylibczirw) wheels for MacOS, which makes it possible to read and write CZI files on MacOS.


## Reading the metadata

Please check [use_metadata_tools.py](https://github.com/sebi06/czitools/blob/main/demo/scripts/use_metadata_tools.py) for some examples.

```python
from czitools.metadata_tools.czi_metadata import CziMetadata, writexml
from czitools.metadata_tools.dimension import CziDimensions
from czitools.metadata_tools.boundingbox import CziBoundingBox
from czitools.metadata_tools.channel import CziChannelInfo
from czitools.metadata_tools.scaling import CziScaling
from czitools.metadata_tools.sample import CziSampleInfo
from czitools.metadata_tools.objective import CziObjectives
from czitools.metadata_tools.microscope import CziMicroscope
from czitools.metadata_tools.add_metadata import CziAddMetaData
from czitools.metadata_tools.detector import CziDetector
from czitools.read_tools import read_tools

try:
    import napari
    from napari.utils.colormaps import Colormap

    show_napari = True
except ImportError:
    print("Napari not installed, skipping napari import")
    show_napari = False

# get the metadata_tools at once as one big class
mdata = CziMetadata(filepath)

# get only specific metadata_tools
czi_dimensions = CziDimensions(filepath)
print("SizeS: ", czi_dimensions.SizeS)
print("SizeT: ", czi_dimensions.SizeT)
print("SizeZ: ", czi_dimensions.SizeZ)
print("SizeC: ", czi_dimensions.SizeC)
print("SizeY: ", czi_dimensions.SizeY)
print("SizeX: ", czi_dimensions.SizeX)

# try to write XML to file
xmlfile = writexml(filepath)

# get info about the channels
czi_channels = CziChannelInfo(filepath)

# get the complete metadata_tools from the CZI as one big object
czimd_complete = get_metadata_as_object(filepath)

# get an object containing only the dimension information
czi_scale = CziScaling(filepath)

# get an object containing information about the sample
czi_sample = CziSampleInfo(filepath)

# get info about the objective, the microscope and the detectors
czi_objectives = CziObjectives(filepath)
czi_detectors = CziDetector(filepath)
czi_microscope = CziMicroscope(filepath)

# get info about the sample carrier
czi_sample = CziSampleInfo(filepath)

# get additional metainformation
czi_addmd = CziAddMetaData(filepath)

# get the complete data about the bounding boxes
czi_bbox = CziBoundingBox(filepath)
```

## Reading CZI pixel data

While the [pylibCZIrw](https://pypi.org/project/pylibCZIrw/) is focussing on reading individual planes it is also helpful to read CZI pixel data as a STCZYX(A) stack. Please check [use_read_tools.py](https://github.com/sebi06/czitools/blob/main/demo/scripts/use_read_tools.py) for some examples.

```python
# return a dask or numpy array with dimension order STCZYX(A)
array6d, mdata = read_tools.read_6darray(filepath, use_xarray=True)

if show_napari:

    # show in napari (requires napari to be installed!)
    viewer = napari.Viewer()

    # loop over all channels
    for ch in range(0, array6d.sizes["C"]):

        # extract channel subarray
        sub_array = array6d.sel(C=ch)

        # get the scaling factors for that channel and adapt Z-axis scaling
        scalefactors = [1.0] * len(sub_array.shape)
        scalefactors[sub_array.get_axis_num("Z")] = mdata.scale.ratio["zx_sf"]

        # remove the last scaling factor in case of an RGB image
        if "A" in sub_array.dims:
            # remove the A axis from the scaling factors
            scalefactors.pop(sub_array.get_axis_num("A"))

        # get colors and channel name
        chname = mdata.channelinfo.names[ch]

        # inside the CZI metadata_tools colors are defined as ARGB hexstring
        rgb = "#" + mdata.channelinfo.colors[ch][3:]
        ncmap = Colormap(["#000000", rgb], name="cm_" + chname)

        # add the channel to the viewer
        viewer.add_image(
            sub_array,
            name=chname,
            colormap=ncmap,
            blending="additive",
            scale=scalefactors,
            gamma=0.85,
        )

        # set the axis labels based on the dimensions
        viewer.dims.axis_labels = sub_array.dims

    napari.run()

```


![5D CZI inside Napari](https://github.com/sebi06/czitools/raw/main/images/czi_napari2.png)

## Colab Notebooks

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

### Read a CZI and segment using Voroni-Otsu provided by PyClesperanto GPU processing

The basic usage can be inferred from this sample notebook:&nbsp;
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/read_czi_segment_voroni_otsu.ipynb)

## Remarks

The code to read multi-dimensional with delayed reading using Dask array was heavily inspired by input from: [Pradeep Rajasekhar](https://github.com/pr4deepr).
