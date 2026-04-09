# czitools

[![PyPI](https://img.shields.io/pypi/v/czitools.svg?color=green)](https://pypi.org/project/czitools)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/czitools)](https://pypistats.org/packages/czitools)
[![License](https://img.shields.io/pypi/l/czitools.svg?color=green)](https://github.com/sebi06/czitools/raw/master/LICENSE)
[![codecov](https://codecov.io/github/sebi06/czitools/graph/badge.svg?token=WK1KIMZARL)](https://codecov.io/github/sebi06/czitools)
[![Python Version](https://img.shields.io/pypi/pyversions/czitools.svg?color=green)](https://python.org)
[![Development Status](https://img.shields.io/pypi/status/czitools.svg)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Alpha)

This repository provides a collection of tools to simplify reading CZI (Carl Zeiss Image) pixel and metadata in Python. It is available as a [Python Package on PyPi](https://pypi.org/project/czitools/).

For full documentation see **[sebi06.github.io/czitools](https://sebi06.github.io/czitools/)**.

## Quick Start

```bash
pip install czitools
```

```python
from czitools.metadata_tools.czi_metadata import CziMetadata
from czitools.read_tools import read_tools

# read all metadata
mdata = CziMetadata("path/to/file.czi")

# read pixel data as a labelled STCZYX(A) array
array6d, mdata = read_tools.read_6darray("path/to/file.czi", use_dask=True, use_xarray=True)
```

For installation options (PyPI, editable, conda) see the [Installation docs](https://sebi06.github.io/czitools/install/).
For detailed usage examples see the [Usage docs](https://sebi06.github.io/czitools/usage/).

**CZI inside NDV**

![5D CZI inside NDV](https://github.com/sebi06/czitools/raw/main/_images/czi_ndv1.png)

**CZI inside Napari**

![5D CZI inside Napari](https://github.com/sebi06/czitools/raw/main/_images/czi_napari2.png)

## Colab Notebooks

| Topic                      | Link                                                                                                                                                                                               |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Read CZI metadata          | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/read_czi_metadata.ipynb)            |
| Read CZI pixel data        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/read_czi_pixeldata.ipynb)           |
| Write OME-ZARR from CZI    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/omezarr_from_czi_5d.ipynb)          |
| Save with ZSTD compression | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/save_with_ZSTD_compression.ipynb)   |
| Show planetable as surface | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/show_czi_surface.ipynb)             |
| Segment with Voronoi-Otsu  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/read_czi_segment_voroni_otsu.ipynb) |
