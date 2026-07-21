# czitools

[![PyPI](https://img.shields.io/pypi/v/czitools.svg?color=green)](https://pypi.org/project/czitools)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/czitools)](https://pypistats.org/packages/czitools)
[![License](https://img.shields.io/pypi/l/czitools.svg?color=green)](https://github.com/sebi06/czitools/raw/master/LICENSE)
[![codecov](https://codecov.io/github/sebi06/czitools/graph/badge.svg?token=WK1KIMZARL)](https://codecov.io/github/sebi06/czitools)
[![Python Version](https://img.shields.io/pypi/pyversions/czitools.svg?color=green)](https://python.org)
[![Development Status](https://img.shields.io/pypi/status/czitools.svg)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Alpha)

This repository provides a collection of tools to simplify reading CZI (Carl Zeiss Image) pixel and metadata in Python. It is available as a [Python Package on PyPi](https://pypi.org/project/czitools/).

For full documentation see **[sebi06.github.io/czitools](https://sebi06.github.io/czitools/)**.

## Installation

### Basic Installation

Install the core package from PyPI:

```bash
pip install czitools
```

### Optional Features

Install with additional functionality using optional extras:

```bash
# OME-ZARR export (conversion + validation)
pip install czitools[omezarr]

# OME-ZARR export with GUI converter application
pip install czitools[omezarr-gui]

# HCS plate analysis and visualization
pip install czitools[analysis]

# Everything (all optional dependencies)
pip install czitools[all]
```

### Development Installation

For development or to get the latest unreleased features:

```bash
# Clone the repository
git clone https://github.com/sebi06/czitools.git
cd czitools

# Install in editable mode with all extras
pip install -e .[all]
```

### Conda/Pixi Installation

For conda users (recommended for complex scientific Python environments):

```bash
# Using conda
conda install -c conda-forge czitools

# Using pixi (modern conda alternative)
pixi add czitools
```

For more details see the [Installation docs](https://sebi06.github.io/czitools/install/).

## Quick Start

```python
from czitools.metadata_tools.czi_metadata import CziMetadata
from czitools.read_tools import read_tools

# read all metadata
mdata = CziMetadata("path/to/file.czi")

# read pixel data as a labelled STCZYX(A) array
array6d, mdata = read_tools.read_6darray("path/to/file.czi", use_dask=True, use_xarray=True)
```

For detailed usage examples see the [Usage docs](https://sebi06.github.io/czitools/usage/).

## Features

### Analysis Tools

The `analysis_tools` package provides image processing and HCS plate analysis utilities:

```python
from czitools.analysis_tools import ArrayProcessor, process_hcs_omezarr, create_well_plate_heatmap

# Process 2D images with filters and object detection
proc = ArrayProcessor(image_2d)
filtered = proc.apply_gaussian_filter(sigma=2).apply_threshold(value=100)
labelled, count, props = proc.label_objects(min_size=50, measure_params=True)

# Analyze HCS OME-ZARR plates
results = process_hcs_omezarr("plate.ome.zarr", channel2analyze=0)

# Visualize results as heatmap
fig = create_well_plate_heatmap(results, num_rows=8, num_cols=12)
```

**Requires:** `pip install czitools[analysis]`

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
| Show planetable as surface | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/show_czi_surface.ipynb)             |
| Segment with Voronoi-Otsu  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/read_czi_segment_voroni_otsu.ipynb) |
