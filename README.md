# czitools

[![PyPI](https://img.shields.io/pypi/v/czitools.svg?color=green)](https://pypi.org/project/czitools)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/czitools)](https://pypistats.org/packages/czitools)
[![License](https://img.shields.io/pypi/l/czitools.svg?color=green)](https://github.com/sebi06/czitools/raw/main/LICENSE)
[![codecov](https://codecov.io/github/sebi06/czitools/graph/badge.svg?token=WK1KIMZARL)](https://codecov.io/github/sebi06/czitools)
[![Python Version](https://img.shields.io/pypi/pyversions/czitools.svg?color=green)](https://python.org)
[![Development Status](https://img.shields.io/pypi/status/czitools.svg)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Beta)

This repository provides tools for reading CZI (Carl Zeiss Image) pixel data
and metadata in Python, interpreting CZI well plates as an HCS
Plate → Well → Field model, and converting CZI data to OME-Zarr. It is
available as a [Python package on PyPI](https://pypi.org/project/czitools/).

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
# OME-Zarr export (conversion + validation)
pip install "czitools[omezarr]"

# OME-Zarr export with GUI converter application
pip install "czitools[omezarr-gui]"

# HCS plate analysis and visualization
pip install "czitools[analysis]"

# Everything (all optional dependencies)
pip install "czitools[all]"
```

### Development Installation

For development or to get the latest unreleased features:

```bash
# Clone the repository
git clone https://github.com/sebi06/czitools.git
cd czitools

# Install in editable mode with all extras
pip install -e ".[all]"
```

### Conda/Pixi Development Environment

The cloned repository includes both a conda environment file and a Pixi
workspace:

```bash
# Create the provided conda environment
conda env create -f env_czitools.yml
conda activate czitools
python -m pip install -e ".[all]"

# Or install the locked Pixi workspace (Windows and Linux)
pixi install
```

For more details see the [Installation docs](https://sebi06.github.io/czitools/install/).

## Quick Start

```python
from czitools.metadata_tools import CziMetadata
from czitools.read_tools import read_6darray, read_stacks_list

# Read metadata without loading pixels.
mdata = CziMetadata("path/to/file.czi")
print(mdata.image_required.SizeC)
print(mdata.scale_required.X)

# Read regular, equal-sized scenes eagerly as a labelled STCZYX(A) array.
array6d, mdata = read_6darray("path/to/file.czi", use_xarray=True)

# For true on-demand Dask reads, keep scenes as a list.
scenes, dims, scene_count, mdata = read_stacks_list(
    "path/to/file.czi",
    use_dask=True,
    use_xarray=True,
)
first_plane = scenes[0].isel(T=0, C=0, Z=0).compute()
```

`read_6darray(..., use_dask=True)` produces a Dask-backed result but still
reads the CZI eagerly. Use `read_stacks(..., use_dask=True)` or
`read_stacks_list(..., use_dask=True)` for genuinely lazy pixel access.

For detailed usage examples see the [Usage docs](https://sebi06.github.io/czitools/usage/).

## Features

### CZI Well Plates and OME-Zarr HCS

```python
from czitools.export_tools import convert_czi2hcs_ngff, validate_ome_zarr
from czitools.metadata_tools import CziMetadata
from czitools.read_tools import read_field

filepath = "path/to/plate.czi"
mdata = CziMetadata(filepath)

if mdata.hcs is None:
    raise ValueError(mdata.hcs_status.reason)

well = mdata.hcs.get_well("B04")
field, _ = read_field(filepath, well="B04", field=0)

# Requires: pip install "czitools[omezarr]"
output = convert_czi2hcs_ngff(filepath, overwrite=True)
assert validate_ome_zarr(output)
```

Well names accept forms such as `B4`, `b04`, and `B/4`. Field indices are
zero-based within a well. The OME-Zarr converter writes the HCS hierarchy
plate → well → field image → multiscale level.

### OME-Zarr Converter GUI

The experimental converter GUI exports individual CZI images and HCS plates
using either `ome-zarr-py` or `ngff-zarr`. It provides controls for compression,
legacy OME-NGFF v0.4/Zarr v2 output, supported single-file `.ozx` workflows,
parallel I/O, and optional napari viewing. The metadata preview lets you verify
the detected dimensions and scenes before starting the conversion, while the
log panel shows its progress.

Install and launch it with:

```bash
pip install "czitools[omezarr-gui]"
czitools-omezarr-gui
```

From the repository's Pixi environment, use the equivalent task:

```bash
pixi run omezarr-gui
```

![CZI to OME-Zarr converter GUI](https://github.com/sebi06/czitools/raw/main/_images/czi_omezarr_gui.png)

See the [usage documentation](https://sebi06.github.io/czitools/usage/#ome-zarr-converter-gui)
for the workflow and Python/napari integration examples.

### Analysis Tools

The `analysis_tools` package provides image processing and HCS plate analysis utilities:

```python
from czitools.analysis_tools import ArrayProcessor, process_hcs_omezarr, create_well_plate_heatmap

# Process 2D images with filters and object detection
proc = ArrayProcessor(image_2d)
filtered = proc.apply_gaussian_filter(sigma=2)
binary = ArrayProcessor(filtered).apply_threshold(value=100)
labelled, count, props = ArrayProcessor(binary).label_objects(
    min_size=50,
    measure_params=True,
)

# Analyze HCS OME-Zarr plates
results = process_hcs_omezarr("plate.ome.zarr", channel2analyze=0)

# Visualize results as heatmap
fig = create_well_plate_heatmap(results, num_rows=8, num_cols=12)
```

**Requires:** `pip install "czitools[analysis]"`

**CZI inside NDV**

![5D CZI inside NDV](https://github.com/sebi06/czitools/raw/main/_images/czi_ndv1.png)

**CZI inside Napari**

![5D CZI inside Napari](https://github.com/sebi06/czitools/raw/main/_images/czi_napari2.png)

## Colab Notebooks

| Topic                      | Link                                                                                                                                                                                               |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| General usage czitools     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/czitools_usage_demo.ipynb)          |
| Read CZI metadata          | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/read_czi_metadata.ipynb)            |
| Read CZI pixel data        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/read_czi_pixeldata.ipynb)           |
| Read CZI well-plate data   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/read_czi_wellplate_data.ipynb)      |
| Process OME-Zarr HCS plate | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/process_omezarr_HCS_plate.ipynb)    |
| Show planetable as surface | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/show_czi_surface.ipynb)             |
| Segment with Voronoi-Otsu  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/read_czi_segment_voroni_otsu.ipynb) |

## Contributing

The Pixi workspace is the recommended development setup on Windows and Linux.
After cloning the repository, install the locked environment and run the local
quality checks:

```bash
pixi install
pixi run lint
pixi run test-no-net
```

Please keep changes focused, add or update tests for behavioral changes, and
open an issue before starting a large API or dependency change.
