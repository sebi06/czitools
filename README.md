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
print(mdata.scene_shape_is_consistent)  # True when all scenes can be stacked

# scene_shape_tolerance (default=1) controls the maximum allowed pixel
# difference in width or height between scenes before they are considered
# inconsistent. A value of 1 absorbs the ±1-pixel rounding that commonly
# occurs with HCS plate coordinates and allows those scenes to be stacked.
mdata_plate = CziMetadata("path/to/plate.czi", scene_shape_tolerance=1)

# Read regular, equal-sized scenes eagerly as a labelled STCZYX(A) array.
array6d, mdata = read_6darray("path/to/file.czi", use_xarray=True)

# For HCS plate files whose scenes differ by ±1 pixel due to coordinate
# rounding, pass scene_stack_tolerance=1 so read_stacks crops them to a
# common shape and stacks them into one array (default=0, strict equality).
from czitools.read_tools import read_stacks
stacked, dims, n, mdata = read_stacks(
    "path/to/plate.czi",
    use_dask=True,
    use_xarray=True,
    stack_scenes=True,
    scene_stack_tolerance=1,
)

# For irregular scenes, keep genuinely lazy reads as a list.
scenes, dims, scene_count, mdata = read_stacks_list(
    "path/to/file.czi",
    use_dask=True,
    use_xarray=True,
)
first_plane = scenes[0].isel(T=0, C=0, Z=0).compute()
```

`read_6darray(..., use_dask=True)` also provides genuinely lazy pixel access
when the CZI has equal-sized scenes and consistent pixel types.
Both eager and lazy reads use each scene's full-resolution, non-pyramid
bounding rectangle, so rounded pyramid coverage cannot pad or change the
regular STCZYX(A) shape.
`read_stacks(..., use_dask=True)` groups up to 64 planes per task by default to
reduce file-open and scheduler overhead. Set `lazy_read_strategy="plane"` for
the finest-grained random access, or tune the group with `planes_per_chunk`.

For gigapixel CZIs (single planes over ~256 MB uncompressed) `read_stacks`
automatically activates **spatial Y/X tiling**: each dask chunk becomes one
ROI-based read via `pylibCZIrw`, so viewers such as napari only load the tiles
that intersect the visible viewport instead of full planes. Tune the tile edge
with `tile_size` (default 4096) or the trigger threshold with
`chunk_memory_limit`. Small planes keep the fast whole-plane path.

For interactive viewers that need a **multiscale pyramid** (napari's
`add_image(..., multiscale=True)`, gigapixel whole-slide display, etc.) use
`read_stacks_multiscale`:

```python
from czitools.read_tools import get_pyramid_zooms, read_stacks_multiscale

# Inspect the stored pyramid without reading pixels.
print(get_pyramid_zooms("path/to/large.czi"))
# -> [1.0, 0.5, 0.25, 0.125, 0.0625]   (standard 2x pyramid)

# One lazy dask array per level, ready for napari.
levels, infos, dims, num_stacks, mdata = read_stacks_multiscale(
    "path/to/large.czi",
    max_coarse_edge=8192,   # force coarser synthetic levels if needed
)
```

Levels detected on disk are served directly from their subblocks (no
resampling). If the coarsest stored level is still larger than
`max_coarse_edge` on any edge, additional coarser levels are synthesized
via libCZI's C++ downsampler so the top of the pyramid always fits in one
GPU texture.

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

### HCS Plate Inspection CLI

Quickly inspect CZI well-plate metadata from the command line:

```bash
# Inspect entire plate (all wells and fields)
python -m czitools.demo.scripts.czi_hcs_check -f plate.czi

# Inspect a specific well
python -m czitools.demo.scripts.czi_hcs_check -f plate.czi --well B4

# Hide the well summary table (useful for large plates)
python -m czitools.demo.scripts.czi_hcs_check -f plate.czi --no-well-table

# Get help
python -m czitools.demo.scripts.czi_hcs_check --help
```

Or use the utility functions in Python:

```python
from czitools.utils import print_hcs_plate_info, print_sample_metadata, print_well_fields
from czitools.metadata_tools import CziMetadata

mdata = CziMetadata("plate.czi")

# Print plate hierarchy with well summary
print_hcs_plate_info(mdata)

# Print sample metadata and scene details
print_sample_metadata(mdata)

# Print field information for a specific well
print_well_fields(mdata, well_name="B4")
```

### OME-Zarr Converter GUI

The experimental converter GUI exports individual CZI images and HCS plates
using either `ome-zarr-py` or `ngff-zarr`. All conversions use Zarr v3. The GUI
provides controls for compression, supported single-file `.ozx` workflows,
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

![5D CZI inside NDV](https://github.com/sebi06/czitools/raw/main/_images/ndv.png)

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
