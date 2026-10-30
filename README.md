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

## 0.27.2 Highlights

- Both export backends default to OME-NGFF 0.6 for individual images and
    HCS plates, including native CZI pyramids and ngff-zarr `.ozx` output.
- Low-level ome-zarr-py metadata is converted to genuine 0.6 metadata,
    preserving channels, physical scaling, and pyramid translations.
- Explicit version profiles and axis checks reject unsupported exports before
    overwriting output. Validation rejects mixed-version hierarchies and
    incomplete channel display metadata; draft 0.9 output remains disabled.
- OZX packaging preserves the source store's declared NGFF version instead
    of silently downgrading to 0.5. Existing stores are not migrated.

## Installation

Supported Python versions: **3.12, 3.13, and 3.14**.

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
print(mdata.image_required.SizeS)
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

Image dimension sizes (`SizeS`, `SizeT`, `SizeC`, `SizeZ`, `SizeY`, `SizeX`,
and optional dimensions) describe the full-resolution data physically stored
in CZI subblocks. They do not use numeric XML `Size*` declarations. This is
important for split acquisitions, where every child file can retain XML for
the complete acquisition while storing only a subset of the global scene
indices. Use `mdata.stored_scene_indices` to inspect those physical `S` keys.

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

Both CZI export backends default to OME-NGFF v0.6 for individual images and
HCS plates, including ngff-zarr `.ozx` output. Exports contain named coordinate
systems, explicit transform references, physical scaling, and channel display
metadata. Low-level ome-zarr-py metadata is converted using ngff-zarr's v0.6
model after pixel writing. New exports do not silently fall back to v0.5;
older versions remain explicit Python compatibility options.

OME-NGFF v0.6 output requires `ngff-zarr>=0.47.0`. Validation selects the image
or HCS reader from the layout and declared version, with validation enabled.
Packaging an existing directory as `.ozx` preserves its NGFF version rather
than relabeling it. Existing stores are unchanged; reconvert a CZI to obtain
v0.6 metadata.

NGFF compatibility is controlled by explicit internal version profiles, not
the installed library's latest-version setting. Single-image writers require
TCZYX data; named arrays in another order must be explicitly transposed before
export. Unsupported versions and axis layouts fail before replacing output.
Validation and OZX packaging reject mixed-version hierarchies. Channel colors
and complete display windows remain required when channel metadata is present.

The exporter is prepared for future axis-model extensions, but does not enable
`0.9.dev1` or `0.9.dev2` output. Draft support requires an exact schema snapshot
and dedicated round-trip and consumer compatibility tests. A future draft mode
must be opt-in, use separate output stores, and retain v0.6 as the stable default;
changing a version tag alone is not a conversion.

All four CZI export APIs accept `preserve_original_metadata=True` (default:
`False`). This stores the original CZI XML at `CZI/metadata.xml`, referenced
from root `czitools.originalMetadata` attributes, including in `.ozx` output.
The converter GUI offers **Preserve original CZI XML**; conversion demo
scripts and the benchmark accept `--preserve-original-metadata`.
This is a czitools convention, not an NGFF-standard XML location.
`OME/METADATA.ome.xml` remains reserved for genuine OME-XML.
Preserving the XML also retains source stage positions, acquisition timing,
and axis information when present, for future standardized mappings. It does
not invent calibration or automatically map CZI scenes to NGFF spatial scenes.

For HCS plates, `CZI` is an auxiliary group beside the row groups, not a row
or well. Readers should follow the declared `ome.plate.wells` paths rather
than treat every root folder as a row. The custom
`czitools.originalMetadata` reference locates the XML; it does not define or
change plate membership.
See [usage details](https://sebi06.github.io/czitools/usage/#preserving-original-czi-xml).

### HCS Plate Inspection CLI

Use `czi_hcs_check` to inspect CZI well-plate metadata from the command line
with colorized output. The inspector displays all
full-resolution, subblock-derived dimension sizes and, by default, limits the
HCS hierarchy to fields physically stored in the selected file:

```bash
# Inspect dimensions and HCS fields physically stored in this file
czi_hcs_check -f plate.czi

# Inspect a specific well
czi_hcs_check -f plate.czi --well B4

# Omit the well summary table while retaining plate, sample, and field details
czi_hcs_check -f plate.czi --no-well-table

# Show the complete acquisition model declared by the CZI XML
czi_hcs_check -f plate.czi --show-declared

# Get help
czi_hcs_check --help
```

Use `--no-well-table` for large plates when the per-well summary would make
the terminal output unnecessarily long. `--show-declared` is useful for split
acquisitions when you want the planned plate layout rather than only the
payload of the current file.

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
using either `ome-zarr-py` or `ngff-zarr`. All conversions default to OME-NGFF
v0.6 with Zarr v3 storage. The GUI
loads the selected file's metadata automatically and enables **Write HCS
layout** when a plate layout is detected. For multi-scene, non-HCS files, the
**Scene ID** selector appears directly below the HCS option. A single **Create
Single-File OME-ZARR (.ozx)** checkbox selects archive output for both HCS and
non-HCS conversions. Additional controls configure compression and optional
napari viewing. The `ngff-zarr` backend uses zarrista automatically. The
metadata preview shows the detected dimensions and scenes before conversion,
while the log panel reports progress.

For individual images, **Spatial chunk size (pixels)** selects **256, 512,
or 1024**, with **512** as the default. The choice applies to both backends;
larger chunks reduce Dask
task overhead; 256 can be useful when smaller storage chunks are required.

**Reuse CZI pyramid** is enabled by default for individual images with both
`ngff-zarr` and `ome-zarr-py`. HCS conversion also reuses stored levels with
both backends. Native levels, including ZEN's approximately 3x pyramid
steps, are exported without generating replacement 2x levels. Their pixels,
dimensions, physical scales, and translations are preserved. Only a missing
coarse tail is added through the lazy CZI reader, targeting a longest edge
of 512 pixels within libCZI's supported zoom range. Files without stored
lower-resolution levels use that reader's synthetic pyramid fallback.
Reuse still requires decoding CZI data and encoding the destination arrays;
it is not a compressed-byte copy.

To regenerate an individual image's pyramid, uncheck **Reuse CZI pyramid**.
**Pyramid levels** then selects the number of additional 2x Y/X levels.
The `ngff-zarr` quality and fast/balanced presets select Gaussian and
bin-shrink downsampling, respectively. These regeneration controls are
disabled while reuse is enabled; compression remains independently selectable.
Python callers of `write_omezarr` and `write_omezarr_ngff` can supply lazy
levels through `source_multiscales`; raw-array calls retain their existing
generated-pyramid behavior.

Individual scenes are read lazily with spatial CZI ROI tiles instead of
allocating entire channel planes. CZI read tiles are up to **4096 pixels**
with a **32 MiB per-tile budget**, independent of the selected output chunk
size. Larger read tiles reduce CZI file opens and ROI task overhead; the
reader reduces tile dimensions when needed to stay within the budget.
For individual images, **Dask workers** defaults to **Auto**. At conversion
start, Auto reads currently available RAM, reserves the larger of **2 GiB**
or **20% of total RAM**, and budgets **1 GiB per worker**. It selects at most
half the logical CPU count and caps automatic concurrency at **8 workers**,
with a minimum of one. When RAM metrics are unavailable, it uses at most two.
The selected count and memory budget appear in the conversion log.
**Selected Dask workers** also shows the resolved count beside the worker
control during conversion, for example **8 (Auto)**. This is the selected
pool size, not an instantaneous count of busy tasks. Manual selections are
shown with **(manual)**, and changing files clears the previous status.

Manual values from **1 to 32** override Auto; the Python writers accept
`worker_count=0` for Auto or a positive count for a manual override. Higher
counts can overlap CZI reads, compression, and writes, but increase active
memory use and disk contention. This is a conservative starting heuristic,
not a hard memory limit or a throughput guarantee, and does not resize the
pool during a conversion. HCS retains its existing field-concurrency policy.

For `ngff-zarr` directory output, the base image is written directly to the
final store. Reused CZI levels are read and appended directly. When regeneration
is selected, each next level is built from the preceding written level and
appended to that store, avoiding an extra intermediate pyramid copy and
source-dependent graphs spanning the entire pyramid. Single-file `.ozx`
output writes supplied CZI levels directly; regenerated pyramids retain
disk staging before final archive packaging.
Export requires `ngff-zarr>=0.47.0` for incremental directory writes.

The terminal and GUI log show a progress bar for each directly written
directory level. Archive output shows base-image caching, each pyramid-level
cache when regenerating, and final output writing. Task percentages refer to the current
region, not the entire conversion. Initial graph preparation may show
elapsed time before task counts become available.

Directory output does not require an intermediate cache. For regenerated
`.ozx` pyramids, temporary caches are created beside the output and removed on success or
failure. Allow additional free disk space for an intermediate pyramid
(approximately 1.33 times the uncompressed base image for a 2x Y/X pyramid,
before compression), as well as the final output. `.ozx` packaging may need
additional temporary space. These limits bound active pixel processing, not
all Python/scheduler memory; very large datasets still need sufficient RAM
for task metadata. HCS conversion uses stored levels with both backends and
adds only the missing coarse tail; the chunk selector applies to individual images.

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
