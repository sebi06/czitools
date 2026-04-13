# Usage

!!! warning "Work in Progress"
    This documentation is still incomplete and actively being updated.
    Some sections may be missing or subject to change.

## Reading Metadata

All metadata can be retrieved at once via `CziMetadata`, or selectively using the individual classes:

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

filepath = "path/to/file.czi"

# Get all metadata at once
mdata = CziMetadata(filepath)

# Or get only specific metadata
czi_dimensions = CziDimensions(filepath)
print("SizeS:", czi_dimensions.SizeS)
print("SizeT:", czi_dimensions.SizeT)
print("SizeZ:", czi_dimensions.SizeZ)
print("SizeC:", czi_dimensions.SizeC)
print("SizeY:", czi_dimensions.SizeY)
print("SizeX:", czi_dimensions.SizeX)

# Write the CZI XML metadata to a file
xmlfile = writexml(filepath)

# Channel information
czi_channels = CziChannelInfo(filepath)

# Scaling (values in microns)
czi_scale = CziScaling(filepath)

# Objectives, detectors, microscope
czi_objectives = CziObjectives(filepath)
czi_detectors = CziDetector(filepath)
czi_microscope = CziMicroscope(filepath)

# Sample carrier info
czi_sample = CziSampleInfo(filepath)

# Additional metadata
czi_addmd = CziAddMetaData(filepath)

# Bounding box information
czi_bbox = CziBoundingBox(filepath)
```

### Using Box for Attribute-Style Access

```python
from czitools.utils.box import get_czimd_box

czi_box = get_czimd_box(filepath)
scaling = czi_box.ImageDocument.Metadata.Scaling.Items.Distance
```

## Reading Pixel Data

### `read_6darray` — Full 6D Stack

Returns the entire image as a single array with dimension order **STCZYX(A)**:

```python
from czitools.read_tools import read_tools

# NumPy array
array6d, mdata = read_tools.read_6darray(filepath)

# Dask (lazy) array — recommended for large files
array6d, mdata = read_tools.read_6darray(filepath, use_dask=True)

# xarray with labelled dimensions
array6d, mdata = read_tools.read_6darray(filepath, use_dask=True, use_xarray=True)

# Downscale to 50 %
array6d, mdata = read_tools.read_6darray(filepath, zoom=0.5)
```

### `read_stacks` — Scene-Wise Reading

`read_stacks` reads the image scene by scene and optionally stacks the results:

```python
result, dims, num_stacks, mdata = read_tools.read_stacks(
    filepath,
    use_dask=True,
    use_xarray=True,
    stack_scenes=True,   # attempt to stack all scenes into one array
)
```

Return behaviour:

| `stack_scenes` | Scenes compatible? | Return type                         |
| -------------- | ------------------ | ----------------------------------- |
| `False`        | —                  | `list` (one array per scene)        |
| `True`         | Yes                | Single stacked array (with `S` dim) |
| `True`         | No                 | `list` (with warning)               |

For strict return contracts:

```python
# Always returns a list
result_list, dims, n, mdata = read_tools.read_stacks_list(filepath, ...)

# Raises ValueError if scenes cannot be stacked
stacked, dims, n, mdata = read_tools.read_stacks_stacked(filepath, ...)
```

## Displaying in Napari

### Single Array

```python
from czitools.utils.napari_helpers import display_xarray_in_napari

subset_planes = array6d.attrs.get("subset_planes", {})
display_xarray_in_napari(array6d, mdata, subset_planes)
```

### List of Scene Stacks

```python
from czitools.utils.napari_helpers import display_xarray_list_in_napari

display_xarray_list_in_napari(result_list, mdata)
```

To display only one scene from a list:

```python
idx = 0
subset_planes = result_list[idx].attrs.get("subset_planes", {})
display_xarray_in_napari(result_list[idx], mdata, subset_planes)
```

### Recommended Parameters Helper

```python
from czitools.utils.napari_helpers import _get_recommended_read_params

params = _get_recommended_read_params()
array6d, mdata = read_tools.read_6darray(filepath, **params)
```

## NDV Viewer Integration

```python
from czitools.metadata_tools.czi_metadata import CziMetadata
from czitools.utils.ndv_tools import _create_luts_ndv, _create_scales_ndv

mdata = CziMetadata(filepath)
luts = _create_luts_ndv(mdata)
scales = _create_scales_ndv(mdata)
```

## Array Dimension Order

CZI arrays always follow the dimension order **STCZYX(A)**:

| Dim | Meaning                          |
| --- | -------------------------------- |
| S   | Scene                            |
| T   | Time                             |
| C   | Channel                          |
| Z   | Z-slice                          |
| Y   | Y (height)                       |
| X   | X (width)                        |
| A   | Alpha / RGB component (optional) |

## Colab Notebooks

| Topic                      | Link                                                                                                                                                                                               |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Read CZI metadata          | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/read_czi_metadata.ipynb)            |
| Read CZI pixel data        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/read_czi_pixeldata.ipynb)           |
| Write OME-ZARR from CZI    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/omezarr_from_czi_5d.ipynb)          |
| Save with ZSTD compression | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/save_with_ZSTD_compression.ipynb)   |
| Show planetable as surface | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/show_czi_surface.ipynb)             |
| Segment with Voronoi-Otsu  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/read_czi_segment_voroni_otsu.ipynb) |
