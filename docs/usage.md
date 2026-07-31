# Usage

!!! warning "Work in Progress"
    This documentation is still incomplete and actively being updated.
    Some sections may be missing or subject to change.

The examples below assume that `filepath` points to a local CZI:

```python
from pathlib import Path

filepath = Path("data/WP96_4Pos_B4-10_DAPI.czi")
```



For most workflows, create one `CziMetadata` object and use its grouped
properties. The metadata object returned by a pixel reader is the same type, so
there is no need to read it twice:

```python
from czitools.metadata_tools import CziMetadata

mdata = CziMetadata(filepath)
image = mdata.image_required
scale = mdata.scale_required
channels = mdata.channelinfo_required

print(f"S/T/C/Z: {image.SizeS}/{image.SizeT}/{image.SizeC}/{image.SizeZ}")
print(f"Y/X: {image.SizeY}/{image.SizeX}")
print(f"physical pixel size X/Y/Z: {scale.X}/{scale.Y}/{scale.Z}")
print(mdata.pixeltypes)
print(mdata.scene_shape_is_consistent)
```

The `*_required` properties return a non-optional value or raise a clear error.
Frequently used groups are:

| Property                              | Contents                                     |
| ------------------------------------- | -------------------------------------------- |
| `image`                               | Dimension sizes                              |
| `bbox`                                | Total and per-scene bounding boxes           |
| `channelinfo`                         | Channel acquisition and display information  |
| `scale`                               | Physical scaling                             |
| `objective`, `detector`, `microscope` | Instrument information                       |
| `sample`                              | Per-scene sample and well information        |
| `attachments`                         | CZI attachment availability                  |
| `hcs`, `hcs_status`                   | Validated HCS hierarchy and detection result |

Metadata can also be read selectively using the individual classes:

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

Choose the reader according to the regularity of the data and whether loading
must be genuinely lazy:

| Requirement                               | Recommended function          | Result                        |
| ----------------------------------------- | ----------------------------- | ----------------------------- |
| Equal-sized scenes; eager read            | `read_6darray`                | One `STCZYX(A)` array         |
| Regular array wrapped as Dask             | `read_6darray(use_dask=True)` | Dask-backed, but eagerly read |
| True on-demand reads                      | `read_stacks(use_dask=True)`  | Per-scene arrays by default   |
| True lazy reads with equal scenes stacked | `read_stacks_stacked`         | One array with `S`            |
| Scenes that may differ in shape           | `read_stacks_list`            | Stable list of scene arrays   |
| A known HCS well or field                 | `read_well` / `read_field`    | HCS-aware field arrays        |

### `read_6darray` — Full 6D Stack

Returns the image as a single array with dimension order **STCZYX(A)**.
It requires equal-sized scenes and consistent pixel types:

```python
from czitools.read_tools import read_6darray

# NumPy array
array6d, mdata = read_6darray(filepath)

# Dask-backed array. The CZI data is still read eagerly.
array6d, mdata = read_6darray(filepath, use_dask=True)

# xarray with labelled dimensions
array6d, mdata = read_6darray(filepath, use_xarray=True)

# Downscale to 50 %
array6d, mdata = read_6darray(filepath, zoom=0.5)

# Read zero-based inclusive S/T/C/Z ranges
subset, mdata = read_6darray(
    filepath,
    planes={"S": (0, 0), "T": (0, 1), "C": (0, 0), "Z": (0, 4)},
    adapt_metadata=True,
)
```

!!! important "Dask-backed is not necessarily lazy"
    `read_6darray(..., use_dask=True)` wraps the eagerly read result in a Dask
    array. For true on-demand CZI reads, use
    `read_stacks(..., use_dask=True)`.

### `read_stacks` — Scene-Wise Reading

`read_stacks` supports all CZI dimensions and optionally stacks compatible
scenes. With `use_dask=True`, pixel planes are read only when indexed or
computed:

```python
from czitools.read_tools import read_stacks

result, dims, num_stacks, mdata = read_stacks(
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
from czitools.read_tools import read_stacks_list, read_stacks_stacked

# Always returns a list
result_list, dims, n, mdata = read_stacks_list(
    filepath,
    use_dask=True,
)

# Raises ValueError if scenes cannot be stacked
stacked, dims, n, mdata = read_stacks_stacked(
    filepath,
    use_dask=True,
)
```

`read_stacks_list` is the safest interface for files whose scenes may have
different shapes. Call `.compute()` on a Dask-backed selection when its pixels
are needed.

For example, this reads only one selected plane:

```python
scenes, dims, scene_count, mdata = read_stacks_list(
    filepath,
    use_dask=True,
    use_xarray=True,
    planes={"T": (0, 0), "C": (0, 0)},
)

first_plane = scenes[0].isel(T=0, C=0, Z=0).compute()
```

## Reading Well-Plate Metadata

`CziMetadata` provides both an HCS detection result and, when detection is
successful, an immutable `Plate -> Well -> Field` hierarchy:

```python
from czitools.metadata_tools import CziMetadata

mdata = CziMetadata(filepath)
print(mdata.hcs_status.detected)
print(mdata.hcs_status.reason)

if mdata.hcs is not None:
    plate = mdata.hcs
    print(plate.id, plate.name, plate.schema_version)
    print(plate.declared_rows, plate.declared_columns)
    print(plate.observed_row_indices)       # normalized, zero-based
    print(plate.observed_column_indices)    # normalized, zero-based

    for well in plate.wells:
        print(
            well.canonical_name,
            well.canonical_path,
            well.row_index,
            well.column_index,
            len(well.fields),
        )

    # Capitalization, zero padding, and path notation are normalized.
    well = plate.get_well("b04")
    for field in well.fields:
        print(
            field.field_index,       # zero-based within the well
            field.scene_index,       # global CZI scene
            field.region_id,
            field.scene_center_x,
            field.scene_center_y,
        )
```

The model retains the original CZI well indices as well as normalized
zero-based indices. Field indices are local to a well; scene indices are global
to the CZI.

Resolver functions map selectors to the model without reading pixels:

```python
from czitools.metadata_tools.hcs import resolve_field, resolve_well

well = resolve_well(plate, "B/04")
field = resolve_field(plate, "B/04", 0)

# A source RegionId string can select the same field.
if field.region_id is not None:
    same_field = resolve_field(plate, "B04", field.region_id)
```

The compatibility-oriented `mdata.sample` object exposes per-scene
collections. Prefer `sample.field_centerX` and `sample.field_centerY`: they
preserve valid `0.0` coordinates and use `None` for missing positions. The
deprecated `scene_stageX` and `scene_stageY` properties convert missing values
to `0.0`.

HCS detection is an additional interpretation. If `mdata.hcs` is `None`,
general metadata and ordinary CZI pixel reading remain available.

### Optional stage-position enrichment

Scene-center positions come from scene XML. Subblock stage/focus coordinates
can be added explicitly from the planetable for a local CZI:

```python
enriched_plate = mdata.enrich_hcs_positions(position_tolerance=1.0)

if enriched_plate is not None:
    well = enriched_plate.get_well("B04")
    for field in well.fields:
        print(
            field.stage_x,
            field.stage_y,
            field.acquisition_z,
            field.position_conflict,
        )
```

Enrichment returns a new immutable plate and updates `mdata.hcs`. It keeps
scene-center and subblock-stage coordinates separate and is unavailable for URL
sources.

Position helpers expose well-relative scene-center offsets and absolute
coordinates. They return `None` if any required coordinate is missing:

```python
from czitools.metadata_tools.hcs import (
    well_absolute_field_positions,
    well_relative_field_positions,
)

if enriched_plate is None:
    raise ValueError("Stage positions are unavailable.")

well = enriched_plate.get_well("B04")
relative = well_relative_field_positions(well)
scene_centers = well_absolute_field_positions(well, source="scene_center")
stage_positions = well_absolute_field_positions(well, source="stage")
```

## Reading by Well / Field (HCS Plates)

For high-content-screening plates, wells and fields can be read directly by name
without tracking scene indices. These reads use the canonical HCS model
(`CziMetadata.hcs`) and reuse the single-scene read path.

```python
from czitools.read_tools import read_field, read_well

# Read a single field of a well (well names accept "B4", "b04" or "B/4").
# `field` is the well-local 0-based index, or a source-scoped RegionId string.
array, mdata = read_field(filepath, well="B4", field=0)

# Read all fields of a well as a list of per-field arrays (shapes may differ).
arrays, mdata = read_well(filepath, well="B4")

# Stack the fields along the S axis (requires identical field shapes).
stacked, mdata = read_well(filepath, well="B4", stack=True)
```

If the CZI has no usable HCS plate metadata, both functions raise a `ValueError`
explaining why (from `CziMetadata.hcs_status.reason`).

These functions reuse `read_6darray`. Their `use_dask=True` results are
Dask-backed, but are not true on-demand reads from the CZI.

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
from czitools.read_tools import read_6darray

params = _get_recommended_read_params()
array6d, mdata = read_6darray(filepath, **params)
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

`read_6darray`, `read_field`, and `read_well` use **STCZYX(A)**:

| Dim | Meaning                         |
| --- | ------------------------------- |
| S   | Scene                           |
| T   | Time                            |
| C   | Channel                         |
| Z   | Z-slice                         |
| Y   | Y (height)                      |
| X   | X (width)                       |
| A   | RGB sample/component (optional) |

`read_stacks` tracks `S` separately unless scenes are stacked. Optional CZI
dimensions `V`, `R`, `I`, `H`, and `M` precede the always-present core
dimensions `T`, `C`, and `Z`; spatial dimensions and optional `A` follow.

## Exporting to OME-Zarr

!!! note "Requires the `omezarr` extra"
    OME-Zarr export lives in `czitools.export_tools` and needs the optional
    dependencies: `pip install "czitools[omezarr]"` (or `"czitools[omezarr-gui]"`
    for the GUI). See the [Installation docs](install.md).

### HCS plate export

The converter writes the logical hierarchy plate → row/column well → field
image → multiscale level. It resolves the layout from `CziMetadata.hcs` and
uses complete, unambiguous sample metadata only as a fallback. Fields are
written individually, so fields with different shapes are supported.

```python
from pathlib import Path

from czitools.export_tools import (
    convert_czi2hcs_ngff,      # ngff-zarr backend, OME-NGFF v0.5
    convert_czi2hcs_omezarr,   # ome-zarr-py backend, Zarr v3 by default
    validate_ome_zarr,
)

# Write an HCS plate (rows/wells/fields) with the ngff-zarr backend.
out = convert_czi2hcs_ngff(
    "path/to/plate.czi",
    output_dir=Path("exports"),
    overwrite=True,
    pad_columns=True,
)

# Validate the result against the OME-NGFF v0.5 schema.
assert validate_ome_zarr(out)
```

The ome-zarr-py backend can write its default Zarr v3 store or a legacy
OME-NGFF v0.4/Zarr v2 store:

```python
# Use zarr_format=2 only when a legacy reader requires Zarr v2.
legacy_out = convert_czi2hcs_omezarr(
    "path/to/plate.czi",
    overwrite=True,
    zarr_format=2,
)
assert validate_ome_zarr(legacy_out)
```

Both backends route through a canonical layout resolver that prefers the Stage 1
HCS model (`CziMetadata.hcs`) and falls back to `CziSampleInfo` only when it is
complete and unambiguous. Sparse plates and variable field counts per well are
supported.

`pad_columns=True` produces well paths such as `B/04`; `False` produces `B/4`.
The NGFF backend accepts `output_dir` and can write a single `.ozx` archive
with `write_ozx_directly=True`. The ome-zarr-py HCS converter currently writes
beside the source CZI.

The repository also includes a command-line example:

```bash
python demo/scripts/omezarr_convert_hcs.py path/to/plate.czi
python demo/scripts/omezarr_convert_hcs.py path/to/plate.czi --backend omezarr
python demo/scripts/omezarr_convert_hcs.py path/to/plate.czi --no-pad-columns
```

Validation should normally remain enabled. Use `overwrite=False` when an
existing export must be preserved.

### Single-image export

```python
from czitools.read_tools import read_6darray
from czitools.export_tools import write_omezarr_ngff, write_omezarr

array, mdata = read_6darray("image.czi", planes={"S": (0, 0)}, use_xarray=True)
array = array.squeeze("S")  # 6D -> 5D (T, C, Z, Y, X)

# ngff-zarr backend (multi-scale pyramid, OME-NGFF v0.5)
write_omezarr_ngff(array, "image_ngff.ome.zarr", mdata, scale_factors=[2, 4], overwrite=True)

# ome-zarr-py backend (Zarr v3 by default; pass zarr_format=2 for legacy v2)
write_omezarr(array, zarr_path="image.ome.zarr", metadata=mdata, overwrite=True)
```

### Converter GUI

Install the GUI extra (`pip install "czitools[omezarr-gui]"`) and launch it via the
console script or the Python API:

```bash
czitools-omezarr-gui
```

```python
# Launch standalone
from czitools.export_tools import run_gui
run_gui()

# Or embed the widget in napari
from czitools.export_tools import create_gui
viewer.window.add_dock_widget(create_gui(), name="CZI Converter")
```

## End-to-End HCS Example

This workflow detects a plate, inspects one well, reads a small field subset,
converts the complete plate, and validates the output:

```python
from pathlib import Path

from czitools.export_tools import convert_czi2hcs_ngff, validate_ome_zarr
from czitools.metadata_tools import CziMetadata
from czitools.read_tools import read_field

czi_path = Path("data/WP96_4Pos_B4-10_DAPI.czi")

# Metadata access does not load the image pixels.
mdata = CziMetadata(czi_path)
if mdata.hcs is None:
    raise ValueError(f"Not an unambiguous HCS plate: {mdata.hcs_status.reason}")

well = mdata.hcs.get_well("B04")
print(mdata.hcs.name, well.canonical_name, len(well.fields))

# Read one T/C/Z plane from the first field.
field, _ = read_field(
    czi_path,
    well="B04",
    field=0,
    planes={"T": (0, 0), "C": (0, 0), "Z": (0, 0)},
    use_xarray=True,
)
if field is None:
    raise ValueError("The selected field could not be read.")
print(field.shape)

# Convert and validate the complete plate.
output = convert_czi2hcs_ngff(
    czi_path,
    output_dir=Path("exports"),
    overwrite=True,
)
if not validate_ome_zarr(output):
    raise RuntimeError(f"Validation failed: {output}")

print(f"Wrote {output}")
```

## Practical Guidance

- Start with `CziMetadata` when deciding how to process an unfamiliar CZI.
- Use `read_6darray` for regular data that comfortably fits in memory.
- Use `read_stacks_list(..., use_dask=True)` for genuinely lazy access and
  unequal scene shapes.
- Prefer the HCS model over manually correlating scenes and well names.
- Preserve missing positions as `None`; do not reinterpret them as the origin.
- Keep scene-center and stage positions as separate coordinate sources.
- Validate every OME-Zarr HCS export before downstream analysis or publication.

## Colab Notebooks

| Topic                      | Link                                                                                                                                                                                               |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Read CZI metadata          | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/read_czi_metadata.ipynb)            |
| Read CZI pixel data        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/read_czi_pixeldata.ipynb)           |
| Read CZI well-plate data   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/read_czi_wellplate_data.ipynb)      |
| Process OME-Zarr HCS plate | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/process_omezarr_HCS_plate.ipynb)    |
| Write OME-ZARR from CZI    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/omezarr_from_czi_5d.ipynb)          |
| Save with ZSTD compression | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/save_with_ZSTD_compression.ipynb)   |
| Show planetable as surface | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/show_czi_surface.ipynb)             |
| Segment with Voronoi-Otsu  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/read_czi_segment_voroni_otsu.ipynb) |
