# Reading CZI Data and Converting Well Plates to OME-Zarr HCS

This guide presents a consistent workflow for:

1. reading CZI pixel data, including eager and lazy/Dask options;
2. reading general CZI metadata;
3. reading and interpreting CZI well-plate metadata; and
4. converting a CZI well plate to an OME-Zarr HCS layout.

The examples use `pathlib.Path`, return labeled `xarray.DataArray` objects where
possible, and assume that `czi_path` points to a local CZI file:

```python
from pathlib import Path

czi_path = Path("data/WP96_4Pos_B4-10_DAPI.czi")
```

OME-Zarr conversion requires the optional export dependencies:

```bash
pip install "czitools[omezarr]"
```

## 1. Choosing a CZI pixel reader

The most important choice is whether the CZI can be represented as one regular
six-dimensional array and whether pixel loading must be genuinely lazy.

| Requirement | Recommended function | Result |
| --- | --- | --- |
| Regular, equal-sized scenes; eager read | `read_6darray` | One `STCZYX(A)` array |
| Regular array wrapped as Dask after reading | `read_6darray(use_dask=True)` | One Dask-backed array, but the CZI is still read eagerly |
| True lazy, on-demand reads | `read_stacks(use_dask=True)` | A list of per-scene arrays by default |
| True lazy reads with equal scenes stacked | `read_stacks_stacked(use_dask=True)` | One array with an `S` dimension |
| Scenes that may have different shapes | `read_stacks_list` | A stable list of per-scene arrays |
| A field or all fields from a known well | `read_field` / `read_well` | HCS-aware field arrays |

`A` is present only for RGB data. In `read_stacks`, optional CZI dimensions
`V`, `R`, `I`, `H`, and `M` precede the core `T`, `C`, and `Z` dimensions.

### 1.1 Eager reading as one regular array

`read_6darray` is convenient when all scenes have the same shape and the CZI
uses consistent pixel types:

```python
from czitools.read_tools import read_6darray

image, metadata = read_6darray(
    czi_path,
    use_xarray=True,
    use_dask=False,
)

if image is None:
    raise ValueError("The CZI cannot be represented as one regular 6D array.")

print(image.dims)   # typically: ("S", "T", "C", "Z", "Y", "X")
print(image.shape)
print(image.dtype)
```

Use labeled xarray selection to make dimension intent explicit:

```python
plane = image.isel(S=0, T=0, C=0, Z=0)
```

The `planes` argument limits the read. Ranges are zero-based and inclusive:

```python
subset, metadata = read_6darray(
    czi_path,
    planes={
        "S": (0, 0),  # first scene
        "T": (0, 1),  # time points 0 and 1
        "C": (0, 0),  # first channel
        "Z": (0, 4),  # Z planes 0 through 4
    },
    zoom=0.5,
    use_xarray=True,
    adapt_metadata=True,
)
```

`zoom` is an XY downscaling factor in the range `0.01` to `1.0`.
`adapt_metadata=True` updates the returned metadata dimensions to describe the
selected subset.

### 1.2 Dask-backed versus truly lazy reading

These two modes are deliberately different:

```python
# Dask-backed, but all CZI pixels are read before the function returns.
dask_wrapped, metadata = read_6darray(
    czi_path,
    use_dask=True,
    use_xarray=True,
)
```

Use `read_stacks` for true lazy loading. Pixel planes are read only when they
are indexed or computed:

```python
from czitools.read_tools import read_stacks_list

scenes, dims, scene_count, metadata = read_stacks_list(
    czi_path,
    use_dask=True,
    use_xarray=True,
    planes={"T": (0, 0), "C": (0, 0)},
)

first_scene = scenes[0]          # still lazy
first_plane = first_scene.isel(T=0, C=0, Z=0).compute()
```

`read_stacks_list` is the safest general-purpose interface because it supports
different scene shapes and always returns a list. If every scene must have the
same shape, require a single stacked result instead:

```python
from czitools.read_tools import read_stacks_stacked

plate_array, dims, scene_count, metadata = read_stacks_stacked(
    czi_path,
    use_dask=True,
    use_xarray=True,
)

# Trigger only the selected computation.
one_plane = plate_array.isel(S=0, T=0, C=0, Z=0).compute()
```

`read_stacks_stacked` raises `ValueError` when the scenes cannot be stacked.
Use `read_stacks_list` in that case. Dask must be installed in the environment
when `use_dask=True` is used.

### 1.3 Reading a non-plate, multi-scene CZI

HCS detection is an additional interpretation and does not restrict normal CZI
access. A non-plate file can still be read normally:

```python
from czitools.metadata_tools import CziMetadata
from czitools.read_tools import read_6darray

metadata = CziMetadata(czi_path)
print(metadata.hcs_status.detected)
print(metadata.hcs_status.reason)

pixels, metadata = read_6darray(
    czi_path,
    planes={"S": (0, 0), "T": (0, 0), "C": (0, 0), "Z": (0, 0)},
    use_xarray=True,
)
```

## 2. Reading general CZI metadata

`CziMetadata` reads the major metadata categories once and exposes them through
one object:

```python
from czitools.metadata_tools import CziMetadata

metadata = CziMetadata(czi_path)

print(metadata.filename)
print(metadata.acquisition_date)
print(metadata.pixeltypes)
print(metadata.has_scenes)
print(metadata.scene_shape_is_consistent)
```

Some metadata groups are optional. The corresponding `*_required` properties
give a non-optional value or raise a clear error:

```python
image = metadata.image_required
scale = metadata.scale_required
channels = metadata.channelinfo_required

print(f"S/T/C/Z: {image.SizeS}/{image.SizeT}/{image.SizeC}/{image.SizeZ}")
print(f"Y/X: {image.SizeY}/{image.SizeX}")
print(f"physical pixel size X/Y/Z: {scale.X}/{scale.Y}/{scale.Z}")
```

Frequently useful metadata groups include:

| Property | Contents |
| --- | --- |
| `image` | Dimension sizes |
| `bbox` | Total and per-scene bounding boxes |
| `channelinfo` | Channel names and display/acquisition information |
| `scale` | Physical scaling |
| `objective`, `detector`, `microscope` | Instrument information |
| `sample` | Per-scene sample and well information |
| `attachments` | CZI attachment availability |
| `hcs`, `hcs_status` | Validated HCS hierarchy and detection result |

For example, inspect subblock-derived bounds with:

```python
bbox = metadata.bbox_required
print(bbox.total_bounding_box)
print(bbox.scenes_bounding_rect)
```

The metadata object returned by a pixel reader is a `CziMetadata` instance too,
so a second construction is unnecessary when pixels and metadata are needed
together.

To export the embedded CZI XML for low-level inspection:

```python
from czitools.metadata_tools.czi_metadata import writexml

xml_path = writexml(czi_path)
print(xml_path)
```

## 3. Reading CZI well-plate metadata

Well-plate metadata is available at two levels:

- `metadata.sample` is a compatibility-oriented, per-scene view.
- `metadata.hcs` is the preferred validated and immutable
  `Plate -> Well -> Field` model.

Always inspect `hcs_status` before accessing the HCS model:

```python
metadata = CziMetadata(czi_path)

print(f"HCS detected: {metadata.hcs_status.detected}")
print(f"Reason: {metadata.hcs_status.reason}")

if metadata.hcs is None:
    raise ValueError("This CZI does not contain an unambiguous HCS plate.")

plate = metadata.hcs
```

Failed HCS detection does not invalidate general metadata or pixel access.

### 3.1 The Plate -> Well -> Field model

```python
print(plate.id, plate.name, plate.schema_version)
print(plate.declared_rows, plate.declared_columns)
print(plate.observed_row_indices)       # normalized, zero-based
print(plate.observed_column_indices)    # normalized, zero-based

for well in plate.wells:
    print(
        well.canonical_name,     # for example "B4"
        well.canonical_path,     # for example "B/4"
        well.row_index,          # normalized zero-based index
        well.column_index,       # normalized zero-based index
        len(well.fields),
    )
```

The model retains both original CZI indices and normalized zero-based indices.
Well lookup accepts capitalization, zero padding, and path notation:

```python
well = plate.get_well("b04")  # resolves to canonical well B4

for field in well.fields:
    print(
        field.field_index,       # zero-based within this well
        field.scene_index,       # corresponding global CZI scene
        field.region_id,         # source RegionId, if present
        field.scene_center_x,
        field.scene_center_y,
        field.position_unit,
    )
```

Pure resolver functions map well/field selectors without reading pixels:

```python
from czitools.metadata_tools.hcs import resolve_field, resolve_well

well = resolve_well(plate, "B/04")
field = resolve_field(plate, "B/04", 0)  # local, zero-based field index

# A RegionId string can also select a field:
if field.region_id is not None:
    same_field = resolve_field(plate, "B04", field.region_id)
```

### 3.2 The per-scene sample view

The sample collections have one entry per scene:

```python
sample = metadata.sample

if sample is not None:
    print(sample.scene_count)
    print(sample.well_unique_number)
    print(sample.multipos_per_well)

    for scene in range(sample.scene_count):
        print(
            scene,
            sample.well_array_names[scene],
            sample.well_region_ids[scene],
            sample.field_centerX[scene],
            sample.field_centerY[scene],
        )
```

Prefer `field_centerX` and `field_centerY` over the deprecated
`scene_stageX` and `scene_stageY` compatibility properties. The preferred
fields preserve a real coordinate of `0.0` and use `None` for absent or
malformed `Scene.CenterPosition` metadata. The legacy properties convert
missing values to `0.0`, losing that distinction.

### 3.3 Reading pixels by well and field

`read_field` maps a well-local field to its CZI scene and returns an
`STCZYX(A)` array with `S == 1`:

```python
from czitools.read_tools import read_field, read_well

field_image, metadata = read_field(
    czi_path,
    well="B04",
    field=0,
    planes={"T": (0, 0), "C": (0, 0)},
    use_xarray=True,
)
```

Read every field from a well as a list because field shapes may differ:

```python
field_images, metadata = read_well(czi_path, "B04")

if isinstance(field_images, list):
    for index, field_image in enumerate(field_images):
        print(index, field_image.shape)
```

When all field shapes are equal, they can be concatenated along `S`:

```python
stacked_fields, metadata = read_well(czi_path, "B04", stack=True)
```

`read_field` and `read_well` currently reuse `read_6darray`; consequently their
`use_dask=True` option creates Dask-backed results but does not provide true
on-demand reading from the CZI.

### 3.4 Optional stage-position enrichment

Scene-center coordinates come from `Scene.CenterPosition`. Subblock stage and
focus positions are separate information and are added only by explicit
planetable enrichment:

```python
enriched_plate = metadata.enrich_hcs_positions(position_tolerance=1.0)

if enriched_plate is not None:
    well = enriched_plate.get_well("B04")
    for field in well.fields:
        print(
            field.field_index,
            field.stage_x,
            field.stage_y,
            field.acquisition_z,
            field.position_conflict,
        )
```

Enrichment scans local CZI subblock metadata, returns a new immutable plate,
and updates `metadata.hcs` to that enriched copy. It is unavailable for URL
sources. Representative positions are medians across matching subblocks; value
ranges are retained, and `position_conflict` is set when a range exceeds the
specified tolerance. Scene-center and subblock-stage coordinates are never
merged.

Position helpers expose either well-relative scene-center offsets or absolute
coordinates:

```python
from czitools.metadata_tools.hcs import (
    well_absolute_field_positions,
    well_relative_field_positions,
)

well = enriched_plate.get_well("B04")
relative = well_relative_field_positions(well)
scene_centers = well_absolute_field_positions(well, source="scene_center")
stage_positions = well_absolute_field_positions(well, source="stage")
```

Each helper returns `None` if a required coordinate is missing for any field.

## 4. Converting a CZI well plate to OME-Zarr HCS

The converter resolves the canonical HCS layout from `CziMetadata.hcs` and
uses unambiguous sample metadata as a fallback. It writes every field as an
image below its well, preserving plates whose fields have different shapes.

A simplified logical output hierarchy is:

```text
plate
└── row B
    └── column 04 (well B04)
        ├── field 0
        │   ├── multiscale level 0
        │   └── multiscale level 1
        └── field 1
            └── ...
```

The metadata keys and storage files differ between Zarr formats, but the
logical hierarchy is always plate → row/column well → field image →
multiscale level.

### 4.1 Recommended NGFF v0.5 conversion

```python
from czitools.export_tools import convert_czi2hcs_ngff, validate_ome_zarr

output_path = convert_czi2hcs_ngff(
    czi_path,
    plate_name="Automated Plate",
    output_dir=Path("exports"),
    overwrite=True,
    pad_columns=True,
)

if not validate_ome_zarr(output_path):
    raise RuntimeError(f"OME-Zarr validation failed: {output_path}")

print(output_path)
```

This backend writes an OME-NGFF v0.5, Zarr v3 HCS store. It can also write a
single-file `.ozx` archive with `write_ozx_directly=True`.

### 4.2 ome-zarr-py conversion

```python
from czitools.export_tools import convert_czi2hcs_omezarr, validate_ome_zarr

output_path = convert_czi2hcs_omezarr(
    czi_path,
    overwrite=True,
    pad_columns=True,
    zarr_format=3,
)

assert validate_ome_zarr(output_path)
```

Use `zarr_format=2` for legacy readers that require OME-NGFF v0.4/Zarr v2.
Unlike the NGFF backend, this function currently writes beside the input CZI
and does not accept `output_dir`.

`pad_columns=True` produces paths such as `B/04`; setting it to `False`
produces `B/4`. Both represent the same logical well, but consumers may have a
path-format preference.

### 4.3 Command-line conversion

The repository demo provides the same backend choice:

```bash
python demo/scripts/omezarr_convert_hcs.py path/to/plate.czi
python demo/scripts/omezarr_convert_hcs.py path/to/plate.czi --backend omezarr
python demo/scripts/omezarr_convert_hcs.py path/to/plate.czi --no-pad-columns
python demo/scripts/omezarr_convert_hcs.py path/to/plate.czi --no-validate
```

Validation should normally remain enabled. Also keep `overwrite=False` when an
existing export must be preserved; `overwrite=True` removes the converter's
previous output at the target path.

## 5. End-to-end example

The following compact workflow detects the plate, inspects one field, reads a
small pixel subset, converts the complete plate, and validates the result:

```python
from pathlib import Path

from czitools.export_tools import convert_czi2hcs_ngff, validate_ome_zarr
from czitools.metadata_tools import CziMetadata
from czitools.read_tools import read_field

czi_path = Path("data/WP96_4Pos_B4-10_DAPI.czi")

# Inspect metadata without loading the image pixels.
metadata = CziMetadata(czi_path)
if metadata.hcs is None:
    raise ValueError(f"Not an unambiguous HCS plate: {metadata.hcs_status.reason}")

plate = metadata.hcs
well = plate.get_well("B04")
print(plate.name, well.canonical_name, len(well.fields))

# Read only one T/C/Z plane from the first field.
field, _ = read_field(
    czi_path,
    well="B04",
    field=0,
    planes={"T": (0, 0), "C": (0, 0), "Z": (0, 0)},
    use_xarray=True,
)
print(field.shape)

# Convert the entire plate and validate the written hierarchy.
output = convert_czi2hcs_ngff(
    czi_path,
    output_dir=Path("exports"),
    overwrite=True,
)
if not validate_ome_zarr(output):
    raise RuntimeError(f"Validation failed: {output}")

print(f"Wrote {output}")
```

## 6. Practical guidance

- Use `CziMetadata` first when deciding how to process an unfamiliar CZI.
- Use `read_6darray` for simple, regular data that comfortably fits in memory.
- Use `read_stacks_list(..., use_dask=True)` for genuinely lazy access and for
  files with unequal scene shapes.
- Prefer the HCS model over manually correlating scene arrays and well names.
- Treat field indices as well-local and scene indices as file-global.
- Preserve missing positions as `None`; do not silently reinterpret them as
  coordinates at the origin.
- Keep stage positions and scene-center positions as separate coordinate
  sources.
- Validate every OME-Zarr HCS export before publishing or starting downstream
  analysis.

## Source examples in this repository

This guide consolidates the behavior demonstrated by:

- `demo/scripts/hcs_non_plate_multiscene.py`
- `demo/scripts/hcs_plate_model.py`
- `demo/scripts/hcs_position_enrichment.py`
- `demo/scripts/hcs_read_fields.py`
- `demo/scripts/hcs_sample_metadata.py`
- `demo/scripts/omezarr_convert_hcs.py`
- `demo/scripts/process_hcsplate_example.py`
- `demo/notebooks/read_czi_wellplate_data.ipynb`
- `demo/notebooks/process_omezarr_HCS_plate.ipynb`
