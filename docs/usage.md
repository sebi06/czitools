# Usage

!!! warning "Work in Progress"
    This documentation is still incomplete and actively being updated.
    Some sections may be missing or subject to change.

The examples below assume that `filepath` points to a local CZI:

```python
from pathlib import Path

filepath = Path("data/WP96_4Pos_B4-10_DAPI.czi")
```

## Inspecting HCS Well Plates

### Command-Line Tool

The `czi_hcs_check.py` script provides a convenient way to inspect CZI
well-plate metadata from the terminal with rich, colorized output. It first
shows every full-resolution dimension size derived from stored subblocks. The
HCS plate view contains only physically stored fields by default.

#### Inspect entire plate

```bash
python demo/scripts/czi_hcs_check.py plate.czi
```

For split acquisitions, the default output may contain fewer fields than the
XML declares. The split child files retain global scene indices, so a child
that stores only scene `S=3` is matched to scene 3 in the XML-derived HCS
model. The inspector reports the stored and declared field counts when they
differ.

#### Show the complete XML-declared acquisition

```bash
python demo/scripts/czi_hcs_check.py plate.czi --show-declared
```

`--show-declared` disables payload filtering and shows every HCS field in the
acquisition plan, including fields stored in other split files.

#### Inspect a specific well

```bash
python demo/scripts/czi_hcs_check.py -f plate.czi --well B4
```

When you specify a well with `--well`, the tool displays:

- The requested well's field information in the **Well Fields** section
- The **First Scene Details** from that specific well (not the overall first scene)

This provides consistent context when inspecting a particular well.

#### Using the --filepath flag

```bash
python demo/scripts/czi_hcs_check.py --filepath plate.czi
```

#### View all options

```bash
python demo/scripts/czi_hcs_check.py --help
```

#### Hide well summary table for large plates

```bash
python demo/scripts/czi_hcs_check.py -f plate.czi --no-well-table
```

The `--no-well-table` flag omits the per-well summary table while retaining
the plate information, sample metadata, and selected well's field details.
This is useful for large plates where the summary would make terminal output
unnecessarily long. It can be combined with `--well`:

```bash
python demo/scripts/czi_hcs_check.py -f plate.czi --well B4 --no-well-table
```

**Example output** (showing HCS plate information with rich formatting):

```text
Full-Resolution Subblock Dimensions
Dimension  Meaning       Size
S          Scene         1
T          Time          1
C          Channel       2
Z          Z-slice       5
Y          Height        1024
X          Width         1024
M          Mosaic        not present
R/I/H/V/B                 not present
First stored scene Y × X: 1024 × 1024 px

╭─────────────────────────────────────────────────────────────────╮
│ 🔬 High-Content Screening (HCS) Plate Information              │
╰─────────────────────────────────────────────────────────────────╯
HCS Detected: True
Reason: Valid HCS hierarchy detected

Plate ID: 596fa2ec-0844-4cef-999f-8a27cf3c85dd
Plate Name: Test Plate 96
Schema Version: 2.0
Dimensions: 8 rows × 12 columns (declared)
Row Indices: [0, 1, 2, 3, 4, 5, 6, 7]
Column Indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

Total Wells: 96
Total Fields: 384

╭─────────────────────────────────────────────────────────────────╮
│ Well Summary                                                    │
├──────────┬─────────────┬───────────┬────────────┬────────────┤
│ Well     │ Path        │ CZI Index │ Normalized │ Fields     │
├──────────┼─────────────┼───────────┼────────────┼────────────┤
│ A1       │ Row0/Col0   │ (0,0)     │ (0,0)      │ 4          │
│ A2       │ Row0/Col1   │ (0,1)     │ (0,1)      │ 4          │
│ ...[continues for all wells]...
│ H12      │ Row7/Col11  │ (7,11)    │ (7,11)     │ 4          │
╰──────────┴─────────────┴───────────┴────────────┴────────────╯
```

### Python API

For programmatic access, use the HCS display utilities from `czitools.utils`:

#### Print plate information

```python
from czitools.utils import print_hcs_plate_info
from czitools.metadata_tools import CziMetadata

mdata = CziMetadata("plate.czi")
print_hcs_plate_info(mdata)
```

#### Print sample metadata

```python
from czitools.utils import print_sample_metadata

mdata = CziMetadata("plate.czi")

# Print sample metadata (shows first scene overall)
print_sample_metadata(mdata)

# Print sample metadata with first scene from a specific well
print_sample_metadata(mdata, well_name="B5")
```

The optional `well_name` parameter shows the first scene from that specific well, providing consistent context when inspecting a particular well.

**Example output** (sample metadata):

```text
╭─────────────────────────────────────────────────────────────────╮
│ 📊 Sample Metadata                                              │
╰─────────────────────────────────────────────────────────────────╯

Scene Count:        384
Unique Wells:       96
Fields per Well:    4

Per-Scene Collections (8 entries)
  well names          384
  well indices        384
  position names      384
  row indices         384
  column indices      384
  field center X      384
  field center Y      384
  region IDs          384

╭─────────────────────────────────────────────────────────────────╮
│ 🎬 First Scene Details                                          │
╰─────────────────────────────────────────────────────────────────╯

Well:              A1
Region ID:         1
Field Center:      (1234.56, 5678.90) µm
Stage Position:    (0.0, 0.0)
```

#### Print well fields

```python
from czitools.utils import print_well_fields

mdata = CziMetadata("plate.czi")
print_well_fields(mdata, well_name="B4")
```

**Example output** (well fields):

```text
╭─────────────────────────────────────────────────────────────────╮
│ 🔎 Well Fields                                                  │
╰─────────────────────────────────────────────────────────────────╯

╭────────────────────────────────────────────────────────────────╮
│ Fields in well 'B4'                                            │
├──────────┬────────┬────┬─────────┬─────────────┬─────────────┤
│ Local    │ Scene  │ ID │ Region  │ Center X    │ Center Y    │
├──────────┼────────┼────┼─────────┼─────────────┼─────────────┤
│ 0        │ 52     │ 1  │ 1       │ 1234.56     │ 5678.90     │
│ 1        │ 53     │ 2  │ 1       │ 2345.67     │ 5678.90     │
│ 2        │ 54     │ 3  │ 1       │ 1234.56     │ 6789.01     │
│ 3        │ 55     │ 4  │ 1       │ 2345.67     │ 6789.01     │
╰──────────┴────────┴────┴─────────┴─────────────┴─────────────╯
```

#### Get well by name and access fields

```python
from czitools.metadata_tools import CziMetadata

mdata = CziMetadata("plate.czi")

# Access HCS plate hierarchy
plate = mdata.hcs
if plate is not None:
    # Get a well by name (supports B4, b04, B/4 formats)
    well = plate.get_well("B4")
    print(f"Well: {well.canonical_name}")
    print(f"Position: Row {well.row_index}, Column {well.column_index}")
    print(f"Fields: {len(well.fields)}")
    
    # Access field information
    for field in well.fields:
        print(f"  Field {field.field_index}: "
              f"Scene {field.scene_index}, "
              f"Center: ({field.scene_center_x}, {field.scene_center_y}) µm")
```

## General Metadata Access

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

`CziDimensions` obtains index ranges and spatial extents from
`pylibCZIrw`'s full-resolution, non-pyramid subblock-derived bounding boxes.
The numeric XML `Size*` values are not authoritative. `SizeS` is the number of
stored scene rectangles, not the largest scene index plus one; global scene
keys can therefore be sparse. `SizeX_scene` and `SizeY_scene` describe the
first stored scene, while `SizeX` and `SizeY` describe the total stored spatial
extent.

XML remains the source for acquisition-plan semantics such as well names,
plate layout, and scene-center positions. For split acquisitions, use these
attributes to compare both views:

```python
mdata = CziMetadata(filepath, filter_hcs_to_stored_scenes=True)

print(mdata.stored_scene_indices)  # physical global S keys
print(mdata.hcs)                   # physically stored HCS subset
print(mdata.hcs_declared)          # complete XML-declared HCS model
```

`filter_hcs_to_stored_scenes` defaults to `False` in the Python API for
backward compatibility. The HCS inspector enables it by default and offers
`--show-declared` for the complete XML model.

The `*_required` properties return a non-optional value or raise a clear error.
Frequently used groups are:

| Property                              | Contents                                    |
| ------------------------------------- | ------------------------------------------- |
| `image`                               | Full-resolution subblock-derived dimensions |
| `bbox`                                | Total and per-scene bounding boxes          |
| `channelinfo`                         | Channel acquisition and display information |
| `scale`                               | Physical scaling                            |
| `objective`, `detector`, `microscope` | Instrument information                      |
| `sample`                              | Per-scene sample and well information       |
| `attachments`                         | CZI attachment availability                 |
| `hcs`, `hcs_status`                   | Active HCS hierarchy and detection result   |

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
| Equal-sized scenes; lazy read             | `read_6darray(use_dask=True)` | One lazy `STCZYX(A)` array    |
| Irregular scenes; lazy read               | `read_stacks(use_dask=True)`  | Per-scene arrays by default   |
| True lazy reads with equal scenes stacked | `read_stacks_stacked`         | One array with `S`            |
| Scenes that may differ in shape           | `read_stacks_list`            | Stable list of scene arrays   |
| Gigapixel planes (whole-slide, large 2D)  | `read_stacks(use_dask=True)`  | Automatic spatial Y/X tiling  |
| Multiscale pyramid for napari             | `read_stacks_multiscale`      | List of dask arrays per level |
| A known HCS well or field                 | `read_well` / `read_field`    | HCS-aware field arrays        |

### `read_6darray` — Full 6D Stack

Returns the image as a single array with dimension order **STCZYX(A)**.
It requires equal-sized scenes and consistent pixel types:

```python
from czitools.read_tools import read_6darray

# NumPy array
array6d, mdata = read_6darray(filepath)

# Genuinely lazy Dask-backed array; planes are read during computation.
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

!!! note "Lazy reads"
    `read_6darray(..., use_dask=True)` defers each CZI plane read until the
    corresponding Dask task is computed. Use `read_stacks(..., use_dask=True)`
    when scenes may have different shapes or extra dimensions.

!!! note "Full-resolution bounds"
  Eager and lazy `read_6darray` reads are constrained to each selected
  scene's non-pyramid bounding rectangle in native layer-0 coordinates.
  Coarse pyramid tiles can round their logical coverage outward by a few
  pixels; those storage-level overhangs do not change the regular array's
  Y/X shape.

### `read_stacks` — Scene-Wise Reading

`read_stacks` supports all CZI dimensions and optionally stacks compatible
scenes. With `use_dask=True`, it reads one representative plane per scene while
constructing the result; the remaining pixel reads are deferred until indexed
or computed:

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

**HCS plate files with ±1-pixel rounding differences** — scenes from
different wells often have bounding rectangles that differ by one pixel
because stage coordinates round differently at each well position. Pass
`scene_stack_tolerance=1` (default `0`) to crop all scenes to the smallest
common Y/X shape before stacking:

```python
stacked, dims, n, mdata = read_stacks(
    filepath,
    use_dask=True,
    use_xarray=True,
    stack_scenes=True,
    scene_stack_tolerance=1,  # crop ±N-px differences; 0 = strict equality
)
```

The crop removes at most `scene_stack_tolerance` pixels from the right/bottom
edge of oversized scenes and is silent for files where all scenes already have
identical shapes. `read_stacks_multiscale` accepts the same parameter and
forwards it to every pyramid level.

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

By default, lazy reads group up to 64 planes into each task. This substantially
reduces scheduler and repeated file-open overhead. It may read neighbouring
planes when only a small selection is computed. Use `planes_per_chunk` to tune
the group size, or opt into one-plane tasks when minimal random-access reads are
more important than throughput:

```python
scenes, dims, scene_count, mdata = read_stacks_list(
    filepath,
    use_dask=True,
    use_xarray=True,
    planes={"T": (0, 0), "C": (0, 0)},
    lazy_read_strategy="plane",
)

# With the plane strategy, this computes only the selected plane task.
first_plane = scenes[0].isel(T=0, C=0, Z=0).compute()
```

#### Spatial Y/X tiling for very large planes

Some CZI files, especially whole-slide or high-magnification scans, contain
individual 2D planes that are tens of gigabytes when uncompressed. A single
`93,555 × 138,996` `uint16` plane is about **24 GB** per channel. If the dask
graph uses one task per plane, the very first tile fetch a viewer performs
loads the entire plane into RAM.

To keep those files usable, `read_stacks(..., use_dask=True)` automatically
tiles Y/X into a grid when a single plane would exceed `chunk_memory_limit`
(default 256 MB). Each tile becomes its own dask chunk backed by a
ROI-based read via
[`pylibCZIrw.CziReader.read(roi=(x, y, w, h), ...)`](https://github.com/ZEISS/pylibczirw),
so viewers such as napari only fetch the tiles that intersect the current
viewport.

Behaviour:

- **Trigger:** `plane_bytes = spatial_y × spatial_x × dtype.itemsize × components`;
  tiling is used when `plane_bytes > chunk_memory_limit`.
- **Tile size:** `tile_size` (default 4096) sets the nominal square edge in
  *zoomed* pixels. The chosen tile is halved iteratively so a single tile
  never exceeds `chunk_memory_limit`. Very small tiles (< ~256 px) are not
  recommended: libCZI's resampler is ROI-aware, so tile-boundary pixels can
  disagree slightly with a whole-plane read on files without an on-disk
  pyramid. At the file's stored pyramid zooms every ROI is served directly
  from a subblock and this effect does not occur.
- **Coordinate spaces:** `pylibCZIrw.CziReader.read(roi=...)` interprets the
  ROI in **native (layer-0) coordinates** and returns an array of shape
  `int(roi.w * zoom) × int(roi.h * zoom)` (libCZI uses truncation, not
  rounding). czitools converts the requested zoomed tile size back to
  layer-0 via `ceil(tile / zoom)` and matches libCZI's truncation for the
  declared dask chunk shape, so the graph is exact at any pyramid zoom.
- **Grouping:** spatial tiling forces `lazy_read_strategy="plane"` for the
  affected stack. Small planes always keep the whole-plane path (no overhead).
- **Where the change lives:** entirely in `czitools`. Downstream code (for
  example `napari-czitools`) needs no changes because it already passes
  `use_dask=True` and napari renders chunked dask arrays natively.

```python
from czitools.read_tools import read_stacks_stacked

# A 93k x 139k uint16 plane triggers 4096x4096 tiling (~32 MB per chunk).
stacked, dims, n, mdata = read_stacks_stacked(
    "path/to/huge.czi",
    use_dask=True,
    use_xarray=True,
    tile_size=4096,               # optional; 4096 is the default
    chunk_memory_limit=256 * 1024 * 1024,  # optional; 256 MB is the default
)

# Nothing has been read yet. Only the tiles hit by this ROI are loaded.
viewport = stacked.isel(S=0, T=0, C=0, Z=0, Y=slice(0, 4096), X=slice(0, 4096))
viewport.compute()
```

#### Multiscale pyramid reading

Interactive viewers such as napari need a **multiscale pyramid** to render
gigapixel planes efficiently: the coarsest level is uploaded to a single GPU
texture and finer tiles stream in on zoom. `read_stacks_multiscale` wraps
`read_stacks` and returns one lazy dask array per pyramid level, in a shape
that napari's `viewer.add_image(..., multiscale=True)` accepts directly.

CZI reality:

- Some files store no pyramid at all (only `1.0`).
- Some store standard powers of two (e.g. `[1.0, 0.5, 0.25, 0.125, 0.0625]`).
- Others use application-specific factors (ZEN's 3x pyramid produces
  `[1.0, 0.333, 0.111, ...]`).
- `pylibCZIrw.CziReader.read(zoom=z)` reads directly from the matching
  subblocks when `z` is a stored level (fast). Non-stored zooms trigger
  libCZI's C++ resampler (slower, but still much cheaper than materialising
  layer 0 in Python).
- Levels below `zoom = 0.01` cannot be read (`pylibCZIrw` clamps to `0.01`)
  and are automatically dropped from the returned list.

Inspect the stored pyramid without reading any pixel data:

```python
from czitools.read_tools import get_pyramid_zooms

print(get_pyramid_zooms("path/to/large.czi"))
# -> [1.0, 0.5, 0.25, 0.125, 0.0625]
```

Build a multiscale pyramid ready for napari:

```python
from czitools.read_tools import read_stacks_multiscale

levels, infos, dims, num_stacks, mdata = read_stacks_multiscale(
    "path/to/large.czi",
    use_xarray=True,
    stack_scenes=True,
    max_coarse_edge=8192,   # coarsest level's longest edge target in px
)

for lvl, info in zip(levels, infos):
    print(f"zoom={info.zoom:.4f} stored={info.stored} shape={lvl.shape}")
```

Behaviour:

- `levels` is a list, coarsest last. Every element has the same `S/T/C/Z`
  shape as level 0 but progressively smaller `Y/X`.
- `infos` contains a `PyramidLevel(zoom, stored, y, x)` per level so callers
  can build per-level scale metadata (napari infers this from shape ratios
  in most cases).
- If the coarsest stored level's longer edge is still greater than
  `max_coarse_edge` (default 8192), additional coarser levels are appended
  by repeatedly halving the zoom until the top of the pyramid fits within
  the target. Those synthetic levels use libCZI's resampler.
- The read graph stays fully lazy — nothing is fetched until a caller
  triggers `.compute()` (or napari renders the tiles it needs).

```python
import napari

viewer = napari.Viewer()
viewer.add_image(levels, multiscale=True)
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

These functions reuse `read_6darray`, so `use_dask=True` also provides
on-demand CZI plane reads.

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
    convert_czi2hcs_omezarr,   # ome-zarr-py backend, Zarr v3
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

Both export backends write Zarr v3 stores and retain the pyramid paths produced
by their underlying writer libraries.

See [HCS NGFF Conversion Workflow](hcs-ngff-conversion.md) for the optimized
reader/writer pipeline, Mermaid architecture diagrams, sharding behavior, and
bounded-concurrency memory guidance.

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
from czitools.export_tools import compression_type, write_omezarr, write_omezarr_ngff

array, mdata = read_6darray("image.czi", planes={"S": (0, 0)}, use_xarray=True)
array = array.squeeze("S")  # 6D -> 5D (T, C, Z, Y, X)

# ngff-zarr backend (multi-scale pyramid, OME-NGFF v0.5)
write_omezarr_ngff(array, "image_ngff.ome.zarr", mdata, scale_factors=[2, 4], overwrite=True)

# ome-zarr-py backend (Zarr v3)
write_omezarr(array, zarr_path="image.ome.zarr", metadata=mdata, overwrite=True)
```

For both directory-backed `.ome.zarr` stores and file-backed `.ozx` archives,
omitting `chunks` makes `write_omezarr_ngff()` use bounded TCZYX chunks:
`(1, 1, 1, min(512, Y), min(512, X))`. Each time point, channel, and Z plane
is chunked independently, while large spatial planes are divided into tiles no
larger than 512 x 512 pixels. This avoids passing a complete multidimensional
image volume to Blosc as one compression buffer and keeps large exports within
the codec's supported chunk size.

Pass `chunks=(T, C, Z, Y, X)` to override the automatic policy. Explicit
chunks are preserved and applied consistently to the source Dask array and
generated multiscales. Choose values that keep each uncompressed chunk well
below available memory and codec limits; for example:

```python
write_omezarr_ngff(
  array,
  "image_ngff.ome.zarr",
  mdata,
  chunks=(1, 1, 1, 256, 256),
  overwrite=True,
)
```

For a faster intensity-image conversion with balanced storage granularity,
use bin-shrink pyramids, larger chunks, and spatial-only sharding:

```python
import ngff_zarr as nz

write_omezarr_ngff(
  array,
  "image_fast.ome.zarr",
  mdata,
  chunks=(1, 1, 4, 1024, 1024),
  chunks_per_shard={"y": 2, "x": 2},
  compression=compression_type.BLOSC,
  downsampling_method=nz.Methods.DASK_BIN_SHRINK,
  overwrite=True,
)
```

The converter GUI uses this configuration by default as the **Fast balanced**
preset for non-HCS ngff-zarr exports. It supports both directory-backed
`.ome.zarr` and single-file `.ozx` output, enforces Blosc compression, and
disables the conflicting compression control. Select **Quality** to retain
512 x 512 spatial chunks and Dask Gaussian pyramids. Bin-shrink computes local
means and is appropriate for intensity images, but Gaussian downsampling
generally produces fewer aliasing artifacts. OZX conversion remains slower
than directory output because ngff-zarr packages a temporary directory store
into the final archive after writing the pyramid.

For file-backed `.ozx` output, `overwrite=True` removes either a completed or
incomplete existing archive before writing. This makes retries after an
interrupted or failed conversion behave the same as retries for directory
stores.

### OME-Zarr Converter GUI

The experimental CZI to OME-Zarr converter provides a graphical interface for
exporting individual images and HCS plates. It supports the `ome-zarr-py` and
`ngff-zarr` backends, Zarr v3 output, compression selection, and single-file
`.ozx` output where supported. ngff-zarr 0.45 uses its zarrista backend
automatically; no storage-engine control is required. The GUI can optionally
open directory outputs in napari.

Install the GUI extra and launch the application with its console command:

```bash
pip install "czitools[omezarr-gui]"
czitools-omezarr-gui
```

From a Pixi checkout, use the equivalent task:

```bash
pixi run omezarr-gui
```

Select a CZI file, choose the backend and output options, and click **Read
Metadata**. Review the detected dimensions and scene information before
clicking **Convert to OME-ZARR**. Conversion progress is displayed in the log
panel at the bottom of the window.

![CZI to OME-Zarr converter GUI](https://github.com/sebi06/czitools/raw/main/_images/czi_omezarr_gui.png){ width="800" }

The application can also be launched from Python or embedded as a napari dock
widget:

```python
# Launch standalone
from czitools.export_tools import run_gui
run_gui()

# Or embed the widget in napari
from czitools.export_tools import create_gui
import napari

viewer = napari.Viewer()
viewer.window.add_dock_widget(create_gui(), name="CZI OME-ZARR Converter")

napari.run()
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
- Use `read_6darray` for regular data, with `use_dask=True` when pixel reads
  should remain lazy.
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
| Show planetable as surface | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/show_czi_surface.ipynb)             |
| Segment with Voronoi-Otsu  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sebi06/czitools/blob/main/demo/notebooks/read_czi_segment_voroni_otsu.ipynb) |
