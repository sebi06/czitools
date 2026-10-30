# HCS NGFF Conversion Workflow

The ngff-zarr HCS exporter converts a CZI plate into an OME-NGFF v0.6,
Zarr v3 hierarchy by default, including `.ozx` output. It requires
`ngff-zarr>=0.47.0` and does not silently downgrade new exports. The
ome-zarr-py HCS backend also defaults to v0.6, using ngff-zarr's metadata
converter after pixel writing. The optimized path is designed around three
constraints:

- parse expensive CZI metadata once and reuse it;
- read stored CZI pyramid levels directly, using native libCZI zoom only when
  the source has no suitable coarse level;
- overlap a bounded number of field reads and writes without materializing the
  complete plate in memory.

## Conversion pipeline

```mermaid
--8<-- "docs/diagrams/hcs-ngff-conversion-workflow.mmd"
```

The exporter resolves the complete HCS layout before scheduling writes. Each
job selects one scene and calls `read_stacks_multiscale()` for that field. The
returned Dask arrays remain lazy until ngff-zarr writes them.

For every requested level:

1. A stored CZI pyramid level is used when available.
2. Otherwise, libCZI creates the level during the CZI read with its native
   `zoom` implementation. The exporter does not read level 0 and downsample it
   later in Python.
3. Y and X are rechunked toward `512 x 512` pixels. ngff-zarr clamps chunk and
   shard geometry to the actual array shape.
4. ngff-zarr writes OME-NGFF metadata and Zarr v3 arrays through zarrista.
5. `chunks_per_shard={"y": 4, "x": 4}` requests up to four inner chunks per
   spatial shard axis. Blosc compresses the inner chunks.

TensorStore is not part of this path. ngff-zarr uses zarrista and receives
filesystem or `.ozx` paths directly. HCS validation selects the version-aware
HCS reader with validation enabled for v0.6 stores. Older versions remain
available through explicit Python compatibility options and legacy validation.

## Version compatibility

Version profiles keep NGFF v0.6 as the stable default independently of the
installed backend's latest version. Explicit v0.5 compatibility output remains
available; ngff-zarr also supports v0.4 directory output, but not RFC-9 archives.
Known v0.6 prerelease aliases are recognized only for reading, not writing.
Unknown versions, including `0.9.dev1` and `0.9.dev2`, are rejected rather than
validated against an older schema or relabeled.

Internal axis descriptions retain names, types, units, and discrete semantics
without treating unknown axes as time. Released profiles retain their 2-5D,
time/channel/spatial ordering constraints. Single-image APIs still require
TCZYX; future arbitrary-axis support must use an explicit profile and cannot
silently squeeze, reorder, or discard dimensions to produce v0.6 output.

Validation checks metadata across the hierarchy without reading pixel chunks.
All NGFF nodes must declare the same exact version. Display metadata retains
complete channel colors and min/max/start/end windows, even where a draft might
allow fewer fields. Packaging an existing store also checks these guardrails
before writing an archive or removing its source; it never upgrades metadata.

Adding a draft profile requires verified support for the exact draft version,
pinned schema fixtures, negative axis/layout tests, image and HCS round trips,
and tests with target readers. Draft output must remain opt-in in a separate
store, while v0.6 pixels, scaling, pyramid alignment, channels, and layout stay
covered by released-profile regression tests. No draft writer is enabled by
this preparation.

## Metadata reuse

Constructing `CziMetadata` repeatedly is expensive for large HCS files. The
exporter creates it once, then passes it through `read_stacks_multiscale()` to
each `read_stacks()` call.

`read_stacks()` uses a shallow metadata clone and copies only the scale object
that it updates for the current zoom. It also copies image dimensions when
`adapt_metadata=True`. The large parsed metadata graph and bounding-box model
remain shared and read-only. This avoids both repeated CZI metadata parsing and
a full `deepcopy()` for every field and pyramid level.

With `preserve_original_metadata=True`, both HCS backends also read the raw
CZI XML once and store it at root-relative `CZI/metadata.xml`, referenced by
`czitools.originalMetadata` in root attributes. This is opt-in and does not
duplicate XML per field. For `.ozx`, the directory store and XML are staged
before archive finalization; allow space for both the staging store and archive.

The root-level `CZI` group is auxiliary metadata alongside row groups such
as `A` and `B`, not an HCS row or well. It is not included in
`ome.plate.rows` or `ome.plate.wells`; HCS readers should follow the declared
well paths rather than treat every root group as a row. The custom
`czitools.originalMetadata` reference makes the XML discoverable, but does
not define plate membership or make the HCS layout valid. The existing
plate metadata defines the layout independently of this reference.
See [Preserving original CZI XML](usage.md#preserving-original-czi-xml) for the
application convention, CLI flag, and GUI checkbox.

## Bounded concurrency and memory

```mermaid
--8<-- "docs/diagrams/hcs-ngff-bounded-concurrency.mmd"
```

`max_workers=4` is the default. Four field jobs may therefore build and write
their lazy graphs concurrently. This overlaps CZI reads, compression, and
filesystem writes while bounding the number of active fields.

This differs from loading the complete plate into one in-memory array. Peak
memory depends on field shape, pixel type, T/C/Z depth, chunk geometry, codec
buffers, and the number of concurrent workers. As a practical approximation,
increasing `max_workers` can increase the active field-processing memory by a
similar factor, although Dask and the writer operate chunk by chunk.

Use the default for typical HCS fields. Set `max_workers=1` when fields are very
large, T or Z is deep, memory is constrained, or the source storage performs
poorly under concurrent random reads. Intermediate values such as 2 provide a
middle ground.

## Storage layout

The output hierarchy is:

```text
plate/
  <row>/<column>/
    <field>/
      scale0/<image-name>
      scale1/<image-name>
      ...
```

Each image array uses dimension order `(t, c, z, y, x)`. For Zarr v3 output,
indexed sharding groups compressed inner chunks into fewer outer objects. This
reduces filesystem metadata overhead and file count; it does not itself reduce
CZI decoding work.

For a `2000 x 2000` field, a nominal spatial chunk size of 512 and four chunks
per shard can produce one spatial shard with approximately `500 x 500` inner
chunks. For smaller fields, ngff-zarr clamps the geometry. A `708 x 980` field,
for example, can use a `708 x 980` outer shard with `354 x 490` inner chunks.

## API example

```python
from pathlib import Path

from czitools.export_tools import convert_czi2hcs_ngff, validate_ome_zarr

output = convert_czi2hcs_ngff(
    "plate.czi",
    output_dir=Path("exports"),
    spatial_chunk_size=512,
    chunks_per_shard={"y": 4, "x": 4},
    planes_per_chunk=64,
    max_workers=4,
)

if not validate_ome_zarr(output):
    raise RuntimeError(f"OME-NGFF validation failed: {output}")
```

## Performance interpretation

Conversion time is workload- and hardware-specific. The largest improvements
come from eliminating repeated metadata parsing and full metadata deep copies.
Stored or native CZI pyramid reads avoid Python-side resampling, while bounded
field concurrency and sharding improve throughput and storage behavior.

Always compare outputs as well as timings. A useful verification includes:

- the expected plate, well, field, and scale counts;
- OME-NGFF schema validation;
- array shapes, data types, dimension names, and coordinate transforms;
- representative pixel comparisons against direct CZI reads;
- output size, file count, compression, and sharding codecs.