# HCS NGFF Conversion Workflow

The ngff-zarr HCS exporter converts a CZI plate into an OME-NGFF v0.5,
Zarr v3 hierarchy. The optimized path is designed around three constraints:

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

TensorStore is not part of this path. ngff-zarr 0.45 uses zarrista and receives
filesystem or `.ozx` paths directly.

## Metadata reuse

Constructing `CziMetadata` repeatedly is expensive for large HCS files. The
exporter creates it once, then passes it through `read_stacks_multiscale()` to
each `read_stacks()` call.

`read_stacks()` uses a shallow metadata clone and copies only the scale object
that it updates for the current zoom. It also copies image dimensions when
`adapt_metadata=True`. The large parsed metadata graph and bounding-box model
remain shared and read-only. This avoids both repeated CZI metadata parsing and
a full `deepcopy()` for every field and pyramid level.

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