# czitools — Modernization & Improvement ToDo

> Findings from a repo-wide review (code quality, architecture, typing, dask/zarr
> lazy reading, packaging/versioning, CI, README, MkDocs). Work through these
> **top to bottom** — they are ordered by priority. Each item lists *why*, the
> concrete *action*, and a rough *effort* (S/M/L).

**Priority legend:** `P0` = correctness/release blocker · `P1` = high value ·
`P2` = medium · `P3` = polish.

**Model-tier legend** (which model to spend on each step, per the
`token-anxiety-manager` skill):
`cheap` = mechanical, low-risk edits (config, CI YAML, find/replace) ·
`standard` = focused code changes across a few files with tests ·
`premium` = architectural redesign / non-trivial dask-lazy reasoning.
Each item below is tagged with a recommended tier.

---

## P0 — Blockers

### 1. CI runs tests **only on tags** — PRs and pushes are never tested `P0` `S`

- **Where:** [.github/workflows/test_and_deploy.yml](../.github/workflows/test_and_deploy.yml)
- **Problem:** the `test` job has `if: contains(github.ref, 'tags')`. The workflow
  triggers on `pull_request` and `push`, but the condition skips the entire test
  matrix unless the ref is a tag. Result: **no PR or branch push is ever validated**;
  regressions only surface at release time.
- **Action:** remove the `if: contains(github.ref, 'tags')` from the `test` job
  (keep it on `deploy`). Optionally split into `ci.yml` (test on push/PR) and
  `release.yml` (build+publish on tags) for clarity.
- **Effort:** S · **Model tier:** `cheap` (YAML-only edit)

---

## P1 — High value

### 2. Automatic version bumping (single source of truth) `P1` `M`

- **Where:** [pyproject.toml](../pyproject.toml) (`version = "0.20.1"`),
  [pixi.toml](../pixi.toml) (`version = "0.20.1"`) — version is duplicated and
  bumped by hand.
- **Recommended:** adopt **`setuptools-scm`** so the package version is derived
  from the git tag at build time. Release becomes: tag `vX.Y.Z` → CI builds with
  the correct version automatically. No file edits, no drift.
  
  ```toml
  [build-system]
  requires = ["setuptools>=77", "setuptools-scm>=8"]

  [project]
  dynamic = ["version"]

  [tool.setuptools_scm]
  version_file = "src/czitools/_version.py"
  ```
  
  Expose it: `from czitools._version import version as __version__` in
  [src/czitools/__init__.py](../src/czitools/__init__.py).
- **Alternative** (if a static file is preferred): `bump-my-version` or
  `commitizen` to bump `pyproject.toml` + `pixi.toml` + changelog in one command.
- **Note:** align this with the existing release skill in
  `.claude/skills/release-new-version/`.
- **Effort:** M · **Model tier:** `standard` (build-backend config + one import wire-up; verify a build)

### 3. `read_6darray(use_dask=True)` is not actually lazy `P1` `M`

- **Where:** [src/czitools/read_tools/read_tools.py](../src/czitools/read_tools/read_tools.py)
  (~L352 `da.empty(...)` then per-plane assignment).
- **Problem:** it allocates `da.empty(array_shape, chunks=array_shape)` (one giant
  chunk) and assigns each plane eagerly. This materializes the whole array in RAM
  and defeats dask — as the docstring admits. Users reaching for `use_dask=True`
  expecting lazy behavior get eager memory use instead.
- **Action:** either (a) build the eager path as NumPy only and route lazy users
  to `read_stacks(..., use_dask=True)` (already the true lazy path via
  `da.from_delayed`), or (b) reimplement `read_6darray`'s dask branch with
  `da.from_delayed`/`da.map_blocks` so nothing is read until `.compute()`.
  Update the docstring to stop advertising a non-lazy `use_dask`.
- **Effort:** M · **Model tier:** `premium` (dask laziness/correctness reasoning; easy to get subtly wrong)

### 3b. Spatial Y/X tiling for very large planes `P1` `M`

- **Where:** [src/czitools/read_tools/stacks.py](../src/czitools/read_tools/stacks.py)
  (`_read_plane_delayed`, `_read_plane_chunk`, `read_stacks` dask branch).
- **Problem:** even with `use_dask=True`, each dask task in `read_stacks` reads
  one **whole 2D plane** at full resolution via `czidoc.read(plane=..., scene=...)`.
  For gigapixel CZIs (for example a `93,555 × 138,996` uint16 plane ≈ 24 GB per
  channel) a single tile fetch — including the very first tile napari asks for —
  allocates the entire plane, so viewers OOM despite the graph being lazy. The
  effective chunk size is the plane size, not a user-friendly tile size.
- **Concept:** switch the dask graph from one-task-per-plane to a **grid of
  spatial tiles per plane** using pylibCZIrw's existing ROI support
  ([`CziReader.read(roi=Rectangle(x, y, w, h), ...)`](https://github.com/ZEISS/pylibczirw/blob/main/pylibCZIrw/czi.py#L912)).
  Each tile task reads only the pixels inside its ROI, so viewers only pay for
  the tiles that actually intersect the viewport.
- **Threshold — only kicks in when needed:**
  - Estimate `plane_bytes = spatial_y × spatial_x × dtype.itemsize` (× 3 for RGB).
  - Reuse the existing `chunk_memory_limit` parameter (default 256 MB). When
    `plane_bytes > chunk_memory_limit`, build a tiled dask graph; otherwise keep
    the current whole-plane behaviour. Small planes therefore pay **no overhead**.
  - Default tile size: 4096 × 4096 (~32 MB uint16), configurable via a new
    `tile_size: int = 4096` parameter. Actual tile dims are clamped to
    `chunk_memory_limit`.
- **How the graph is built:**
  1. Compute the tile grid `(n_rows, n_cols)` from `(spatial_y, spatial_x)`.
  2. Add `_read_tile_delayed(filepath, plane, scene_index, roi, zoom, ...)` and
     `_read_tile_chunk(filepath, tiles_batch, ...)` helpers. Both call
     `czidoc.read(plane=..., scene=..., roi=Rectangle(x, y, w, h))`. ROI
     coordinates are expressed in the **scene's** coordinate space: the base
     offset is `stack_rect.x/y` (or `total_bounding_rectangle` for scene-less
     files), so each tile's `roi.x = base_x + col * tile_w`, `roi.y = base_y +
     row * tile_h`, with the last row/column clipped to the remaining extent.
  3. Wrap each tile in `da.from_delayed(shape=(tile_h, tile_w[, A]), dtype=...)`.
  4. Compose the plane with `da.block([[t00, t01, ...], [t10, t11, ...], ...])`
     so the resulting dask array has shape `(spatial_y, spatial_x[, A])` with
     chunks `((tile_h, ..., last_tile_h), (tile_w, ..., last_tile_w))`.
  5. Combine per-plane tiled arrays across T/C/Z (and extra dims) using the
     existing `da.stack`/`build_dask_stack` pattern.
- **Where the change lives:**
  - **czitools only** — extends `stacks.py`. `napari-czitools` needs no changes
    because it already forwards `use_dask=True` and napari natively renders
    chunked dask arrays.
  - Existing `lazy_read_strategy` / `planes_per_chunk` still control T/C/Z-plane
    grouping; `tile_size` is orthogonal and controls the Y/X grid.
- **Impact on the 105 GB / 93k × 139k uint16 file:**
  - Chunk size drops from ~24 GB → ~32 MB (4096² tile).
  - Initial napari render loads only the few tiles that intersect the viewport
    (a couple hundred MB) instead of every full plane (~72 GB for 3 channels).
- **Effort:** M · **Model tier:** `premium` (dask graph shape + ROI coordinate
  arithmetic must be correct; add tests covering last-tile clamping, RGB (A dim),
  scene-less files, and cross-check pixels against a whole-plane read for a
  small file).

### 3c. Multiscale pyramid reader for gigapixel display `P1` `L`

- **Where:** new module `src/czitools/read_tools/pyramid.py` (multiscale) plus
  the existing [stacks.py](../src/czitools/read_tools/stacks.py) tiling. Also
  extends [napari-czitools](../../napari-czitools/) to consume the multiscale
  list.
- **Problem — the second-order rendering wall:** the Y/X tiling in item 3b
  removed the OOM at construction time and made viewport-scale reads cheap,
  but a single gigapixel plane still has to be rendered by napari. Without a
  multiscale representation, napari's default 2D image layer must materialize
  the full plane once to build a display texture (the array is far larger
  than any GPU texture, so downsampling happens on the CPU) and then keep it
  cached for interactive contrast changes. Observed on the 93,555 × 138,996
  `uint16` file: first-frame latency ≈ 90 s and RAM oscillates between about
  44 GB and 90 GB (multiple ~24 GB copies during the render pipeline).
  Contrast auto-detection is no longer to blame — it now uses the CZI's
  embedded display settings for free — so only a multiscale pyramid can
  eliminate the ~24 GB spikes.
- **CZI reality — no assumptions about level count or ratio:**
  - Some CZIs store no pyramid at all (only zoom `1.0`).
  - Some store standard powers of two (`1.0, 0.5, 0.25, ...` — for example
    `DTScan_ID4.czi` has 5 such levels).
  - Others use application-specific factors. The 105 GB `Mouse Kidney` file
    stores only `[1.0, 0.333]`, so its coarsest stored level is still
    ~31k × 46k and will not fit in one GPU texture.
  - Reading a stored zoom via `pylibCZIrw.CziReader.read(zoom=z)` serves
    pixels directly from the matching subblocks (**cheap**). Requesting a
    zoom that is not stored triggers libCZI's C++ sampler
    (**not free** but still much cheaper than materializing layer 0 in
    Python and downsampling).
- **Detection is pylibCZIrw-only (no `czifile` needed):**
  ```python
  from pylibCZIrw import czi as pyczi

  zooms: set[float] = set()

  def _cb(idx, info) -> bool:
      zooms.add(round(float(info.get_zoom()), 6))
      return True

  with pyczi.open_czi(filepath) as doc:
      doc.enumerate_subblocks(_cb)
  # sorted(zooms, reverse=True) -> [1.0, 0.5, 0.25, ...]
  ```
  `CziReader.enumerate_subblocks` walks subblock headers only (no pixel I/O)
  and `SubBlockInfo.get_zoom()` returns the stored physical/logical ratio
  directly. Verified against `CellDivision`, `Tumor_HE_RGB`, `WellD6_S1`,
  `DTScan_ID4`, and the `Mouse Kidney` gigapixel file.
- **Action / design:**
  1. Add `get_pyramid_zooms(filepath) -> list[float]` in a small
     `pyramid.py` helper (sorted largest-first, `[1.0]` for files without a
     stored pyramid).
  2. Add `read_stacks_multiscale(filepath, ..., extra_coarse_edge=8192)`
     that returns a `list[array]` — one per pyramid level, same S/T/C/Z
     shape, progressively smaller Y/X. Each level is built by calling the
     existing `read_stacks(..., zoom=z, use_dask=True, use_xarray=...)` so
     spatial Y/X tiling from item 3b still applies per level and the plumbing
     stays in one place. Cache the parsed `CziMetadata` between level reads
     to avoid re-parsing XML.
  3. **GPU-safety synthesis.** After the stored levels, extend the list with
     synthetic coarser levels (each half the previous edge) until the
     coarsest edge is ≤ `extra_coarse_edge` (default 8192, safely under the
     typical 16k GPU texture limit). Each synthetic level is a call to
     `read_stacks(..., zoom=z_synth)` — libCZI resamples in C++ from the
     nearest stored level. The synthetic reads only happen when napari
     actually asks for those tiles, so the graph itself stays lazy.
  4. Return a helper `list[LevelInfo]` alongside the arrays with per-level
     zoom, physical pixel size, and shape so napari-czitools can pass a
     matching `scale` per level.
- **napari-czitools integration:**
  1. `ChannelLayer.sub_array` accepts either `xr.DataArray` (as today) or
     `list[xr.DataArray | dask.array.Array]` (multiscale). Add matching
     `scales: list[list[float]] | None` for per-level scale vectors.
  2. `CZIDataLoader.add_to_viewer` detects the list case and calls
     `viewer.add_image(sub_array_list, multiscale=True, scale=scales, ...)`
     while continuing to forward the pre-computed `contrast_limits`.
  3. Turn multiscale on automatically when the file has more than one
     detected level. Add an option to force it off (fallback to the current
     single-array path) for debugging or files where the pyramid is broken.
- **Where the change lives:**
  - Core detection + multiscale reader: `czitools`.
  - Consumer wiring (widget checkbox, `add_image(multiscale=True)`, per-level
    scale metadata): `napari-czitools`.
- **Expected impact on the 105 GB / 93k × 139k file:**
  - First-frame latency drops from ~90 s → ~1–3 s (napari renders the coarse
    level immediately).
  - Peak RAM drops from ~90 GB → baseline + a few hundred MB (the coarse
    level is a few MB; finer tiles stream only when the user zooms in).
  - Pan/zoom stays fluid because only visible tiles at the current zoom level
    are read on demand.
- **Tests to add:**
  - `get_pyramid_zooms` matches expected sets for the probed files above.
  - `read_stacks_multiscale` returns a list of length ≥ 1 with monotonically
    shrinking Y/X.
  - Pixel-equality between level 0 of the multiscale output and the current
    `read_stacks_stacked` result.
  - Synthetic-level path: when the coarsest stored level is larger than
    `extra_coarse_edge`, the returned list includes additional lazy levels
    with correct shapes.
- **Effort:** L · **Model tier:** `premium` (detection + graph construction
  are simple; the fiddly parts are non-uniform pyramid ratios, per-level
  `scale`, and integration with napari's multiscale renderer).

### 4. Split the monolithic `read_tools.py` (~1550 lines) `P1` `M`

- **Where:** [src/czitools/read_tools/read_tools.py](../src/czitools/read_tools/read_tools.py)
- **Problem:** one file holds `read_6darray`, `read_field`, `read_well`,
  `read_attachments`, `read_tiles`, `read_stacks*`, plus private delayed/chunk
  helpers. Hard to navigate, test, and extend (violates SRP/SoC).
- **Action:** split into a package while keeping `read_tools/__init__.py` as the
  stable public API (re-export unchanged, so imports don't break):
  - `_helpers.py` — `_as_int/_as_float`, coord/axis helpers, dim constants
  - `array6d.py` — `read_6darray`
  - `field_well.py` — `read_field`, `read_well`, HCS resolution
  - `attachments.py` — `read_attachments`
  - `tiles.py` — `read_tiles`
  - `stacks.py` — `read_stacks*` + `_read_plane_delayed`, `_read_plane_chunk`
- **Effort:** M (mechanical; guard with existing tests + `pylanceCheckSignatureCompatibility`) · **Model tier:** `standard` (mostly moves + re-exports; low reasoning, but verify imports/signatures)

### 5. Modernize typing to PEP 585 / 604 `P1` `M`

- **Where:** [read_tools.py](../src/czitools/read_tools/read_tools.py) L19
  (`from typing import Dict, Tuple, Optional, Union, List, ...`) and other modules.
- **Problem:** legacy `Dict`/`List`/`Optional`/`Union` conflict with the project's
  own convention (copilot-instructions: prefer `list[str]`, `dict[str,int]`,
  `X | None`).
- **Action:** migrate to builtin generics and `X | None`; promote the module-level
  aliases (`CziPath`, `Array6D`, …) to `typing.TypeAlias`. Automate with Ruff
  `UP` rules (see #8) + `pylanceInvokeRefactoring addTypeAnnotation`.
- **Effort:** M · **Model tier:** `cheap`→`standard` (Ruff `--fix` does most of it on `cheap`; escalate to `standard` only for aliases/edge cases)

---

## P2 — Medium

### 6. Reading-speed improvements `P2` `M`

- **Observations & actions:**
  - The eager loop already opens the CZI once (good). The lazy per-plane path
    (`_read_plane_delayed`) re-opens the file **per plane**; prefer the existing
    chunked reader (`_read_plane_chunk`, opens once per chunk) as the default and
    make plane-per-task opt-in.
  - Consider `da.map_blocks` over many tiny `da.from_delayed` tasks to cut
    scheduler overhead for large T/Z/C stacks.
  - Parallelism: reads are single-threaded per scene. Evaluate a threaded reader
    (pylibCZIrw releases the GIL during I/O) or dask threaded scheduler defaults.
  - Avoid the `tempfile`/`shutil` copy path where possible (URL/remote reads);
    confirm it isn't taken for local files.
  - Cache `CziMetadata` construction — several `*_required` properties re-parse.
- **Implemented decision:** lazy stack reads now use delayed multi-plane chunks
  (64 planes by default) and retain `lazy_read_strategy="plane"` for minimal
  random-access reads. This maps each task directly to one CZI open/close cycle
  while permitting a smaller final chunk; `da.map_blocks` adds no clear benefit
  for that variable final task. Both synchronous and threaded schedulers were
  benchmarked against the plane strategy with pixel-equivalent results.
- **Review outcome:** local paths use pylibCZIrw's standard reader directly and
  do not take the attachment copy path. The `CziMetadata.*_required` properties
  return components built during initialization and do not re-parse the file, so
  no shared cache was added; caching these mutable metadata objects across calls
  would make `adapt_metadata` and zoom-derived scaling leak between reads.
- **Effort:** M (measure first with `pylancePythonProfiling` on a representative CZI) · **Model tier:** `premium` (profiling-driven perf work; measure before/after)

### 7. Optional Zarr-backed lazy reading / caching `P2` `L`

- **Benefit is workload-dependent:** point #6 already makes a sequential CZI
  scan very efficient. A persistent Zarr cache is therefore not a general
  replacement for CZI reading; it is valuable when the same dataset is revisited
  through random planes, small spatial ROIs, visualization, or repeated ML
  epochs.
- **Measured result:** local warm-cache benchmark on
  `CellDivision_T10_Z15_CH2_DCV_small.czi` (300 planes, 39 MiB source), using
  pixel-equivalent Zstd-compressed, one-plane chunks:

  | Workload            | Optimized CZI | Zarr chunk I/O | Result                    |
  | ------------------- | ------------: | -------------: | ------------------------- |
  | Full 300-plane scan |       0.063 s |        0.068 s | Zarr was about 9% slower  |
  | 20 random planes    |       0.357 s |       0.0047 s | Zarr was about 76x faster |

  The cache read-and-write step took about 0.24 s after the common metadata
  setup, so it broke even after roughly 14 random plane reads. The compressed
  full-resolution chunks occupied about 21 MiB. Pixel checksums matched. These
  figures measure the underlying local chunk I/O and are not a guarantee for
  other CZI compression, chunk shapes, storage devices, or remote stores.
- **Current blocker:** with the installed `zarr 3.2.1`, opening existing local
  Zarr v2 and v3 arrays did not complete within 15 seconds, and prototype writes
  stalled for more than one minute. Investigate and resolve this backend/version
  behaviour before building an API around `da.from_zarr`; otherwise the actual
  integration could erase the measured chunk-I/O benefit.
- **Decision:** keep this as a conditional P2 item for interactive and repeated
  random/ROI workloads. Defer it for one-shot batch pipelines and repeated full
  scans, where it adds conversion time, storage duplication, and invalidation
  complexity without a measured speed benefit.
- **Action, after the blocker is resolved:**
  1. Establish a tested Zarr/Numcodecs/Dask version combination and benchmark
     real `da.from_zarr` reads and writes, including process restart and cold
     filesystem-cache cases.
  2. Prototype an explicit optional cache API (no transparent auto-conversion),
     reusing `export_tools` where appropriate and keeping the dependency behind
     `czitools[omezarr]`.
  3. Support configurable Z/spatial chunks and evaluate Zarr v3 sharding so
     random access does not create an excessive number of small files or remote
     requests.
  4. Add source fingerprinting/invalidation, atomic cache creation, metadata and
     dimension preservation, pixel-parity tests, and documented cleanup/overwrite
     behaviour.
  5. Accept the feature only if end-to-end benchmarks show a useful break-even
     point for the intended random/ROI workloads without regressing normal CZI
     reads or importing optional dependencies from the core package.
- **Effort:** L · **Model tier:** `premium` (new lazy API design + chunking strategy)

### 8. Expand Ruff beyond the release gate `P2` `S`

- **Where:** [pyproject.toml](../pyproject.toml) `[tool.ruff.lint] select = ["E4","E7","E9","F"]`
- **Action:** incrementally add `I` (import sort), `UP` (pyupgrade → supports #5),
  `B` (bugbear), `SIM`, `RUF`. Run `ruff check --fix` per module to keep diffs
  reviewable. The comment already anticipates this ("after 0.20.0").
- **Effort:** S (setup) + incremental cleanup · **Model tier:** `cheap` (rule config + `ruff --fix`; skim the diffs)

### 9. Consistent GitHub Actions versions `P2` `S`

- **Where:** [test_and_deploy.yml](../.github/workflows/test_and_deploy.yml) uses
  `checkout@v6`/`setup-python@v6`; [docs.yml](../.github/workflows/docs.yml) uses
  `checkout@v4`/`setup-python@v5`.
- **Action:** pin a single current major across all workflows; delete the stale
  `test_and_deploy_testpypi.yml.old`.
- **Effort:** S · **Model tier:** `cheap` (YAML pins + file delete)

---

## P3 — Polish / docs

### 10. Remove stale "purely experimental" disclaimers `P3` `S`

- Legacy headers (e.g. top of `read_tools.py`: *"This code is purely
  experimental. Use at your own risk."*) undercut a published, tested package.
  Replace with a short module docstring.
- **Model tier:** `cheap`

### 11. MkDocs nav & README review `P3` `S`

- **Where:** [mkdocs.yml](../mkdocs.yml) — `motivation.md` and `env_var.md` are
  commented out of `nav`; decide to include or delete the source files.
- **README:** verify the quick-start uses the current recommended lazy path
  (`read_stacks(..., use_dask=True, use_xarray=True)`) rather than
  `read_6darray(use_dask=True)`; confirm install extras (`[all]`, `[omezarr]`)
  and the STCZYX(A) dimension note are accurate.
- **Docs build** already runs `strict: true` — good; keep it.
- **Model tier:** `standard` (prose accuracy + verifying code snippets run)

### 12. Public API surface & `__all__` hygiene `P3` `S`

- Confirm each subpackage `__init__.py` exports a curated, documented API
  (`read_tools/__init__.py` looks good). Apply the same pattern to
  `metadata_tools` and `export_tools` so `mkdocstrings` autodoc stays clean.
- **Model tier:** `cheap`

---

## Suggested execution order

1. **#1** (fix CI gating) — unblocks safe iteration on everything else. `cheap`
2. **#2** (setuptools-scm) — simplifies every subsequent release. `standard`
3. **#5 + #8** (typing + Ruff `UP`) — do together; one automated sweep. `cheap`→`standard`
4. **#4** (split `read_tools.py`) — easier to review once linting is clean. `standard`
5. **#3** (true-lazy `read_6darray`) — build on the split modules. `premium`
6. **#6 / #7** (perf + Zarr) — measure, then optimize. `premium`
7. **#9–#12** (CI/doc polish) — batch at the end. `cheap` (except #11 `standard`)

### Model-tier summary

| Tier       | Items                                     | Rationale                                                                  |
| ---------- | ----------------------------------------- | -------------------------------------------------------------------------- |
| `cheap`    | #1, #8, #9, #10, #12 (+ first pass of #5) | Mechanical config/YAML/lint edits; low risk, `ruff --fix` does the work.   |
| `standard` | #2, #4, #11 (+ tail of #5)                | Focused multi-file code changes needing test/build verification.           |
| `premium`  | #3, #6, #7                                | Architectural + dask-lazy/perf reasoning where subtle mistakes are costly. |

> Rule of thumb: start each item on the lowest tier listed; escalate only if the
> change turns out to need deeper reasoning than expected.

---

*Generated review notes — update the checkboxes as items land.*

- [x] #1 CI test gating
- [x] #2 setuptools-scm versioning
- [x] #3 true-lazy read_6darray
- [x] #3b spatial Y/X tiling for very large planes
- [x] #3c multiscale pyramid reader for gigapixel display
- [x] #4 split read_tools.py
- [x] #5 modern typing
- [x] #6 reading speed
- [ ] #7 zarr-backed lazy
- [ ] #8 ruff expansion
- [ ] #9 actions versions
- [ ] #10 disclaimers
- [ ] #11 mkdocs/readme
- [ ] #12 API `__all__`
