# Changelog

All notable changes to **czitools** are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) conventions,
and the project adheres to [Semantic Versioning](https://semver.org/).

---

## [0.24.0] — 2026-09-03

### Added

- The `czi_hcs_check` inspector now displays all full-resolution,
  subblock-derived dimension sizes and the first stored scene's Y/X size.
- HCS metadata can be filtered to physically stored global scene indices with
  `filter_hcs_to_stored_scenes=True`. The complete XML model remains available
  as `hcs_declared`, and `stored_scene_indices` exposes the layer-0 `S` keys.
- The HCS inspector now shows physically stored fields by default;
  `--show-declared` displays the complete XML-declared acquisition model.
- The converter GUI now offers a **Fast balanced** preset for non-HCS
  `ngff-zarr` exports. It uses Blosc compression,
  `(1, 1, 4, 1024, 1024)` TCZYX chunks, 2-by-2 spatial sharding, and Dask
  bin-shrink pyramids and is selected by default. The existing Gaussian path
  remains available as the **Quality** preset. Both directory-backed
  `.ome.zarr` and single-file `.ozx` output are supported.
- `write_omezarr_ngff()` now accepts a `downsampling_method` argument using
  `ngff_zarr.Methods`, allowing Python callers to select bin-shrink, Gaussian,
  or another method supported by ngff-zarr.
- Non-HCS `ngff-zarr` conversions now log an always-visible scale progress bar,
  periodic elapsed-time updates, and the total conversion time on completion.
- HCS conversions now report field progress and total time. Direct `.ozx`
  output also reports archive-finalization activity and removes incomplete
  archives when finalization is interrupted.
- NGFF HCS conversion now reuses pyramid levels stored in the CZI instead of
  regenerating every lower-resolution level with Gaussian downsampling. When
  no lower levels are stored, coarse levels use libCZI's native zoom reader.
- NGFF HCS conversion now uses larger spatial chunks, 4-by-4 Zarr v3 spatial
  sharding tuned for typical 2000-by-2000 fields, grouped CZI plane reads,
  preloaded metadata across fields and levels without deeply copying the full
  parsed metadata graph, and up to four concurrent field writes by default.
  Set `max_workers=1` for very large fields or deep T/Z stacks to minimize peak
  memory use.

### Fixed

- Dimension sizes now come from full-resolution CZI subblocks instead of
  numeric XML `Size*` declarations. Scene-shape checks support sparse global
  scene indices, as found in split-acquisition child files.
- Non-HCS `ngff-zarr` exports, for both directory-backed `.ome.zarr` stores
  and file-backed `.ozx` archives, now default to bounded TCZYX chunks of
  `(1, 1, 1, min(512, Y), min(512, X))` instead of compressing an entire
  C/Z/Y/X volume as one chunk. This prevents Blosc failures for chunks larger
  than its supported buffer size and keeps large-image writes memory-bounded.
  Explicit `chunks` values remain supported and are applied unchanged.
- Retrying a non-HCS `.ozx` export with `overwrite=True` now removes an
  existing or incomplete file-backed archive correctly. Directory-backed
  Zarr stores continue to be removed recursively.

## [0.23.1] — 2026-08-31

### Changed

- Raised the optional `ngff-zarr` dependency floor to 0.45.0 and migrated
  exports to its always-on zarrista backend.
- Removed the obsolete TensorStore dependency, writer argument, and converter
  GUI option. The independent ome-zarr-py backend and direct `zarr` dependency
  remain supported.

## [0.23.0] — 2026-08-28

### Added

- **OME-ZARR GUI improvements**: Unified single-file `.ozx` checkbox for both
  HCS and non-HCS conversions, automatic metadata loading, and improved Scene ID
  visibility controls.
- **HCS plate inspection tool** (`czi_hcs_check`): Provides rich-formatted
  terminal output for plate overview, well information, and field listings with
  optional filtering and table suppression.
- **ngff-zarr compressor compatibility**: Fixed codec object construction for
  zarrista layer compatibility, ensuring valid OZX archive generation.

### Changed

- OME-ZARR converter GUI now auto-loads metadata and shows HCS mode/Scene ID
  controls conditionally based on file content.
- Scene ID selector moves directly below the HCS checkbox for improved UX.
- Single `.ozx` output option replaces mode-specific timing controls.

### Fixed

- 3D rendering issues in napari viewer integration.
- Multiscale pyramid handling and level constraints.
- Non-pyramid bounds enforcement for regular 6D reads ensures compatible array
  shapes.
- ngff-zarr Blosc/Zstd compressor codec compatibility with zarrista layer.

## [0.22.1] — 2026-08-27

### Fixed

- `CziMicroscope` now handles missing or partial nested microscope metadata.
  Files whose `Instrument.Microscopes` node is absent now return `None` for
  microscope fields instead of raising `AttributeError`.
- `read_6darray` now constrains eager and lazy plane reads to scene-aware,
  non-pyramid layer-0 bounding rectangles. This prevents rounded coarse-pyramid
  coverage from returning padded planes that cannot fit the regular STCZYX(A)
  array shape.

### Changed

- Simplified legacy and archived code while retaining the supported read,
  metadata, export, and GUI APIs.

## [0.22.0] — 2026-08-25

### Added

- `czi_hcs_check` — New command-line tool for inspecting High-Content Screening
  (HCS) well plates in CZI files. Provides rich-formatted terminal output showing
  plate overview, well information, sample metadata, and detailed field listings.
  Supports filtering by well name and optional suppression of the well summary
  table.
- `czitools.utils.hcs_tools` — Module providing rich-formatted display utilities
  for HCS plate metadata. Includes functions for displaying plate information,
  sample metadata, and well field details with colored terminal output.

### Changed

- Scene shape tolerance improved when dimensions differ by only a few pixels
  (e.g., rounding errors across scenes). The reader now handles minor dimension
  mismatches gracefully.

### Fixed

- Resolved circular import between `hcs_tools` and `metadata_tools` modules
  using Python's standard `TYPE_CHECKING` pattern (PEP 484/563).
- Fixed pyramid level enumeration to correctly identify all stored zoom levels.
- Fixed all static analysis violations (ruff F821, F541) for enhanced code quality.

See [_release_notes/v0.22.0.md](_release_notes/v0.22.0.md) for detailed release notes.

---

## [0.21.0] — 2026-08-24

### Added

- `czitools.read_tools.get_pyramid_zooms(filepath)` — enumerates a CZI's
  stored pyramid levels via `pylibCZIrw.CziReader.enumerate_subblocks` and
  `SubBlockInfo.get_zoom()`. Runs in ~20 ms even on gigapixel files with
  thousands of subblocks. Uses only `pylibCZIrw`; no `czifile` dependency.
- `czitools.read_tools.read_stacks_multiscale(filepath, ...)` — returns one
  lazy Dask array per pyramid level in the shape napari's
  `add_image(..., multiscale=True)` accepts. Detected on-disk levels are
  served directly from their subblocks. If the coarsest stored level
  exceeds `max_coarse_edge` (default 8192 px), additional coarser levels
  are synthesised via libCZI's C++ resampler so the top of the pyramid
  always fits in one GPU texture.
- `czitools.read_tools.PyramidLevel` — dataclass exposing per-level
  `zoom`, `stored`, `y`, `x`.
- `read_stacks(..., tile_size=4096)` — new parameter for spatial Y/X
  tiling of gigapixel planes. Automatically activated when a single 2D
  plane exceeds `chunk_memory_limit` (default 256 MB). Each Dask chunk
  becomes one ROI-based read via
  `pylibCZIrw.CziReader.read(roi=(x, y, w, h))`, so viewers only fetch
  the tiles that intersect the current viewport instead of full planes.

### Changed

- The monolithic `src/czitools/read_tools/read_tools.py` (~1550 lines) is
  split into focused modules: `_helpers.py`, `array6d.py`, `stacks.py`,
  `field_well.py`, `attachments.py`, `tiles.py`, and the new
  `pyramid.py`. The old `read_tools.py` remains as a backward-compat
  facade that re-exports every public function, so existing imports like
  `from czitools.read_tools import read_tools; read_tools.read_stacks(...)`
  continue to work unchanged.
- `read_stacks` no longer materialises a full plane just to probe dtype
  and shape. It now issues a 1×1 ROI probe read and derives spatial
  extents from `stack_rect × zoom`. This alone drops multiscale
  construction on a 93,555 × 138,996 file from ~164 s to ~2.5 s.
- Typing modernised to PEP 604 (`X | None`) and PEP 585 builtin generics
  in the split modules.
- README, `docs/usage.md`, and `_notes/MODERNIZATION_TODO.md` describe
  the new tiling and multiscale reading APIs, including coordinate-space
  and rounding caveats.

### Fixed

- `read_stacks` now uses `int()` truncation (matching libCZI's
  convention) when predicting the shape of a zoomed read. `round()`
  overshoots by one pixel for zooms like `0.037037` and caused
  `ValueError: could not broadcast input array from shape (X-1, Y-1)
  into shape (X, Y)` at chunk assembly.
- The tiled reader now expresses each ROI in **layer-0 native
  coordinates** and computes the declared Dask chunk shape with the same
  `int(roi × zoom)` formula libCZI uses. Previously the tile grid was
  laid out in zoomed coordinates, which silently produced wrong pixels
  at any non-1.0 zoom.
- A single `_plan_tile_grid` helper is now the source of truth for the
  per-row and per-column tile sizes. Both the Dask graph and the outer
  coordinate arrays consume this plan, eliminating the
  `xarray.CoordinateValidationError: conflicting sizes for dimension
  'Y'` that occurred at zooms like `1/3` where the sum of per-tile
  truncated sizes differs from the truncation of the total size.
- The napari-czitools plugin (companion release) now forwards
  `use_dask=True` whenever the "Lazy Loading" checkbox is checked and
  passes an explicit `contrast_limits` (from the CZI's embedded display
  settings) so napari does not auto-scan the entire Dask array to derive
  the display range.

See [_release_notes/v0.21.0.md](_release_notes/v0.21.0.md) for detailed release notes.

---

## [0.20.1] — 2026-08-02

### Added

- Regression coverage for nested HCS export directories, local ngff-zarr
  compression handling, and lazy top-level package imports.
- A consolidated usage notebook and a napari OME-Zarr GUI demo script.

### Changed

- Top-level `czitools` subpackages are imported lazily to reduce startup cost
  and avoid loading optional visualization dependencies unnecessarily.
- Napari display helpers now return the viewer and support non-blocking notebook
  use through `run=False`.
- The Pixi workspace now enforces a PyQt6 backend per platform and includes the
  corrected `cmap`, `tensorstore`, NumPy, and validators dependency declarations.
- README, usage guidance, Copilot instructions, demos, and notebooks were
  refreshed for the current metadata, reader, HCS, and OME-Zarr APIs.

### Fixed

- HCS conversion now creates parent directories before opening its log file,
  allowing exports to new nested output directories.
- ngff-zarr compression is passed through the writer's `compressor` argument
  instead of being misrouted as an FSSpec storage option.
- The converter GUI selects PyQt6 before importing QtPy and MagicGUI.

See [_release_notes/v0.20.1.md](_release_notes/v0.20.1.md) for detailed release notes.

---

## [0.20.0] - 2026-07-30

### Added

- Immutable CZI HCS Plate → Well → Field metadata model with explicit detection status.
- Well- and field-based pixel readers.
- True lazy, scene-wise Dask readers with stable list and stacked return contracts.
- OME-Zarr HCS conversion through ngff-zarr and ome-zarr-py backends.
- OME-Zarr validation, converter GUI, HCS analysis, and plate heatmap helpers.
- Well-plate metadata, position-enrichment, conversion, and analysis examples.

### Changed

- Expanded CZI sample metadata with lossless missing-value handling.
- Added support for unequal scene and field shapes in scene-wise reads and HCS conversion.
- Split optional OME-Zarr, GUI, analysis, documentation, and full dependency groups.
- Updated installation, usage, and README documentation for current reader and HCS behavior.
- Updated the Pixi environment to use compatible `imagecodecs` and `czifile` releases.

### Fixed

- TestPyPI deployment is now restricted to `testpypi-v*` tags.
- Dask, Requests, Zarr, and Matplotlib are declared as core dependencies
  because they are imported through the public package path.
- Correctness-focused Ruff findings, including undefined type names and stale imports.

See [_release_notes/v0.20.0.md](_release_notes/v0.20.0.md) for detailed release notes.

---

## [0.17.2] — 2026-05-22

### Fixed

- Converted all docstrings across the entire `src/` tree to Google-style format, fixing
  incorrect rendering with `mkdocstrings` (`docstring_style: "google"` in `mkdocs.yml`).
- Fixed 15 files across `metadata_tools/`, `read_tools/`, `utils/`, `visu_tools/`, and `_tests/`:
  removed NumPy-style dashed underlines, replaced `Parameters:` with `Args:`, removed
  non-standard `Methods:` sections, fixed malformed `Attributes:` headers, and added
  missing blank lines between docstring sections.

### Changed

- `.github/copilot-instructions.md`: expanded Docstrings section with comprehensive
  Google-style rules, function and dataclass examples, and explicit anti-patterns.
- Version bumped to `0.17.2` in `pyproject.toml` and `src/czitools/__init__.py`.

---

## [0.17.1] — 2026-05-18

### Fixed

- `get_planetable()` now skips pyramid (reduced-resolution) subblocks when iterating CZI
  subblocks, preventing spurious rows and incorrect tile counters for mosaic/tiled CZI files.

### Changed

- Demo script `read_planetable.py`: updated default `zplane` selection from `3` to `0`.
- Version bumped to `0.17.1` in `pyproject.toml` and `src/czitools/__init__.py`.

### Added

- `data/testwell96_small.czi` — additional test CZI file covering the fixed planetable code path.

---

## [0.17.0] — 2026-05

### Highlights

- **Refactored public API** — explicit, stable public symbols exported from all sub-package
  `__init__.py` files; internal helpers prefixed with `_`.
- **Circular-import fix** — resolved startup `ImportError` caused by a circular dependency
  between `utils.misc` and `metadata_tools.czi_metadata`.
- **`calc_scaling` zarr fix** — zarr arrays are wrapped in a dask view before reduction so
  min/max are computed chunk-by-chunk without loading the full array into memory.

### Added

- Explicit `__all__` exports in `metadata_tools`, `read_tools`, `utils`, and `visu_tools` packages.
- `src/czitools/__init__.py` re-exports `visu_tools` helpers and `napari_helpers` at the top level.

### Changed

- `utils/misc.py`: fixed `calc_scaling` type hint and zarr reduction path; deferred circular
  import in `md2dataframe`.
- `metadata_tools/czi_metadata.py`: internal helpers renamed with `_` prefix.
- `read_tools/read_tools.py`: internal helpers renamed with `_` prefix.
- `utils/planetable.py`: minor refactor and clarified docstrings.
- `visu_tools/vis_tools.py`: CI fix for optional import guard.
- Demo notebooks and scripts refreshed to use the new public API.

### Fixed

- `ImportError: cannot import name 'CziAddMetaData'` on package startup (circular import).
- `AttributeError: 'Array' object has no attribute 'min'` in `calc_scaling` for `zarr.Array`.
- `KeyError` in test suite caused by stale internal function references after rename.

---

## [0.16.0]

### Highlights

- **Dropped Python 3.11 support** — `czifile >= 2026` requires Python 3.12+.
- Removed all old-`czifile` API compatibility shims.
- Added automated GitHub Actions workflow for building and publishing docs to GitHub Pages.
- Removed `aicspylibczi` dependency; all pixel reading now uses `pylibCZIrw` and `czifile`.

### Added

- `.github/workflows/docs.yml` — CI workflow deploying MkDocs docs via `mike` to GitHub Pages.
- `docs/install.md` and `docs/usage.md` — full installation and usage guides.
- `src/czitools/utils/threading_helpers.py` and `utils/napari_helpers.py`.
- `.github/copilot-instructions.md` — Copilot coding guidelines.

### Changed

- `pyproject.toml`: `requires-python` bumped to `>=3.12,<3.14`.
- `utils/misc.py`: simplified dimension helpers; dropped all `dimension_entries` fallback paths.
- `read_tools/read_tools.py`: `read_tiles()` uses `de.dims`, `de.is_pyramid`, `de.stored_shape` directly.
- `utils/planetable.py`: `sb.metadata()` treated as XML string only; removed dict-metadata code path.
- MkDocs configuration updated to `google` docstring style with richer API reference output.

### Removed

- Python 3.11 support.
- Old-`czifile` API compatibility code.
- `_getsbinfo_from_dict` helper in `planetable.py`.
- `aicspylibczi` dependency.

---

## [0.15.0]

### Highlights

- Enhanced `read_stacks(...)` with explicit metadata return and improved stacking/chunk behavior.
- Added clearer typed scene-reading wrappers for stable return contracts.
- Aligned tests with current APIs and improved zoom/lazy-read coverage.

### Added

- `read_stacks_list(...)` and `read_stacks_stacked(...)` in `read_tools.py`.
- `demo/scripts/read_bioio_dask_compare.py`.

### Changed

- `read_stacks(...)` now returns `(result, dims, num_stacks, mdata)`.
- Tests updated for current `read_stacks(...)` and `read_6darray(..., use_dask=True)` usage.
- README and Copilot instructions updated.

### Removed

- `demo/scripts/read_lazy_demo.py` and `diagnose_czi.py`.

---

## [0.14.0]

### Highlights

- Added NDV utility helpers for LUT and scale generation from CZI metadata.
- Added typed stack-reading wrappers for clearer API contracts.
- Improved Napari display helpers for channel-label handling.

### Added

- `src/czitools/utils/ndv_tools.py`: `normalize_luts()`, `create_luts_ndv()`, `create_scales_ndv()`.
- `read_stacks_list(...)` and `read_stacks_stacked(...)` in `read_tools.py`.
- `src/czitools/_tests/test_ndv_tools.py`.

### Changed

- `read_stacks(...)` typing and ergonomics improved.
- `utils/napari_tools.py`: robust channel selection; `display_xarray_list_in_napari()` added.
- `_tests/test_url_metadata.py`: retries candidate URLs; skips on transient SSL/network failures.
- Demo notebooks and scripts updated.

---

## [0.13.2] — 2026-01-26

### Highlights

- **Critical fix for Linux + Napari threading crashes** — `aicspylibczi` caused crashes when
  used concurrently with Napari (PyQt) on Linux.

### Added

- `CZITOOLS_DISABLE_AICSPYLIBCZI` environment variable (safe mode) to disable the problematic library.
- Global `RLock` protection in `read_tiles()` and `get_planetable()`.
- Platform detection with automatic user warnings on Linux.
- `src/czitools/utils/napari_helpers.py`: `enable_napari_safe_mode()`, `is_napari_safe_mode()`,
  `check_napari_compatibility()`, `get_recommended_read_params()`, `warn_if_unsafe_for_napari()`.
- `src/czitools/utils/threading_helpers.py`: `with_aics_lock()`, `is_napari_safe()`.
- `src/czitools/_tests/test_napari_safe_mode.py` — 7 new tests.

### Changed

- `read_tools/read_tools.py`, `utils/planetable.py`, `metadata_tools/czi_metadata.py`:
  safe mode + thread-lock support added.
- README updated with a prominent Linux/Napari warning section.

---

## [0.13.0] — 2025-12-12

### Highlights

- `read_stacks`: added `planes` support, `chunk_policy` option, and safe `chunk_memory_limit`
  heuristic; fixed Dask reshape issues by building dask arrays via recursive stacking.
- `read_6darray`: no longer mutates caller-provided `planes` dicts.
- Added `get_pyczi_readertype()` integration for URL vs local file reading paths.
- New utility `display_xarray_in_napari()` to centralize Napari display logic.

### Added

- `read_stacks(...)`: new `chunk_policy` and `chunk_memory_limit` arguments.
- `utils/napari_tools.py`: `display_xarray_in_napari()`.
- `_tests/test_read_6darray_no_mutation.py`.

### Changed

- `read_6darray(...)`: normalized `planes` is now available via `attrs['subset_planes']` on the
  returned xarray; input dict is no longer mutated.
- `read_stacks(...)`: improved handling for CZI files without explicit scenes.
- Demo scripts and notebooks updated.

---

[0.20.0]: https://github.com/sebi06/czitools/compare/v0.17.2...v0.20.0
[0.17.2]: https://github.com/sebi06/czitools/releases/tag/v0.17.2
[0.17.1]: https://github.com/sebi06/czitools/releases/tag/v0.17.1
[0.17.0]: https://github.com/sebi06/czitools/releases/tag/v0.17.0
[0.16.0]: https://github.com/sebi06/czitools/releases/tag/v0.16.0
[0.15.0]: https://github.com/sebi06/czitools/releases/tag/v0.15.0
[0.14.0]: https://github.com/sebi06/czitools/releases/tag/v0.14.0
[0.13.2]: https://github.com/sebi06/czitools/releases/tag/v0.13.2
[0.13.0]: https://github.com/sebi06/czitools/releases/tag/v0.13.0
