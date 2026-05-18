# Changelog

All notable changes to **czitools** are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) conventions,
and the project adheres to [Semantic Versioning](https://semver.org/).

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

[0.17.1]: https://github.com/sebi06/czitools/releases/tag/v0.17.1
[0.17.0]: https://github.com/sebi06/czitools/releases/tag/v0.17.0
[0.16.0]: https://github.com/sebi06/czitools/releases/tag/v0.16.0
[0.15.0]: https://github.com/sebi06/czitools/releases/tag/v0.15.0
[0.14.0]: https://github.com/sebi06/czitools/releases/tag/v0.14.0
[0.13.2]: https://github.com/sebi06/czitools/releases/tag/v0.13.2
[0.13.0]: https://github.com/sebi06/czitools/releases/tag/v0.13.0
