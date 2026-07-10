# czitools HCS / Wellplate Metadata — Capability Review & Improvement Plan

> Source proposal: `_notes/HCS_Metadata_OME_CZI_Proposal_Balu.docx`
> Status legend: `[ ]` not started · `[~]` in progress · `[x]` done

---

## 1. Verification of `sample.py` findings

All findings and observations were verified against
[sample.py](../src/czitools/metadata_tools/sample.py) and are **correct**.

### "Partially covered" list — confirmed accurate

| Claim                                                     | Verified in code                                                | Correct? |
| --------------------------------------------------------- | --------------------------------------------------------------- | -------- |
| Well from `Scene.ArrayName`, fallback `Scene.Shape.Name`  | `get_well_info` checks `well.ArrayName`, else `well.Shape.Name` | yes      |
| Row/col from `Scene.Shape.RowIndex/ColumnIndex`           | `well.Shape.ColumnIndex/RowIndex` -> `well_colID/well_rowID`    | yes      |
| Each Scene ~ Field/FOV candidate                          | S dimension drives the loop; one entry per scene                | yes      |
| Scene->well grouping via `well_scene_indices`             | built from `get_scenes_for_well`                                | yes      |
| Multi-position per well via `multipos_per_well`           | derived from `Counter(self.well_counter.values())`              | yes      |
| Field XY from `Scene.CenterPosition`                      | `well.CenterPosition.split(",")`                                | yes      |
| Fallback image XY from planetable `X[micron]`/`Y[micron]` | fallback branch when `SizeS is None`                            | yes      |

### Observations — confirmed as real issues

1. **`well_total_number` counts entries, not unique wells** — `self.well_total_number = len(self.well_array_names)`. Unique wells would be `len(self.well_counter)`.
2. **Verbose-gated appends cause list-length skew** — real bug. When `verbose=False` and a field is missing, the fallback `.append(...)` for `well_indices`, `well_position_names`, `well_colID/rowID`, and the `CenterPosition is None` branch never executes (nested inside `if self.verbose:`). This desynchronizes the parallel lists.
3. **`scene_stageX/Y` is misnamed** — value comes from `Scene.CenterPosition` (scene/field center), not a verified stage position. True stage position lives in the planetable (`StageXPosition/StageYPosition`).
4. **`RegionId` missing** — not extracted anywhere; present in the sample XML and is the natural stable link between a semantic Field/FOV and a CZI acquisition region.

---

## 2. Existing wellplate capabilities

**Metadata (read):**
- [sample.py](../src/czitools/metadata_tools/sample.py) — `CziSampleInfo`: well names, indices, position names, row/col IDs, per-well scene grouping, multi-position detection, scene-center XY, planetable XY fallback.
- [scene.py](../src/czitools/metadata_tools/scene.py) — `CziScene`: per-scene pixel bounding box via `pylibCZIrw`.
- [dimension.py](../src/czitools/metadata_tools/dimension.py) — `CziDimensions`: `SizeS` and STCZYX(A) sizes, per-scene `SizeX_scene/SizeY_scene`.
- [boundingbox.py](../src/czitools/metadata_tools/boundingbox.py) — scene/total bounding boxes.
- [czi_metadata.py](../src/czitools/metadata_tools/czi_metadata.py) — aggregates all (`has_scenes`, `scene_shape_is_consistent`, `sample`).
- [planetable.py](../src/czitools/utils/planetable.py) — per-subblock DataFrame with `S, M, T, C, Z`, stage `X/Y/Z[micron]`, `Time[s]`, pixel bbox.

**Pixel data (read):** [read_tools.py](../src/czitools/read_tools/read_tools.py)
- `read_6darray` (STCZYX(A), dask/xarray, `planes` substack incl. `S`, `zoom`).
- `read_stacks` / `read_stacks_list` / `read_stacks_stacked` (lazy per-scene reading for inconsistent scene shapes).
- Single-scene reads allowed even when `scene_shape_is_consistent` is `False`.

**Summary:** czitools can enumerate wells, map scenes<->wells, detect multi-FOV wells, get per-scene geometry, and read pixels per scene/well. It lacks an explicit, OME-aligned Plate -> Well -> Field model with stable IDs and verbose-independent data.

---

## 3. Gap analysis vs. the OME/HCS proposal

| Proposal concept                                           | OME basis                | czitools today                     | Gap                                                        |
| ---------------------------------------------------------- | ------------------------ | ---------------------------------- | ---------------------------------------------------------- |
| **Experiment / Screen**                                    | `Screen`                 | none                               | Cross-plate/campaign context missing                       |
| **Plate** (rows, cols, naming, barcode, origin)            | `Plate`                  | none                               | No plate entity, no barcode/LIMS ID                        |
| **PlateAcquisition** (run, start/end, max fields)          | `PlateAcquisition`       | times in planetable only           | No acquisition-run entity                                  |
| **Well** (explicit ID, Row, Column, ExternalIdentifier)    | `Well`                   | inferred from ArrayName/Shape      | Identity inferred, not explicit; no external ID            |
| **Field / FOV / WellSample** (stable ID, well-local index) | `WellSample`             | Scene used implicitly              | No stable FieldId, no well-local FieldIndex, no `RegionId` |
| **Field position** (X/Y/Z + coord system)                  | `WellSample/PositionX/Y` | scene center + planetable stage XY | Z per field, coordinate-system origin unclear              |
| **Deskew / lightsheet geometry**                           | NGFF RFC-5 affine/shear  | none                               | No geometry, angle, shear axis, or affine matrix           |
| **OME-Zarr HCS export** (`plate/A/1/0`)                    | OME-NGFF                 | none                               | No export/mapping path                                     |

---

## 4. Improvement plan

### Stage 0 — Fix `sample.py` correctness (no new features)
- [ ] Move all fallback `.append(...)` calls out of `if self.verbose:` so lists stay index-aligned regardless of verbosity.
- [ ] Add `well_unique_number = len(self.well_counter)`; document `well_total_number` as "field/scene count" (or deprecate).
- [ ] Rename `scene_stageX/Y` -> `field_centerX/Y` (keep old names as deprecated aliases/properties).
- [ ] Extract `Scene.RegionId` into a new `well_regionIDs: List[str]` (aligned with the other per-scene lists).
- [ ] Add unit tests for a plate CZI with missing fields at `verbose=False` to lock in list-alignment.

### Stage 1 — Explicit, OME-aligned data model
- [ ] New module (e.g. `metadata_tools/hcs.py`) with `CziPlate` (rows, columns, naming convention, optional barcode/`ExternalIdentifier`, well origin XY).
- [ ] `CziWell` (id, name `A1`, `RowIndex`, `ColumnIndex` with documented 0-based vs 1-based semantics, external id, fields list).
- [ ] `CziField` (stable `FieldId`, well-local `FieldIndex`, global `SceneIndex`, `RegionId`, `PositionX/Y/Z` + unit + coordinate-system note).
- [ ] Build hierarchy by reshaping existing `CziSampleInfo` lists + planetable stage positions.
- [ ] Expose as `CziMetadata.hcs` (optional, populated only when `has_scenes`).

### Stage 2 — Position/coordinate clarity
- [ ] Populate `CziField.PositionZ` and true stage XY by joining scene metadata with the planetable (`X/Y/Z[micron]`).
- [ ] Document coordinate origin and relation to `WellOriginX/Y`; add plate-absolute <-> well-relative conversion helper.

### Stage 3 — Convenience reads keyed by plate/well/field
- [ ] `read_well(filepath, well="A1")` resolving to scene indices via the HCS model.
- [ ] `read_field(filepath, well="A1", field=0)` reusing existing single-scene reading.

### Stage 4 — Deskew / lattice-lightsheet geometry
- [ ] `CziLightSheetGeometry` (acquisition mode, shear axis, angle, Z step, pixel sizes).
- [ ] `DeskewTransform` (affine matrix) parsed from CZI metadata; modeled as a coordinate transform, not a boolean flag.

### Stage 5 — OME-Zarr / OME-NGFF export (adapt from `omezarr_playground`)

The local repository `omezarr_playground` already contains a working CZI → OME-Zarr exporter in `czi_omezarr_utils/`.
That code currently depends on czitools for:

- pixel reads: `czitools.read_tools.read_tools.read_6darray(..., use_xarray=True)`
- metadata: `CziMetadata` including `mdata.scale`, `mdata.channelinfo`, and **plate-ish** metadata via `mdata.sample.*`

Goal: **fully incorporate** the `omezarr_playground` conversion tools into **czitools** as an *optional* (extras-gated) feature set. The exporter should *prefer* the future explicit HCS model (`CziMetadata.hcs`, Stage 1) when it is available, but must **always** fall back to the current `CziSampleInfo` heuristics so conversion works even before/without the explicit model.

#### 5.1. Integration approach — vendor into czitools
- [ ] **Vendor** the relevant `czi_omezarr_utils` modules into czitools (single-package UX; this is the chosen approach).
- [ ] Define the new czitools namespace/package path (e.g. `czitools/export_tools/omezarr/` or `czitools/omezarr_tools/`).
- [ ] Preserve the public function names from `czi_omezarr_utils` so existing `omezarr_playground` scripts can migrate to czitools with minimal changes.
- [ ] After vendoring, update `omezarr_playground` to re-export from czitools (or depend on the new czitools export API) to avoid code duplication.

#### 5.2. Add optional dependencies (“extras”)
- [ ] Add a `pyproject.toml` optional-dependency group (e.g. `omezarr`) to avoid forcing these heavy packages on all users:
	- `ngff-zarr>=0.34.0`
	- `ome-zarr>=0.16.0`
	- `zarr>=3`
	- `ome-zarr-models`
	- plus any plotting/analysis deps only if czitools will ship the plotting helpers
- [ ] Implement a consistent error message when the export API is used without extras installed (raise `ImportError` with install hint).

#### 5.3. Port core export functions (from `omezarr_playground/czi_omezarr_utils/conversion.py`)
- [ ] `convert_czi2hcs_ngff(...)` (ngff-zarr backend, OME-NGFF v0.5).
- [ ] `convert_czi2hcs_omezarr(...)` (ome-zarr-py backend; note: writes OME-NGFF v0.4 in practice).
- [ ] `write_omezarr_ngff(...)` (single image multiscales).
- [ ] `write_omezarr(...)` (single image, ome-zarr-py).

#### 5.4. Port/merge supporting helpers
- [ ] `extract_well_coordinates(...)` — consider relocating to czitools metadata/HCS utilities, so both metadata and export share a single well-path normalization.
- [ ] `create_channel_list(...)` + `get_display(...)` — decide whether this belongs in czitools core (metadata-derived display settings) or stays export-specific.
- [ ] `get_fieldimage(...)` — decide whether to keep as an internal export helper (multiscales per field) or expose publicly.
- [ ] `validate_ome_zarr(...)` — add as an optional validation helper (depends on `ome-zarr-models`).
- [ ] `convert_hcs_omezarr2ozx(...)` — include the OZX conversion helper, including the Windows workaround described in `convert_czi2hcs_ngff`.

#### 5.5. Align exporter with czitools HCS model (Stage 1) — with `CziSampleInfo` fallback
- [ ] Add a resolver that yields the well/field layout (well IDs, row/column indices, per-well field list with `SceneIndex`, optional `RegionId`) from **either** source:
	- **Preferred:** the explicit HCS model (`CziMetadata.hcs`) when populated.
	- **Fallback (always available):** the current `CziSampleInfo` heuristics (`mdata.sample.well_counter`, `mdata.sample.well_scene_indices`), including the existing “strip leading zeros” well-ID normalization.
- [ ] Route both `convert_czi2hcs_ngff` and `convert_czi2hcs_omezarr` through this resolver so neither backend hard-codes a metadata source.
- [ ] Allow variable field counts per well (current exporter assumes a single `field_paths` length derived from the first well); the fallback path must handle this too.
- [ ] Keep behavior identical to today when only `CziSampleInfo` is available (no regressions for existing files).

#### 5.6. Performance & memory improvements for large plates
- [ ] Avoid reading full 6D arrays into memory for large plates:
	- Prefer `read_stacks(..., use_dask=True)` or scene-by-scene iteration.
	- Write each well/field incrementally to the target store.
- [ ] Decide chunking strategy per backend (ome-zarr-py chunk hints vs ngff-zarr shard/chunk settings).

#### 5.7. Tests and validation in czitools
- [ ] Add tests that run a minimal HCS conversion on an included test CZI (e.g. `data/WP96_4Pos_B4-10_DAPI.czi`) and validate the output:
	- Use `validate_ome_zarr(...)` when extras are installed.
	- Mark tests to skip automatically when export extras are absent.
- [ ] Add regression tests for:
	- padded vs non-padded well columns
	- variable fields per well
	- Windows `.ozx` workaround path

#### 5.8. User-facing API and docs
- [ ] Define a stable import path (e.g. `from czitools.export_tools import convert_czi2hcs_ngff`).
- [ ] Add a short usage section in docs demonstrating:
	- CZI → HCS OME-Zarr
	- CZI → standard (non-HCS) OME-Zarr
	- validation step

### Suggested ordering & risk
1. **Stage 0** (bug fixes, tests) — first, high value, low risk.
2. **Stage 1** (explicit model) — foundation for everything else.
3. **Stages 2–3** (positions + convenience reads) — high user value.
4. **Stages 4–5** (deskew, NGFF export) — larger scope, can follow.
