# czitools HCS / Wellplate Metadata — Capability Review & Improvement Plan

> Source proposal: `_notes/HCS_Metadata_OME_CZI_Proposal_Balu.docx` (not in the repo anymore)
> Status legend: `[ ]` not started · `[~]` in progress · `[x]` done
> Reviewed against the repository on 2026-07-10. Changes made during the review
> are explained in **Review decisions** below.

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
3. **`scene_stageX/Y` is imprecisely named** — the value comes from `Scene.CenterPosition` and is a physical scene/field center in the CZI metadata, while the planetable contains per-subblock `StageXPosition/StageYPosition`. These sources must not be treated as interchangeable without validation.
4. **`RegionId` missing** — not extracted anywhere; present in the sample XML and is a useful source-scoped link between a semantic Field/FOV and a CZI acquisition region. It must not be advertised as globally stable without evidence.

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

**Summary:** czitools can enumerate wells, map scenes<->wells, detect multi-FOV wells, get per-scene geometry, and read pixels per scene. It lacks an explicit, OME-aligned Plate -> Well -> Field model with source-scoped IDs, declared coordinate semantics, and verbose-independent data. There is not yet a public read-by-well API.

---

## 3. Review decisions (2026-07-10)

The overall sequence makes sense: repair the current extraction first, introduce one
canonical HCS model, then build reads and export on that model. The following changes
are needed before implementation:

1. **Separate OME-XML concepts from OME-NGFF requirements.** `Screen`,
   `PlateAcquisition`, and `WellSample` are OME-XML concepts; an OME-NGFF HCS plate
   uses plate/well/image groups and does not require the same object graph. The core
   model should preserve useful acquisition semantics without claiming one-to-one
   compliance with both specifications.
2. **Do not invent plate dimensions or identifiers.** A CZI may contain only a subset
   of a plate. Store declared row/column counts and barcode only when metadata supplies
   them; keep observed row/column extents as separate derived values. The sample CZI
   uses 1-based `Shape.RowIndex/ColumnIndex` (`B4` is `2,4`), whereas OME/NGFF array
   indices are 0-based, so both source and normalized indices must be explicit.
3. **Use missing values, not fabricated zeroes.** `0`, `0.0`, `1`, and `"P1"` can be
   valid values. New structures should use `None` for absent metadata. Legacy lists may
   retain compatibility defaults temporarily, but they must remain aligned and expose
   validity/provenance.
4. **Treat positions as sourced measurements.** `Scene.CenterPosition` is a physical
   scene center; planetable positions are per-subblock stage/focus values and can vary
   by tile, Z, or time. Stage 2 must define an aggregation policy and detect conflicts,
   rather than blindly joining one arbitrary row. OME `WellSample` directly models X/Y;
   Z should be retained as CZI acquisition metadata unless the chosen target schema
   supports it.
5. **Keep one canonical layout resolver.** Reads and both exporters should consume the
   same normalized layout. A fallback to `CziSampleInfo` is appropriate only when it
   yields an unambiguous well mapping; otherwise HCS export should fail clearly instead
   of manufacturing a plate.
6. **Add a dependency/API compatibility spike.** `pyproject.toml` already includes
   `ngff-zarr` variants in the `all` extra, while the playground is currently a set of
   scripts (`scripts_and_notebooks/ome_zarr_utils.py`), not the proposed
   `czi_omezarr_utils` package. Backend versions, Zarr v2/v3 support, output NGFF
   versions, and validation support must be verified before pinning dependencies or
   promising preserved public names.
7. **Deskew is independent and metadata-dependent.** It is not a prerequisite for HCS
   metadata or export. Implement it only after representative light-sheet metadata and
   expected transforms are available as fixtures.

---

## 4. Gap analysis vs. the OME/HCS proposal

| Proposal concept                                             | OME basis                | czitools today                     | Gap                                                        |
| ------------------------------------------------------------ | ------------------------ | ---------------------------------- | ---------------------------------------------------------- |
| **Experiment / Screen**                                      | `Screen`                 | none                               | Cross-plate/campaign context missing                       |
| **Plate** (rows, cols, naming, barcode, origin)              | `Plate`                  | none                               | No plate entity; declared values may be absent             |
| **PlateAcquisition** (run, start/end, max fields)            | `PlateAcquisition`       | times in planetable only           | No acquisition-run entity                                  |
| **Well** (explicit ID, Row, Column, ExternalIdentifier)      | `Well`                   | inferred from ArrayName/Shape      | Source indices appear 1-based; normalized identity absent  |
| **Field / FOV / WellSample** (source-scoped ID, local index) | `WellSample`             | Scene used implicitly              | No local FieldIndex or extracted `RegionId`                |
| **Field position** (X/Y + coord system; optional source Z)   | `WellSample/PositionX/Y` | scene center + subblock positions  | Provenance, aggregation, units, and origin unclear         |
| **Deskew / lightsheet geometry**                             | NGFF RFC-5 affine/shear  | none                               | No geometry, angle, shear axis, or affine matrix           |
| **OME-Zarr HCS export** (`plate/A/1/0`)                      | OME-NGFF                 | none                               | No export/mapping path                                     |

---

## 5. Improvement plan

### Stage 0 — Fix `sample.py` correctness (no new features)

- [x] Move all fallback `.append(...)` calls out of `if self.verbose:` so lists stay index-aligned regardless of verbosity.
- [x] Add `well_unique_number = len(self.well_counter)`; introduce `scene_count`; deprecate the ambiguous `well_total_number` while preserving it for compatibility.
- [x] Add `field_centerX/Y` as the preferred names, retaining `scene_stageX/Y` as deprecated aliases. Record units and source (`Scene.CenterPosition`).
- [x] Extract `Scene.RegionId` into `well_region_ids: List[Optional[str]]` (aligned with the other per-scene lists); keep it as a string to avoid numeric-ID assumptions.
- [x] Use `None` in new/normalized fields for missing values. If legacy lists retain defaults, add a documented migration path rather than silently changing their types.
- [x] Compute counters and `multipos_per_well` once after all scenes are parsed, not after every append.
- [x] Add XML/`Box` unit fixtures for missing and malformed fields at both verbosity settings, plus an integration test using the included wellplate CZI. Assert equal per-scene list lengths and correct `RegionId` extraction.

**Stage 0 implementation notes (2026-07-10):**

- `field_centerX/Y` are the lossless fields: missing or malformed centers are
  `None`, their unit is micrometers, and their source is `Scene.CenterPosition`.
  Deprecated `scene_stageX/Y` compatibility properties still expose missing values
  as `0.0`, matching the previous API. This deliberately avoids changing existing
  callers while giving new code an unambiguous missing-value representation.
- Existing `well_indices`, position-name, and row/column lists retain their historical
  defaults (`1`, `"P1"`, and `0`) for compatibility, but appends are now independent
  of logging verbosity. Stage 1 must use normalized optional values rather than copy
  those legacy sentinels into the HCS model.
- A scene without a usable well name receives an empty legacy name so every per-scene
  list remains aligned; empty names are excluded from well counts and grouping.
- `well_total_number` remains readable and now consistently mirrors `scene_count`.
  New code should use `scene_count` or `well_unique_number`, depending on intent.
- Tests in `test_sample.py` cover both verbosity modes, missing/malformed values,
  compatibility properties, aggregate calculation, string-preserved `RegionId`, and
  the included `WP96_4Pos_B4-10_DAPI.czi` fixture (28 scenes across 7 wells).

### Stage 1 — Explicit, OME-aligned data model

- [x] New module (e.g. `metadata_tools/hcs.py`) with typed immutable/value-like models and explicit schema/version documentation. Avoid naming it fully OME-compliant unless serialization tests prove that claim.
- [x] `CziPlate`: optional declared rows/columns, naming convention, barcode/external identifier, plus separately derived observed row/column extents. Do not infer full plate size from occupied wells.
- [x] `CziWell`: canonical name/path, original source name, source row/column indices, normalized 0-based row/column indices, optional external ID, and fields.
- [x] `CziField`: deterministic source-scoped ID, well-local 0-based `field_index`, global 0-based `scene_index`, optional string `region_id`, scene-center X/Y with unit/provenance, and optional acquisition Z metadata.
- [x] Define deterministic ID and duplicate-handling rules (for example, source identity + `RegionId`, falling back to scene index); validate duplicate wells, regions, and scene indices.
- [x] Build the base hierarchy from scene XML only. Enrich it from the planetable in Stage 2, rather than making construction depend on an expensive full subblock scan.
- [x] Expose as `CziMetadata.hcs` when an unambiguous HCS mapping exists; add a detection/result status explaining why it is absent. `has_scenes` alone is insufficient because non-plate CZIs also have scenes.

**Stage 1 implementation notes (2026-07-10):**

- `metadata_tools/hcs.py` defines frozen `CziPlate`, `CziWell`, `CziField`, and
  `CziHcsResult` value models under internal schema version `1.0`. They are explicitly
  OME/NGFF-oriented domain models, not OME-XML or NGFF serializers.
- `build_hcs_metadata()` reads scene and plate-template XML directly. It does not use
  planetable data or copy Stage 0's legacy sentinel values. Template `ShapeRows` and
  `ShapeColumns` become optional declared dimensions; observed 0-based row and column
  index sets are stored separately. Barcode and external IDs remain `None` when the CZI
  does not provide a trustworthy value.
- CZI `Shape.RowIndex/ColumnIndex` are retained as 1-based source indices and converted
  to normalized 0-based indices. Well names are canonicalized (`B04` -> `B4`) and must
  agree with the source indices. NGFF-style paths use `row/column`, for example `B/4`.
- Field order is deterministic by global scene index. A field ID uses `field:<RegionId>`
  when available and falls back to `scene:<scene_index>`; these IDs are source-scoped,
  not globally persistent identifiers. Missing/malformed scene centers remain `None`,
  with unit and XML provenance recorded on each field. Acquisition Z stays `None` until
  the Stage 2 enrichment policy exists.
- HCS detection rejects missing/invalid well names, non-positive or conflicting source
  indices, duplicate scene indices, duplicate non-empty `RegionId` values, and wells
  outside declared template bounds. `CziMetadata.hcs_status` always contains the result
  and explanation; `CziMetadata.hcs` contains a plate only for a valid mapping.
- The public metadata package exports the four models, builder, and well-name normalizer.
  Tests in `test_hcs.py` cover normalization, immutability, deterministic IDs, missing
  positions, rejection paths, the included 96-well CZI (7 wells/28 fields), and a
  multi-scene non-plate rejection.

### Stage 2 — Position/coordinate clarity

- [ ] Join by explicit `S`/scene index and group all matching subblocks. Define representative-value and tolerance rules across `M/T/C/Z`; preserve ranges or report conflicts instead of selecting the first row silently.
- [ ] Keep `Scene.CenterPosition`, subblock stage X/Y, and focus Z as separate sourced values until their relationship is verified on fixtures.
- [ ] Document unit, axis direction, coordinate origin, and source for every coordinate. Add plate-absolute <-> well-relative conversion only when a trustworthy well origin/center is present; otherwise return an unavailable result.
- [ ] Make planetable enrichment lazy/opt-in because it scans subblocks and is unavailable for URL sources.

### Stage 3 — Convenience reads keyed by plate/well/field

- [ ] First add pure resolvers (`resolve_well`, `resolve_field`) with normalization, duplicate, and out-of-range tests.
- [ ] `read_field(filepath, well="A1", field=0)` should reuse the existing single-scene path and define whether `field` is the well-local index or a `RegionId`.
- [ ] `read_well(filepath, well="A1")` should return a list/mapping when field shapes differ; stack only on explicit request after validating compatible shapes.

### Stage 4 — Deskew / lattice-lightsheet geometry

- [ ] Gate this stage on acquiring representative CZI metadata fixtures and a known-good transformed result.
- [ ] `CziLightSheetGeometry` (acquisition mode, shear axis, angle, Z step, pixel sizes).
- [ ] `DeskewTransform` (affine matrix) derived from documented metadata; modeled as a coordinate transform with axis/order/unit conventions, not a boolean flag.

### Stage 5 — OME-Zarr / OME-NGFF export (adapt from `omezarr_playground`)

The local repository `omezarr_playground` contains prototype CZI → OME-Zarr scripts, primarily
`scripts_and_notebooks/ome_zarr_utils.py`. Treat them as migration input, not yet as a packaged/stable API.
That code currently depends on czitools for:

- pixel reads: `czitools.read_tools.read_tools.read_6darray(..., use_xarray=True)`
- metadata: `CziMetadata` including `mdata.scale`, `mdata.channelinfo`, and **plate-ish** metadata via `mdata.sample.*`

Goal: incorporate the reusable conversion core into **czitools** as an optional feature. HCS export should use the canonical HCS layout resolver. The resolver may adapt legacy `CziSampleInfo` data when the mapping is complete and unambiguous; otherwise it must return a clear validation error.

#### 5.0. Compatibility and scope spike (must precede vendoring)

- [ ] Inventory prototype functions, licenses, tests, private imports, GUI/plotting dependencies, and actual callers.
- [ ] Test a small matrix of `ngff-zarr`, `ome-zarr`, `zarr` v2/v3, Python 3.12/3.13, and validators; record the NGFF version each backend really writes.
- [ ] Choose one primary writer backend. Keep a second backend only if it provides a tested capability the primary one lacks.
- [ ] Define the supported output contract (NGFF version, array axes, chunks, labels, overwrite/resume behavior) before declaring stable public functions.

#### 5.1. Integration approach — vendor into czitools

- [ ] **Vendor** the relevant `czi_omezarr_utils` modules into czitools (single-package UX; this is the chosen approach).
- [ ] Define the new czitools namespace/package path (e.g. `czitools/export_tools/omezarr/` or `czitools/omezarr_tools/`).
- [ ] Preserve prototype function names through temporary compatibility wrappers only where callers exist; define a smaller stable czitools API.
- [ ] After vendoring, update `omezarr_playground` to re-export from czitools (or depend on the new czitools export API) to avoid code duplication.

#### 5.2. Add optional dependencies (“extras”)

- [ ] Add a focused `omezarr` optional-dependency group. Reconcile it with the existing `all` extra, which already contains unpinned `ngff-zarr` variants.
- [ ] Pin compatible ranges only after the 5.0 spike; do not assume `zarr>=3` is compatible with both writers.
- [ ] Keep plotting/GUI/analysis dependencies out of the export extra unless those helpers are intentionally shipped and tested.
- [ ] Implement a consistent error message when the export API is used without extras installed (raise `ImportError` with install hint).

#### 5.3. Port core export functions (from `omezarr_playground/scripts_and_notebooks/ome_zarr_utils.py`)

- [ ] `convert_czi2hcs_ngff(...)` (ngff-zarr backend, OME-NGFF v0.5).
- [ ] `convert_czi2hcs_omezarr(...)` (ome-zarr-py backend; note: writes OME-NGFF v0.4 in practice).
- [ ] `write_omezarr_ngff(...)` (single image multiscales).
- [ ] `write_omezarr(...)` (single image, ome-zarr-py).

#### 5.4. Port/merge supporting helpers

- [ ] `extract_well_coordinates(...)` — consider relocating to czitools metadata/HCS utilities, so both metadata and export share a single well-path normalization.
- [ ] `create_channel_list(...)` + `get_display(...)` — decide whether this belongs in czitools core (metadata-derived display settings) or stays export-specific.
- [ ] `get_fieldimage(...)` — decide whether to keep as an internal export helper (multiscales per field) or expose publicly.
- [ ] Add a new `validate_ome_zarr(...)` helper backed by the validator selected in the 5.0 spike; this function does not currently exist in the playground prototype.
- [ ] `convert_hcs_omezarr2ozx(...)` — include the OZX conversion helper, including the Windows workaround described in `convert_czi2hcs_ngff`.

#### 5.5. Align exporter with czitools HCS model (Stage 1) — with `CziSampleInfo` fallback

- [ ] Add one resolver that yields normalized well paths and per-well fields (`scene_index`, local `field_index`, optional `region_id`) from either source:
  - **Preferred:** the explicit HCS model (`CziMetadata.hcs`) when populated.
  - **Legacy adapter (conditional):** `CziSampleInfo` heuristics only when names, indices, and scene mapping are complete and unambiguous.
- [ ] Route both `convert_czi2hcs_ngff` and `convert_czi2hcs_omezarr` through this resolver so neither backend hard-codes a metadata source.
- [ ] Allow variable field counts per well (current exporter assumes a single `field_paths` length derived from the first well); the fallback path must handle this too.
- [ ] Preserve proven legacy outputs with golden tests. Reject ambiguous metadata explicitly instead of preserving accidental behavior such as blind leading-zero stripping.

#### 5.6. Performance & memory improvements for large plates

- [ ] Avoid reading full 6D arrays into memory. Iterate scene/field-wise and keep arrays lazy through multiscale generation and writing; verify with a bounded-memory test.
- [ ] Decide chunking strategy per backend (ome-zarr-py chunk hints vs ngff-zarr shard/chunk settings).

#### 5.7. Tests and validation in czitools

- [ ] Add tests that run a minimal HCS conversion on an included test CZI (e.g. `data/WP96_4Pos_B4-10_DAPI.czi`) and validate the output:
  - Use `validate_ome_zarr(...)` when extras are installed.
- Mark tests to skip automatically when export extras are absent.
- [ ] Add regression tests for:
  - padded vs non-padded well columns
  - variable fields per well
  - missing/duplicate `RegionId` and ambiguous well metadata
  - 1-based CZI source indices to 0-based normalized/NGFF indices
  - differing field shapes and bounded-memory incremental writes
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
3. **Stage 3** (convenience reads) — high user value and depends only on the base model.
4. **Stage 2** (coordinate enrichment) — valuable, but requires explicit aggregation/provenance rules.
5. **Stage 5.0, then Stage 5** (NGFF export) — spike first to control dependency and format risk.
6. **Stage 4** (deskew) — independent; schedule when representative fixtures and expected transforms exist.
