# czitools HCS / Wellplate Metadata — Capability Review & Improvement Plan

> Source proposal: `_notes/HCS_Metadata_OME_CZI_Proposal_Balu.docx`
> Status legend: `[ ]` not started · `[~]` in progress · `[x]` done
> Reviewed against the repository on 2026-07-10. Changes made during the review
> are explained in **Review decisions** below.
> Follow-up review 2026-07-14: Stages 0 and 1 confirmed implemented and integrated
> (`CziMetadata.hcs` / `hcs_status`). Stage 5 revised because `omezarr_playground` is
> now an installable `czi_omezarr_utils` package, not loose scripts. Stage 2/3 notes
> added for immutable enrichment, the planetable `S` join key, and the `hcs is None`
> resolver guard.
> Implementation 2026-07-14: Stages 2 and 3 implemented (planetable position
> enrichment, well/field resolvers, `read_field`/`read_well`) with tests passing in the
> `omezarr` environment. See the Stage 2/3 implementation notes below.

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
   `ngff-zarr` variants in the `all` extra. The playground is now an installable
   `czi_omezarr_utils` package (modules `conversion.py`, `hcs.py`, `display.py`,
   `processing.py`, `plotting.py`, `validation.py`, `logging_utils.py`) — the older
   `scripts_and_notebooks/ome_zarr_utils.py` no longer exists. Stage 5 is therefore a
   *vendor + reconcile* effort, not a port of prototype scripts. Backend versions,
   Zarr v2/v3 support, output NGFF versions, and validation support must still be
   verified before pinning dependencies or promising preserved public names. The
   package already carries known-good pins (`ngff-zarr>=0.34.0` because 0.29.0 shipped
   null bytes; `dask>=2024.1,<2025.11` for ngio compatibility) that czitools should
   adopt directly.
7. **Deskew is independent and metadata-dependent.** It is not a prerequisite for HCS
   metadata or export. Implement it only after representative light-sheet metadata and
   expected transforms are available as fixtures.

---

## 4. Gap analysis vs. the OME/HCS proposal

| Proposal concept                                             | OME basis                | czitools today                    | Gap                                                       |
| ------------------------------------------------------------ | ------------------------ | --------------------------------- | --------------------------------------------------------- |
| **Experiment / Screen**                                      | `Screen`                 | none                              | Cross-plate/campaign context missing                      |
| **Plate** (rows, cols, naming, barcode, origin)              | `Plate`                  | none                              | No plate entity; declared values may be absent            |
| **PlateAcquisition** (run, start/end, max fields)            | `PlateAcquisition`       | times in planetable only          | No acquisition-run entity                                 |
| **Well** (explicit ID, Row, Column, ExternalIdentifier)      | `Well`                   | inferred from ArrayName/Shape     | Source indices appear 1-based; normalized identity absent |
| **Field / FOV / WellSample** (source-scoped ID, local index) | `WellSample`             | Scene used implicitly             | No local FieldIndex or extracted `RegionId`               |
| **Field position** (X/Y + coord system; optional source Z)   | `WellSample/PositionX/Y` | scene center + subblock positions | Provenance, aggregation, units, and origin unclear        |
| **Deskew / lightsheet geometry**                             | NGFF RFC-5 affine/shear  | none                              | No geometry, angle, shear axis, or affine matrix          |
| **OME-Zarr HCS export** (`plate/A/1/0`)                      | OME-NGFF                 | none                              | No export/mapping path                                    |

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

- [x] Join by explicit `S`/scene index and group all matching subblocks. Define representative-value and tolerance rules across `M/T/C/Z`; preserve ranges or report conflicts instead of selecting the first row silently.
- [x] Use the planetable `S` DataFrame column as the join key (note: `get_planetable`'s `planes` dict uses different key names such as `scene`/`zplane`, but the returned columns are `S, M, T, C, Z`). Confirm each grouped `S` value aligns with `CziField.scene_index` before enriching.
- [x] Keep `Scene.CenterPosition`, subblock stage X/Y, and focus Z as separate sourced values until their relationship is verified on fixtures.
- [x] Enrich immutably. `CziField`/`CziWell`/`CziPlate` are `frozen=True` and `CziField.acquisition_z` is a reserved slot; enrichment must rebuild new instances (return a new plate), never mutate in place.
- [x] Document unit, axis direction, coordinate origin, and source for every coordinate. Add plate-absolute <-> well-relative conversion only when a trustworthy well origin/center is present; otherwise return an unavailable result.
- [x] Make planetable enrichment lazy/opt-in because it scans subblocks and is unavailable for URL sources.

**Stage 2 implementation notes (2026-07-14):**

- `enrich_hcs_with_planetable(plate, filepath, position_tolerance=1.0)` in
  [hcs.py](../src/czitools/metadata_tools/hcs.py) returns a **new** `CziPlate`
  (via `dataclasses.replace`); the input plate is never mutated. `get_planetable`
  is imported lazily inside the function to keep the pandas/subblock cost off the
  base metadata path and avoid import cycles.
- Fields are matched to planetable rows by the `S` column (`groupby("S")`) against
  each `CziField.scene_index`, never by row order. Scenes absent from the
  planetable are left unenriched (`stage_* is None`).
- The representative value for each of stage X, stage Y and focus Z is the
  **median** across all `M/T/C/Z` subblocks of a scene; the full `(min, max)`
  range is preserved in `stage_x_range`/`stage_y_range`/`acquisition_z_range`.
  `position_conflict` is `True` when any range exceeds `position_tolerance`
  (micrometers). `subblock_count` records how many subblocks were aggregated.
- Provenance is explicit and kept separate by source: `scene_center_x/y`
  (`Scene.CenterPosition`) vs. `stage_x/y` + `acquisition_z`
  (`planetable(StageXPosition,StageYPosition,FocusPosition)`), each with a unit
  attribute. The two sources are never merged.
- URL sources and files without subblock positions return the plate unchanged
  (empty planetable). `CziMetadata.enrich_hcs_positions(...)` is the opt-in
  convenience entry point that replaces `self.hcs` with the enriched copy.
- `well_relative_field_positions(well)` returns each field's XY offset from the
  centroid of the well's field scene-centers, or `None` when any center is
  missing (explicit "unavailable" result), since no trustworthy origin exists then.
- Tests in `test_hcs.py` cover median/range aggregation, the conflict flag,
  `subblock_count`, immutability of the source plate, unenriched absent scenes,
  the empty-planetable (URL) path, and both well-relative outcomes.
- Prerequisite fix: the planetable `S` column previously came from
  `directory_entry.start[S]`, which is always `0` (each subblock spans a single
  scene), so every subblock collapsed to scene 0. It now uses the authoritative
  `directory_entry.scene_index` (see `_get_scene_index` in
  [planetable.py](../src/czitools/utils/planetable.py)). This is what makes the
  `S`-keyed join correct; the mosaic tile `M` counter now also resets per scene.

### Stage 3 — Convenience reads keyed by plate/well/field

- [x] First add pure resolvers (`resolve_well`, `resolve_field`) with normalization, duplicate, and out-of-range tests.
- [x] Resolvers must fail clearly when `CziMetadata.hcs is None`. Only fall back to `CziSampleInfo` when it yields an unambiguous mapping (same rule as Stage 5.5); otherwise raise a precise error rather than manufacturing a plate.
- [x] `read_field(filepath, well="A1", field=0)` should reuse the existing single-scene path (`read_6darray` with `planes={"S": (scene, scene)}`) and return the same `Tuple[array, CziMetadata]` shape. Define whether `field` is the well-local index or a `RegionId`.
- [x] `read_well(filepath, well="A1")` should return a list/mapping when field shapes differ; stack only on explicit request after validating compatible shapes.

**Stage 3 implementation notes (2026-07-14):**

- `resolve_well(plate, well)` and `resolve_field(plate, well, field)` are pure
  functions in [hcs.py](../src/czitools/metadata_tools/hcs.py). `resolve_well`
  accepts `"B4"`, `"b04"` and NGFF-style `"B/4"` (the `/` is stripped before
  normalization) and delegates duplicate/absent handling to `CziPlate.get_well`.
- `resolve_field` treats an `int` as a well-local 0-based `field_index` (raises
  `IndexError` out of range) and a `str` as a `RegionId` (raises `KeyError`
  unless exactly one field matches). `bool` is rejected with `TypeError` because
  it is an `int` subtype.
- `read_field` / `read_well` in
  [read_tools.py](../src/czitools/read_tools/read_tools.py) build `CziMetadata`,
  call `_require_hcs_plate` (raises `ValueError` with the `hcs_status.reason`
  when `mdata.hcs is None`), resolve the scene index, then reuse `read_6darray`
  with `planes={"S": (scene, scene)}`. They return the same
  `Tuple[array, CziMetadata]` shape as `read_6darray`. The `CziSampleInfo`
  fallback is deferred to Stage 5.5 to keep a single resolver path.
- `read_well` returns a **list** of per-field arrays by default (fields may have
  different shapes); `stack=True` concatenates along the `S` axis and raises
  `ValueError` if shapes differ. A `fields=[...]` selector accepts local indices
  and/or `RegionId` strings. Unreadable fields are logged and skipped; an
  all-empty result returns `(None, metadata)`.
- New public names are exported from `czitools.read_tools`
  (`read_field`, `read_well`) and `czitools.metadata_tools`
  (`resolve_well`, `resolve_field`, `enrich_hcs_with_planetable`,
  `well_relative_field_positions`).
- Tests in `test_read_well.py` cover single-field reads, `RegionId` vs index
  equivalence, the per-field list, `stack=True`, a field selector, and the
  non-plate `ValueError` path, using the included `WP96_4Pos_B4-10_DAPI.czi`.

### Stage 4 — Deskew / lattice-lightsheet geometry

- [ ] Gate this stage on acquiring representative CZI metadata fixtures and a known-good transformed result.
- [ ] `CziLightSheetGeometry` (acquisition mode, shear axis, angle, Z step, pixel sizes).
- [ ] `DeskewTransform` (affine matrix) derived from documented metadata; modeled as a coordinate transform with axis/order/unit conventions, not a boolean flag.

### Stage 5 — OME-Zarr / OME-NGFF export (adapt from `omezarr_playground`)

The local repository `omezarr_playground` now ships an installable `czi_omezarr_utils`
package (modules `conversion.py`, `hcs.py`, `display.py`, `processing.py`,
`plotting.py`, `validation.py`, `logging_utils.py`). The older
`scripts_and_notebooks/ome_zarr_utils.py` no longer exists. Treat the package as the
migration source. Its conversion core depends on czitools for:

- pixel reads: `czitools.read_tools.read_tools.read_6darray(..., use_xarray=True)`
- metadata: `CziMetadata` including `mdata.scale`, `mdata.channelinfo`, and **plate-ish** metadata via `mdata.sample.*`

Goal: incorporate the reusable conversion core into **czitools** as an optional feature. HCS export should use the canonical HCS layout resolver. The resolver may adapt legacy `CziSampleInfo` data when the mapping is complete and unambiguous; otherwise it must return a clear validation error.

> **Note (2026-07-14):** Because the playground is already a structured package, Stage 5
> is a *vendor + reconcile* effort, not a port of loose scripts. Most functions listed
> below already exist with the stated names, `validate_ome_zarr(...)` already exists, and
> known-good dependency pins are already declared in the package. The main new work is
> reconciling the playground's own `PlateType`/`PlateConfiguration` plate model with
> czitools' Stage 1 `CziPlate`/`CziWell` (see 5.5).

#### 5.0. Compatibility and scope spike (must precede vendoring)

- [ ] Audit the existing `czi_omezarr_utils` package: public API (`__init__.py`), licenses, tests, private imports, GUI/plotting dependencies (`display.py`, `plotting.py`, `processing.py`), and actual callers.
- [ ] Test a small matrix of `ngff-zarr`, `ome-zarr`, `zarr` v2/v3, Python 3.12/3.13, and validators; record the NGFF version each backend really writes. Reuse the package's known-good pins (`ngff-zarr>=0.34.0`, `dask>=2024.1,<2025.11`) as the starting point.
- [ ] Choose one primary writer backend (ngff-zarr writes OME-NGFF v0.5; ome-zarr-py writes v0.4). Keep the second backend only if it provides a tested capability the primary one lacks.
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

#### 5.3. Vendor core export functions (from `czi_omezarr_utils.conversion`)

These already exist with the stated names; vendor them into czitools rather than rewriting.

- [ ] `convert_czi2hcs_ngff(...)` (ngff-zarr backend, OME-NGFF v0.5).
- [ ] `convert_czi2hcs_omezarr(...)` (ome-zarr-py backend; note: writes OME-NGFF v0.4 in practice).
- [ ] `write_omezarr_ngff(...)` (single image multiscales).
- [ ] `write_omezarr(...)` (single image, ome-zarr-py).

#### 5.4. Vendor/merge supporting helpers

- [ ] `extract_well_coordinates(...)` (from `czi_omezarr_utils.hcs`) — relocate to czitools metadata/HCS utilities so both metadata and export share a single well-path normalization. Reconcile its `pad_columns` behavior with the Stage 1 canonical path (`row/column`).
- [ ] `create_channel_list(...)` + `get_display(...)` (from `czi_omezarr_utils.display`) — decide whether this belongs in czitools core (metadata-derived display settings) or stays export-specific.
- [ ] `get_fieldimage(...)` (from `czi_omezarr_utils.display`) — decide whether to keep as an internal export helper (multiscales per field) or expose publicly.
- [ ] `validate_ome_zarr(...)` (from `czi_omezarr_utils.validation`, OME-NGFF v0.5) — vendor as-is; it already exists, so no new implementation is required.
- [ ] `convert_hcs_omezarr2ozx(...)` (from `czi_omezarr_utils.hcs`) — include the OZX conversion helper, including the Windows workaround described in `convert_czi2hcs_ngff`.
- [ ] Explicitly exclude GUI/analysis-only helpers (`processing.py`, `plotting.py`, and Qt/napari code) from the export extra unless intentionally shipped and tested.

#### 5.5. Align exporter with czitools HCS model (Stage 1) — with `CziSampleInfo` fallback

- [ ] **Reconcile the two plate models.** `czi_omezarr_utils.hcs` has its own `PlateType`, `PlateConfiguration`, `define_plate`, and `define_plate_by_well_count`, which overlap and conflict with czitools' Stage 1 `CziPlate`/`CziWell`. Route the exporter through the Stage 1 canonical resolver and drop the playground's parallel plate abstraction rather than vendoring both.
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
