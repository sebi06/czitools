# OME-NGFF v0.5 Pyramid Level Path Normalization

## Summary

Both ome-zarr-py and ngff-zarr write non-standard pyramid level paths that
deviate from the convention used by the OME-NGFF v0.5 reference implementations
(IDR/EBI datasets, `ome2024-ngff-challenge`). A post-processing normalization
step is applied after writing to rename those paths to plain integers.

---

## OME-NGFF Spec vs Zarr Storage Format

These are two orthogonal versioning axes that are often confused:

| Axis                                | v0.4                                             | v0.5                                                                   |
| ----------------------------------- | ------------------------------------------------ | ---------------------------------------------------------------------- |
| **OME-NGFF spec** (metadata schema) | `multiscales` at attrs root; no `ome` wrapper    | All metadata under `attributes.ome`; `"version": "0.5"` at `ome` level |
| **Zarr storage format**             | zarr v2 (`.zattrs`, `.zarray`, chunks `0.0.0.0`) | zarr v3 (`zarr.json`, chunks in `c/0/0/0/` directories)                |

OME-NGFF v0.5 **requires** zarr v3 storage. However, some older datasets in the
wild use OME-NGFF v0.5 metadata schema stored in zarr v2 — these are unofficial
hybrids and not spec-compliant.

---

## The Pyramid Level Naming Problem

The OME-NGFF v0.5 spec says dataset `"path"` values are *"arbitrary"*, but every
reference implementation uses **plain integers** (`"0"`, `"1"`, `"2"`, …):

```json
"datasets": [
  { "path": "0", "coordinateTransformations": [...] },
  { "path": "1", "coordinateTransformations": [...] },
  { "path": "2", "coordinateTransformations": [...] }
]
```

Both backend libraries deviate from this convention.

---

## ome-zarr-py Deviation

**Root cause**: `_write_pyramid_to_zarr` in `ome_zarr/writer.py` hardcodes
`f"s{idx}"` for both the zarr array path and the metadata `"path"` field.
This is not overridable via any public API parameter.

**What is written on disk:**
```
B/04/0/
├── zarr.json          ← ome.multiscales.datasets[].path = "s0", "s1", …
├── s0/
│   └── zarr.json      ← actual array
└── s1/
    └── zarr.json
```

**Fix**: `_normalize_multiscale_level_names_v3()` (or `_v2` for zarr_format=2)
walks all `zarr.json` (or `.zattrs`) files, renames `sN/` directories to `N/`,
and updates the `datasets[].path` entries in-place.

---

## ngff-zarr Deviation

**Root cause**: ngff-zarr uses `f"scale{N}/{NgffImage.name}"` as the zarr path
for every pyramid level, where `name` is the `NgffImage.name` parameter (which
defaults to the CZI filename). The library creates an **intermediate zarr group**
`scaleN/` and places the actual array one level deeper as `scaleN/{name}`.

**What is written on disk:**
```
B/04/0/
├── zarr.json          ← ome.multiscales.datasets[].path = "scale0/WP96.czi", "scale1/WP96.czi"
├── scale0/
│   ├── zarr.json      ← intermediate group node (not an image!)
│   └── WP96.czi/
│       └── zarr.json  ← actual array
└── scale1/
    ├── zarr.json
    └── WP96.czi/
        └── zarr.json
```

Additionally, ngff-zarr embeds `consolidated_metadata` in each group `zarr.json`
which also needs updating after the rename.

**Fix**: `_normalize_multiscale_level_names_ngff()`:
1. Finds all `zarr.json` group files with `ome.multiscales` containing paths
   matching `scale\d+/…`.
2. Moves the array directory (`scaleN/name/`) to `N/`.
3. Removes the vacated `scaleN/` scaffold (including its `zarr.json`).
4. Updates `datasets[].path` from `"scaleN/name"` to `"N"`.
5. Rewrites `consolidated_metadata.metadata`: drops `"scaleN"` intermediate-group
   entries, renames `"scaleN/name"` array entries to `"N"`.

---

## Why Normalization Instead of Fixing at the Source

Passing a custom level-name format is not possible via the public API of either
library. Alternatives considered:

| Approach                                         | Verdict                                                   |
| ------------------------------------------------ | --------------------------------------------------------- |
| Post-process normalization                       | ✅ Chosen — minimal code change, no library fork           |
| Bypass `write_image`, write zarr arrays directly | Feasible but requires reimplementing pyramid downsampling |
| Monkey-patch library internals at runtime        | Fragile, breaks across library versions                   |
| Fork ome-zarr-py / ngff-zarr                     | Too heavy, maintenance burden                             |

---

## ContainsArrayError on Windows (related bug, also fixed)

When the conversion runs on Windows, a transient `PermissionError` (WinError 5,
"Access is denied") can occur during zarr's atomic rename of a `.partial`
metadata file. The `_retry_io` wrapper catches this and retries. However, on
retry, ome-zarr-py's `write_image` tries to re-create the zarr array that was
partially written during the first attempt, causing `zarr.errors.ContainsArrayError`.

**Fix**: `storage_options=dict(chunks=chunks, overwrite=True)` is passed to
`write_image`. For zarr v3, `overwrite=True` is forwarded to `zarr.create_array`
(via dask's `to_zarr`), allowing the retry to overwrite the partially-created
array instead of raising.

---

## GUI Option

The normalization is controlled by the **"Normalize pyramid level paths"**
checkbox in the converter GUI (`normalize_level_paths`, default `True`).
Disable it only to inspect the raw library output for debugging.

The parameter is propagated as:

```
GUI checkbox
  → perform_conversion(normalize_level_paths=...)
      → convert_czi2hcs_omezarr(normalize_level_paths=...)
      → convert_czi2hcs_ngff(normalize_level_paths=...)
```

---

## Verification

To confirm the output conforms to the v0.5 convention, check a field group
`zarr.json`:

```powershell
# Expected after normalization: paths "0", "1", "2" ...
Get-Content "path/to/plate.ome.zarr/B/04/0/zarr.json" |
  ConvertFrom-Json |
  Select-Object -ExpandProperty attributes |
  Select-Object -ExpandProperty ome |
  Select-Object -ExpandProperty multiscales
```

The reference IDR dataset (`uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.5/idr0090/190129.zarr`)
was used as the ground truth: it uses zarr v3 storage (`"zarr_format": 3`),
OME-NGFF v0.5 metadata (`"version": "0.5"` under `"ome"`), and numeric level
paths (`"path": "0"`, `"path": "1"`, …).
