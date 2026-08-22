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
- **Effort:** M (measure first with `pylancePythonProfiling` on a representative CZI) · **Model tier:** `premium` (profiling-driven perf work; measure before/after)

### 7. Optional Zarr-backed lazy reading / caching `P2` `L`

- **Problem:** dask reads always go back to the CZI. For repeated access,
  converting once to a chunked Zarr store gives fast, truly-lazy random access.
- **Action:** add an optional `read_to_zarr(filepath, store, chunks=...)` helper
  (reuse `export_tools`) and a `da.from_zarr` fast path. Keep it optional
  (`czitools[omezarr]`), out of the core import.
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
- [ ] #3 true-lazy read_6darray
- [ ] #4 split read_tools.py
- [ ] #5 modern typing
- [ ] #6 reading speed
- [ ] #7 zarr-backed lazy
- [ ] #8 ruff expansion
- [ ] #9 actions versions
- [ ] #10 disclaimers
- [ ] #11 mkdocs/readme
- [ ] #12 API `__all__`
