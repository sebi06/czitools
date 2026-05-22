---
name: release-new-version
description: "Release a new version of czitools to PyPI (or TestPyPI). Use when: release new version, publish to PyPI, bump version, cut a release, deploy package, TestPyPI release."
---

# Release a New Version to PyPI

This skill walks through all the steps required to release a new version of **czitools** to PyPI (or TestPyPI). Follow every step in order.

## Prerequisites

- You are on the `main` branch with a clean working tree.
- All tests pass locally (`pytest src/czitools/_tests/`).
- The GitHub CLI (`gh`) is authenticated.
- The repository secret `TWINE_PYPI_TOKEN` (or `TWINE_TESTPYPI_TOKEN` for TestPyPI) is configured in the GitHub repo settings.

## Release Checklist

### 1. Determine the New Version

Ask the user for the new version string if not provided (e.g. `0.18.0`). The project uses [Semantic Versioning](https://semver.org/):
- **MAJOR** — incompatible API changes
- **MINOR** — new functionality, backwards-compatible
- **PATCH** — backwards-compatible bug fixes

### 2. Bump the Version in Two Places

Update the version string in **both** of these files — they must match:

| File | Field |
|---|---|
| `pyproject.toml` | `version = "X.Y.Z"` (under `[project]`) |
| `src/czitools/__init__.py` | `__version__ = "X.Y.Z"` |

### 3. Update the Changelog

Edit `CHANGELOG.md` and add a new section **above** the previous release, following [Keep a Changelog](https://keepachangelog.com/) format:

```markdown
## [X.Y.Z] — YYYY-MM-DD

### Added
- ...

### Changed
- ...

### Fixed
- ...
```

Use today's date. Only include the sub-sections (`Added`, `Changed`, `Fixed`, `Removed`, `Deprecated`, `Security`) that have entries. Separate sections with `---`.

### 4. Create Release Notes File

Create `_release_notes/vX.Y.Z.md` following the established format:

```markdown
# vX.Y.Z

**Release Date:** YYYY-MM-DD

## Highlights

- **Short title** — One-sentence summary of the main change.

## Added
- `path/to/file.py` — description

## Changed
- `path/to/file.py` — description

## Fixed
- `path/to/file.py` — description
```

Only include the sections that have entries. Provide file-level detail for each change.

### 5. Commit and Push

```bash
git add pyproject.toml src/czitools/__init__.py CHANGELOG.md _release_notes/vX.Y.Z.md
git commit -m "chore: add CHANGELOG.md and release notes for vX.Y.Z"
git push origin main
```

### 6. Create the GitHub Release

This is the step that **triggers the CI/CD pipeline** to publish to PyPI.

#### For PyPI (production)

```bash
gh release create vX.Y.Z --title "vX.Y.Z" --notes-file _release_notes/vX.Y.Z.md
```

This creates a git tag `vX.Y.Z` and triggers the workflow `.github/workflows/test_and_deploy.yml`, which:
1. Runs tests on Ubuntu, Windows, and macOS with Python 3.12 and 3.13
2. Builds an sdist and wheel with `python -m build`
3. Publishes to **PyPI** via `pypa/gh-action-pypi-publish` using the `TWINE_PYPI_TOKEN` secret

#### For TestPyPI (pre-release / dry-run)

```bash
gh release create testpypi-vX.Y.Z --title "testpypi-vX.Y.Z" --notes-file _release_notes/vX.Y.Z.md --prerelease
```

This triggers `.github/workflows/test_and_deploy_testpypi.yml`, which publishes to **TestPyPI** using the `TWINE_TESTPYPI_TOKEN` secret.

### 7. Verify the Release

After the CI workflow completes:

1. Check the GitHub Actions tab for green status on the `test_and_deploy_pypi` workflow.
2. Verify the package on PyPI: https://pypi.org/project/czitools/X.Y.Z/
3. Test installation: `pip install czitools==X.Y.Z`

For TestPyPI:
- https://test.pypi.org/project/czitools/X.Y.Z/
- `pip install -i https://test.pypi.org/simple/ czitools==X.Y.Z`

## Important Notes

- **The tag triggers deployment.** Do NOT create a `v*` tag unless you intend to publish to PyPI. Use `testpypi-v*` tags for dry runs.
- **Ask for confirmation** before pushing commits and creating the GitHub release — these actions affect the shared repository and trigger deployments.
- The CI only runs the deploy job when the ref contains a tag (`if: contains(github.ref, 'tags')`).
- Always ask the user whether this is a **PyPI** or **TestPyPI** release if not specified.