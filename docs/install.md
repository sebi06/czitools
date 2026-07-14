# Installation

!!! warning "Work in Progress"
    This documentation is still incomplete and actively being updated.
    Some sections may be missing or subject to change.

## Requirements

| Requirement      | Details               |
| ---------------- | --------------------- |
| Python           | 3.12, 3.13            |
| Operating System | Windows, Linux, macOS |

## Install from PyPI

Install the **core** package (metadata + pixel reading only — no OME-Zarr export or GUI):

```bash
pip install czitools
```

### Feature extras

czitools keeps the base install lean. Heavier features are **opt-in** extras:

| Extra         | Install                               | Adds                                                                                           |
| ------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------- |
| _(core)_      | `pip install czitools`                | Metadata + pixel reading (`CziMetadata`, `read_6darray`, `read_field`, `read_well`, HCS model) |
| `omezarr`     | `pip install "czitools[omezarr]"`     | OME-Zarr / OME-NGFF export + validation (`czitools.export_tools`)                              |
| `omezarr-gui` | `pip install "czitools[omezarr-gui]"` | Everything in `omezarr` plus the MagicGUI converter app and napari preview                     |
| `all`         | `pip install "czitools[all]"`         | All optional features (visualization, BioIO, NDV, export, GUI)                                 |
| `docs`        | `pip install "czitools[docs]"`        | Documentation build toolchain (MkDocs)                                                         |

!!! note "Lazy imports — core stays lean"
    `import czitools.export_tools` never fails on a core-only install. The optional
    dependencies are only required when you actually call an export function; otherwise
    a clear error is raised:

    ```text
    ImportError: OME-Zarr export requires optional dependencies.
    Install them with: pip install "czitools[omezarr]" (or "czitools[omezarr-gui]" for the GUI).
    ```

After installing `omezarr-gui`, launch the converter GUI via the console script:

```bash
czitools-omezarr-gui
```

Install with **all** optional dependencies (visualization, OME-Zarr, GUI, NDV, etc.):

```bash
pip install "czitools[all]"
```

## Local / Editable Install

Useful for development or when working directly from a cloned repository.

Core only:

```bash
pip install -e .
```

With a single feature extra (for example OME-Zarr export):

```bash
pip install -e ".[omezarr]"
```

Full functionality:

```bash
pip install -e ".[all]"
```

## Dependencies

### Core dependencies

These are installed automatically with `pip install czitools`:

| Package               | Purpose                           |
| --------------------- | --------------------------------- |
| `pylibCZIrw>=5`       | Reading and writing CZI files     |
| `czifile`             | Subblock-level CZI access         |
| `numpy`               | Array operations                  |
| `xarray[complete]`    | Labelled multi-dimensional arrays |
| `pandas`              | Planetable data manipulation      |
| `python-box[all]`     | Attribute-style metadata access   |
| `pydantic`            | Data validation                   |
| `loguru` / `colorlog` | Logging                           |
| `progressbar2`        | Progress reporting                |
| `python-dateutil`     | Date parsing                      |

### Optional dependencies (`[all]`)

| Package                                  | Purpose                    |
| ---------------------------------------- | -------------------------- |
| `seaborn` / `plotly`                     | Plotting and visualization |
| `qtpy` / `pyqtgraph`                     | Qt-based visualization     |
| `colormap`                               | Colormap utilities         |
| `ngff-zarr`                              | OME-ZARR export support    |
| `bioio` / `bioio-czi` / `bioio-ome-zarr` | BioIO readers              |
| `ndv[pyqt,vispy]`                        | NDV viewer integration     |

### OME-Zarr export dependencies (`[omezarr]`)

Required by `czitools.export_tools` for CZI → OME-Zarr conversion and validation:

| Package             | Purpose                                 |
| ------------------- | --------------------------------------- |
| `ngff-zarr>=0.34.0` | Primary write backend (OME-NGFF v0.5)   |
| `ome-zarr>=0.16.0`  | Secondary write backend (OME-NGFF v0.4) |
| `zarr>=3.0`         | Zarr v3 storage backend                 |
| `ome-zarr-models`   | OME-NGFF v0.5 schema validation         |

### GUI dependencies (`[omezarr-gui]`)

Adds the MagicGUI converter application on top of `[omezarr]`:

| Package             | Purpose                               |
| ------------------- | ------------------------------------- |
| `magicgui` / `qtpy` | GUI widgets and Qt abstraction        |
| `napari`            | Optional in-app preview of the result |
| `napari-ome-zarr`   | napari reader plugin for OME-Zarr     |

## Conda / Pixi Environment

A ready-to-use conda environment file is provided:

```bash
conda env create -f env_czitools.yml
conda activate czitools
```

## Verify Installation

```python
import czitools
print(czitools.__version__)
```
