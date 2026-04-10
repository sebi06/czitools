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

Install the core package:

```bash
pip install czitools
```

Install with all optional dependencies (visualization, OME-ZARR, NDV, etc.):

```bash
pip install "czitools[all]"
```

## Local / Editable Install

Useful for development or when working directly from a cloned repository.

Core only:

```bash
pip install -e .
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
