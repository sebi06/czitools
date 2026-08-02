# Copilot Instructions for czitools

This document provides guidelines for GitHub Copilot when working with the czitools repository.

## Project Overview

**czitools** reads CZI (Carl Zeiss Image) pixel data and metadata, models CZI
well plates as plate → well → field hierarchies, and optionally converts CZI
data to OME-Zarr or analyses HCS plates.

### Key Dependencies

- Core: `pylibCZIrw`, `czifile`, `numpy`, `dask`, `xarray`, `pandas`,
  `python-box`, `pydantic`, `requests`, `validators`, and `zarr`.
- OME-Zarr export: `ngff-zarr`, `ome-zarr`, `ome-zarr-models`, and
  `tensorstore` (install with `czitools[omezarr]`).
- Analysis and visualization: `scikit-image`, `matplotlib`, `seaborn`,
  `napari`, and `ndv` (install the relevant extra or `czitools[all]`).

### Supported Python Versions

- Python 3.12 and 3.13

### Supported Operating Systems

- Windows
- Linux
- macOS (with manual pylibCZIrw wheel installation)

## Project Structure

```
src/czitools/
├── metadata_tools/  # Metadata dataclasses and the HCS plate model
├── read_tools/      # Eager and lazy CZI pixel readers
├── export_tools/    # Optional OME-Zarr conversion, validation, and GUI
├── analysis_tools/  # Optional image processing and HCS analysis
├── utils/           # Logging, planetable, NDV, and napari helpers
├── visu_tools/      # Matplotlib and Plotly visualization helpers
└── _tests/          # Pytest suite
```

## Coding Conventions

### General Guidelines
- Write clear, maintainable, and well-documented code.
- Use SOLID design principles where they improve the code:
  - Single-responsibility principle (SRP) or Separation of concerns (SoC)
  - Open–closed principle (OCP)
  - Liskov substitution principle (LSP)
  - Interface segregation principle (ISP)
  - Dependency inversion principle (DIP)

### Python Style
- Use Python 3.12+ syntax and type hints.
- Prefer built-in generics (`list[str]`, `dict[str, int]`) and `X | None` for
  new code. Preserve a module's established annotation style in focused edits.
- Follow PEP 8; the configured Ruff gate selects `E4`, `E7`, `E9`, and `F`.
- Use `@dataclass` for metadata value objects and
  `field(init=False, default=None)` for computed fields.
- Accept `str | os.PathLike[str]` for filesystem paths where practical.
- Prefer `Protocol` for lightweight structural typing across utility modules.

### Type Annotations
```python
import os
from dataclasses import dataclass, field

@dataclass
class ExampleMetadata:
    filepath: str | os.PathLike[str]
    value: float | None = field(init=False, default=None)
    items: list[str] = field(init=False, default_factory=list)
```

### Imports Organization
1. Standard library imports
2. Third-party imports (numpy, pandas, etc.)
3. Local imports from czitools

```python
# Standard library
import os
from dataclasses import dataclass, field
from pathlib import Path

# Third-party
import numpy as np
from box import Box
from pylibCZIrw import czi as pyczi

# Local
from czitools.utils import logging_tools
from czitools.metadata_tools.helper import ValueRange
```

### Logging
- Use the custom logging setup from `czitools.utils.logging_tools`
- Initialize logger at module level: `logger = logging_tools.set_logging()`
- Use `logger.info()`, `logger.warning()`, `logger.error()` for messages
- Use `verbose` parameter in classes to control logging output

```python
from czitools.utils import logging_tools
logger = logging_tools.set_logging()

if self.verbose:
    logger.info("Processing completed successfully")
```

### File Path Handling
- Accept both `str` and `os.PathLike[str]` (Path objects)
- Convert paths to strings only at library boundaries that require strings
- Use `pathlib.Path` for path manipulations
- Use the existing URL helpers in `czitools.utils.misc` instead of duplicating
  URL detection

```python
path = Path(filepath)
filename = path.name
```

### Error Handling
- Validate inputs at public boundaries and guard against `None` and division by
  zero where metadata may be incomplete.
- Catch only expected exceptions. Preserve the original exception as the cause
  when raising a clearer domain error.
- Use a documented fallback only when partial metadata is an accepted state;
  otherwise fail clearly instead of silently returning `None`.

```python
# Safe extraction when a missing value is an accepted metadata state.
raw_value = getattr(data, "Value", None)
value = float(raw_value) * 1_000_000 if raw_value is not None else None
```

### Docstrings
- **Use Google-style docstrings** — this project uses `mkdocstrings` with `docstring_style: "google"`. Non-Google formats (NumPy, Sphinx/reST) will render as a single unformatted text block on the docs site.
- Use `Args:` for parameters (never `Parameters:`).
- Use `Attributes:` in class docstrings to document fields with `name (type): Description.` format.
- Use `Returns:`, `Raises:`, `Notes:`, `Examples:` for other sections.
- **Do NOT** add a `Methods:` section in class docstrings — mkdocstrings auto-discovers methods.
- **Always** leave a blank line between the summary and the first section, and between sections.
- **Always** put section content on the next line (indented), never on the same line as the header.
- **Never** use dashed underlines (`------`) under section headers.
- **Never** use Sphinx/reST style (`:param:`, `:type:`, `:return:`, `:rtype:`).
- Include `(type)` in Args entries: `name (type): Description.`

#### Function docstring example
```python
def read_6darray(
    filepath: CziPath,
    use_dask: bool = False,
    zoom: float = 1.0,
) -> tuple[Array6D | None, CziMetadata]:
    """Read a CZI image file as 6D array.

    Args:
        filepath (CziPath): Path to the CZI image file.
        use_dask (bool): Return a Dask-backed result after the eager read.
        zoom (float): Downscale factor from 0.01 through 1.0.

    Returns:
        tuple[Array6D | None, CziMetadata]: Array and metadata pair; the array
            can be `None` when the CZI cannot form one regular 6D array.
    """
```

#### Dataclass docstring example
```python
@dataclass
class CziScaling:
    """A class to handle scaling information from CZI image data.

    Attributes:
        czisource (str | os.PathLike[str] | Box): The CZI metadata source.
        X (float | None): The X scaling value in microns.
        Y (float | None): The Y scaling value in microns.
        verbose (bool): Flag to enable verbose logging.
    """
```

## Testing Guidelines

### Test Location
- Tests are in `src/czitools/_tests/`
- Test files follow pattern: `test_*.py`
- Use pytest as the test framework
- Keep utility-specific tests close to the module naming (for example `test_ndv_tools.py` for `utils/ndv_tools.py`)

### Test Structure
```python
from pathlib import Path
from typing import Any

import pytest

basedir = Path(__file__).resolve().parents[3]

@pytest.mark.parametrize(
    "czifile, expected_value",
    [
        ("CellDivision_T3_Z5_CH2_X240_Y170.czi", [None, 3, 5, 2, 170, 240])
    ]
)
def test_example(czifile: str, expected_value: list[Any]) -> None:
    filepath = basedir / "data" / czifile
    # Test implementation
    assert result == expected_value
```

### Test Data
- Test CZI files are in `data/` directory
- Use parametrized tests for multiple test cases
- Reference test files relative to `basedir`

### Running Tests
```bash
pixi run test
pixi run test-no-net
pixi run lint
```

## Common Patterns

### Reading Metadata
```python
from czitools.metadata_tools import CziDimensions, CziMetadata, CziScaling

# Get all metadata at once
mdata = CziMetadata(filepath)

# Or get specific metadata
scaling = CziScaling(filepath)
dimensions = CziDimensions(filepath)
```

### Reading Pixel Data
```python
from czitools.read_tools import read_stacks_list

# Use read_stacks_list for true lazy access and differently sized scenes.
scenes, dims, scene_count, mdata = read_stacks_list(
    filepath,
    use_dask=True,
    use_xarray=True,
    zoom=0.5,
)
first_plane = scenes[0].isel(T=0, C=0, Z=0).compute()
```

`read_6darray(..., use_dask=True)` still reads pixel data eagerly before
wrapping the result. Do not recommend it as the lazy path for large files.

### Using Box for Metadata
```python
from czitools.utils.box import get_czimd_box

# Get metadata as Box object for attribute-style access
czi_box = get_czimd_box(filepath)
scaling = czi_box.ImageDocument.Metadata.Scaling.Items.Distance
```

## Array Dimension Order

CZI arrays returned by `read_6darray`, `read_field`, and `read_well` use the
dimension order **STCZYX(A)**.

- S = Scene
- T = Time
- C = Channel
- Z = Z-slice
- Y = Y dimension
- X = X dimension
- A = Alpha/RGB component (optional)

`read_stacks` can additionally preserve the optional `V`, `R`, `I`, `H`, and
`M` dimensions before `T`, `C`, and `Z`.

## Additional Notes

### Metadata Classes Pattern
Most metadata classes follow this pattern:

1. Accept `czisource` as filepath, Path, or Box object
2. Use `@dataclass` with `field(init=False)` for computed attributes
3. Implement `__post_init__` for initialization logic
4. Support `verbose` parameter for logging control

Keep optional feature imports lazy so `import czitools` and the core metadata
and read APIs work without GUI, analysis, or OME-Zarr extras installed.

### Scaling Units
- Internal scaling values are in **microns**
- Conversion from CZI values: `value * 1000000` (meters to microns)

### RGB Support
- Check `isRGB` dictionary for RGB status per channel
- RGB images have an additional 'A' dimension

### Scene Handling
- CZI files may have multiple scenes
- Check `has_scenes` and `SizeS` for scene information
- Use `bbox.total_bounding_box` for combined bounds

<!-- mermaid-ai-skills:start -->
## Mermaid Diagrams

When the user asks to create, edit, or visualize a diagram, follow the
instructions in `.github/instructions/mermaid.instructions.md`.
<!-- mermaid-ai-skills:end -->
