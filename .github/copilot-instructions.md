# Copilot Instructions for czitools

This document provides guidelines for GitHub Copilot when working with the czitools repository.

## Project Overview

**czitools** is a Python package for reading CZI (Carl Zeiss Image) pixel and metadata. It simplifies working with CZI microscopy image files by providing tools for metadata extraction and pixel data reading.

### Key Dependencies
- `pylibCZIrw` - Core library for reading/writing CZI files
- `aicspylibczi` - Additional CZI functionality
- `numpy` - Array operations
- `dask` - Lazy/delayed array operations
- `xarray` - Labeled multi-dimensional arrays
- `pandas` - Data manipulation (planetables)
- `python-box` - Dictionary access via attributes
- `pydantic` - Data validation
- `loguru` / `colorlog` - Logging

### Supported Python Versions
- Python 3.10, 3.11, 3.12, 3.13

### Supported Operating Systems
- Windows
- Linux
- macOS (with manual pylibCZIrw wheel installation)

## Project Structure

```
src/czitools/
├── metadata_tools/       # Classes for extracting CZI metadata
│   ├── czi_metadata.py   # Main CziMetadata class
│   ├── dimension.py      # CziDimensions
│   ├── scaling.py        # CziScaling
│   ├── channel.py        # CziChannelInfo
│   ├── boundingbox.py    # CziBoundingBox
│   ├── objective.py      # CziObjectives
│   ├── detector.py       # CziDetector
│   ├── microscope.py     # CziMicroscope
│   ├── sample.py         # CziSampleInfo
│   └── add_metadata.py   # CziAddMetaData
├── read_tools/           # Functions for reading pixel data
│   └── read_tools.py     # read_6darray, read_mdarray, etc.
├── utils/                # Utility modules
│   ├── logging_tools.py  # Logging configuration
│   ├── box.py            # Box utilities for metadata
│   ├── misc.py           # Miscellaneous helpers
│   ├── pixels.py         # Pixel type utilities
│   └── planetable.py     # Planetable generation
├── visu_tools/           # Visualization utilities
└── _tests/               # Test suite
```

## Coding Conventions

### General Guidelines
- write clear, maintainable, and well-documented code
- use SOLID Design Principles in python
  - Single-responsibility principle (SRP) or Separation of concerns (SoC)
  - Open–closed principle (OCP)
  - Liskov substitution principle (LSP)
  - Interface segregation principle (ISP)
  - Dependency inversion principle (DIP)

### Python Style
- Use Python 3.10+ syntax and type hints
- Follow PEP 8 style guidelines
- Use `dataclass` for metadata classes with `@dataclass` decorator
- Use `field(init=False, default=None)` for computed fields in dataclasses
- Prefer `Optional[Type]` for nullable types
- Use `Union[str, os.PathLike[str]]` for file paths

### Type Annotations
```python
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field

@dataclass
class ExampleMetadata:
    filepath: Union[str, os.PathLike[str]]
    value: Optional[float] = field(init=False, default=None)
    items: Optional[List[str]] = field(init=False, default_factory=lambda: [])
```

### Imports Organization
1. Standard library imports
2. Third-party imports (numpy, pandas, etc.)
3. Local imports from czitools

```python
# Standard library
from typing import Dict, Tuple, Optional, Union
import os
from pathlib import Path
from dataclasses import dataclass, field

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
- Convert Path to string when needed: `str(filepath)`
- Use `pathlib.Path` for path manipulations
- Support URL paths using `validators.url()` check

```python
from pathlib import Path

if isinstance(self.filepath, Path):
    self.filepath = str(self.filepath)
```

### Error Handling
- Use defensive programming with fallback values
- Guard against None values and division by zero
- Use `try/except` blocks for external library calls
- Return None or sensible defaults instead of raising exceptions when appropriate

```python
# Safe value extraction with fallback
try:
    value = float(data.Value) * 1000000
    if value == 0.0:
        value = 1.0  # fallback
except (AttributeError, TypeError):
    value = None
```

### Docstrings
- Use Google-style docstrings
- Include Args, Returns, and Raises sections
- Document class attributes in class docstring

```python
def read_6darray(
    filepath: Union[str, os.PathLike[str]],
    use_dask: Optional[bool] = False,
    zoom: Optional[float] = 1.0,
) -> Tuple[Optional[np.ndarray], CziMetadata]:
    """Read a CZI image file as 6D array.

    Args:
        filepath: Path to the CZI image file.
        use_dask: Option to use dask for delayed reading.
        zoom: Downscale factor [0.01 - 1.0].

    Returns:
        Tuple of (array6d, metadata) where array6d may be None on error.
    """
```

## Testing Guidelines

### Test Location
- Tests are in `src/czitools/_tests/`
- Test files follow pattern: `test_*.py`
- Use pytest as the test framework

### Test Structure
```python
from czitools.metadata_tools import czi_metadata as czimd
from pathlib import Path
import pytest
from typing import List, Any

basedir = Path(__file__).resolve().parents[3]

@pytest.mark.parametrize(
    "czifile, expected_value",
    [
        ("CellDivision_T3_Z5_CH2_X240_Y170.czi", [None, 3, 5, 2, 170, 240])
    ]
)
def test_example(czifile: str, expected_value: List[Any]) -> None:
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
pytest src/czitools/_tests/
pytest -m "not network"  # Skip network tests
```

## Common Patterns

### Reading Metadata
```python
from czitools.metadata_tools.czi_metadata import CziMetadata
from czitools.metadata_tools.scaling import CziScaling
from czitools.metadata_tools.dimension import CziDimensions

# Get all metadata at once
mdata = CziMetadata(filepath)

# Or get specific metadata
scaling = CziScaling(filepath)
dimensions = CziDimensions(filepath)
```

### Reading Pixel Data
```python
from czitools.read_tools import read_tools

# Read as 6D array (STCZYX order)
array6d, mdata = read_tools.read_6darray(
    filepath,
    use_dask=True,      # For large files
    use_xarray=True,    # For labeled dimensions
    zoom=0.5            # Downscale
)
```

### Using Box for Metadata
```python
from czitools.utils.box import get_czimd_box

# Get metadata as Box object for attribute-style access
czi_box = get_czimd_box(filepath)
scaling = czi_box.ImageDocument.Metadata.Scaling.Items.Distance
```

## Array Dimension Order

CZI arrays use the dimension order: **STCZYX(A)**
- S = Scene
- T = Time
- C = Channel
- Z = Z-slice
- Y = Y dimension
- X = X dimension
- A = Alpha/RGB component (optional)

## Additional Notes

### Metadata Classes Pattern
All metadata classes follow a similar pattern:
1. Accept `czisource` as filepath, Path, or Box object
2. Use `@dataclass` with `field(init=False)` for computed attributes
3. Implement `__post_init__` for initialization logic
4. Support `verbose` parameter for logging control

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
