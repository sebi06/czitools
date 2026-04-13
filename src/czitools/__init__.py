"""czitools – tools for reading CZI (Carl Zeiss Image) pixel and metadata.

Provides three sub-packages:

- `metadata_tools`: dataclasses for extracting all major CZI metadata sections.
- `read_tools`: functions for reading CZI pixel data as NumPy, Dask or xarray arrays.
- `utils`: logging, scaling, planetable, pixel-type, and napari helpers.
"""

# __init__.py
# version of the czitools package
__version__ = "0.17.0"

from . import metadata_tools, read_tools, utils, visu_tools

__all__ = ["metadata_tools", "read_tools", "utils", "visu_tools"]
