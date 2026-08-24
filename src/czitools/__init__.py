"""czitools – tools for reading CZI (Carl Zeiss Image) pixel and metadata.

Provides three sub-packages:

- `metadata_tools`: dataclasses for extracting all major CZI metadata sections.
- `read_tools`: functions for reading CZI pixel data as NumPy, Dask or xarray arrays.
- `utils`: logging, scaling, planetable, pixel-type, and napari helpers.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _metadata_version
from types import ModuleType
from typing import TYPE_CHECKING

try:
    __version__: str = _metadata_version("czitools")
except PackageNotFoundError:
    # editable install without dist-info; fall back to _version.py
    from czitools._version import version as __version__  # type: ignore[assignment]

__all__ = ["__version__", "metadata_tools", "read_tools", "utils", "visu_tools"]

if TYPE_CHECKING:
    from . import metadata_tools, read_tools, utils, visu_tools


def __getattr__(name: str) -> ModuleType:
    """Load public subpackages only when they are first accessed."""
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazily exposed subpackages in interactive discovery."""
    return sorted(set(globals()) | set(__all__))
