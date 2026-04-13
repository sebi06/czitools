"""Functions for reading CZI pixel data as NumPy, Dask or xarray arrays.

Provides `read_6darray`, `read_stacks`, `read_stacks_list`,
`read_stacks_stacked`, and `read_attachments` for loading CZI
images with dimension order STCZYX(A).
"""

from .read_tools import (
    read_6darray,
    read_attachments,
    read_stacks,
    read_stacks_list,
    read_stacks_stacked,
    read_tiles,
)

__all__ = [
    "read_6darray",
    "read_attachments",
    "read_tiles",
    "read_stacks",
    "read_stacks_list",
    "read_stacks_stacked",
]
