"""Functions for reading CZI pixel data as NumPy, Dask or xarray arrays.

Provides `read_6darray`, `read_stacks`, `read_stacks_list`,
`read_stacks_stacked`, `read_field`, `read_well`, and `read_attachments`
for loading CZI images with dimension order STCZYX(A).
"""

from .array6d import read_6darray
from .attachments import read_attachments
from .field_well import read_field, read_well
from .stacks import read_stacks, read_stacks_list, read_stacks_stacked
from .tiles import read_tiles

__all__ = [
    "read_6darray",
    "read_attachments",
    "read_field",
    "read_tiles",
    "read_stacks",
    "read_stacks_list",
    "read_stacks_stacked",
    "read_well",
]
