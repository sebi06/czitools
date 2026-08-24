"""Backward-compatible facade for the split CZI reader modules.

New code may import functions directly from czitools.read_tools. This module
remains available for existing legacy module imports.
"""

from .array6d import read_6darray
from .attachments import read_attachments
from .field_well import read_field, read_well
from .pyramid import PyramidLevel, get_pyramid_zooms, read_stacks_multiscale
from .stacks import (
    _read_plane_chunk as _read_plane_chunk,
    _read_plane_delayed as _read_plane_delayed,
    read_stacks,
    read_stacks_list,
    read_stacks_stacked,
)
from .tiles import read_tiles

__all__ = [
    "PyramidLevel",
    "get_pyramid_zooms",
    "read_6darray",
    "read_attachments",
    "read_field",
    "read_stacks",
    "read_stacks_list",
    "read_stacks_multiscale",
    "read_stacks_stacked",
    "read_tiles",
    "read_well",
]
