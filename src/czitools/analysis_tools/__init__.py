# -*- coding: utf-8 -*-
"""Analysis and visualisation tools for czitools.

Image-analysis helpers (2D processing, HCS OME-Zarr object counting) and
well-plate visualisation. These build on top of the OME-Zarr export tools and
are useful for analysing high-content-screening (HCS) plate data.

These features require optional dependencies. Install them with::

    pip install "czitools[analysis]"

Public API (import lazily; a clear :class:`ImportError` is raised when the
optional dependencies are missing)::

    from czitools.analysis_tools import (
        ArrayProcessor,
        process_hcs_omezarr,
        create_well_plate_heatmap,
    )
"""

from __future__ import annotations

import importlib
from typing import Any

_INSTALL_HINT = (
    "czitools analysis tools require optional dependencies. " 'Install them with: pip install "czitools[analysis]".'
)

# public name -> submodule that defines it
_EXPORTS = {
    "ArrayProcessor": "processing",
    "process_hcs_omezarr": "hcs_analysis",
    "create_well_plate_heatmap": "plotting",
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(f".{module_name}", __name__)
    except ModuleNotFoundError as error:
        raise ImportError(f"{_INSTALL_HINT}\nMissing dependency: {error.name}") from error
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)
