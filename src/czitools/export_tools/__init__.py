# -*- coding: utf-8 -*-
"""OME-Zarr / OME-NGFF export tools for czitools (Stage 5).

Convert CZI files to OME-Zarr, including HCS (high-content-screening) plate
layouts, using either the ngff-zarr (OME-NGFF v0.5) or ome-zarr-py
(OME-NGFF v0.4) backend. A MagicGUI application is also provided.

These features require optional dependencies. Install them with::

    pip install "czitools[omezarr]"        # conversion + validation
    pip install "czitools[omezarr-gui]"    # additionally the MagicGUI app

Public API (import lazily; a clear :class:`ImportError` is raised when the
optional dependencies are missing)::

    from czitools.export_tools import (
        convert_czi2hcs_ngff,
        convert_czi2hcs_omezarr,
        write_omezarr,
        write_omezarr_ngff,
        convert_hcs_omezarr2ozx,
        validate_ome_zarr,
        resolve_hcs_layout,
        omezarr_package,
        setup_logging,
        run_gui,
    )
"""

from __future__ import annotations

import importlib
from typing import Any

_INSTALL_HINT = (
    "OME-Zarr export requires optional dependencies. "
    'Install them with: pip install "czitools[omezarr]" '
    '(or "czitools[omezarr-gui]" for the GUI).'
)

# public name -> submodule that defines it
_EXPORTS = {
    "omezarr_package": "_logging",
    "setup_logging": "_logging",
    "extract_well_coordinates": "plate",
    "PlateType": "plate",
    "PlateConfiguration": "plate",
    "define_plate": "plate",
    "define_plate_by_well_count": "plate",
    "convert_hcs_omezarr2ozx": "plate",
    "get_fieldimage": "display",
    "get_display": "display",
    "create_channel_list": "display",
    "resolve_hcs_layout": "resolver",
    "HcsLayout": "resolver",
    "HcsWellLayout": "resolver",
    "convert_czi2hcs_omezarr": "conversion",
    "convert_czi2hcs_ngff": "conversion",
    "write_omezarr": "conversion",
    "write_omezarr_ngff": "conversion",
    "validate_ome_zarr": "validation",
    "run_gui": "gui",
    "create_gui": "gui",
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
