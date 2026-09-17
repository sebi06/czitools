# -*- coding: utf-8 -*-
"""OME-Zarr / OME-NGFF export tools for czitools (Stage 5).

Convert CZI files to OME-Zarr, including HCS (high-content-screening) plate
layouts. The default ngff-zarr backend writes OME-NGFF 0.6 images; its HCS
and RFC-9 OZX writers use 0.5. The ome-zarr-py compatibility backend writes
its current 0.5 format. A MagicGUI application is also provided.

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
        compression_type,
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
_GUI_INSTALL_HINT = (
    "The OME-Zarr GUI requires a Qt binding and other optional dependencies. "
    'Install them with: python -m pip install "czitools[omezarr-gui]".'
)

# public name -> submodule that defines it
_EXPORTS = {
    "compression_type": "_logging",
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
    except Exception as error:
        if module_name == "gui" and error.__class__.__name__ == "QtBindingsNotFoundError":
            raise ImportError(_GUI_INSTALL_HINT) from error
        raise
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)
