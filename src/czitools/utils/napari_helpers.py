# -*- coding: utf-8 -*-

#################################################################
# File        : napari_helpers.py
# Author      : sebi06
#
# Disclaimer: The code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

"""
Helper functions for safe Napari integration.

This module provides utilities to ensure thread-safe CZI reading
when working with Napari, especially on Linux systems where
aicspylibczi can cause threading conflicts.
"""

import os
import warnings
from typing import Union, Optional, Tuple
from pathlib import Path
from czitools.utils import logging_tools

logger = logging_tools.set_logging()


def enable_napari_safe_mode() -> None:
    """
    Enable Napari-safe mode by disabling aicspylibczi.

    This function sets the CZITOOLS_DISABLE_AICSPYLIBCZI environment variable
    to prevent threading conflicts between aicspylibczi and Napari's PyQt event loop.

    **MUST be called BEFORE importing any czitools modules that read CZI files.**

    Example:
        >>> from czitools.utils.napari_helpers import enable_napari_safe_mode
        >>> enable_napari_safe_mode()  # Call this FIRST
        >>> from czitools.read_tools import read_tools  # Now safe to import
        >>> array, mdata = read_tools.read_6darray(filepath, use_dask=True)

    Note:
        This disables:
        - read_tiles() function (will raise RuntimeError)
        - get_planetable() function (returns empty DataFrame)
        - Some auxiliary metadata fields from aicspylibczi

        These remain fully functional:
        - read_6darray() - RECOMMENDED for Napari
        - read_stacks()
        - CziMetadata (core metadata)
    """
    os.environ["CZITOOLS_DISABLE_AICSPYLIBCZI"] = "1"
    logger.info("Napari-safe mode enabled: aicspylibczi disabled to prevent threading conflicts")


def is_napari_safe_mode() -> bool:
    """
    Check if Napari-safe mode is currently enabled.

    Returns:
        bool: True if aicspylibczi is disabled, False otherwise.

    Example:
        >>> from czitools.utils.napari_helpers import is_napari_safe_mode
        >>> if is_napari_safe_mode():
        ...     print("Running in thread-safe mode")
    """
    return os.environ.get("CZITOOLS_DISABLE_AICSPYLIBCZI", "0") == "1"


def check_napari_compatibility() -> Tuple[bool, str]:
    """
    Check if the current configuration is compatible with Napari.

    Returns:
        Tuple[bool, str]: A tuple of (is_compatible, message) where:
            - is_compatible: True if configuration is safe for Napari
            - message: Explanation or recommendation

    Example:
        >>> from czitools.utils.napari_helpers import check_napari_compatibility
        >>> compatible, msg = check_napari_compatibility()
        >>> if not compatible:
        ...     print(f"Warning: {msg}")
    """
    if is_napari_safe_mode():
        return True, "Thread-safe mode enabled - compatible with Napari"
    else:
        return (
            False,
            "WARNING: aicspylibczi is enabled and may cause crashes with Napari on Linux. "
            "Call enable_napari_safe_mode() before importing czitools reading functions.",
        )


def get_recommended_read_params() -> dict:
    """
    Get recommended parameters for reading CZI files in Napari.

    Returns:
        dict: Dictionary of recommended parameters for read_6darray() or read_stacks()

    Example:
        >>> from czitools.read_tools import read_tools
        >>> from czitools.utils.napari_helpers import get_recommended_read_params
        >>> params = get_recommended_read_params()
        >>> array, mdata = read_tools.read_6darray(filepath, **params)
    """
    return {
        "use_dask": True,  # Lazy loading - essential for large files
        "use_xarray": True,  # Labeled dimensions - easier to work with
        "chunk_zyx": True,  # Optimize chunking for performance
    }


def warn_if_unsafe_for_napari() -> None:
    """
    Issue a warning if the current configuration may not be safe for Napari.

    This function checks if aicspylibczi is enabled and issues a warning
    if it is, as this can cause threading conflicts with Napari on Linux.

    Example:
        >>> from czitools.utils.napari_helpers import warn_if_unsafe_for_napari
        >>> warn_if_unsafe_for_napari()
    """
    compatible, message = check_napari_compatibility()
    if not compatible:
        warnings.warn(
            f"\n{'='*70}\n"
            f"{message}\n\n"
            f"To fix this issue:\n"
            f"1. Import: from czitools.utils.napari_helpers import enable_napari_safe_mode\n"
            f"2. Call: enable_napari_safe_mode() BEFORE importing read functions\n"
            f"3. Use: read_6darray(filepath, use_dask=True) for optimal Napari integration\n"
            f"{'='*70}\n",
            RuntimeWarning,
            stacklevel=2,
        )


# Automatically check compatibility when this module is imported
# This provides early warning if there might be issues
if __name__ != "__main__":
    # Only warn if we're being imported as a module, not run as script
    try:
        # Check if napari is already imported (indicates we're in a Napari context)
        import sys

        if "napari" in sys.modules and not is_napari_safe_mode():
            logger.warning(
                "napari is already loaded but aicspylibczi is not disabled. "
                "This may cause threading conflicts on Linux. "
                "Call enable_napari_safe_mode() to fix."
            )
    except Exception:
        pass  # Silently ignore any errors during compatibility check
