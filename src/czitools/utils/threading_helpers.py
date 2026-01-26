# -*- coding: utf-8 -*-

#################################################################
# File        : threading_helpers.py
# Author      : sebi06
#
# Disclaimer: The code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

"""
Thread-safety helpers for aicspylibczi operations.

This module provides thread-safe wrappers for aicspylibczi operations
to prevent conflicts with PyQt/Napari event loops on Linux.

IMPORTANT: The threading lock approach provides basic thread-safety but may
NOT fully resolve PyQt event loop conflicts on Linux with Napari.

For maximum safety with Napari on Linux, set CZITOOLS_DISABLE_AICSPYLIBCZI=1
to completely disable aicspylibczi and use only pylibCZIrw (read_6darray).
"""

import os
import threading
from typing import Callable
from czitools.utils import logging_tools

logger = logging_tools.set_logging()

# Global lock for thread-safe aicspylibczi operations
_AICS_LOCK = threading.RLock()


def with_aics_lock(func: Callable) -> Callable:
    """
    Decorator to wrap aicspylibczi operations with a thread lock.

    This provides basic thread-safety by ensuring only one thread
    accesses aicspylibczi at a time.

    **WARNING**: This may NOT fully resolve PyQt/Napari conflicts on Linux.
    For maximum safety, use CZITOOLS_DISABLE_AICSPYLIBCZI=1 instead.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function with lock protection
    """

    def wrapper(*args, **kwargs):
        with _AICS_LOCK:
            return func(*args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


def is_napari_safe() -> bool:
    """
    Check if the current configuration is safe for use with Napari.

    Returns:
        bool: True if aicspylibczi is disabled (safest configuration)
    """
    return os.environ.get("CZITOOLS_DISABLE_AICSPYLIBCZI", "0") == "1"


def warn_if_unsafe_for_napari():
    """
    Warn the user if using aicspylibczi with Napari on Linux.

    This is called automatically by read_tiles() and get_planetable().
    """
    if not is_napari_safe():
        import platform

        if platform.system() == "Linux":
            logger.warning(
                "WARNING: read_tiles() and get_planetable() use aicspylibczi which may crash "
                "when used with Napari on Linux. If you experience crashes, set "
                "CZITOOLS_DISABLE_AICSPYLIBCZI=1 and use read_6darray() instead."
            )
