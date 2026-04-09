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
Helper functions for Napari integration with czitools.
"""


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
