# -*- coding: utf-8 -*-

#################################################################
# File        : write_tools.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

# the ome_zarr imports we require
from pathlib import Path
import dask.array as da
import zarr
import ome_zarr.reader
import ome_zarr.scale
import ome_zarr.writer
import ome_zarr.format
from ome_zarr.io import parse_url
from typing import Union
import shutil
import numpy as np
from czitools.utils import logging_tools

#logger = logging_tools.get_logger()
logger = logging_tools.set_logging()


def write_omezarr(
    array5d: Union[np.ndarray, da.Array],
    zarr_path: str,
    axes: str = "tczyx",
    overwrite: bool = False,
) -> str:
    """
    Simple function to write OME-ZARR from an 5D numpy yor dask array

    Args:
        array5d (Union[np.ndarary, da.Array]): Up-to 5 dimensional array to be written as OME-ZARR
        zarr_path (str): Path for the OME-ZARR folder to be created
        axes (str): Defines the dimension order (lower case string). Defaults to "STCZYX"
        overwrite (False): If True, then an existing folder will be overwritten. Defaults to False

    Returns:
        str: Path for location of OME-ZARR folder
    """

    # check number of dimension of input array
    if len(array5d.shape) > 5:
        logger.warning("Input array as more than 5 dimensions.")
        return None

    # make sure lower case is use for axes order
    axes = axes.lower()

    # check for invalid dimensions and clean up
    for character in ["b", "h", "s", "i", "v", "a"]:
        axes = axes.replace(character, "")

    # check if zarr_path already exits
    if Path(zarr_path).exists() and overwrite:
        shutil.rmtree(zarr_path, ignore_errors=False, onerror=None)

    # show currently used version of NGFF specification
    ngff_version = ome_zarr.format.CurrentFormat().version
    logger.info(f"Using ngff format version: {ngff_version}")

    # write the image data
    store = parse_url(zarr_path, mode="w").store
    root = zarr.group(store=store)
    # root.info

    # TODO: Add Channel information etc. to the root along those lines
    """
    # add omero metadata_tools: the napari ome-zarr plugin uses this to pass rendering
    # options to napari.
    root.attrs['omero'] = {
        'channels': [{
                'color': 'ffffff',
                'label': 'LS-data',
                'active': True,
                }]
        }
    
    """

    # write the OME-ZARR file
    ome_zarr.writer.write_image(
        image=array5d,
        group=root,
        axes=axes,
        storage_options=dict(chunks=array5d.shape),
    )

    logger.info(f"Finished writing OME-ZARR to: {zarr_path}")

    if Path(zarr_path).exists():
        return zarr_path
    if not Path(zarr_path).exists():
        return None
