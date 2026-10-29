# -*- coding: utf-8 -*-

#################################################################
# File        : write_omezarr.py
# Author      : sebi06
#
# Requires: ome-zarr, ngff-zarr
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

import argparse

from czitools.read_tools import read_tools
from czitools.metadata_tools import czi_metadata as czimd
from czitools.export_tools.conversion import _write_original_czi_metadata
import ngff_zarr as nz
from pathlib import Path
import dask.array as da
import zarr
import ome_zarr.writer
import ome_zarr.format
from ome_zarr.io import parse_url
from typing import Union
import shutil
import numpy as np
from czitools.utils import logging_tools

logger = logging_tools.set_logging()


def write_omezarr(
    array5d: Union[np.ndarray, da.Array],
    zarr_path: str,
    axes: str = "tczyx",
    overwrite: bool = False,
    czi_filepath: str | Path | None = None,
    preserve_original_metadata: bool = False,
) -> str:
    """Write a 5D image with optional original CZI XML preservation.

    Args:
        array5d (np.ndarray | da.Array): Pixels in the selected axis order.
        zarr_path (str): Destination directory store.
        axes (str): Axis order, defaulting to TCZYX.
        overwrite (bool): Replace an existing output store.
        czi_filepath (str | Path | None): Source CZI path for XML preservation.
        preserve_original_metadata (bool): Embed source XML at CZI/metadata.xml.

    Returns:
        str: Output path on success, or None when the write is skipped.

    Raises:
        ValueError: XML preservation is requested without a source path.
    """

    if preserve_original_metadata and czi_filepath is None:
        raise ValueError("czi_filepath is required to preserve original CZI XML.")

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
    elif Path(zarr_path).exists() and not overwrite:
        logger.warning(f"File already exists at {zarr_path}. Set overwrite=True to remove.")
        return None

    # show currently used version of NGFF specification
    ngff_version = ome_zarr.format.CurrentFormat().version
    logger.info(f"Using ngff format version: {ngff_version}")

    # write the image data
    store = parse_url(zarr_path, mode="w").store
    root = zarr.group(store=store, overwrite=overwrite)
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

    if preserve_original_metadata:
        _write_original_czi_metadata(czi_filepath, Path(zarr_path))

    logger.info(f"Finished writing OME-ZARR to: {zarr_path}")

    return zarr_path


# open s simple dialog to select a CZI file
defaultdir = Path(Path(__file__).resolve().parents[2]) / "data"
filepath = defaultdir / "CellDivision_T3_Z5_CH2_X240_Y170.czi"
parser = argparse.ArgumentParser(description="Write a CZI scene with both OME-Zarr backends.")
parser.add_argument("filepath", nargs="?", type=Path, default=filepath)
parser.add_argument(
    "--preserve-original-metadata",
    action="store_true",
    help="Embed the original CZI XML at CZI/metadata.xml in both output stores.",
)
args = parser.parse_args()
filepath = args.filepath

# get the metadata_tools at once as one big class
mdata = czimd.CziMetadata(filepath)
print("Number of Scenes: ", mdata.image.SizeS)
scene_id = 0

# return a 6D array with dimension order STCZYX(A)
array, mdata = read_tools.read_6darray(filepath, use_dask=True)
array = array[scene_id, ...]

# Approach 1: Use ome-zarr-py to write OME-ZARR
zarr_path1 = Path(str(filepath)[:-4] + "_1.ome.zarr")

# write OME-ZARR using utility function
zarr_path1 = write_omezarr(
    array,
    zarr_path=str(zarr_path1),
    axes="tczyx",
    overwrite=True,
    czi_filepath=filepath,
    preserve_original_metadata=args.preserve_original_metadata,
)

print(f"Written OME-ZARR using ome-zarr.py: {zarr_path1}")

# Approach 2: Use ngff-zarr to create NGFF structure and write using ome-zarr-py
zarr_path2 = Path(str(filepath)[:-4] + "_2.ome.zarr")

# create NGFF image from the array
image = nz.to_ngff_image(
    array.data,
    dims=["t", "c", "z", "y", "x"],
    scale={"y": mdata.scale.Y, "x": mdata.scale.X, "z": mdata.scale.Z},
    name=mdata.filename,
)

# create multi-scaled, chunked data structure from the image
multiscales = nz.to_multiscales(image, [2, 4], method=nz.Methods.DASK_IMAGE_GAUSSIAN)

# write using ngff-zarr
nz.to_ngff_zarr(zarr_path2, multiscales)
if args.preserve_original_metadata:
    _write_original_czi_metadata(filepath, zarr_path2)
print(f"NGFF Image: {image}")
print(f"Written OME-ZARR using ngff-zarr: {zarr_path2}")
