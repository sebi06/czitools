# -*- coding: utf-8 -*-
"""Image display helpers for the OME-Zarr export tools.

Vendored (with light edits) from ``czi_omezarr_utils.display`` in the
``omezarr_playground`` repository as part of czitools Stage 5.

Contents:
  - ``get_fieldimage`` — extract a scene from a 6D array as a multi-scale NgffImage
  - ``get_display`` — per-channel display-range settings from CZI metadata
  - ``create_channel_list`` — OMERO channel list used by both write backends
"""

from typing import Union
import logging

import numpy as np
import dask.array as da
import xarray as xr
import ngff_zarr as nz
from czitools.metadata_tools.czi_metadata import CziMetadata

logger = logging.getLogger(__name__)


def get_fieldimage(
    array6d: Union[xr.DataArray, np.ndarray, da.Array],
    scene_index: int,
    metadata: CziMetadata,
) -> "nz.Multiscales":
    """Extract a field image from a 6D array as a multi-scale representation.

    Args:
        array6d (Union[xr.DataArray, np.ndarray, da.Array]): 6D array with dimensions
            ``[scene, t, c, z, y, x]``.
        scene_index (int): Index of the scene to extract.
        metadata (CziMetadata): Metadata with scale information and filename.

    Returns:
        nz.Multiscales: Multi-scale representation using Gaussian downsampling with
            scale factors ``[2, 2, 2]``.
    """
    if isinstance(array6d, xr.DataArray):
        data = array6d[scene_index, ...].data
    else:
        data = array6d[scene_index, ...]

    _scale = metadata.scale
    current_field_image = nz.NgffImage(
        data=data,  # type: ignore[arg-type]
        dims=["t", "c", "z", "y", "x"],
        scale={
            "y": float(_scale.Y) if (_scale is not None and _scale.Y is not None) else 1.0,
            "x": float(_scale.X) if (_scale is not None and _scale.X is not None) else 1.0,
            "z": float(_scale.Z) if (_scale is not None and _scale.Z is not None) else 1.0,
        },
        translation={"t": 0.0, "c": 0.0, "z": 0.0, "y": 0.0, "x": 0.0},
        name=metadata.filename if metadata.filename is not None else "image.czi",
    )

    return nz.to_multiscales(  # type: ignore[attr-defined]
        current_field_image,
        scale_factors=[2, 2, 2],
        method=nz.Methods.DASK_IMAGE_GAUSSIAN,  # type: ignore[attr-defined]
    )


def get_display(metadata: CziMetadata, channel_index: int) -> tuple[float, float, float]:
    """Extract display-range settings for a channel from CZI metadata.

    Args:
        metadata (CziMetadata): Metadata with channel display settings.
        channel_index (int): Zero-based channel index.

    Returns:
        tuple[float, float, float]: ``(lower, higher, maxvalue)``. Falls back to
            ``(0, maxvalue, maxvalue)`` when metadata is missing or corrupted.
    """
    channelinfo = metadata.channelinfo
    maxvalue_list = metadata.maxvalue_list
    try:
        assert channelinfo is not None, "channelinfo is None"
        assert maxvalue_list is not None, "maxvalue_list is None"
        lower = np.round(channelinfo.clims[channel_index][0] * maxvalue_list[channel_index], 0)
        higher = np.round(channelinfo.clims[channel_index][1] * maxvalue_list[channel_index], 0)
        maxvalue = maxvalue_list[channel_index]
    except (IndexError, AssertionError):
        logger.warning("Display setting from CZI unavailable. Using 0-Max instead.")
        lower = 0.0
        higher = float(maxvalue_list[channel_index]) if maxvalue_list is not None else 0.0
        maxvalue = higher

    return lower, higher, maxvalue


def create_channel_list(metadata: CziMetadata) -> list:
    """Build the OMERO channel list used by both write backends.

    Args:
        metadata (CziMetadata): Metadata with ``channelinfo`` and ``maxvalue_list``.

    Returns:
        list: Channel dicts with keys ``color``, ``label``, ``active``, ``window``.
    """
    channels_list: list = []

    image = metadata.image
    if image is None:
        return channels_list

    channelinfo = metadata.channelinfo
    if channelinfo is None:
        return channels_list

    for ch_index in range(image.SizeC or 0):
        rgb = channelinfo.colors[ch_index][3:]
        chname = channelinfo.names[ch_index]
        lower, higher, maxvalue = get_display(metadata, ch_index)
        channels_list.append(
            {
                "color": rgb,
                "label": chname,
                "active": True,
                "window": {
                    "min": lower,
                    "start": lower,
                    "end": higher,
                    "max": maxvalue,
                },
            }
        )

    return channels_list
