# -*- coding: utf-8 -*-
"""Image display helpers for the OME-Zarr export tools.

Vendored (with light edits) from ``czi_omezarr_utils.display`` in the
``omezarr_playground`` repository as part of czitools Stage 5.

Contents:
  - ``compute_pyramid_scale_factors`` — size-aware multiscale scale factors (Y/X only)
  - ``get_fieldimage`` — extract a scene from a 6D array as a multi-scale NgffImage
  - ``get_display`` — per-channel display-range settings from CZI metadata
  - ``create_channel_list`` — OMERO channel list used by both write backends
"""

import math
from typing import Union
import logging

import numpy as np
import dask.array as da
import xarray as xr
import ngff_zarr as nz
from czitools.metadata_tools.czi_metadata import CziMetadata

logger = logging.getLogger(__name__)


def compute_pyramid_levels(size_y: int, size_x: int, min_size: int = 512, max_levels: int = 6) -> int:
    """Compute the number of resolution levels based on the 2D plane size.

    The pyramid keeps halving the XY plane until the largest XY dimension of the
    coarsest level is roughly ``<= min_size`` pixels (i.e. it fits in about one
    chunk/tile). Small planes therefore get few (or no) extra levels, avoiding
    wasted tiny levels, while large planes get enough levels for smooth zoomed-out
    viewing.

    Args:
        size_y (int): Height of the base-resolution plane in pixels.
        size_x (int): Width of the base-resolution plane in pixels.
        min_size (int): Target maximum XY size of the coarsest level. Defaults to 512.
        max_levels (int): Hard cap on the number of levels. Defaults to 6.

    Returns:
        int: Number of resolution levels (>= 1, where 1 means base only, no pyramid).
    """
    largest = int(max(size_y, size_x))
    if largest <= min_size or min_size <= 0:
        return 1
    # Number of halvings needed for the largest XY dim to reach <= min_size,
    # plus the base level. ceil() picks the first level that fits within min_size.
    n_downsamples = math.ceil(math.log2(largest / min_size))
    n_levels = n_downsamples + 1
    return max(1, min(int(n_levels), int(max_levels)))


def compute_pyramid_scale_factors(size_y: int, size_x: int, min_size: int = 512, max_levels: int = 6) -> list:
    """Build size-aware, Y/X-only downscale factors for ``ngff_zarr.to_multiscales``.

    Returns cumulative factors ``[2, 4, 8, ...]`` applied to the Y and X axes only
    (Z is not downsampled). The number of factors is ``levels - 1`` where ``levels``
    comes from :func:`compute_pyramid_levels`. An empty list means "base only".

    Args:
        size_y (int): Height of the base-resolution plane in pixels.
        size_x (int): Width of the base-resolution plane in pixels.
        min_size (int): Target maximum XY size of the coarsest level. Defaults to 512.
        max_levels (int): Hard cap on the number of levels. Defaults to 6.

    Returns:
        list: A list of per-level dicts ``{"z": 1, "y": 2**i, "x": 2**i}`` (Y/X only,
            Z factor fixed at 1 = no Z downsampling), or an empty list when no pyramid
            is warranted.
    """
    n_levels = compute_pyramid_levels(size_y, size_x, min_size=min_size, max_levels=max_levels)
    # z factor is fixed at 1 (no Z downsampling). ngff-zarr requires every spatial
    # dim (z, y, x) to be present in each factor dict, so z=1 must be included.
    return [{"z": 1, "y": 2**i, "x": 2**i} for i in range(1, n_levels)]


def get_fieldimage(
    array6d: Union[xr.DataArray, np.ndarray, da.Array],
    scene_index: int,
    metadata: CziMetadata,
    min_size: int = 512,
    max_levels: int = 6,
) -> "nz.Multiscales":
    """Extract a field image from a 6D array as a multi-scale representation.

    The number of resolution levels is derived from the 2D (Y, X) plane size via
    :func:`compute_pyramid_scale_factors` (downsampling Y/X only, never Z), so
    small fields get few/no extra levels and large fields get enough for smooth
    zoomed-out viewing.

    Args:
        array6d (Union[xr.DataArray, np.ndarray, da.Array]): 6D array with dimensions
            ``[scene, t, c, z, y, x]``.
        scene_index (int): Index of the scene to extract.
        metadata (CziMetadata): Metadata with scale information and filename.
        min_size (int): Target maximum XY size of the coarsest level. Defaults to 512.
        max_levels (int): Hard cap on the number of levels. Defaults to 6.

    Returns:
        nz.Multiscales: Multi-scale representation using Gaussian downsampling with
            size-aware, Y/X-only scale factors.
    """
    if isinstance(array6d, xr.DataArray):
        data = array6d[scene_index, ...].data
    else:
        data = array6d[scene_index, ...]

    size_y, size_x = int(data.shape[-2]), int(data.shape[-1])
    scale_factors = compute_pyramid_scale_factors(size_y, size_x, min_size=min_size, max_levels=max_levels)

    _scale = metadata.scale
    current_field_image = nz.NgffImage(
        data=data,  # type: ignore[arg-type]
        dims=["t", "c", "z", "y", "x"],
        scale={
            "t": 1.0,
            "c": 1.0,
            "z": float(_scale.Z) if (_scale is not None and _scale.Z is not None) else 1.0,
            "y": float(_scale.Y) if (_scale is not None and _scale.Y is not None) else 1.0,
            "x": float(_scale.X) if (_scale is not None and _scale.X is not None) else 1.0,
        },
        axes_units={
            "t": "second",
            "z": "micrometer",
            "y": "micrometer",
            "x": "micrometer",
        },
        translation={"t": 0.0, "c": 0.0, "z": 0.0, "y": 0.0, "x": 0.0},
        name=metadata.filename if metadata.filename is not None else "image.czi",
    )

    return nz.to_multiscales(  # type: ignore[attr-defined]
        current_field_image,
        scale_factors=scale_factors,
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


def create_ngff_omero_channels(metadata: CziMetadata) -> list:
    """Build ngff-zarr ``OmeroChannel`` objects from CZI metadata.

    These are attached to a multiscale image so that OME-NGFF readers (e.g. ngio /
    napari-ome-zarr-navigator) can resolve per-channel display settings. Without
    OMERO channel metadata, ngio's ``channels_meta`` is ``None`` and the navigator
    ROI loader fails with ``'NoneType' object has no attribute 'channels'``.

    Args:
        metadata (CziMetadata): Metadata with ``channelinfo`` and ``maxvalue_list``.

    Returns:
        list: A list of :class:`ngff_zarr.OmeroChannel` objects (empty if no
            channel metadata is available).
    """
    channels = []
    for ch in create_channel_list(metadata):
        channels.append(
            nz.OmeroChannel(
                color=ch["color"],
                window=nz.OmeroWindow(
                    min=ch["window"]["min"],
                    max=ch["window"]["max"],
                    start=ch["window"]["start"],
                    end=ch["window"]["end"],
                ),
                label=ch["label"],
            )
        )
    return channels
