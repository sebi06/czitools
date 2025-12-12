"""Napari display helpers for czitools.

This module provides a convenience function to display an xarray
DataArray with canonical CZITools axes in Napari. It centralizes
color/scale handling and keeps demo scripts small.
"""

from typing import Dict

import numpy as np


def display_xarray_in_napari(array6d, mdata, subset_planes: Dict[str, int] | Dict[str, int] = None):
    """Display an xarray DataArray (STCZYX[A]) in Napari.

    Args:
        array6d: xarray.DataArray with axes including C and Z.
        mdata: CziMetadata instance (used for channel names/colors/scaling).
        subset_planes: dict describing subset planes (as returned in
            `array6d.attrs['subset_planes']`). If provided, used to map
            channel indices and names.

    Notes:
        - If Napari is not installed this function raises ImportError.
    """
    # Import napari lazily so czitools does not require it as a dependency.
    try:
        import napari
        from napari.utils.colormaps import Colormap
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Napari is required for display_xarray_in_napari but is not installed. "
            "Install napari or skip viewer display. Original error: " + str(exc)
        )

    if subset_planes is None:
        subset_planes = array6d.attrs.get("subset_planes", {})

    viewer = napari.Viewer()

    # Loop channels and add to viewer
    for ch in range(0, array6d.sizes["C"]):
        sub_array = array6d.sel(C=ch)

        # scaling factors: keep 1.0 and adapt Z axis to metadata ratio
        scalefactors = [1.0] * len(sub_array.shape)
        z_axis = sub_array.get_axis_num("Z")
        scalefactors[z_axis] = mdata.scale.ratio["zx_sf"]

        # remove A axis scale if present
        if "A" in sub_array.dims:
            scalefactors.pop(sub_array.get_axis_num("A"))

        ch_index = (subset_planes.get("C", [0, 0])[0] + ch) if isinstance(subset_planes.get("C"), list) else ch
        chname = mdata.channelinfo.names[ch_index]

        # ARGB stored in metadata as hexstring; convert to #RRGGBB
        rgb = "#" + mdata.channelinfo.colors[ch_index][3:]
        ncmap = Colormap(["#000000", rgb], name="cm_" + chname)

        viewer.add_image(
            sub_array,
            name=chname,
            colormap=ncmap,
            blending="additive",
            scale=scalefactors,
            gamma=0.85,
        )

        viewer.dims.axis_labels = sub_array.dims

    napari.run()
