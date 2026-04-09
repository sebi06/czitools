"""Napari display helpers for czitools.

This module provides a convenience function to display an xarray
DataArray with canonical CZITools axes in Napari. It centralizes
color/scale handling and keeps demo scripts small.
"""

from typing import Any, Mapping, Sequence


def _add_xarray_to_viewer(
    viewer: Any, array6d: Any, mdata: Any, subset_planes_map: Mapping[str, Any], layer_prefix: str = ""
) -> None:
    """Add one xarray stack to an existing napari viewer."""

    # Loop channels and add to viewer
    for ch in range(0, array6d.sizes["C"]):
        # Prefer label-based selection. This supports numeric labels and
        # string labels (for example channel names).
        sub_array = None
        c_coord = array6d.coords.get("C")
        if c_coord is not None:
            try:
                ch_label = c_coord.values[ch]
                sub_array = array6d.sel(C=ch_label)
            except Exception:
                sub_array = None

        # Fallback for arrays without usable C labels.
        if sub_array is None:
            sub_array = array6d.isel(C=ch)

        # scaling factors: keep 1.0 and adapt Z axis to metadata ratio
        scalefactors = [1.0] * len(sub_array.shape)
        z_axis = sub_array.get_axis_num("Z")
        scalefactors[z_axis] = mdata.scale.ratio["zx_sf"]

        # remove A axis scale if present
        if "A" in sub_array.dims:
            scalefactors.pop(sub_array.get_axis_num("A"))

        c_range = subset_planes_map.get("C")
        if isinstance(c_range, (list, tuple)) and len(c_range) >= 1:
            ch_index = int(c_range[0]) + ch
        else:
            ch_index = ch

        names = list(getattr(mdata.channelinfo, "names", None) or [])
        colors = list(getattr(mdata.channelinfo, "colors", None) or [])
        chname = str(names[ch_index]) if ch_index < len(names) else f"ch{ch_index}"

        # ARGB stored in metadata as hexstring; convert to #RRGGBB
        color_argb = str(colors[ch_index]) if ch_index < len(colors) else "FF00FF00"
        rgb = f"#{color_argb[-6:]}" if len(color_argb) >= 8 else "#00FF00"

        from napari.utils.colormaps import Colormap

        ncmap = Colormap(["#000000", rgb], name="cm_" + chname)

        viewer.add_image(
            sub_array,
            name=f"{layer_prefix}{chname}",
            colormap=ncmap,
            blending="additive",
            scale=scalefactors,
            gamma=0.85,
        )

        viewer.dims.axis_labels = sub_array.dims


def display_xarray_in_napari(array6d: Any, mdata: Any, subset_planes: Mapping[str, Any] | None = None) -> None:
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
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Napari is required for display_xarray_in_napari but is not installed. "
            "Install napari or skip viewer display. Original error: " + str(exc)
        )

    if subset_planes is None:
        subset_planes_map: Mapping[str, Any] = array6d.attrs.get("subset_planes", {})
    else:
        subset_planes_map = subset_planes

    viewer = napari.Viewer()
    _add_xarray_to_viewer(viewer, array6d, mdata, subset_planes_map)

    napari.run()


def display_xarray_list_in_napari(
    arrays: Sequence[Any],
    mdata: Any,
    subset_planes_list: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Display a list of xarray stacks in one Napari viewer.

    This supports stacks with different spatial shapes by adding each stack as
    its own set of layers.
    """
    try:
        import napari
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Napari is required for display_xarray_list_in_napari but is not installed. "
            "Install napari or skip viewer display. Original error: " + str(exc)
        )

    viewer = napari.Viewer()
    for idx, arr in enumerate(arrays):
        if subset_planes_list is not None and idx < len(subset_planes_list):
            subset_planes_map = subset_planes_list[idx]
        else:
            subset_planes_map = arr.attrs.get("subset_planes", {})
        _add_xarray_to_viewer(viewer, arr, mdata, subset_planes_map, layer_prefix=f"S{idx}_")

    napari.run()
