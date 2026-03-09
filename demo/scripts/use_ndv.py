# -*- coding: utf-8 -*-

#################################################################
# File        : use_read_tools.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
# Requires napari & magicgui to be installed!
#
# Requires: ndv-0.5.0rc3-py3-none-any.whl installed
#
#################################################################

from czitools.metadata_tools.czi_metadata import CziMetadata
from czitools.read_tools import read_tools
from pathlib import Path
from magicgui import magicgui
from magicgui.types import FileDialogMode
from czitools.utils.napari_tools import display_xarray_in_napari
import ndv
from cmap import Colormap
import xarray as xr
from typing import Any
from ndv._types import AxisKey

show_napari = False
show_ndv = True
use_mdtsack = True
# option to toggle using a file open dialog or a hardcoded filename
use_dialog = True

# adapt to your needs
defaultdir = Path(Path(__file__).resolve().parents[2]) / "data"

# Keep shared variables initialized for both code paths.
array6d: Any = None
subset_planes: dict[str, Any] = {}
result: Any = None


def normalize_luts(luts_like):
    """Normalize LUT definitions to NDV's channel-indexed mapping format."""
    if isinstance(luts_like, dict):
        return luts_like

    if isinstance(luts_like, list):
        return {i: {"cmap": cmap} for i, cmap in enumerate(luts_like)}

    raise TypeError("luts must be a dict or list")


# open simple dialog to select a CZI file
@magicgui(
    filepath={
        "label": "Choose CZI files:",
        "mode": FileDialogMode.EXISTING_FILE,
        "filter": "*.czi",
        "value": defaultdir,
    },
    call_button="Open CZI File",
)
def filespicker(filepath: Path) -> Path:
    """Take a filename and do something with it."""
    # Close the dialog after the file is selected and the button was pressed
    # This will also return the filepath to the caller
    filespicker.close()

    return filepath


if use_dialog:

    filespicker.filepath.changed.connect(print)
    filespicker.show(run=True)

    filepath = str(filespicker.filepath.value)
    print(f"Selected file: {filepath}")

elif not use_dialog:

    filepath = str(defaultdir / "CellDivision_T10_Z15_CH2_DCV_small.czi")


if not use_mdtsack:
    # return a xarray with dimension order STCZYX(A)
    # when no planes are specified the complete dataset
    # when planes are specified the metadata will be adapted accordingly for SizeS, SizeT, SizeC and SizeZ
    array6d, mdata = read_tools.read_6darray(
        filepath,
        zoom=1.0,
        # planes={"S": (0, 0), "T": (0, 2), "C": (0, 0), "Z": (0, 4)},
        adapt_metadata=True,
        use_dask=False,
        use_xarray=True,
    )

    if array6d is None:
        raise RuntimeError("read_6darray returned no data.")

    # get the planes
    attrs = getattr(array6d, "attrs", {})
    subset_planes = attrs.get("subset_planes", {}) if isinstance(attrs, dict) else {}

    # print the shape of the xarray etc.
    print(f"Shape: {array6d.shape}")
    print(f"Dimensions: {array6d.dims}")
    print(f"Subset Planes: {subset_planes}")

    if isinstance(attrs, dict):
        for k, v in attrs.items():
            print(f"{k} :  {v}")

elif use_mdtsack:

    mdata = CziMetadata(filepath)

    result, dims, num_stacks = read_tools.read_stacks(
        filepath,
        use_dask=False,
        use_xarray=True,
        stack_scenes=True,
    )

    print("\n=== Results ===")
    print(f"Number of stacks: {num_stacks}")
    print(f"Dimension order: {dims}")

    if isinstance(result, list):
        # List of per-scene arrays
        for idx, arr in enumerate(result):
            if isinstance(arr, xr.DataArray):
                print(f"Stack {idx}: dims={arr.dims}, shape={arr.shape}, dtype={arr.dtype}")
            else:
                print(f"Stack {idx}: shape={arr.shape}, dims={dims}, dtype={arr.dtype}")
    else:
        # Single stacked array
        if isinstance(result, xr.DataArray):
            print(f"Stacked: dims={result.dims}, shape={result.shape}, dtype={result.dtype}")
        else:
            print(f"Stacked: shape={result.shape}, dims={dims}, dtype={result.dtype}")

        # With use_dask=True, result is backed by dask - no data loaded yet
        print(f"\nArray shape (no data loaded): {result.shape}")

###  SHOW RESULTS IN VIEWER

if show_napari:

    # Delegate Napari display to the utility function
    if array6d is not None:
        display_xarray_in_napari(array6d, mdata, subset_planes)

if show_ndv:
    luts = {}

    size_c = int(getattr(mdata.image, "SizeC", 0) or 0)
    names = getattr(mdata.channelinfo, "names", None) or []
    colors = getattr(mdata.channelinfo, "colors", None) or []

    for ch_index in range(0, size_c):
        chname = names[ch_index] if ch_index < len(names) else f"ch{ch_index}"
        # ARGB stored in metadata as hexstring; convert to #RRGGBB
        color_argb = colors[ch_index] if ch_index < len(colors) else "FF00FF00"
        rgb = "#" + str(color_argb)[3:]
        luts[ch_index] = {"cmap": Colormap(["#000000", rgb], name="cm_" + chname)}

    imshow_kwargs = {
        "channel_mode": "composite",
        "channel_axis": "C",
        "luts": normalize_luts(luts),
    }

    # ndv >= commit c7955c8 supports `scales`; older versions require coords fallback.
    # {"Z": z_scale, "Y": y_scale, "X": x_scale}
    scales: dict[AxisKey, float] = {
        "Z": float(getattr(mdata.scale, "Z", 1.0) or 1.0),
        "Y": float(getattr(mdata.scale, "Y", 1.0) or 1.0),
        "X": float(getattr(mdata.scale, "X", 1.0) or 1.0),
    }

    viewer_data = None
    if isinstance(result, list):
        viewer_data = result[0] if len(result) > 0 else None
    elif result is not None:
        viewer_data = result
    elif array6d is not None:
        viewer_data = array6d

    if viewer_data is None:
        raise RuntimeError("No data available for NDV display.")

    viewer = ndv.imshow(viewer_data, scales=scales, **imshow_kwargs)
