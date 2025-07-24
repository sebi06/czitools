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
#################################################################

from czitools.read_tools import read_tools
from pathlib import Path
from magicgui import magicgui
from magicgui.types import FileDialogMode

try:
    import napari
    from napari.utils.colormaps import Colormap

    show_napari = True
except ImportError:
    print("Napari not installed, skipping napari import")
    show_napari = False

# adapt to your needs
defaultdir = Path(Path(__file__).resolve().parents[2]) / "data"


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


filespicker.filepath.changed.connect(print)
filespicker.show(run=True)

filepath = str(filespicker.filepath.value)
print(f"Selected file: {filepath}")

# return an array with dimension order STCZYX(A)
array6d, mdata = read_tools.read_6darray(
    filepath,
    # use_dask=False,
    # chunk_zyx=False,
    # zoom=1.0,
    # planes={"S": (0, 0), "T": (1, 2), "C": (0, 0), "Z": (0, 2)},
    # use_xarray=True,
)

# print the shape of the array etc.
print(f"Shape: {array6d.shape}")
print(f"Dimensions: {array6d.dims}")

for k, v in array6d.attrs.items():
    print(f"{k} :  {v}")

if show_napari:

    # show in napari (requires napari to be installed!)
    viewer = napari.Viewer()

    # loop over all channels
    for ch in range(0, array6d.sizes["C"]):

        # extract channel subarray
        sub_array = array6d.sel(C=ch)

        # get the scaling factors for that channel and adapt Z-axis scaling
        scalefactors = [1.0] * len(sub_array.shape)
        scalefactors[sub_array.get_axis_num("Z")] = mdata.scale.ratio["zx_sf"]  # * 1.00001

        # remove the last scaling factor in case of an RGB image
        if "A" in sub_array.dims:
            # remove the A axis from the scaling factors
            scalefactors.pop(sub_array.get_axis_num("A"))

        # get colors and channel name
        chname = mdata.channelinfo.names[ch]

        # inside the CZI metadata_tools colors are defined as ARGB hexstring
        rgb = "#" + mdata.channelinfo.colors[ch][3:]
        ncmap = Colormap(["#000000", rgb], name="cm_" + chname)

        # add the channel to the viewer
        viewer.add_image(
            sub_array,
            name=chname,
            colormap=ncmap,
            blending="additive",
            scale=scalefactors,
            gamma=0.85,
        )

        # set the axis labels based on the dimensions
        viewer.dims.axis_labels = sub_array.dims

    napari.run()
