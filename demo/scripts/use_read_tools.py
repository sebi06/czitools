# -*- coding: utf-8 -*-

#################################################################
# File        : use_read_tools.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
# Requires napari to be installed!
#
#################################################################

from czitools.read_tools import read_tools
from czitools.utils import misc
from pathlib import Path

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
filepath = misc.openfile(
    directory=defaultdir,
    title="Open CZI Image File",
    ftypename="CZI Files",
    extension="*.czi",
)

print(filepath)

# return an array with dimension order STCZYX(A)
array6d, mdata = read_tools.read_6darray(
    filepath,
    use_dask=False,
    chunk_zyx=False,
    zoom=1.0,
    # planes={"S": (0, 0), "T": (1, 2), "C": (0, 0), "Z": (0, 2)},
    use_xarray=True,
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
        scalefactors[sub_array.get_axis_num("Z")] = mdata.scale.ratio[
            "zx_sf"
        ]  # * 1.00001

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
