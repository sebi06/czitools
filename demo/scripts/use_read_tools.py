# -*- coding: utf-8 -*-

#################################################################
# File        : use_read_tools.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from czitools import read_tools
from czitools import napari_tools
from czitools import misc_tools
import napari
from pathlib import Path

# adapt to your needs
defaultdir = Path(Path(__file__).resolve().parents[2]) / "data"

# open s simple dialog to select a CZI file
filepath = misc_tools.openfile(
    directory=defaultdir,
    title="Open CZI Image File",
    ftypename="CZI Files",
    extension="*.czi",
)

print(filepath)

# return a array with dimension order STZCYX(A)
array6d, mdata, dim_string6d = read_tools.read_6darray(
    filepath,
    output_order="STCZYX",
    use_dask=False,
    chunk_zyx=False,
    # planes={"S": (0, 1), "T": (1, 2), "C": (0, 0), "Z": (0, 2)},
    # planes={"Z": (2, 2)},
    # planes={"S": (4, 6)},
)

if array6d is None:
    print("Empty array6d. Nothing to display in Napari")
else:
    # show array inside napari viewer
    viewer = napari.Viewer()
    layers = napari_tools.show(
        viewer,
        array6d,
        mdata,
        dim_string=dim_string6d,
        blending="additive",
        contrast="from_czi",
        gamma=0.85,
        show_metadata="tree",
        name_sliders=True,
    )

    napari.run()
