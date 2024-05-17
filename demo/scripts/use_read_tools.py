# -*- coding: utf-8 -*-

#################################################################
# File        : use_read_tools.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from czitools.read_tools import read_tools
from czitools.napari_tools import napari_tools
from czitools.utils import misc
import napari
from pathlib import Path

# adapt to your needs
defaultdir = Path(Path(__file__).resolve().parents[2]) / "data"

# open s simple dialog to select a CZI file
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
    # planes={"Z": (2, 2)},
    # planes={"S": (4, 6)},
    # zoom=1.0,
)

# this is a rather experimental function

# return an array with dimension order STCZYX(A)
# array6d, mdata = read_tools.read_6darray_lazy(filepath, chunk_zyx=True)

if array6d is None:
    print("Empty array6d. Nothing to display in Napari")
else:
    # show array inside napari viewer
    viewer = napari.Viewer()
    layers = napari_tools.show(
        viewer,
        array6d,
        mdata,
        blending="additive",
        contrast="from_czi",
        gamma=0.85,
        show_metadata="table",
        name_sliders=True,
    )

    napari.run()
