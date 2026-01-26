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
from czitools.utils.napari_tools import display_xarray_in_napari

show_napari = True

# adapt to your needs
defaultdir = Path(Path(__file__).resolve().parents[2]) / "data"

# option to toogle using a file open dialog or a hardcoded filename
use_dialog = True


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

# return a xarray with dimension order STCZYX(A)
# when no planes are specified the complete dataset
# when planes are specified the metadata will be adapted accordingly for SizeS, SizeT, SizeC and SizeZ
array6d, mdata = read_tools.read_6darray(
    filepath,
    zoom=1.0,
    # planes={"S": (0, 0), "T": (0, 2), "C": (0, 0), "Z": (0, 4)},
    adapt_metadata=True,
)

# get the planes
subset_planes = array6d.attrs["subset_planes"]

# print the shape of the xarray etc.
print(f"Shape: {array6d.shape}")
print(f"Dimensions: {array6d.dims}")
print(f"Subset Planes: {subset_planes}")

for k, v in array6d.attrs.items():
    print(f"{k} :  {v}")

if show_napari:

    # Delegate Napari display to the utility function
    display_xarray_in_napari(array6d, mdata, subset_planes)
