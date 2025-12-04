# -*- coding: utf-8 -*-

#################################################################
# File        : read_planetable.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
# Requires magicgui to be installed!
#################################################################

from czitools.metadata_tools.czi_metadata import CziMetadata
from czitools.metadata_tools.dimension import CziDimensions
from czitools.utils import planetable
from pathlib import Path
from magicgui import magicgui
from magicgui.types import FileDialogMode
from czitools.utils import logging_tools

logger = logging_tools.set_logging()

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
    call_button="Read Plantable from CZI File",
)
def filespicker(filepath: Path) -> Path:
    """Take a filename and do something with it."""
    # Close the dialog after the file is selected and the button was pressed
    # This will return the filepath to the caller
    filespicker.close()

    return filepath


filespicker.filepath.changed.connect(print)
filespicker.show(run=True)

# Get the filepath and convert to string
filepath = str(filespicker.filepath.value)
logger.info(f"Selected file: {filepath}")

# get the metadata_tools at once as one big class
mdata = CziMetadata(filepath)

# get only specific metadata_tools
czi_dimensions = CziDimensions(filepath)
print("SizeS: ", czi_dimensions.SizeS)
print("SizeT: ", czi_dimensions.SizeT)
print("SizeZ: ", czi_dimensions.SizeZ)
print("SizeC: ", czi_dimensions.SizeC)
print("SizeY: ", czi_dimensions.SizeY)
print("SizeX: ", czi_dimensions.SizeX)

# get the planetable for the CZI file with "normed" timestamps
pt1, _ = planetable.get_planetable(filepath,
                                         norm_time=True,
                                         save_table=False,
                                         planes={"time": 0, "channel": 0, "zplane": 3})

print(pt1)

# get the planetable for the CZI file with actual timestamps
pt2, _ = planetable.get_planetable(filepath,
                                         norm_time=False,
                                         save_table=False,
                                         planes={"time": 0, "channel": 0, "zplane": 3})

print(pt2)


