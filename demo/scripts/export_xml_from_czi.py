# -*- coding: utf-8 -*-

#################################################################
# File        : export_xml_from_czi.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
# Requires magicgui to be installed!
#
#################################################################

# import the required libraries
from czitools.metadata_tools import czi_metadata as czimd
from czitools.utils import misc
from pathlib import Path
from magicgui import magicgui
from magicgui.types import FileDialogMode


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

filepath = filespicker.filepath.value
print(f"Selected file: {filepath}")

# write XML to disk
xmlfile = czimd.writexml(filepath)

print("XML File written to:", xmlfile)
