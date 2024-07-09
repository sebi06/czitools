# -*- coding: utf-8 -*-

#################################################################
# File        : export_xml_from_czi.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

# import the required libraries
from czitools.metadata_tools import czi_metadata as czimd
from czitools.utils import misc
from pathlib import Path


# adapt to your needs
defaultdir = Path(Path(__file__).resolve().parents[2]) / "data"

# open s simple dialog to select a CZI file
filepath = misc.openfile(
    directory=str(defaultdir),
    title="Open CZI Image File",
    ftypename="CZI Files",
    extension="*.czi",
)

# write XML to disk
xmlfile = czimd.writexml(filepath)

print("XML File written to:", xmlfile)
