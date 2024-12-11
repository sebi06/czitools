# -*- coding: utf-8 -*-

#################################################################
# File        : read_write_czi.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from pylibCZIrw import czi as pyczi
from czitools.metadata_tools.czi_metadata import CziChannelInfo
import numpy as np


def hex_to_rgb(hex_string: str) -> tuple[int, int, int]:
    """
    Convert a hexadecimal color string to an RGB tuple.
    Args:
        hex_string (str): A string representing a color in hexadecimal format (e.g., '#RRGGBB').
    Returns:
        tuple: A tuple containing the RGB values (r, g, b) as integers.
    """
    # Remove the '#' character
    hex_string = hex_string.lstrip("#")

    # Convert hex string to integer values
    r = int(hex_string[0:2], 16)
    g = int(hex_string[2:4], 16)
    b = int(hex_string[4:6], 16)

    return r, g, b


filepath = r"\Testdata_Zeiss\CZI_Testfiles\A01sm.czi"
filepath_new = r"F:\Testdata_Zeiss\CZI_Testfiles\A01sm_disp.czi"


ch_info = CziChannelInfo(filepath)

channel_names = {index: value for index, value in enumerate(ch_info.names)}
channel_disp = {index: value for index, value in enumerate(ch_info.colors)}
channel_clims = {index: value for index, value in enumerate(ch_info.clims)}


display_settings = {}

for channel_index in range(len(ch_info.names)):

    # inside the CZI metadata_tools colors are defined as ARGB hexstring
    r, g, b = hex_to_rgb(channel_disp[channel_index][3:])

    display_settings[channel_index] = pyczi.ChannelDisplaySettingsDataClass(
        True,
        pyczi.TintingMode.Color,
        pyczi.Rgb8Color(np.uint8(r), np.uint8(g), np.uint8(b)),
        channel_clims[channel_index][0],
        channel_clims[channel_index][1],
    )


with pyczi.open_czi(filepath) as czidoc:

    ch0 = czidoc.read(plane={"T": 0, "Z": 0, "C": 0})
    ch1 = czidoc.read(plane={"T": 0, "Z": 0, "C": 1})

print(f"ch_info: {ch_info}")

with pyczi.create_czi(filepath_new, exist_ok=True) as czidoc:

    # write the 2D plane to the new CZI file
    czidoc.write(data=ch0, plane={"T": 0, "Z": 0, "C": 0})
    czidoc.write(data=ch1, plane={"T": 0, "Z": 0, "C": 1})

    # write the document title, channel names and dispaly settings
    czidoc.write_metadata(
        document_name="A01sm_disp",
        channel_names=channel_names,
        display_settings=display_settings,
    )
