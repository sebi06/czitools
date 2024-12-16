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


# def hex_to_rgb(hex_string: str) -> tuple[int, int, int]:
#     """
#     Convert a hexadecimal color string to an RGB tuple.
#     Args:
#         hex_string (str): A string representing a color in hexadecimal format (e.g., '#RRGGBB').
#     Returns:
#         tuple: A tuple containing the RGB values (r, g, b) as integers.
#     """
#     # Remove the '#' character
#     hex_string = hex_string.lstrip("#")

#     # Convert hex string to integer values
#     r = int(hex_string[0:2], 16)
#     g = int(hex_string[2:4], 16)
#     b = int(hex_string[4:6], 16)

#     return r, g, b

# filepath = r"F:\Github\czitools\data\Tumor_HE_Orig_small.czi"
filepath = r"F:\Github\czitools\data\CellDivision_T3_Z5_CH2_X240_Y170.czi"
# filepath = r"F:\Github\czitools\data\Al2O3_SE_020_sp.czi"
# filepath = r"F:\Github\czitools\data\w96_A1+A2.czi"
filepath_new = filepath[:-4] + "_disp.czi"

write_new = False

ch_info = CziChannelInfo(filepath, verbose=True)

channel_names_dict = {index: value for index, value in enumerate(ch_info.names)}

display_settings_dict = ch_info.czi_disp_settings

# display_settings_dict = {}

# for channel_index in range(len(ch_info.names)):

#     # inside the CZI metadata_tools colors are defined as ARGB hexstring
#     r, g, b = hex_to_rgb(ch_info.colors[channel_index][3:])

#     display_settings_dict[channel_index] = pyczi.ChannelDisplaySettingsDataClass(
#         True,
#         pyczi.TintingMode.Color,
#         pyczi.Rgb8Color(np.uint8(r), np.uint8(g), np.uint8(b)),
#         ch_info.clims[channel_index][0],
#         ch_info.clims[channel_index][1],
#     )


with pyczi.open_czi(filepath) as czidoc:

    ch0 = czidoc.read(plane={"T": 0, "Z": 0, "C": 0})
    # ch1 = czidoc.read(plane={"T": 0, "Z": 0, "C": 1})

print(f"ch_info: {ch_info}")

if write_new:

    with pyczi.create_czi(filepath_new, exist_ok=True) as czidoc:

        # write the 2D plane to the new CZI file
        czidoc.write(data=ch0, plane={"T": 0, "Z": 0, "C": 0})
        # czidoc.write(data=ch1, plane={"T": 0, "Z": 0, "C": 1})

        # write the document title, channel names and dispaly settings
        czidoc.write_metadata(
            document_name="test",
            channel_names=channel_names_dict,
            display_settings=display_settings_dict,
        )
