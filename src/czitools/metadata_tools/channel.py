# -*- coding: utf-8 -*-

#################################################################
# File        : channel.py
# Author      : sebi06
#
#################################################################

from typing import Union, List, Dict
from dataclasses import dataclass, field
from box import Box, BoxList
import os
from czitools.utils import logging_tools, pixels
from czitools.utils.box import get_czimd_box
from pylibCZIrw import czi as pyczi
import numpy as np

logger = logging_tools.set_logging()


@dataclass
class CziChannelInfo:
    """
    A class to handle channel information from CZI image data.

    Attributes:
        czisource (Union[str, os.PathLike[str], Box]): The source of the CZI image data.
        names (List[str]): List of channel names.
        dyes (List[str]): List of dye names.
        dyes_short (List[str]): List of short dye names.
        channel_descriptions (List[str]): List of channel descriptions.
        colors (List[str]): List of channel colors.
        clims (List[List[float]]): List of channel intensity limits.
        gamma (List[float]): List of gamma values for each channel.
        pixeltypes (Dict[int, str]): Dictionary of pixel types for each channel.
        isRGB (Dict[int, bool]): Dictionary indicating if each channel is RGB.
        consistent_pixeltypes (bool): Indicates if pixel types are consistent across channels.
        czi_disp_settings (Dict[int, pyczi.ChannelDisplaySettingsDataClass]): Dictionary containing the display settings for each channel.
        verbose (bool): Flag to enable verbose logging. Initialized to False.

    Methods:
        __post_init__():
            Initializes the channel information from the CZI image data.
        _get_channel_info(display: Box):
            Extracts and appends channel display information.
        _calculate_display_settings() -> Dict:
            Calculates and returns the display settings for each channel.
    """

    czisource: Union[str, os.PathLike[str], Box]
    names: List[str] = field(init=False, default_factory=lambda: [])
    dyes: List[str] = field(init=False, default_factory=lambda: [])
    dyes_short: List[str] = field(init=False, default_factory=lambda: [])
    channel_descriptions: List[str] = field(init=False, default_factory=lambda: [])
    colors: List[str] = field(init=False, default_factory=lambda: [])
    clims: List[List[float]] = field(init=False, default_factory=lambda: [])
    gamma: List[float] = field(init=False, default_factory=lambda: [])
    pixeltypes: Dict[int, str] = field(init=False, default_factory=lambda: {})
    isRGB: Dict[int, bool] = field(init=False, default_factory=lambda: {})
    consistent_pixeltypes: bool = field(init=False, default=None)
    czi_disp_settings: Dict[int, pyczi.ChannelDisplaySettingsDataClass] = field(
        init=False, default_factory=lambda: {}
    )
    verbose: bool = False

    def __post_init__(self):
        if self.verbose:
            logger.info("Reading Channel Information from CZI image data.")

        if isinstance(self.czisource, Box):
            czi_box = self.czisource
        else:
            czi_box = get_czimd_box(self.czisource)

        # get channels part of dict
        if czi_box.has_channels:
            try:
                # extract the relevant dimension metadata_tools
                channels = (
                    czi_box.ImageDocument.Metadata.Information.Image.Dimensions.Channels.Channel
                )
                if isinstance(channels, Box):
                    # get the data in case of only one channel
                    (
                        self.names.append("CH1")
                        if channels.Name is None
                        else self.names.append(channels.Name)
                    )
                elif isinstance(channels, BoxList):
                    # get the data in case multiple channels
                    for ch in range(len(channels)):
                        (
                            self.names.append("CH1")
                            if channels[ch].Name is None
                            else self.names.append(channels[ch].Name)
                        )

                # get the pixel types for all channels
                with pyczi.open_czi(
                    str(czi_box.filepath), czi_box.czi_open_arg
                ) as czidoc:

                    # get the pixel typed for all channels
                    self.pixeltypes = czidoc.pixel_types
                    self.isRGB, self.consistent_pixeltypes = pixels.check_if_rgb(
                        self.pixeltypes
                    )

            except AttributeError:
                channels = None
        elif not czi_box.has_channels:
            if self.verbose:
                logger.info("Channel(s) information not found.")

        if czi_box.has_disp:
            try:
                # extract the relevant dimension metadata_tools
                disp = czi_box.ImageDocument.Metadata.DisplaySetting.Channels.Channel
                if isinstance(disp, Box):
                    self._get_channel_info(disp)
                elif isinstance(disp, BoxList):
                    for ch in range(len(disp)):
                        self._get_channel_info(disp[ch])
            except AttributeError:
                disp = None

            # calculate the ChannelDisplaySetting
            self.czi_disp_settings = self._calculate_display_settings()

        elif not czi_box.has_disp:
            if self.verbose:
                logger.info("DisplaySetting(s) not inside CZI metadata.")

    def _get_channel_info(self, display: Box):
        if display is not None:
            (
                self.dyes.append("Dye-CH1")
                if display.Name is None
                else self.dyes.append(display.Name)
            )
            (
                self.dyes_short.append("Dye-CH1")
                if display.ShortName is None
                else self.dyes_short.append(display.ShortName)
            )
            (
                self.channel_descriptions.append("")
                if display.ShortName is None
                else self.channel_descriptions.append(display.Description)
            )
            (
                self.colors.append("#80808000")
                if display.Color is None
                else self.colors.append(display.Color)
            )

            low = 0.0 if display.Low is None else float(display.Low)
            high = 0.5 if display.High is None else float(display.High)

            self.clims.append([low, high])
            (
                self.gamma.append(0.85)
                if display.Gamma is None
                else self.gamma.append(float(display.Gamma))
            )
        else:
            self.dyes.append("Dye-CH1")
            self.colors.append("#80808000")
            self.clims.append([0.0, 0.5])
            self.gamma.append(0.85)

    def _calculate_display_settings(self) -> Dict:

        # get the number of channels
        num_channels = len(self.names)

        # initialize the display settings
        display_settings_dict = {}

        for channel_index in range(num_channels):

            # Get RGB values based on a RGB hexstring. Inside a CZI per channel one gets #AARRGGBB.
            r, g, b = hex_to_rgb(self.colors[channel_index][3:])

            if self.isRGB[channel_index]:
                tinting_mode = pyczi.TintingMode.none
                black_point = 0.0
                white_point = 1.0

            else:
                tinting_mode = pyczi.TintingMode.Color
                black_point = self.clims[channel_index][0]
                white_point = self.clims[channel_index][1]

            display_settings_dict[channel_index] = (
                pyczi.ChannelDisplaySettingsDataClass(
                    is_enabled=True,
                    tinting_mode=tinting_mode,
                    tinting_color=pyczi.Rgb8Color(
                        np.uint8(r), np.uint8(g), np.uint8(b)
                    ),
                    black_point=black_point,  # min value for histogram
                    white_point=white_point,  # max value for histogram
                )
            )

        return display_settings_dict


def hex_to_rgb(hex_string: str) -> tuple[int, int, int]:
    """
    Convert a hexadecimal color string to an RGB tuple.
    Args:
        hex_string (str): A string representing a color in hexadecimal format (e.g., '#RRGGBB').
    Returns:
        tuple: A tuple containing the RGB values (r, g, b) as integers.
    """

    # Remove the '#' characters
    hex_string = hex_string.replace("#", "")

    # check the length of the string
    if len(hex_string) != 6:
        # remove alpha values
        hex_string = hex_string[2:]

    try:
        # Convert hex string to integer values
        r = int(hex_string[0:2], 16)
        g = int(hex_string[2:4], 16)
        b = int(hex_string[4:6], 16)

    except ValueError:

        # Set RGB values to 128 if conversion fails
        r = 128
        g = 128
        b = 128

        logger.error(
            f"Invalid RGB values detected. Conversion of hex string {hex_string} to RGB failed. Setting RGB values to 128."
        )

    return r, g, b
