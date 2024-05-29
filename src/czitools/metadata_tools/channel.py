from typing import Union, List
from dataclasses import dataclass, field
from box import Box, BoxList
import os
from czitools.utils.logging_tools import get_logger, set_logging
from czitools.utils.box import get_czimd_box

#logger = get_logger()
logger = set_logging()


@dataclass
class CziChannelInfo:
    czisource: Union[str, os.PathLike[str], Box]
    names: List[str] = field(init=False, default_factory=lambda: [])
    dyes: List[str] = field(init=False, default_factory=lambda: [])
    colors: List[str] = field(init=False, default_factory=lambda: [])
    clims: List[List[float]] = field(init=False, default_factory=lambda: [])
    gamma: List[float] = field(init=False, default_factory=lambda: [])

    def __post_init__(self):
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
            except AttributeError:
                channels = None
        elif not czi_box.has_channels:
            logger.info("Channel(s) information not found.")

        if czi_box.has_disp:
            try:
                # extract the relevant dimension metadata_tools
                disp = czi_box.ImageDocument.Metadata.DisplaySetting.Channels.Channel
                if isinstance(disp, Box):
                    self.get_channel_info(disp)
                elif isinstance(disp, BoxList):
                    for ch in range(len(disp)):
                        self.get_channel_info(disp[ch])
            except AttributeError:
                disp = None

        elif not czi_box.has_disp:
            # print("DisplaySetting(s) not found.")
            logger.info("DisplaySetting(s) not found.")

    def get_channel_info(self, display: Box):
        if display is not None:
            (
                self.dyes.append("Dye-CH1")
                if display.ShortName is None
                else self.dyes.append(display.ShortName)
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

