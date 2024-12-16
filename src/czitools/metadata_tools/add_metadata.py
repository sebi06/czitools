from typing import Union, Optional
from dataclasses import dataclass, field
from box import Box
import os
from czitools.utils import logging_tools
from czitools.utils.box import get_czimd_box

logger = logging_tools.set_logging()


@dataclass
class CziAddMetaData:
    """
    A class to add metadata from CZI image data.
    Attributes:
        czisource (Union[str, os.PathLike[str], Box]): The source of the CZI image data.
        experiment (Optional[Box]): The experiment metadata. Initialized to None.
        hardwaresetting (Optional[Box]): The hardware setting metadata. Initialized to None.
        customattributes (Optional[Box]): The custom attributes metadata. Initialized to None.
        displaysetting (Optional[Box]): The display setting metadata. Initialized to None.
        layers (Optional[Box]): The layers metadata. Initialized to None.
        verbose (bool): Flag to enable verbose logging. Initialized to False.
    Methods:
        __post_init__(): Reads additional metadata from the CZI image data and initializes the attributes.
    """

    czisource: Union[str, os.PathLike[str], Box]
    experiment: Optional[Box] = field(init=False, default=None)
    hardwaresetting: Optional[Box] = field(init=False, default=None)
    customattributes: Optional[Box] = field(init=False, default=None)
    displaysetting: Optional[Box] = field(init=False, default=None)
    layers: Optional[Box] = field(init=False, default=None)
    verbose: bool = False

    def __post_init__(self):
        if self.verbose:
            logger.info("Reading additional Metedata from CZI image data.")

        if isinstance(self.czisource, Box):
            czi_box = self.czisource
        else:
            czi_box = get_czimd_box(self.czisource)

        if czi_box.has_experiment:
            self.experiment = czi_box.ImageDocument.Metadata.Experiment
        else:
            if self.verbose:
                logger.info("No Experiment information found.")

        if czi_box.has_hardware:
            self.hardwaresetting = czi_box.ImageDocument.Metadata.HardwareSetting
        else:
            if self.verbose:
                logger.info("No HardwareSetting information found.")

        if czi_box.has_customattr:
            self.customattributes = czi_box.ImageDocument.Metadata.CustomAttributes
        else:
            if self.verbose:
                logger.info("No CustomAttributes information found.")

        if czi_box.has_disp:
            self.displaysetting = czi_box.ImageDocument.Metadata.DisplaySetting
        else:
            if self.verbose:
                logger.info("No DisplaySetting information found.")

        if czi_box.has_layers:
            self.layers = czi_box.ImageDocument.Metadata.Layers
        else:
            if self.verbose:
                logger.info("No Layers information found.")
