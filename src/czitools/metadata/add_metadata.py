from typing import Union, Optional
from dataclasses import dataclass, field
from box import Box
import os
from czitools.tools.logger import get_logger
from czitools.tools.box import get_czimd_box

logger = get_logger()


@dataclass
class CziAddMetaData:
    czisource: Union[str, os.PathLike[str], Box]
    experiment: Optional[Box] = field(init=False, default=None)
    hardwaresetting: Optional[Box] = field(init=False, default=None)
    customattributes: Optional[Box] = field(init=False, default=None)
    displaysetting: Optional[Box] = field(init=False, default=None)
    layers: Optional[Box] = field(init=False, default=None)

    def __post_init__(self):
        logger.info("Reading additional Metedata from CZI image data.")

        if isinstance(self.czisource, Box):
            czi_box = self.czisource
        else:
            czi_box = get_czimd_box(self.czisource)

        if czi_box.has_experiment:
            self.experiment = czi_box.ImageDocument.Metadata.Experiment
        else:
            # print("No Experiment information found.")
            logger.info("No Experiment information found.")

        if czi_box.has_hardware:
            self.hardwaresetting = czi_box.ImageDocument.Metadata.HardwareSetting
        else:
            # print("No HardwareSetting information found.")
            logger.info("No HardwareSetting information found.")

        if czi_box.has_customattr:
            self.customattributes = czi_box.ImageDocument.Metadata.CustomAttributes
        else:
            # print("No CustomAttributes information found.")
            logger.info("No CustomAttributes information found.")

        if czi_box.has_disp:
            self.displaysetting = czi_box.ImageDocument.Metadata.DisplaySetting
        else:
            # print("No DisplaySetting information found.")
            logger.info("No DisplaySetting information found.")

        if czi_box.has_layers:
            self.layers = czi_box.ImageDocument.Metadata.Layers
        else:
            # print("No Layers information found.")
            logger.info("No Layers information found.")
