from typing import Union, Optional
from dataclasses import dataclass, field
from box import Box
import os
from czitools.utils import logging_tools
from czitools.utils.box import get_czimd_box

logger = logging_tools.set_logging()


@dataclass
class CziAddMetaData:
    """Collect additional (auxiliary) metadata blocks from a CZI.

    This class extracts optional metadata blocks such as Experiment,
    HardwareSetting, CustomAttributes, DisplaySetting and Layers. Each
    attribute is ``None`` if the block is absent. Diagnostic messages
    are emitted only when ``verbose`` is True.
    """

    czisource: Union[str, os.PathLike, Box]
    experiment: Optional[Box] = field(init=False, default=None)
    hardwaresetting: Optional[Box] = field(init=False, default=None)
    customattributes: Optional[Box] = field(init=False, default=None)
    displaysetting: Optional[Box] = field(init=False, default=None)
    layers: Optional[Box] = field(init=False, default=None)
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.verbose:
            logger.info("Reading additional metadata from CZI image data.")

        czi_box = self.czisource if isinstance(self.czisource, Box) else get_czimd_box(self.czisource)

        # Extract optional blocks defensively using hasattr/getattr
        if getattr(czi_box, "has_experiment", False):
            self.experiment = getattr(czi_box.ImageDocument.Metadata, "Experiment", None)
        elif self.verbose:
            logger.info("No Experiment information found.")

        if getattr(czi_box, "has_hardware", False):
            self.hardwaresetting = getattr(czi_box.ImageDocument.Metadata, "HardwareSetting", None)
        elif self.verbose:
            logger.info("No HardwareSetting information found.")

        if getattr(czi_box, "has_customattr", False):
            self.customattributes = getattr(czi_box.ImageDocument.Metadata, "CustomAttributes", None)
        elif self.verbose:
            logger.info("No CustomAttributes information found.")

        if getattr(czi_box, "has_disp", False):
            self.displaysetting = getattr(czi_box.ImageDocument.Metadata, "DisplaySetting", None)
        elif self.verbose:
            logger.info("No DisplaySetting information found.")

        if getattr(czi_box, "has_layers", False):
            self.layers = getattr(czi_box.ImageDocument.Metadata, "Layers", None)
        elif self.verbose:
            logger.info("No Layers information found.")
