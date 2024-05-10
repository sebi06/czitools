from typing import Optional, Union
from dataclasses import dataclass, field
import os
from czitools.utils.logger import get_logger
from pylibCZIrw import czi as pyczi
from pathlib import Path

logger = get_logger()


@dataclass
class CziScene:
    filepath: Union[str, os.PathLike[str]]
    index: int
    bbox: Optional[pyczi.Rectangle] = field(init=False, default=None)
    xstart: Optional[int] = field(init=False, default=None)
    ystart: Optional[int] = field(init=False, default=None)
    width: Optional[int] = field(init=False, default=None)
    height: Optional[int] = field(init=False, default=None)

    def __post_init__(self):
        logger.info("Reading Scene Information from CZI image data.")

        if isinstance(self.filepath, Path):
            # convert to string
            self.filepath = str(self.filepath)

        # get scene information from the CZI file
        with pyczi.open_czi(self.filepath) as czidoc:
            try:
                self.bbox = czidoc.scenes_bounding_rectangle[self.index]
                self.xstart = self.bbox.x
                self.ystart = self.bbox.y
                self.width = self.bbox.w
                self.height = self.bbox.h
            except KeyError:
                # in case an invalid index was used
                logger.info("No Scenes detected.")
