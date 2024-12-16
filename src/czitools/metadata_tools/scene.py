from typing import Optional, Union
from dataclasses import dataclass, field
import os
from czitools.utils import logging_tools
from pylibCZIrw import czi as pyczi
from pathlib import Path

logger = logging_tools.set_logging()


@dataclass
class CziScene:
    """
    CziScene class represents a scene within a CZI (Carl Zeiss Image) file.
    Attributes:
        filepath (Union[str, os.PathLike[str]]): The path to the CZI file.
        index (int): The index of the scene within the CZI file.
        bbox (Optional[pyczi.Rectangle]): The bounding box of the scene. Initialized to None.
        xstart (Optional[int]): The starting x-coordinate of the scene. Initialized to None.
        ystart (Optional[int]): The starting y-coordinate of the scene. Initialized to None.
        width (Optional[int]): The width of the scene. Initialized to None.
        height (Optional[int]): The height of the scene. Initialized to None.
        verbose (bool): Flag to enable verbose logging.
    Methods:
        __post_init__(): Initializes the scene information by reading from the CZI file.
    """

    filepath: Union[str, os.PathLike[str]]
    index: int
    bbox: Optional[pyczi.Rectangle] = field(init=False, default=None)
    xstart: Optional[int] = field(init=False, default=None)
    ystart: Optional[int] = field(init=False, default=None)
    width: Optional[int] = field(init=False, default=None)
    height: Optional[int] = field(init=False, default=None)
    verbose: bool = False

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
            except KeyError as e:
                if self.verbose:
                    # in case an invalid index was used
                    logger.info(f"{e}: No Scenes detected.")
