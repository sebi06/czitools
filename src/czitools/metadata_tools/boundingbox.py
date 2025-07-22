from typing import Optional, Union, Dict
from dataclasses import dataclass, field
from box import Box
import os
from czitools.utils import logging_tools
from pylibCZIrw import czi as pyczi
import validators

logger = logging_tools.set_logging()


@dataclass
class CziBoundingBox:
    """
    A class to represent and handle bounding boxes from CZI image data.
    Attributes:
    -----------
    czisource : Union[str, os.PathLike[str], Box]
        The source of the CZI file, which can be a string, a path-like object, or a Box object.
    scenes_bounding_rect : Optional[Dict[int, pyczi.Rectangle]]
        A dictionary containing the bounding rectangles for each scene in the CZI file.
    scenes_bounding_rect_no_pyramid : Optional[Dict[int, pyczi.Rectangle]]
        A dictionary containing the bounding rectangles for each scene in the CZI file without a pyramid.
    total_rect : Optional[pyczi.Rectangle]
        The total bounding rectangle for the entire CZI file.
    total_rect_no_pyramid : Optional[pyczi.Rectangle]
        The total bounding rectangle for the entire CZI file without a pyramid.
    total_bounding_box : Optional[Dict[str, tuple]]
        A dictionary containing the total bounding box.
    total_bounding_box_no_pyramid : Optional[Dict[str, tuple]]
        A dictionary containing the total bounding box coordinates without a pyramid.
    verbose (bool): Flag to enable verbose logging. Initialized to False.
    Methods:
    --------
    __post_init__():
        Initializes the bounding box attributes by reading from the CZI file.
    """

    czisource: Union[str, os.PathLike[str], Box]
    scenes_bounding_rect: Optional[Dict[int, pyczi.Rectangle]] = field(
        init=False, default_factory=lambda: {}
    )
    scenes_bounding_rect_no_pyramid: Optional[Dict[int, pyczi.Rectangle]] = field(
        init=False, default_factory=lambda: {}
    )
    total_rect: Optional[pyczi.Rectangle] = field(init=False, default=None)
    total_rect_no_pyarmid: Optional[pyczi.Rectangle] = field(init=False, default=None)
    total_bounding_box: Optional[Dict[str, tuple]] = field(
        init=False, default_factory=lambda: {}
    )
    total_bounding_box_no_pyramid: Optional[Dict[str, tuple]] = field(
        init=False, default_factory=lambda: {}
    )
    verbose: bool = False

    # TODO Is this really needed as a separate class or better integrate directly into CziMetadata class?

    def __post_init__(self):
        if self.verbose:
            logger.info("Reading BoundingBoxes from CZI image data.")

        pyczi_readertype = pyczi.ReaderFileInputTypes.Standard

        if isinstance(self.czisource, Box):
            self.czisource = self.czisource.filepath

        if validators.url(str(self.czisource)):
            pyczi_readertype = pyczi.ReaderFileInputTypes.Curl
            logger.info(
                "FilePath is a valid link. Only pylibCZIrw functionality is available."
            )

        with pyczi.open_czi(str(self.czisource), pyczi_readertype) as czidoc:

            # get scenes bounding rectangles
            try:
                self.scenes_bounding_rect = czidoc.scenes_bounding_rectangle
            except Exception as e:
                if self.verbose:
                    self.scenes_bounding_rect = None
                    logger.info("Scenes Bounding Rectangle not found.")

            try:
                self.scenes_bounding_rect_no_pyramid = (
                    czidoc.scenes_bounding_rectangle_no_pyramid
                )
            except Exception as e:
                if self.verbose:
                    self.scenes_bounding_rect_no_pyramid = None
                    logger.info("Scenes Bounding Rectangle no Pyramid not found.")

            # get total bounding rectangles
            try:
                self.total_rect = czidoc.total_bounding_rectangle
            except Exception as e:
                self.total_rect = None
                if self.verbose:
                    logger.info("Total Bounding Rectangle not found.")

            # get total bounding rectangles
            try:
                self.total_rect_no_pyramid = czidoc.total_bounding_rectangle_no_pyramid
            except Exception as e:
                self.total_rect_no_pyramid = None
                if self.verbose:
                    logger.info("Total Bounding Rectangle no pyramid not found.")

            # get total bounding boxes
            try:
                self.total_bounding_box = czidoc.total_bounding_box
            except Exception as e:
                self.total_bounding_box = None
                if self.verbose:
                    logger.info("Total Bounding Box not found.")

            try:
                self.total_bounding_box_no_pyramid = (
                    czidoc.total_bounding_box_no_pyramid
                )
            except Exception as e:
                self.total_bounding_box_no_pyramid = None
                if self.verbose:
                    logger.info("Total Bounding Box no Pyramid not found.")
