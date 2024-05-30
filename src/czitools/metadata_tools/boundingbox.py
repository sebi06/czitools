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
    czisource: Union[str, os.PathLike[str], Box]
    scenes_bounding_rect: Optional[Dict[int, pyczi.Rectangle]] = field(
        init=False, default_factory=lambda: []
    )
    total_rect: Optional[pyczi.Rectangle] = field(init=False, default=None)
    total_bounding_box: Optional[Dict[str, tuple]] = field(
        init=False, default_factory=lambda: []
    )

    # TODO Is this really needed as a separate class or better integrate directly into CziMetadata class?

    def __post_init__(self):
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
            try:
                self.scenes_bounding_rect = czidoc.scenes_bounding_rectangle
            except Exception as e:
                self.scenes_bounding_rect = None
                # print("Scenes Bounding rectangle not found.")
                logger.info("Scenes Bounding rectangle not found.")

            try:
                self.total_rect = czidoc.total_bounding_rectangle
            except Exception as e:
                self.total_rect = None
                # print("Total Bounding rectangle not found.")
                logger.info("Total Bounding rectangle not found.")

            try:
                self.total_bounding_box = czidoc.total_bounding_box
            except Exception as e:
                self.total_bounding_box = None
                # print("Total Bounding Box not found.")
                logger.info("Total Bounding Box not found.")