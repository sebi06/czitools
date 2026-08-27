"""Microscope metadata utilities for CZI files.

Provides `CziMicroscope`, which extracts the microscope system name and
identifier from CZI metadata.
"""

import os
from dataclasses import dataclass, field

from box import Box

from czitools.utils import logging_tools
from czitools.utils.box import get_czimd_box

logger = logging_tools.set_logging()


@dataclass
class CziMicroscope:
    """A class to represent a microscope from CZI image data.

    Attributes:
        czisource (Union[str, os.PathLike[str], Box]): The source of the CZI image data, which can be a file path or a Box object.
        Id (Optional[str]): The identifier of the microscope.
        Name (Optional[str]): The name of the microscope.
        System (Optional[str]): The system of the microscope.
        verbose (bool): Flag to enable verbose logging.
    """

    czisource: str | os.PathLike[str] | Box
    Id: str | None = field(init=False, default=None)
    Name: str | None = field(init=False, default=None)
    System: str | None = field(init=False, default=None)
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.verbose:
            logger.info("Reading Microscope Information from CZI image data.")

        czi_box = self.czisource if isinstance(self.czisource, Box) else get_czimd_box(self.czisource)

        image_document = getattr(czi_box, "ImageDocument", None)
        metadata = getattr(image_document, "Metadata", None)
        information = getattr(metadata, "Information", None)
        instrument = getattr(information, "Instrument", None)
        microscopes = getattr(instrument, "Microscopes", None)
        microscope = getattr(microscopes, "Microscope", None)

        if microscope is None:
            if self.verbose:
                logger.info("No Microscope information found.")
            return

        self.Id = getattr(microscope, "Id", None)
        self.Name = getattr(microscope, "Name", None)
        self.System = getattr(microscope, "System", None)
