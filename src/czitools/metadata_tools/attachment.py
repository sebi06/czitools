"""CZI attachment metadata utilities.

Provides `CziAttachments`, which inspects a CZI file for known attachment
types (label, preview, prescan) using `pylibCZIrw`.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import validators
from box import Box

from czitools.utils import logging_tools

logger = logging_tools.set_logging()


@dataclass
class CziAttachments:
    """CziAttachments class for handling CZI image data attachments.

    Attributes:
        czisource (Union[str, os.PathLike[str], Box]): Source of the CZI image data.
        has_label (Optional[bool]): Indicates if the CZI image has a label attachment.
        has_preview (Optional[bool]): Indicates if the CZI image has a preview attachment.
        has_prescan (Optional[bool]): Indicates if the CZI image has a prescan attachment.
        names (Optional[List[str]]): List of attachment names found in the CZI image.
        verbose (bool): Flag to enable verbose logging.
    """

    czisource: str | os.PathLike[str] | Box
    has_label: bool | None = field(init=False, default=False)
    has_preview: bool | None = field(init=False, default=False)
    has_prescan: bool | None = field(init=False, default=False)
    names: list[str] = field(init=False, default_factory=lambda: [])
    verbose: bool = False

    def __post_init__(self):
        if self.verbose:
            logger.info("Reading AttachmentImages from CZI image data.")

        try:
            import czifile

            if isinstance(self.czisource, Path):
                self.czisource = str(self.czisource)
            elif isinstance(self.czisource, Box):
                self.czisource = str(self.czisource.filepath)

            czisource_str: str = str(self.czisource)

            if validators.url(czisource_str):
                if self.verbose:
                    logger.warning("Reading Attachments from CZI via a link is not supported.")
            else:
                # create CZI-object using czifile library
                with czifile.CziFile(czisource_str) as cz:
                    # iterate over attachments
                    for att in cz.attachments():
                        self.names.append(att.attachment_entry.name)

                    if "SlidePreview" in self.names:
                        self.has_preview = True
                        if self.verbose:
                            logger.info("Attachment SlidePreview found.")
                    if "Label" in self.names:
                        self.has_label = True
                        if self.verbose:
                            logger.info("Attachment Label found.")
                    if "Prescan" in self.names:
                        self.has_prescan = True
                        if self.verbose:
                            logger.info("Attachment Prescan found.")

        except ImportError as e:
            if self.verbose:
                logger.warning(f"{e}: Package czifile not found. Cannot extract information about attached images.")
