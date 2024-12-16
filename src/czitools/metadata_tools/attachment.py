from typing import Optional, Union, List
from dataclasses import dataclass, field
from box import Box
import os
from czitools.utils import logging_tools
from pathlib import Path
import validators

logger = logging_tools.set_logging()


@dataclass
class CziAttachments:
    """
    CziAttachments class for handling CZI image data attachments.
    Attributes:
        czisource (Union[str, os.PathLike[str], Box]): Source of the CZI image data.
        has_label (Optional[bool]): Indicates if the CZI image has a label attachment.
        has_preview (Optional[bool]): Indicates if the CZI image has a preview attachment.
        has_prescan (Optional[bool]): Indicates if the CZI image has a prescan attachment.
        names (Optional[List[str]]): List of attachment names found in the CZI image.
        verbose (bool): Flag to enable verbose logging. Initialized to False.
    Methods:
        __post_init__(): Initializes the CziAttachments object, reads attachment images from the CZI image data, and sets the appropriate flags for label, preview, and prescan attachments.
    """

    czisource: Union[str, os.PathLike[str], Box]
    has_label: Optional[bool] = field(init=False, default=False)
    has_preview: Optional[bool] = field(init=False, default=False)
    has_prescan: Optional[bool] = field(init=False, default=False)
    names: Optional[List[str]] = field(init=False, default_factory=lambda: [])
    verbose: bool = False

    def __post_init__(self):
        if self.verbose:
            logger.info("Reading AttachmentImages from CZI image data.")

        try:
            import czifile

            if isinstance(self.czisource, Path):
                # convert to string
                self.czisource = str(self.czisource)
            elif isinstance(self.czisource, Box):
                self.czisource = self.czisource.filepath

            if validators.url(self.czisource):
                if self.verbose:
                    logger.warning(
                        "Reading Attachments from CZI via a link is not supported."
                    )
            else:
                # create CZI-object using czifile library
                with czifile.CziFile(self.czisource) as cz:
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
                logger.warning(
                    f"{e}: Package czifile not found. Cannot extract information about attached images."
                )
