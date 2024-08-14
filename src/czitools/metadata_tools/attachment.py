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
    czisource: Union[str, os.PathLike[str], Box]
    has_label: Optional[bool] = field(init=False, default=False)
    has_preview: Optional[bool] = field(init=False, default=False)
    has_prescan: Optional[bool] = field(init=False, default=False)
    names: Optional[List[str]] = field(init=False, default_factory=lambda: [])

    def __post_init__(self):
        logger.info("Reading AttachmentImages from CZI image data.")

        try:
            import czifile

            if isinstance(self.czisource, Path):
                # convert to string
                self.czisource = str(self.czisource)
            elif isinstance(self.czisource, Box):
                self.czisource = self.czisource.filepath

            if validators.url(self.czisource):
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
                        logger.info("Attachment SlidePreview found.")
                    if "Label" in self.names:
                        self.has_label = True
                        logger.info("Attachment Label found.")
                    if "Prescan" in self.names:
                        self.has_prescan = True
                        logger.info("Attachment Prescan found.")

        except ImportError as e:
            logger.warning(
                "Package czifile not found. Cannot extract information about attached images."
            )