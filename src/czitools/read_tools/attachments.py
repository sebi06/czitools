"""Read embedded CZI attachment images."""

from pathlib import Path
import os
import shutil
import tempfile

import numpy as np
from pylibCZIrw import czi as pyczi

from czitools.metadata_tools import czi_metadata as czimd
from czitools.metadata_tools.helper import AttachmentType

from ._helpers import logger


def read_attachments(
    czi_filepath: str | os.PathLike,
    attachment_type: AttachmentType = AttachmentType.SlidePreview,
    copy: bool = True,
) -> tuple[np.ndarray | None, str | None]:
    """Read attachment images from a CZI image as numpy array

    Args:
        czi_filepath (str): FilePath of the CZI image file
        attachment_type (str, optional): Type of the attachment to be read. Defaults to "SlidePreview".
        copy (bool, optional): Option to copy the attachments as CZI image directly. Defaults to True.

    Raises:
        Exception: If specified attachment is not found and exception is raised.

    Returns:
        Tuple[nd,array, [Optional[str]]: Tuple containing the 2d image array and optionally the location of the copied image.
    """

    try:
        import czifile

        if attachment_type not in AttachmentType:
            # if attachment_type not in ["SlidePreview", "Label", "]:
            raise Exception(f"{attachment_type} is not supported. Valid types are: SlidePreview, Label or Prescan.")

        att = czimd.CziAttachments(czi_filepath)

        if attachment_type == AttachmentType.Label and not att.has_label:
            # if attachment_type == "Label" and not att.has_label:
            return np.array([]), None

        if attachment_type == AttachmentType.SlidePreview and not att.has_preview:
            # if attachment_type == "SlidePreview" and not att.has_preview:
            return np.array([]), None

        if attachment_type == AttachmentType.Prescan and not att.has_prescan:
            # if attachment_type == "SlidePreview" and not att.has_preview:
            return np.array([]), None

        # create CZI-object using czifile library
        with czifile.CziFile(czi_filepath) as cz:
            with tempfile.TemporaryDirectory() as tmpdirname:
                # save attachments to temporary directory
                cz.save_attachments(directory=tmpdirname)

                # iterate over attachments
                for att in cz.attachments():
                    if att.attachment_entry.name == attachment_type.name:
                        # get the full path of the attachment image
                        full_path = Path(tmpdirname) / att.attachment_entry.filename

                        if copy:
                            # create path to store the attachment image
                            att_path = str(czi_filepath)[:-4] + "_" + att.attachment_entry.name + ".czi"

                            # copy the file
                            dest = shutil.copyfile(full_path, att_path)

                        # open the CZI document to read array
                        with pyczi.open_czi(str(full_path)) as czidoc:
                            img2d = czidoc.read()

                        if copy:
                            return img2d, dest
                        if not copy:
                            return img2d, None

        return None, None

    except ImportError:  # as e:
        logger.warning("Package czifile not found. Cannot extract information about attached images.")

        return None, None
