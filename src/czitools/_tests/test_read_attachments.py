from czitools import read_tools, metadata_tools
from czitools.read_tools import AttachmentType
from pathlib import Path
import numpy as np
import pytest
from typing import List, Dict, Tuple, Optional, Type, Any, Union, Mapping, Literal


basedir = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "cziname, attachment_type, copy, has_label, has_preview, has_prescan, array_shape",
    [
        ("w96_A1+A2.czi", AttachmentType.SlidePreview, True, False, True, False, (2030, 3066, 1)),
        (
            "Tumor_HE_Orig_small.czi",
            AttachmentType.SlidePreview,
            True,
            True,
            True,
            False,
            (758, 1696, 3),
        ),
        ("Tumor_HE_Orig_small.czi", AttachmentType.Label, True, True, True, False, (523, 532, 3)),
        (
            "CellDivision_T10_Z15_CH2_DCV_small.czi",
            AttachmentType.Label,
            True,
            False,
            False,
            False,
            (0,),
        ),
    ],
)
def test_read_attachments_images(
    cziname: str,
    attachment_type: AttachmentType,
    copy: bool,
    has_label: bool,
    has_preview: bool,
    has_prescan: bool,
    array_shape: Tuple[int, int, int],
) -> None:
    # get the filepath and the metadata
    filepath = basedir / "data" / cziname

    # get info about attachments only
    attachments = metadata_tools.CziAttachments(filepath)

    assert attachments.has_label == has_label
    assert attachments.has_preview == has_preview
    assert attachments.has_prescan == has_prescan

    data, location = read_tools.read_attachments(
        filepath, attachment_type=attachment_type, copy=copy
    )

    assert array_shape == data.shape

    # create path to store the attachment image
    att_path = str(filepath)[:-4] + "_" + attachment_type.name + ".czi"

    assert att_path.endswith(cziname[:-4] + "_" + attachment_type.name + ".czi") is True

    # remove the files
    if np.any(data):
        Path.unlink(Path(location))


