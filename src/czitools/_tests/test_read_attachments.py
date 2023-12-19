from czitools import read_tools, metadata_tools
from pathlib import Path
import numpy as np
import pytest
from pylibCZIrw import czi as pyczi
from typing import List, Dict, Tuple, Optional, Type, Any, Union, Mapping


basedir = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "cziname, attachment_type, copy, has_label, has_preview, has_prescan, array_shape",
    [
        ("w96_A1+A2.czi", "SlidePreview", True, False, True, False, (2030, 3066, 1)),
        (
            "Tumor_HE_Orig_small.czi",
            "SlidePreview",
            True,
            True,
            True,
            False,
            (758, 1696, 3),
        ),
        ("Tumor_HE_Orig_small.czi", "Label", True, True, True, False, (523, 532, 3)),
        (
            "CellDivision_T=10_Z=15_CH=2_DCV_small.czi",
            "Label",
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
    attachment_type: str,
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
    att_path = str(filepath)[:-4] + "_" + attachment_type + ".czi"

    assert att_path.endswith(cziname[:-4] + "_" + attachment_type + ".czi") == True

    # remove the files
    if np.any(data):
        Path.unlink(Path(location))


@pytest.mark.parametrize("cziname, attachment_type", [("w96_A1+A2.czi", "BlaBla")])
def test_reading_wrong_type(cziname: str, attachment_type: str) -> None:
    # get the filepath and the metadata
    filepath = basedir / "data" / cziname

    with pytest.raises(Exception):
        data, loc = read_tools.read_attachments(
            filepath, attachment_type=attachment_type, copy=True
        )
