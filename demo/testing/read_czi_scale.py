# -*- coding: utf-8 -*-

#################################################################
# File        : read_czi_scale
# .py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from typing import Union, Optional
import os
import numpy as np
from pylibCZIrw import czi as pyczi
from box import Box, BoxList
import validators
from dataclasses import dataclass, field


@dataclass
class CziScaleXYZ:
    """
    A class to handle scaling information from CZI image data.
    Attributes:
        X (Optional[float]): The scaling value for the X dimension in microns.
        Y (Optional[float]): The scaling value for the Y dimension in microns.
        Z (Optional[float]): The scaling value for the Z dimension in microns.
    Methods:
        __post_init__(): Initializes the scaling values from the CZI image data.
        _safe_get_scale(dist: BoxList, idx: int) -> Optional[float]: Safely retrieves the scaling value for a given dimension.
    """

    czisource: Union[str, os.PathLike[str]]
    X: Optional[float] = field(init=False, default=None)
    Y: Optional[float] = field(init=False, default=None)
    Z: Optional[float] = field(init=False, default=None)

    def __post_init__(self):

        czi_box = get_czimd_box(self.czisource)

        if czi_box.has_scale:
            distances = czi_box.ImageDocument.Metadata.Scaling.Items.Distance

            # get the scaling values for X,Y and Z
            self.X = np.round(self._safe_get_scale(distances, 0), 3)
            self.Y = np.round(self._safe_get_scale(distances, 1), 3)
            self.Z = np.round(self._safe_get_scale(distances, 2), 3)

        elif not czi_box.has_scale:
            self.X = 1.0
            self.Y = 1.0
            self.Z = 1.0

    @staticmethod
    def _safe_get_scale(dist: BoxList, idx: int, verbose: bool = False) -> Optional[float]:

        try:
            # get the scaling value in [micron]
            sc = float(dist[idx].Value) * 1e6

            # check for the value = 0.0
            if sc == 0.0:
                sc = 1.0

            return sc

        except (IndexError, TypeError, AttributeError):

            return 1.0


def get_czimd_box(filepath: Union[str, os.PathLike[str]]) -> Box:
    """
    get_czimd_box: Get CZI metadata_tools as a python-box. For details: https://pypi.org/project/python-box/

    Args:
        filepath (Union[str, os.PathLike[str]]): Filepath of the CZI file

    Returns:
        Box: CZI metadata_tools as a Box object
    """

    readertype = pyczi.ReaderFileInputTypes.Standard

    if validators.url(str(filepath)):
        readertype = pyczi.ReaderFileInputTypes.Curl

    # get metadata_tools dictionary using pylibCZIrw
    with pyczi.open_czi(str(filepath), readertype) as czi_document:
        metadata_dict = czi_document.metadata

    czimd_box = Box(
        metadata_dict,
        conversion_box=True,
        default_box=True,
        default_box_attr=None,
        default_box_create_on_get=True,
    )

    # set the defaults to False
    czimd_box.has_scale = False

    if "Scaling" in czimd_box.ImageDocument.Metadata:
        czimd_box.has_scale = True

    return czimd_box


# filepath = r"F:\Testdata_Zeiss\Mindpeak\28448e7c-3dbb-486e-ab57-c4fb3ade3f1d.czi"
# filepath = r"F:\Testdata_Zeiss\Mindpeak\sample.czi"
# filepath = r"F:\Testdata_Zeiss\CZI_Testfiles\W96_A1+A2_S=2_4x2_Z=5_CH=2.czi"
# filepath = r"F:\Testdata_Zeiss\CZI_Testfiles\W96_A1+A2_S=2_Pos=8_Z=5_CH=2.czi"
filepath = r"F:\Github\czitools\data\WellD6_S1.czi"

if __name__ == "__main__":

    cziscale = CziScaleXYZ(filepath)
    print(f"X: {cziscale.X}, Y: {cziscale.Y}, Z: {cziscale.Z}")
