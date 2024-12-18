from typing import Tuple, Optional, Union, List
from dataclasses import dataclass, field, fields, Field
from box import Box
import os
from czitools.utils import logging_tools
from czitools.utils.box import get_czimd_box
from pylibCZIrw import czi as pyczi

from pydantic.dataclasses import dataclass
from pydantic import BaseModel, ConfigDict

logger = logging_tools.set_logging()


def string_to_float_list(string: str) -> List[float]:
    """Converts a space-separated string of numbers into a list of floats.

    Args:
      string: The input string.

    Returns:
      A list of floats.
    """

    numbers = string.split()
    float_numbers = [float(num) for num in numbers]

    return float_numbers


class Config:
    arbitrary_types_allowed = True


@dataclass(config=Config)
class CziDimensions:
    """
    CziDimensions is a dataclass that encapsulates the dimensions of a CZI image.
    Attributes:
        czisource (Union[str, os.PathLike[str], Box]): Source of the CZI image.
        SizeX (Optional[int]): Total size in the X dimension, including scenes.
        SizeY (Optional[int]): Total size in the Y dimension, including scenes.
        SizeX_scene (Optional[int]): Size in the X dimension per scene (if equal scene sizes).
        SizeY_scene (Optional[int]): Size in the Y dimension per scene (if equal scene sizes).
        SizeS (Optional[int]): Size in the S dimension.
        SizeT (Optional[int]): Size in the T dimension.
        SizeZ (Optional[int]): Size in the Z dimension.
        SizeC (Optional[int]): Size in the C dimension.
        SizeM (Optional[int]): Size in the M dimension.
        SizeR (Optional[int]): Size in the R dimension.
        SizeH (Optional[int]): Size in the H dimension.
        SizeI (Optional[int]): Size in the I dimension.
        SizeV (Optional[int]): Size in the V dimension.
        SizeB (Optional[int]): Size in the B dimension.
        posZ (Optional[List[float]]): List of Z positions in microns, if they exist.
        posT (Optional[List[float]]): List of T positions in seconds, if they exist.
        verbose (bool): Flag to enable verbose logging.
    Methods:
        __post_init__(): Initializes the dimensions by reading from the CZI image data.
        set_dimensions(): Populates the image dimensions with the detected values from the metadata.
        set_dimensions_new(): Populates the image dimensions with the detected values from the metadata using a new method.
    Notes:
        The class contains information about official CZI Dimension Characters:
        "X": "Width"        : width of image [pixel]
        "Y": "Height"       : height of image [pixel]
        "C": "Channel"      : number of channels
        "Z": "Slice"        : number of z-planes
        "T": "Time"         : number of time points
        "R": "Rotation"     :
        "S": "Scene"        : contiguous regions of interest in a mosaic image
        "I": "Illumination" : SPIM direction for LightSheet
        "B": "Block"        : acquisition
        "M": "Mosaic"       : index of tile for compositing a scene
        "V": "View"         : e.g. for SPIM
        In addition, it contains the Z-Positions [microns] and T-Positions [s] if they exist. Otherwise, they
        are set to None.
        "H": "Phase"        : e.g. Airy detector fibers
    """

    czisource: Union[str, os.PathLike[str], Box]
    SizeX: Optional[int] = field(
        init=False, default=None
    )  # total size X including scenes
    SizeY: Optional[int] = field(
        init=False, default=None
    )  # total size Y including scenes
    SizeX_scene: Optional[int] = field(
        init=False, default=None
    )  # size X per scene (if equal scene sizes)
    SizeY_scene: Optional[int] = field(
        init=False, default=None
    )  # size Y per scene (if equal scene sizes)
    SizeS: Optional[int] = field(init=False, default=None)
    SizeT: Optional[int] = field(init=False, default=None)
    SizeZ: Optional[int] = field(init=False, default=None)
    SizeC: Optional[int] = field(init=False, default=None)
    SizeM: Optional[int] = field(init=False, default=None)
    SizeR: Optional[int] = field(init=False, default=None)
    SizeH: Optional[int] = field(init=False, default=None)
    SizeI: Optional[int] = field(init=False, default=None)
    SizeV: Optional[int] = field(init=False, default=None)
    SizeB: Optional[int] = field(init=False, default=None)
    posZ: Optional[List[float]] = field(init=False, default=None)
    posT: Optional[List[float]] = field(init=False, default=None)
    verbose: bool = False

    def __post_init__(self):

        if self.verbose:
            logger.info("Reading Dimensions from CZI image data.")
        self.set_dimensions()

        # set dimensions in XY with respect to possible down scaling
        self.SizeX_sf = self.SizeX
        self.SizeY_sf = self.SizeY

    def set_dimensions(self):
        """
        Populate the image dimensions with the detected values from the metadata.
        This method sets the dimensions of the image based on the metadata extracted from the CZI file.
        It handles various dimensions such as SizeX, SizeY, SizeS, SizeT, SizeZ, SizeC, SizeM, SizeR, SizeH, SizeI, SizeV, and SizeB.
        Additionally, it processes scene dimensions and time (T) and z-position (Z) lists if available.
        Attributes:
            SizeX (int): Width of the image.
            SizeY (int): Height of the image.
            SizeS (int): Size of the S dimension.
            SizeT (int): Size of the T dimension.
            SizeZ (int): Size of the Z dimension.
            SizeC (int): Size of the C dimension.
            SizeM (int): Size of the M dimension.
            SizeR (int): Size of the R dimension.
            SizeH (int): Size of the H dimension.
            SizeI (int): Size of the I dimension.
            SizeV (int): Size of the V dimension.
            SizeB (int): Size of the B dimension.
            SizeX_scene (int): Width of the scene bounding rectangle.
            SizeY_scene (int): Height of the scene bounding rectangle.
            posT (list of float): List of time positions.
            posZ (list of float): List of z positions.
        Raises:
            KeyError: If scene bounding rectangle information is not found.
            Exception: If there is an error converting string to float list for posT or posZ.
        """

        # get the Box and extract the relevant dimension metadata_tools
        if isinstance(self.czisource, Box):
            czi_box = self.czisource
        else:
            czi_box = get_czimd_box(self.czisource)

        dimensions = czi_box.ImageDocument.Metadata.Information.Image

        # define the image dimensions to check for
        dims = [
            "SizeX",
            "SizeY",
            "SizeS",
            "SizeT",
            "SizeZ",
            "SizeC",
            "SizeM",
            "SizeR",
            "SizeH",
            "SizeI",
            "SizeV",
            "SizeB",
        ]

        cls_fields: Tuple[Field, ...] = fields(self.__class__)
        for fd in cls_fields:
            if fd.name in dims:
                if dimensions[fd.name] is not None:
                    setattr(self, fd.name, int(dimensions[fd.name]))

        if czi_box.has_scenes:
            try:
                with pyczi.open_czi(czi_box.filepath, czi_box.czi_open_arg) as czidoc:
                    self.SizeX_scene = czidoc.scenes_bounding_rectangle[0].w
                    self.SizeY_scene = czidoc.scenes_bounding_rectangle[0].h
            except KeyError as e:
                self.SizeX_scene = None
                self.SizeY_scene = None
                if self.verbose:
                    logger.warning(
                        "Scenes Dimension detected but no bounding rectangle information found."
                    )

        if czi_box.has_T:
            # check if there is a list with timepoints (is not in very CZI)
            if dimensions.Dimensions.T.Positions is not None:
                if dimensions.Dimensions.T.Positions.List is not None:
                    try:
                        self.posT = string_to_float_list(
                            dimensions.Dimensions.T.Positions.List.Offsets
                        )
                    except Exception as e:
                        if self.verbose:
                            logger.error(f"{e}")
                else:
                    if self.verbose:
                        logger.warning(
                            "No posT list found under 'dimensions.Dimensions.T.Positions.List'"
                        )
            else:
                if self.verbose:
                    logger.warning(
                        "No posT list found under 'dimensions.Dimensions.T.Positions'"
                    )

        if czi_box.has_Z:
            # check if there is a list with z-positions (is not in very CZI)
            if dimensions.Dimensions.Z.Positions is not None:
                if dimensions.Dimensions.Z.Positions.List is not None:
                    try:
                        self.posZ = string_to_float_list(
                            dimensions.Dimensions.Z.Positions.List.Offsets
                        )
                    except Exception as e:
                        if self.verbose:
                            logger.error(f"{e}")
                else:
                    if self.verbose:
                        logger.warning(
                            "No posZ list found under 'dimensions.Dimensions.Z.Positions.List'"
                        )
            else:
                if self.verbose:
                    logger.warning(
                        "No posZ list found under 'dimensions.Dimensions.Z.Positions'"
                    )
