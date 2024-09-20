from typing import Tuple, Optional, Union, List
from dataclasses import dataclass, field, fields, Field
from box import Box
import os
from czitools.utils import logging_tools
from czitools.utils.box import get_czimd_box
from pylibCZIrw import czi as pyczi

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


@dataclass
class CziDimensions:
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
    """Dataclass containing the image dimensions.

    Information official CZI Dimension Characters:
    "X":"Width"        : width of image [pixel]
    "Y":"Height"       : height of image [pixel]
    "C":"Channel"      : number of channels
    "Z":"Slice"        : number of z-planes
    "T":"Time"         : number of time points
    "R":"Rotation"     :
    "S":"Scene"        : contiguous regions of interest in a mosaic image
    "I":"Illumination" : SPIM direction for LightSheet
    "B":"Block"        : acquisition
    "M":"Mosaic"       : index of tile for compositing a scene
    "H":"Phase"        : e.g. Airy detector fibers
    "V":"View"         : e.g. for SPIM

    In addition in contains the Z-Positions [microns] and T-Positions [s] if they exist. Otherwise they
    are set to None

    """

    def __post_init__(self):

        logger.info("Reading Dimensions from CZI image data.")
        self.set_dimensions()

        # set dimensions in XY with respect to possible down scaling
        self.SizeX_sf = self.SizeX
        self.SizeY_sf = self.SizeY

    def set_dimensions(self):
        """Populate the image dimensions with the detected values from the metadata_tools"""

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
                logger.warning(
                    "Scenes Dimension detected but no bounding rectangle information found."
                )

        if czi_box.has_T:
            # check if there is a list with timepoints (is not in very CZI)
            if dimensions.Dimensions.T.Positions is not None:
                if dimensions.Dimensions.T.Positions.List is not None:
                    try:
                        self.posT = string_to_float_list(dimensions.Dimensions.T.Positions.List.Offsets)
                    except Exception as e:
                        logger.error(f"{e}")
                else:
                    logger.warning("No posT list found under 'dimensions.Dimensions.T.Positions.List'")
            else:
                logger.warning("No posT list found under 'dimensions.Dimensions.T.Positions'")

        if czi_box.has_Z:
            # check if there is a list with z-positions (is not in very CZI)
            if dimensions.Dimensions.Z.Positions is not None:
                if dimensions.Dimensions.Z.Positions.List is not None:
                    try:
                        self.posZ = string_to_float_list(dimensions.Dimensions.Z.Positions.List.Offsets)
                    except Exception as e:
                        logger.error(f"{e}")
                else:
                    logger.warning("No posZ list found under 'dimensions.Dimensions.Z.Positions.List'")
            else:
                logger.warning("No posZ list found under 'dimensions.Dimensions.Z.Positions'")


    def set_dimensions_new(self):
        """Populate the image dimensions with the detected values from the metadata_tools"""

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

        dims_dict = {
            "SizeX": "X",
            "SizeY": "Y",
            "SizeS": "S",
            "SizeT": "T",
            "SizeZ": "Z",
            "SizeC": "C",
            "SizeM": "M",
            "SizeR": "R",
            "SizeH": "H",
            "SizeI": "I",
            "SizeV": "V",
            "SizeB": "B",
        }

        with pyczi.open_czi(czi_box.filepath, czi_box.czi_open_arg) as czidoc:

            total_bounding_box = czidoc.total_bounding_box

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
                logger.warning(
                    "Scenes Dimension detected but no bounding rectangle information found."
                )
