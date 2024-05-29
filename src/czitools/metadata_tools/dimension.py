from typing import Tuple, Optional, Union
from dataclasses import dataclass, field, fields, Field
from box import Box
import os
from czitools.utils.logging_tools import get_logger, set_logging
from czitools.utils.box import get_czimd_box
from pylibCZIrw import czi as pyczi

#logger = get_logger()
logger = set_logging()


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
    """

    def __post_init__(self):

        self.set_dimensions()
        logger.info("Reading Dimensions from CZI image data.")

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