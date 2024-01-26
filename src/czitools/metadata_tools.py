# -*- coding: utf-8 -*-

#################################################################
# File        : metadata_tools.py
# Author      : sebi06
#
# Disclaimer: The code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Type, Any, Union, Mapping, Annotated
import os
from collections import Counter
import xml.etree.ElementTree as ET
from pylibCZIrw import czi as pyczi
from czitools import misc_tools
import numpy as np
from dataclasses import dataclass, field, fields, Field
from pathlib import Path
from box import Box, BoxList
from czitools import logger as LOGGER
import time
import validators
from dataclasses import asdict


logger = LOGGER.get_logger()


@dataclass
class ValueRange:
    lo: float
    hi: float


@dataclass
class CziMetadata:
    filepath: Union[str, os.PathLike[str]]
    filename: Optional[str] = field(init=False, default=None)
    dirname: Optional[str] = field(init=False, default=None)
    is_url: Optional[bool] = field(init=False, default=False)
    software_name: Optional[str] = field(init=False, default=None)
    software_version: Optional[str] = field(init=False, default=None)
    acquisition_date: Optional[str] = field(init=False, default=None)
    czi_box: Optional[Box] = field(init=False, default=None)
    pyczi_dims: Optional[Dict[str, tuple]] = field(
        init=False, default_factory=lambda: {}
    )
    aics_dimstring: Optional[str] = field(init=False, default=None)
    aics_dims_shape: Optional[List[Dict[str, tuple]]] = field(
        init=False, default_factory=lambda: {}
    )
    aics_size: Optional[Tuple[int]] = field(init=False, default_factory=lambda: ())
    aics_ismosaic: Optional[bool] = field(init=False, default=None)
    aics_dim_order: Optional[Dict[str, int]] = field(
        init=False, default_factory=lambda: {}
    )
    aics_dim_index: Optional[List[int]] = field(init=False, default_factory=lambda: [])
    aics_dim_valid: Optional[int] = field(init=False, default=None)
    aics_posC: Optional[int] = field(init=False, default=None)
    pixeltypes: Optional[Dict[int, str]] = field(init=False, default_factory=lambda: {})
    isRGB: Optional[bool] = field(init=False, default=False)
    has_scenes: Optional[bool] = field(init=False, default=False)
    has_label: Optional[bool] = field(init=False, default=False)
    has_preview: Optional[bool] = field(init=False, default=False)
    attachments: Optional[CziAttachments] = field(init=False, default=None)
    npdtype: Optional[List[Any]] = field(init=False, default_factory=lambda: [])
    maxvalue: Optional[List[int]] = field(init=False, default_factory=lambda: [])
    image: Optional[CziDimensions] = field(init=False, default=None)
    bbox: Optional[CziBoundingBox] = field(init=False, default=None)
    channelinfo: Optional[CziChannelInfo] = field(init=False, default=None)
    scale: Optional[CziScaling] = field(init=False, default=None)
    objective: Optional[CziObjectives] = field(init=False, default=None)
    detector: Optional[CziDetector] = field(init=False, default=None)
    microscope: Optional[CziMicroscope] = field(init=False, default=None)
    sample: Optional[CziSampleInfo] = field(init=False, default=None)
    add_metadata: Optional[CziAddMetaData] = field(init=False, default=None)
    pyczi_readertype: Optional[pyczi.ReaderFileInputTypes] = field(
        init=False, default=None
    )
    array6d_size: Optional[Tuple[int]] = field(init=False, default_factory=lambda: ())
    scene_size_consistent: Optional[Tuple[int]] = field(init=False, default_factory=lambda: ())
    """
    Create a CziMetadata object from the filename of the CZI image file.
    """

    def __post_init__(self):

        # check if the location is actually a local file
        if misc_tools.is_local_file(str(self.filepath)):

            self.is_url = False
            self.pyczi_readertype = pyczi.ReaderFileInputTypes.Standard
            if isinstance(self.filepath, Path):
                # convert to string
                self.filepath = str(self.filepath)

        # check if filepath is a valid url
        elif misc_tools.check_url(self.filepath, https_only=True):
            self.is_url = True
            self.pyczi_readertype = pyczi.ReaderFileInputTypes.Curl
            logger.info(
                "FilePath is a valid link. Only pylibCZIrw functionality is available."
            )

        # get directory and filename etc.
        self.dirname = str(Path(self.filepath).parent)
        self.filename = str(Path(self.filepath).name)

        # get the metadata as box
        self.czi_box = get_czimd_box(self.filepath)

        # check for existence of scenes
        self.has_scenes = self.czi_box.has_scenes

        # get acquisition data and SW version
        if self.czi_box.ImageDocument.Metadata.Information.Application is not None:
            self.software_name = (
                self.czi_box.ImageDocument.Metadata.Information.Application.Name
            )
            self.software_version = (
                self.czi_box.ImageDocument.Metadata.Information.Application.Version
            )

        if self.czi_box.ImageDocument.Metadata.Information.Image is not None:
            self.acquisition_date = (
                self.czi_box.ImageDocument.Metadata.Information.Image.AcquisitionDateAndTime
            )

        if self.czi_box.ImageDocument.Metadata.Information.Document is not None:
            self.creation_date = (
                self.czi_box.ImageDocument.Metadata.Information.Document.CreationDate
            )
            self.user_name = (
                self.czi_box.ImageDocument.Metadata.Information.Document.UserName
            )

        # get the dimensions and order
        self.image = CziDimensions(self.czi_box)

        # get metadata using pylibCZIrw
        with pyczi.open_czi(self.filepath, self.pyczi_readertype) as czidoc:
            # get dimensions
            self.pyczi_dims = czidoc.total_bounding_box

            # get the pixel typed for all channels
            self.pixeltypes = czidoc.pixel_types
            self.isRGB, self.consistent_pixeltypes = self.check_if_rgb(self.pixeltypes)

            # check for consistent scene shape
            self.scene_shape_is_consistent = self.check_scenes_shape(
                czidoc, size_s=self.image.SizeS
            )

            # if self.scene_shape_is_consistent:
            #    self.image.SizeX_scene = czidoc.scenes_bounding_rectangle[0].w
            #    self.image.SizeY_scene = czidoc.scenes_bounding_rectangle[0].h

        if not self.is_url:
            # get some additional metadata using aicspylibczi
            try:
                from aicspylibczi import CziFile

                # get the general CZI object using aicspylibczi
                aicsczi = CziFile(self.filepath)

                self.aics_dimstring = aicsczi.dims
                self.aics_dims_shape = aicsczi.get_dims_shape()
                self.aics_size = aicsczi.size
                self.aics_ismosaic = aicsczi.is_mosaic()
                (
                    self.aics_dim_order,
                    self.aics_dim_index,
                    self.aics_dim_valid,
                ) = self.get_dimorder(aicsczi.dims)
                self.aics_posC = self.aics_dim_order["C"]

            except ImportError as e:
                # print("Package aicspylibczi not found. Use Fallback values.")
                logger.warning("Package aicspylibczi not found. Use Fallback values.")

        for ch, px in self.pixeltypes.items():
            npdtype, maxvalue = self.get_dtype_fromstring(px)
            self.npdtype.append(npdtype)
            self.maxvalue.append(maxvalue)

        # try to guess if the CZI is a mosaic file
        if self.image.SizeM is None or self.image.SizeM == 1:
            self.ismosaic = False
        else:
            self.ismosaic = True

        # get the bounding boxes
        self.bbox = CziBoundingBox(self.czi_box)

        # get information about channels
        self.channelinfo = CziChannelInfo(self.czi_box)

        # get scaling info
        self.scale = CziScaling(self.czi_box)

        # get objective information
        self.objective = CziObjectives(self.czi_box)

        # get detector information
        self.detector = CziDetector(self.czi_box)

        # get detector information
        self.microscope = CziMicroscope(self.czi_box)

        # get information about sample carrier and wells etc.
        self.sample = CziSampleInfo(self.czi_box)

        # get additional metainformation
        self.add_metadata = CziAddMetaData(self.czi_box)

        # check for attached label or preview image
        self.attachments = CziAttachments(self.czi_box)

    # can be also used without creating an instance of the class
    @staticmethod
    def get_dtype_fromstring(
        pixeltype: str,
    ) -> Tuple[Optional[np.dtype], Optional[int]]:
        dtype = None
        maxvalue = None

        if pixeltype == "gray16" or pixeltype == "Gray16":
            dtype = np.dtype(np.uint16)
            maxvalue = 65535
        if pixeltype == "gray8" or pixeltype == "Gray8":
            dtype = np.dtype(np.uint8)
            maxvalue = 255
        if pixeltype == "bgr48" or pixeltype == "Bgr48":
            dtype = np.dtype(np.uint16)
            maxvalue = 65535
        if pixeltype == "bgr24" or pixeltype == "Bgr24":
            dtype = np.dtype(np.uint8)
            maxvalue = 255
        if pixeltype == "bgr96float" or pixeltype == "Bgr96Float":
            dtype = np.dtype(np.uint16)
            maxvalue = 65535

        return dtype, maxvalue

    @staticmethod
    def get_dimorder(dim_string: str) -> Tuple[Dict, List, int]:
        """Get the order of dimensions from dimension string

        :param dim_string: string containing the dimensions
        :type dim_string: str
        :return: dims_dict - dictionary with the dimensions and its positions
        :rtype: dict
        :return: dimindex_list - list with indices of dimensions
        :rtype: list
        :return: numvalid_dims - number of valid dimensions
        :rtype: integer
        """

        dimindex_list = []
        dims = ["R", "I", "M", "H", "V", "B", "S", "T", "C", "Z", "Y", "X", "A"]
        dims_dict = {}

        # loop over all dimensions and find the index
        for d in dims:
            dims_dict[d] = dim_string.find(d)
            dimindex_list.append(dim_string.find(d))

        # check if a dimension really exists
        numvalid_dims = sum(i >= 0 for i in dimindex_list)

        return dims_dict, dimindex_list, numvalid_dims

    @staticmethod
    def check_scenes_shape(czidoc: pyczi.CziReader, size_s: Union[int, None]) -> bool:
        """Check if all scenes have the same shape.

        Args:
            czidoc (pyczi.CziReader): CziReader to read the properties
            size_s (Union[int, None]): Size of scene dimension

        Returns:
            bool: True is all scenes have identical XY shape
        """
        scene_width = []
        scene_height = []
        scene_shape_is_consistent = False

        if size_s is not None:
            for s in range(size_s):
                scene_width.append(czidoc.scenes_bounding_rectangle[s].w)
                scene_height.append(czidoc.scenes_bounding_rectangle[s].h)

            # check if all entries in list are the same
            sw = scene_width.count(scene_width[0]) == len(scene_width)
            sh = scene_height.count(scene_height[0]) == len(scene_height)

            # only if entries for X and Y are all the same as the shape is consistent
            if sw is True and sh is True:
                scene_shape_is_consistent = True

        else:
            scene_shape_is_consistent = True

        return scene_shape_is_consistent

    @staticmethod
    def check_if_rgb(pixeltypes: Dict) -> Tuple[bool, bool]:
        is_rgb = False

        for k, v in pixeltypes.items():
            if "Bgr" in v:
                is_rgb = True

        # flag to check if all elements are same
        is_consistant = True

        # extracting value to compare
        test_val = list(pixeltypes.values())[0]

        for ele in pixeltypes:
            if pixeltypes[ele] != test_val:
                is_consistant = False
                break

        return is_rgb, is_consistant


@dataclass
class CziDimensions:
    czisource: Union[str, os.PathLike[str], Box]
    SizeX: Optional[int] = field(init=False, default=None)  # total size X including scenes
    SizeY: Optional[int] = field(init=False, default=None)  # total size Y including scenes
    SizeX_scene: Optional[int] = field(init=False, default=None)  # size X per scene (if equal scene sizes)
    SizeY_scene: Optional[int] = field(init=False, default=None)  # size Y per scene (if equal scene sizes)
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
        """Populate the image dimensions with the detected values from the metadata"""

        # get the Box and extract the relevant dimension metadata
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
            with pyczi.open_czi(czi_box.filepath, czi_box.czi_open_arg) as czidoc:
                self.SizeX_scene = czidoc.scenes_bounding_rectangle[0].w
                self.SizeY_scene = czidoc.scenes_bounding_rectangle[0].h


@dataclass
class CziBoundingBox:
    czisource: Union[str, os.PathLike[str], Box]
    scenes_bounding_rect: Optional[Dict[int, pyczi.Rectangle]] = field(
        init=False, default_factory=lambda: []
    )
    total_rect: Optional[pyczi.Rectangle] = field(init=False, default=None)
    total_bounding_box: Optional[Dict[str, tuple]] = field(
        init=False, default_factory=lambda: []
    )

    # TODO Is this really needed as a separate class or better integrate directly into CziMetadata class?

    def __post_init__(self):
        logger.info("Reading BoundingBoxes from CZI image data.")

        pyczi_readertype = pyczi.ReaderFileInputTypes.Standard

        if isinstance(self.czisource, Box):
            self.czisource = self.czisource.filepath

        # check if czisource is a valid url
        if misc_tools.check_url(self.czisource, https_only=True):
            pyczi_readertype = pyczi.ReaderFileInputTypes.Curl
            logger.info(
                "FilePath is a valid link. Only pylibCZIrw functionality is available."
            )

        # check if the location is actually a local file
        elif misc_tools.is_local_file(str(self.czisource)):

            if isinstance(self.czisource, Path):
                # convert to string
                self.czisource = str(self.czisource)

        with pyczi.open_czi(self.czisource, pyczi_readertype) as czidoc:
            try:
                self.scenes_bounding_rect = czidoc.scenes_bounding_rectangle
            except Exception as e:
                self.scenes_bounding_rect = None
                # print("Scenes Bounding rectangle not found.")
                logger.info("Scenes Bounding rectangle not found.")

            try:
                self.total_rect = czidoc.total_bounding_rectangle
            except Exception as e:
                self.total_rect = None
                # print("Total Bounding rectangle not found.")
                logger.info("Total Bounding rectangle not found.")

            try:
                self.total_bounding_box = czidoc.total_bounding_box
            except Exception as e:
                self.total_bounding_box = None
                # print("Total Bounding Box not found.")
                logger.info("Total Bounding Box not found.")


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


@dataclass
class CziChannelInfo:
    czisource: Union[str, os.PathLike[str], Box]
    names: List[str] = field(init=False, default_factory=lambda: [])
    dyes: List[str] = field(init=False, default_factory=lambda: [])
    colors: List[str] = field(init=False, default_factory=lambda: [])
    clims: List[List[float]] = field(init=False, default_factory=lambda: [])
    gamma: List[float] = field(init=False, default_factory=lambda: [])

    def __post_init__(self):
        logger.info("Reading Channel Information from CZI image data.")

        if isinstance(self.czisource, Box):
            czi_box = self.czisource
        else:
            czi_box = get_czimd_box(self.czisource)

        # get channels part of dict
        if czi_box.has_channels:
            try:
                # extract the relevant dimension metadata
                channels = (
                    czi_box.ImageDocument.Metadata.Information.Image.Dimensions.Channels.Channel
                )
                if isinstance(channels, Box):
                    # get the data in case of only one channel
                    self.names.append(
                        "CH1"
                    ) if channels.Name is None else self.names.append(channels.Name)
                elif isinstance(channels, BoxList):
                    # get the data in case multiple channels
                    for ch in range(len(channels)):
                        self.names.append("CH1") if channels[
                            ch
                        ].Name is None else self.names.append(channels[ch].Name)
            except AttributeError:
                channels = None
        elif not czi_box.has_channels:
            logger.info("Channel(s) information not found.")

        if czi_box.has_disp:
            try:
                # extract the relevant dimension metadata
                disp = czi_box.ImageDocument.Metadata.DisplaySetting.Channels.Channel
                if isinstance(disp, Box):
                    self.get_channel_info(disp)
                elif isinstance(disp, BoxList):
                    for ch in range(len(disp)):
                        self.get_channel_info(disp[ch])
            except AttributeError:
                disp = None

        elif not czi_box.has_disp:
            # print("DisplaySetting(s) not found.")
            logger.info("DisplaySetting(s) not found.")

    def get_channel_info(self, display: Box):
        if display is not None:
            self.dyes.append(
                "Dye-CH1"
            ) if display.ShortName is None else self.dyes.append(display.ShortName)
            self.colors.append(
                "#80808000"
            ) if display.Color is None else self.colors.append(display.Color)

            low = 0.0 if display.Low is None else float(display.Low)
            high = 0.5 if display.High is None else float(display.High)

            self.clims.append([low, high])
            self.gamma.append(0.85) if display.Gamma is None else self.gamma.append(
                float(display.Gamma)
            )
        else:
            self.dyes.append("Dye-CH1")
            self.colors.append("#80808000")
            self.clims.append([0.0, 0.5])
            self.gamma.append(0.85)


@dataclass
class CziScaling:
    czisource: Union[str, os.PathLike[str], Box]
    X: Optional[float] = field(init=False, default=None)
    Y: Optional[float] = field(init=False, default=None)
    Z: Optional[float] = field(init=False, default=None)
    X_sf: Optional[float] = field(init=False, default=None)
    Y_sf: Optional[float] = field(init=False, default=None)
    ratio: Optional[Dict[str, float]] = field(init=False, default=None)
    # ratio_sf: Optional[Dict[str, float]] = field(init=False, default=None)
    # scalefactorXY: Optional[float] = field(init=False, default=None)
    unit: Optional[str] = field(init=True, default="micron")
    zoom: Annotated[float, ValueRange(0.01, 1.0)] = field(init=True, default=1.0)

    def __post_init__(self):
        logger.info("Reading Scaling from CZI image data.")

        if isinstance(self.czisource, Box):
            czi_box = self.czisource
        else:
            czi_box = get_czimd_box(self.czisource)

        if czi_box.has_scale:
            distances = czi_box.ImageDocument.Metadata.Scaling.Items.Distance

            # get the scaling values for X,Y and Z
            self.X = np.round(self.safe_get_scale(distances, 0), 3)
            self.Y = np.round(self.safe_get_scale(distances, 1), 3)
            self.Z = np.round(self.safe_get_scale(distances, 2), 3)

            # calc the scaling values for X,Y when applying downscaling
            self.X_sf = np.round(self.X * (1 / self.zoom), 3)
            self.Y_sf = np.round(self.Y * (1 / self.zoom), 3)

            # calc the scaling ratio
            self.ratio = {
                "xy": np.round(self.X / self.Y, 3),
                "zx": np.round(self.Z / self.X, 3),
                #"zx_sf": np.round(self.Z / self.X_sf, 3),
            }

        elif not czi_box.has_scale:
            logger.info("No scaling information found.")

    @staticmethod
    def safe_get_scale(dist: BoxList, idx: int) -> Optional[float]:
        scales = ["X", "Y", "Z"]

        try:
            # get the scaling value in [micron]
            sc = float(dist[idx].Value) * 1000000

            # check for the value = 0.0
            if sc == 0.0:
                sc = 1.0
                logger.info(
                    "Detected Scaling = 0.0 for "
                    + scales[idx]
                    + " Using default = 1.0 [micron]."
                )
            return sc

        except (IndexError, TypeError, AttributeError):
            logger.info(
                "No " + scales[idx] + "-Scaling found. Using default = 1.0 [micron]."
            )
            return 1.0


@dataclass
class CziObjectives:
    czisource: Union[str, os.PathLike[str], Box]
    NA: List[Optional[float]] = field(init=False, default_factory=lambda: [])
    objmag: List[Optional[float]] = field(init=False, default_factory=lambda: [])
    Id: List[Optional[str]] = field(init=False, default_factory=lambda: [])
    name: List[Optional[str]] = field(init=False, default_factory=lambda: [])
    model: List[Optional[str]] = field(init=False, default_factory=lambda: [])
    immersion: List[Optional[str]] = field(init=False, default_factory=lambda: [])
    tubelensmag: List[Optional[float]] = field(init=False, default_factory=lambda: [])
    totalmag: List[Optional[float]] = field(init=False, default_factory=lambda: [])

    def __post_init__(self):
        logger.info("Reading Objective Information from CZI image data.")

        if isinstance(self.czisource, Box):
            czi_box = self.czisource
        else:
            czi_box = get_czimd_box(self.czisource)

        # check if objective metadata actually exist
        if czi_box.has_objectives:
            try:
                # get objective data
                objective = (
                    czi_box.ImageDocument.Metadata.Information.Instrument.Objectives.Objective
                )
                if isinstance(objective, Box):
                    self.get_objective_info(objective)
                elif isinstance(objective, BoxList):
                    for obj in range(len(objective)):
                        self.get_objective_info(objective[obj])
            except AttributeError:
                objective = None

        elif not czi_box.has_objectives:
            # print("No Objective Information found.")
            logger.info("No Objective Information found.")

        # check if tubelens metadata exist
        if czi_box.has_tubelenses:
            # get tubelenes data
            tubelens = (
                czi_box.ImageDocument.Metadata.Information.Instrument.TubeLenses.TubeLens
            )

            if isinstance(tubelens, Box):

                if tubelens.Magnification is not None:
                    self.tubelensmag.append(float(tubelens.Magnification))
                elif tubelens.Magnification is None:
                    logger.warning("No tubelens magnification found. Use 1.0x instead.")
                    self.tubelensmag.append(1.0)

            elif isinstance(tubelens, BoxList):
                for tl in range(len(tubelens)):
                    self.tubelensmag.append(float(tubelens[tl].Magnification))

            # some additional checks to calc the total magnification
            if self.objmag is not None and self.tubelensmag is not None:
                self.totalmag = [i * j for i in self.objmag for j in self.tubelensmag]

        elif not czi_box.has_tubelens:
            logger.info("No Tublens Information found.")

        if self.objmag is not None and self.tubelensmag == []:
            self.totalmag = self.objmag

    def get_objective_info(self, objective: Box):
        self.name.append(objective.Name)
        self.immersion.append(objective.Immersion)

        if objective.LensNA is not None:
            self.NA.append(float(objective.LensNA))

        if objective.Id is not None:
            self.Id.append(objective.Id)

        if objective.NominalMagnification is not None:
            self.objmag.append(float(objective.NominalMagnification))

        if None in self.name and self.name.count(None) == 1:
            self.name.remove(None)
            self.name.append(objective.Manufacturer.Model)


@dataclass
class CziDetector:
    czisource: Union[str, os.PathLike[str], Box]
    model: List[str] = field(init=False, default_factory=lambda: [])
    name: List[str] = field(init=False, default_factory=lambda: [])
    Id: List[str] = field(init=False, default_factory=lambda: [])
    modeltype: List[str] = field(init=False, default_factory=lambda: [])
    gain: List[float] = field(init=False, default_factory=lambda: [])
    zoom: List[float] = field(init=False, default_factory=lambda: [])
    amplificationgain: List[float] = field(init=False, default_factory=lambda: [])

    def __post_init__(self):
        logger.info("Reading Detector Information from CZI image data.")

        if isinstance(self.czisource, Box):
            czi_box = self.czisource
        else:
            czi_box = get_czimd_box(self.czisource)

        # check if there are any detector entries inside the dictionary
        if czi_box.ImageDocument.Metadata.Information.Instrument is not None:
            # get the data for the detectors
            detectors = (
                czi_box.ImageDocument.Metadata.Information.Instrument.Detectors.Detector
            )

            # check for detector Id, Name, Model and Type
            if isinstance(detectors, Box):
                self.Id.append(detectors.Id)
                self.name.append(detectors.Name)
                self.model.append(detectors.Model)
                self.modeltype.append(detectors.Type)
                self.gain.append(detectors.Gain)
                self.zoom.append(detectors.Zoom)
                self.amplificationgain.append(detectors.AmplificationGain)

            # and do this differently in case of a list of detectors
            elif isinstance(detectors, BoxList):
                for d in range(len(detectors)):
                    self.Id.append(detectors[d].Id)
                    self.name.append(detectors[d].Name)
                    self.model.append(detectors[d].Model)
                    self.modeltype.append(detectors[d].Type)
                    self.gain.append(detectors[d].Gain)
                    self.zoom.append(detectors[d].Zoom)
                    self.amplificationgain.append(detectors[d].AmplificationGain)

        elif czi_box.ImageDocument.Metadata.Information.Instrument is None:
            # print("No Detetctor(s) information found.")
            logger.info("No Detetctor(s) information found.")
            self.model = [None]
            self.name = [None]
            self.Id = [None]
            self.modeltype = [None]
            self.gain = [None]
            self.zoom = [None]
            self.amplificationgain = [None]


@dataclass
class CziMicroscope:
    czisource: Union[str, os.PathLike[str], Box]
    Id: Optional[str] = field(init=False)
    Name: Optional[str] = field(init=False)
    System: Optional[str] = field(init=False)

    def __post_init__(self):
        logger.info("Reading Microscope Information from CZI image data.")

        if isinstance(self.czisource, Box):
            czi_box = self.czisource
        else:
            czi_box = get_czimd_box(self.czisource)

        if czi_box.ImageDocument.Metadata.Information.Instrument is None:
            self.Id = None
            self.Name = None
            self.System = None
            # print("No Microscope information found.")
            logger.info("No Microscope information found.")

        else:
            self.Id = (
                czi_box.ImageDocument.Metadata.Information.Instrument.Microscopes.Microscope.Id
            )
            self.Name = (
                czi_box.ImageDocument.Metadata.Information.Instrument.Microscopes.Microscope.Name
            )
            self.System = (
                czi_box.ImageDocument.Metadata.Information.Instrument.Microscopes.Microscope.System
            )


@dataclass
class CziSampleInfo:
    czisource: Union[str, os.PathLike[str], Box]
    well_array_names: List[str] = field(init=False, default_factory=lambda: [])
    well_indices: List[int] = field(init=False, default_factory=lambda: [])
    well_position_names: List[str] = field(init=False, default_factory=lambda: [])
    well_colID: List[int] = field(init=False, default_factory=lambda: [])
    well_rowID: List[int] = field(init=False, default_factory=lambda: [])
    well_counter: Dict = field(init=False, default_factory=lambda: {})
    scene_stageX: List[float] = field(init=False, default_factory=lambda: [])
    scene_stageY: List[float] = field(init=False, default_factory=lambda: [])
    image_stageX: float = field(init=False, default=None)
    image_stageY: float = field(init=False, default=None)

    def __post_init__(self):
        logger.info("Reading SampleCarrier Information from CZI image data.")

        if isinstance(self.czisource, Box):
            czi_box = self.czisource
        else:
            czi_box = get_czimd_box(self.czisource)

        size_s = CziDimensions(czi_box).SizeS

        if size_s is not None:
            try:
                allscenes = (
                    czi_box.ImageDocument.Metadata.Information.Image.Dimensions.S.Scenes.Scene
                )

                if isinstance(allscenes, Box):
                    self.get_well_info(allscenes)

                if isinstance(allscenes, BoxList):
                    for well in range(len(allscenes)):
                        self.get_well_info(allscenes[well])

            except AttributeError:
                # print("CZI contains no scene metadata.")
                logger.info("CZI contains no scene metadata.")

        elif size_s is None:
            logger.info(
                "No Scene or Well information found. Try to read XY Stage Coordinates from subblocks."
            )

            try:
                # read the data from CSV file
                planetable, csvfile = misc_tools.get_planetable(
                    czi_box.filepath, read_one_only=True, savetable=False
                )

                self.image_stageX = float(planetable["X[micron]"][0])
                self.image_stageY = float(planetable["Y[micron]"][0])

            except Exception as e:
                print(e)

    def get_well_info(self, well: Box):
        # logger = setup_log("CziSampleInfo")

        # check the ArrayName
        if well.ArrayName is not None:
            self.well_array_names.append(well.ArrayName)
            # count the well instances
            self.well_counter = Counter(self.well_array_names)

        if well.Index is not None:
            self.well_indices.append(int(well.Index))
        elif well.Index is None:
            # print("Well Index not found.")
            logger.info("Well Index not found.")
            self.well_indices.append(1)

        if well.Name is not None:
            self.well_position_names.append(well.Name)
        elif well.Name is None:
            # print("Well Position Names not found.")
            logger.info("Well Position Names not found.")
            self.well_position_names.append("P1")

        if well.Shape is not None:
            self.well_colID.append(int(well.Shape.ColumnIndex))
            self.well_rowID.append(int(well.Shape.RowIndex))
        elif well.Shape is None:
            # print("Well Column or Row IDs not found.")
            logger.info("Well Column or Row IDs not found.")
            self.well_colID.append(0)
            self.well_rowID.append(0)

        if well.CenterPosition is not None:
            # get the SceneCenter Position
            sx = well.CenterPosition.split(",")[0]
            sy = well.CenterPosition.split(",")[1]
            self.scene_stageX.append(np.double(sx))
            self.scene_stageY.append(np.double(sy))
        if well.CenterPosition is None:
            # print("Stage Positions XY not found.")
            logger.info("Stage Positions XY not found.")
            self.scene_stageX.append(0.0)
            self.scene_stageY.append(0.0)


@dataclass
class CziAddMetaData:
    czisource: Union[str, os.PathLike[str], Box]
    experiment: Optional[Box] = field(init=False, default=None)
    hardwaresetting: Optional[Box] = field(init=False, default=None)
    customattributes: Optional[Box] = field(init=False, default=None)
    displaysetting: Optional[Box] = field(init=False, default=None)
    layers: Optional[Box] = field(init=False, default=None)

    def __post_init__(self):
        logger.info("Reading additional Metedata from CZI image data.")

        if isinstance(self.czisource, Box):
            czi_box = self.czisource
        else:
            czi_box = get_czimd_box(self.czisource)

        if czi_box.has_experiment:
            self.experiment = czi_box.ImageDocument.Metadata.Experiment
        else:
            # print("No Experiment information found.")
            logger.info("No Experiment information found.")

        if czi_box.has_hardware:
            self.hardwaresetting = czi_box.ImageDocument.Metadata.HardwareSetting
        else:
            # print("No HardwareSetting information found.")
            logger.info("No HardwareSetting information found.")

        if czi_box.has_customattr:
            self.customattributes = czi_box.ImageDocument.Metadata.CustomAttributes
        else:
            # print("No CustomAttributes information found.")
            logger.info("No CustomAttributes information found.")

        if czi_box.has_disp:
            self.displaysetting = czi_box.ImageDocument.Metadata.DisplaySetting
        else:
            # print("No DisplaySetting information found.")
            logger.info("No DisplaySetting information found.")

        if czi_box.has_layers:
            self.layers = czi_box.ImageDocument.Metadata.Layers
        else:
            # print("No Layers information found.")
            logger.info("No Layers information found.")


@dataclass
class CziScene:
    filepath: Union[str, os.PathLike[str]]
    index: int
    bbox: Optional[pyczi.Rectangle] = field(init=False, default=None)
    xstart: Optional[int] = field(init=False, default=None)
    ystart: Optional[int] = field(init=False, default=None)
    width: Optional[int] = field(init=False, default=None)
    height: Optional[int] = field(init=False, default=None)

    def __post_init__(self):
        logger.info("Reading Scene Information from CZI image data.")

        if isinstance(self.filepath, Path):
            # convert to string
            self.filepath = str(self.filepath)

        # get scene information from the CZI file
        with pyczi.open_czi(self.filepath) as czidoc:
            try:
                self.bbox = czidoc.scenes_bounding_rectangle[self.index]
                self.xstart = self.bbox.x
                self.ystart = self.bbox.y
                self.width = self.bbox.w
                self.height = self.bbox.h
            except KeyError:
                # in case an invalid index was used
                # print("No Scenes detected.")
                logger.info("No Scenes detected.")


def get_metadata_as_object(filepath: Union[str, os.PathLike[str]]) -> DictObj:
    """
    Get the complete CZI metadata as an object created based on the
    dictionary created from the XML data.
    """

    if isinstance(filepath, Path):
        # convert to string
        filepath = str(filepath)

    # get metadata dictionary using pylibCZIrw
    with pyczi.open_czi(filepath) as czidoc:
        md_dict = czidoc.metadata

    return DictObj(md_dict)


class DictObj:
    """
    Create an object based on a dictionary. See https://joelmccune.com/python-dictionary-as-object/
    """

    # TODO: is this class still neded because of suing python-box

    def __init__(self, in_dict: dict) -> None:
        assert isinstance(in_dict, dict)

        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(
                    self, key, [DictObj(x) if isinstance(x, dict) else x for x in val]
                )
            else:
                setattr(self, key, DictObj(val) if isinstance(val, dict) else val)


def obj2dict(obj: Any, sort: bool = True) -> Dict[str, Any]:
    """
    obj2dict: Convert a class attributes and their values to a dictionary

    Args:
        obj (Any): Class to be converted
        sort (bool, optional): Sort the resulting dictionary by key names. Defaults to True.

    Returns:
        Dict[str, Any]: The resulting disctionary.
    """

    # https://stackoverflow.com/questions/7963762/what-is-the-most-economical-way-to-convert-nested-python-objects-to-dictionaries

    if not hasattr(obj, "__dict__"):
        return obj

    result = {}

    for key, val in obj.__dict__.items():
        if key.startswith("_"):
            continue

        element = []

        if isinstance(val, list):
            for item in val:
                element.append(obj2dict(item))
        else:
            element = obj2dict(val)

        result[key] = element

    # delete key "czisource"
    if "czisource" in result.keys():
        del result["czisource"]

    if sort:
        return misc_tools.sort_dict_by_key(result)

    elif not sort:
        return result


def writexml(
    filepath: Union[str, os.PathLike[str]], xmlsuffix: str = "_CZI_MetaData.xml"
) -> str:
    """
    writexml: Write XML information of CZI to disk

    Args:
        filepath (Union[str, os.PathLike[str]]): CZI image filename
        xmlsuffix (str, optional): suffix for the XML file that will be created, defaults to '_CZI_MetaData.xml'. Defaults to '_CZI_MetaData.xml'.

    Returns:
        str: filename of the XML file
    """

    if isinstance(filepath, Path):
        # convert to string
        filepath = str(filepath)

    # get the raw metadata as XML or dictionary
    with pyczi.open_czi(filepath) as czidoc:
        metadata_xmlstr = czidoc.raw_metadata

    # change file name
    xmlfile = filepath.replace(".czi", xmlsuffix)

    # get tree from string
    tree = ET.ElementTree(ET.fromstring(metadata_xmlstr))

    # write XML file to same folder
    tree.write(xmlfile, encoding="utf-8", method="xml")

    return xmlfile


def create_md_dict_red(
    metadata: CziMetadata, sort: bool = True, remove_none: bool = True
) -> Dict:
    """
    create_mdict_red: Created a reduced metadata dictionary

    Args:
        metadata: CziMetadata class
        sort: sort the dictionary
        remove_none: Remove None values from dictionary

    Returns: dictionary with the metadata

    """

    # create a dictionary with the metadata
    md_dict = {
        "Directory": metadata.dirname,
        "Filename": metadata.filename,
        "AcqDate": metadata.acquisition_date,
        "CreationDate": metadata.creation_date,
        "UserName": metadata.user_name,
        "SW-App": metadata.software_version,
        "SW-Version": metadata.software_name,
        "SizeX": metadata.image.SizeX,
        "SizeY": metadata.image.SizeY,
        "SizeZ": metadata.image.SizeZ,
        "SizeC": metadata.image.SizeC,
        "SizeT": metadata.image.SizeT,
        "SizeS": metadata.image.SizeS,
        "SizeB": metadata.image.SizeB,
        "SizeM": metadata.image.SizeM,
        "SizeH": metadata.image.SizeH,
        "SizeI": metadata.image.SizeI,
        "isRGB": metadata.isRGB,
        "array6d_size": metadata.array6d_size,
        "has_scenes": metadata.has_scenes,
        "has_label": metadata.attachments.has_label,
        "has_preview": metadata.attachments.has_preview,
        "has_prescan": metadata.attachments.has_prescan,
        "ismosaic": metadata.ismosaic,
        "ObjNA": metadata.objective.NA,
        "ObjMag": metadata.objective.totalmag,
        "TubelensMag": metadata.objective.tubelensmag,
        "XScale": metadata.scale.X,
        "YScale": metadata.scale.Y,
        "XScale_sf": metadata.scale.X_sf,
        "YScale_sf": metadata.scale.Y_sf,
        "ZScale": metadata.scale.Z,
        "ScaleRatio_XYZ": metadata.scale.ratio,
        "ChannelNames": metadata.channelinfo.names,
        "ChannelDyes": metadata.channelinfo.dyes,
        "ChannelColors": metadata.channelinfo.colors,
        "WellArrayNames": metadata.sample.well_array_names,
        "WellIndicies": metadata.sample.well_indices,
        "WellPositionNames": metadata.sample.well_position_names,
        "WellRowID": metadata.sample.well_rowID,
        "WellColumnID": metadata.sample.well_colID,
        "WellCounter": metadata.sample.well_counter,
        "SceneCenterStageX": metadata.sample.scene_stageX,
        "SceneCenterStageY": metadata.sample.scene_stageX,
        "ImageStageX": metadata.sample.image_stageX,
        "ImageStageY": metadata.sample.image_stageX,
        "BoundingBoxX": metadata.bbox
    }

    # check for extra entries when reading mosaic file with a scale factor
    if hasattr(metadata.image, "SizeX_sf"):
        md_dict["XScale_sf"] = metadata.scale.X_sf
        md_dict["YScale_sf"] = metadata.scale.Y_sf

    if metadata.has_scenes:
        md_dict["SizeX_scene"] = metadata.image.SizeX_scene
        md_dict["SizeY_scene"] = metadata.image.SizeY_scene

    if remove_none:
        md_dict = misc_tools.remove_none_from_dict(md_dict)

    if sort:
        return misc_tools.sort_dict_by_key(md_dict)
    if not sort:
        return md_dict


def get_czimd_box(filepath: Union[str, os.PathLike[str]]) -> Box:
    """
    get_czimd_box: Get CZI metadata as a python-box. For details: https://pypi.org/project/python-box/

    Args:
        filepath (Union[str, os.PathLike[str]]): Filepath of the CZI file

    Returns:
        Box: CZI metadata as a Box object
    """

    is_url = False

    # check if the location is actually a local file
    if misc_tools.is_local_file(str(filepath)):

        # check if the input is a path-like object
        if isinstance(filepath, Path) or isinstance(filepath, str):
            # convert to string
            filepath = str(filepath)

        # get metadata dictionary using pylibCZIrw
        with pyczi.open_czi(filepath, pyczi.ReaderFileInputTypes.Standard) as czi_document:
            metadata_dict = czi_document.metadata

    # check if filepath is a valid url
    elif misc_tools.check_url(filepath, https_only=True):
        is_url = True
        # get metadata dictionary using a valid link using pylibCZIrw
        with pyczi.open_czi(filepath, pyczi.ReaderFileInputTypes.Curl) as czi_document:
            metadata_dict = czi_document.metadata

    czimd_box = Box(
        metadata_dict,
        conversion_box=True,
        default_box=True,
        default_box_attr=None,
        default_box_create_on_get=True,
        # default_box_no_key_error=True
    )

    # add the filepath
    czimd_box.filepath = filepath
    czimd_box.is_url = is_url

    if is_url:
        czimd_box.czi_open_arg = pyczi.ReaderFileInputTypes.Curl
    if not is_url:
        czimd_box.czi_open_arg = pyczi.ReaderFileInputTypes.Standard

    # set the defaults to False
    czimd_box.has_customattr = False
    czimd_box.has_experiment = False
    czimd_box.has_disp = False
    czimd_box.has_hardware = False
    czimd_box.has_scale = False
    czimd_box.has_instrument = False
    czimd_box.has_microscopes = False
    czimd_box.has_detectors = False
    czimd_box.has_objectives = False
    czimd_box.has_tubelenses = False
    czimd_box.has_disp = False
    czimd_box.has_channels = False
    czimd_box.has_info = False
    czimd_box.has_app = False
    czimd_box.has_doc = False
    czimd_box.has_image = False
    czimd_box.has_scenes = False
    czimd_box.has_dims = False
    czimd_box.has_layers = False

    if "Experiment" in czimd_box.ImageDocument.Metadata:
        czimd_box.has_experiment = True

    if "HardwareSetting" in czimd_box.ImageDocument.Metadata:
        czimd_box.has_hardware = True

    if "CustomAttributes" in czimd_box.ImageDocument.Metadata:
        czimd_box.has_customattr = True

    if "Information" in czimd_box.ImageDocument.Metadata:
        czimd_box.has_info = True

        if "Application" in czimd_box.ImageDocument.Metadata.Information:
            czimd_box.has_app = True

        if "Document" in czimd_box.ImageDocument.Metadata.Information:
            czimd_box.has_doc = True

        if "Image" in czimd_box.ImageDocument.Metadata.Information:
            czimd_box.has_image = True

            if "Dimensions" in czimd_box.ImageDocument.Metadata.Information.Image:
                czimd_box.has_dims = True

                if (
                    "Channels"
                    in czimd_box.ImageDocument.Metadata.Information.Image.Dimensions
                ):
                    czimd_box.has_channels = True

                #if "S" in czimd_box.ImageDocument.Metadata.Information.Image.Dimensions:
                if "S" in czimd_box.ImageDocument.Metadata.Information.Image:
                    czimd_box.has_scenes = True

        if "Instrument" in czimd_box.ImageDocument.Metadata.Information:
            czimd_box.has_instrument = True

            if "Detectors" in czimd_box.ImageDocument.Metadata.Information.Instrument:
                czimd_box.has_detectors = True

            if "Microscopes" in czimd_box.ImageDocument.Metadata.Information.Instrument:
                czimd_box.has_microscopes = True

            if "Objectives" in czimd_box.ImageDocument.Metadata.Information.Instrument:
                czimd_box.has_objectives = True

            if "TubeLenses" in czimd_box.ImageDocument.Metadata.Information.Instrument:
                czimd_box.has_tubelenses = True

    if "Scaling" in czimd_box.ImageDocument.Metadata:
        czimd_box.has_scale = True

    if "DisplaySetting" in czimd_box.ImageDocument.Metadata:
        czimd_box.has_disp = True

    if "Layers" in czimd_box.ImageDocument.Metadata:
        czimd_box.has_layers = True

    return czimd_box


def create_md_dict_nested(
    metadata: CziMetadata, sort: bool = True, remove_none: bool = True
) -> Dict:
    """
    Create nested dictionary from metadata

    Args:
        metadata (CziMetadata): CzIMetaData object_
        sort (bool, optional): Sort the dictionary_. Defaults to True.
        remove_none (bool, optional): Remove None values from dictionary. Defaults to True.

    Returns:
        Dict: Nested dictionary with reduced set of metadata
    """

    md_array6d = {"array6d": metadata.array6d_size}

    md_box_image = Box(asdict(metadata.image))
    del md_box_image.czisource

    md_box_scale = Box(asdict(metadata.scale))
    del md_box_scale.czisource

    md_box_sample = Box(asdict(metadata.sample))
    del md_box_sample.czisource

    md_box_objective = Box(asdict(metadata.objective))
    del md_box_objective.czisource

    md_box_channels = Box(asdict(metadata.channelinfo))
    del md_box_channels.czisource

    md_box_bbox = Box(asdict(metadata.bbox))
    del md_box_bbox.czisource

    md_box_info = Box(
        {
            "Directory": metadata.dirname,
            "Filename": metadata.filename,
            "AcqDate": metadata.acquisition_date,
            "CreationDate": metadata.creation_date,
            "UserName": metadata.user_name,
            "SW-App": metadata.software_version,
            "SW-Version": metadata.software_name,
        }
    )

    md_box_image_add = Box(
        {
            "isRGB": metadata.isRGB,
            "has_scenes": metadata.has_scenes,
            "ismosaic": metadata.ismosaic,
        }
    )

    md_box_image += md_box_image_add

    IDs = ["array6d", "image", "scale", "sample", "objectives", "channels", "bbox", "info"]

    mds = [
        md_array6d,
        md_box_image.to_dict(),
        md_box_scale.to_dict(),
        md_box_sample.to_dict(),
        md_box_objective.to_dict(),
        md_box_channels.to_dict(),
        md_box_bbox.to_dict(),
        md_box_info.to_dict(),
    ]

    md_dict = dict(zip(IDs, mds))

    if remove_none:
        md_dict = misc_tools.remove_none_from_dict(md_dict)

    if sort:
        return misc_tools.sort_dict_by_key(md_dict)
    if not sort:
        return md_dict

    return md_dict
