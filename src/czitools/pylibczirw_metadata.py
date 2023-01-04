# -*- coding: utf-8 -*-

#################################################################
# File        : pylibczirw_metadata.py
# Author      : sebi06
#
# Disclaimer: The code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Type, Any, Union
import os
from collections import Counter
import xml.etree.ElementTree as ET
from pylibCZIrw import czi as pyczi
from czitools import misc
import numpy as np
import pydash
from dataclasses import dataclass, field, fields, Field
import logging
from pathlib import Path
from box import Box, BoxList

# configure logging
misc.set_logger(name="czitools-logging", level=logging.DEBUG)
logger = misc.get_logger()


class CziMetadataComplete:
    """Get the complete CZI metadata as an object created based on the
    dictionary created from the XML data.
    """

    def __init__(self, filename: Union[str, os.PathLike[str]]) -> None:
        if not isinstance(filename, str):
            filename = filename.as_posix()

        # get metadata dictionary using pylibCZIrw
        with pyczi.open_czi(filename) as czidoc:
            md_dict = czidoc.metadata

        self.md = DictObj(md_dict)


class DictObj:
    """Create an object based on a dictionary
    """

    # based upon: https://joelmccune.com/python-dictionary-as-object/

    def __init__(self, in_dict: dict) -> None:
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, DictObj(val) if isinstance(val, dict) else val)


class CziMetadata:
    """Create a CziMetadata object from the filename
    """

    def __init__(self, filename: Union[str, os.PathLike[str]], dim2none: bool = False) -> None:

        if not isinstance(filename, str):
            filename = filename.as_posix()

        # get metadata dictionary using pylibCZIrw
        with pyczi.open_czi(filename) as czidoc:
            md_dict = czidoc.metadata

            # get directory, filename, SW version and acquisition data
            self.info = CziInfo(filename)

            # get dimensions
            self.pyczi_dims = czidoc.total_bounding_box

            # get some additional metadata using aicspylibczi
            try:
                from aicspylibczi import CziFile

                # get the general CZI object using aicspylibczi
                aicsczi = CziFile(filename)

                self.aics_dimstring = aicsczi.dims
                self.aics_dims_shape = aicsczi.get_dims_shape()
                self.aics_size = aicsczi.size
                self.aics_ismosaic = aicsczi.is_mosaic()
                self.aics_dim_order, self.aics_dim_index, self.aics_dim_valid = self.get_dimorder(
                    aicsczi.dims)
                self.aics_posC = self.aics_dim_order["C"]

            except ImportError as e:
                logger.warning("Package aicspylibczi not found. Use Fallback values.")
                self.aics_dimstring = None
                self.aics_dims_shape = None
                self.aics_size = None
                self.aics_ismosaic = None
                self.aics_dim_order = None
                self.aics_dim_index = None
                self.aics_dim_valid = None
                self.aics_posC = None

            # get the pixel typed for all channels
            self.pixeltypes = czidoc.pixel_types
            self.isRGB, self.consistent_pixeltypes = self.check_if_rgb(self.pixeltypes)

            # try:
            #     # determine pixel type for CZI array by reading XML metadata
            #     self.pixeltype = md_dict["ImageDocument"]["Metadata"]["Information"]["Image"]["PixelType"]
            # except KeyError as e:
            #     logger.error(
            #         "No Pixeltype entry found inside metadata: ImageDocument-Metadata-Information-Image-PixelType")

            # # check if CZI is a RGB file
            # if self.pixeltype in ["Bgr24", "Bgr48", "Bgr96Float"]:
            #     self.isRGB = True

            # determine pixel type for CZI array from first channel

            self.npdtype = []
            self.maxvalue = []

            for ch, px in self.pixeltypes.items():
                npdtype, maxvalue = self.get_dtype_fromstring(px)
                self.npdtype.append(npdtype)
                self.maxvalue.append(maxvalue)

            # self.npdtype, self.maxvalue = self.get_dtype_fromstring(self.pixeltype)

            # get the dimensions and order
            self.image = CziDimensions(filename)

            # try to guess if the CZI is a mosaic file
            if self.image.SizeM is None or self.image.SizeM == 1:
                self.ismosaic = False
            else:
                self.ismosaic = True

            # check for consistent scene shape
            self.scene_shape_is_consistent = self.check_scenes_shape(
                czidoc, size_s=self.image.SizeS)

            # get the bounding boxes
            self.bbox = CziBoundingBox(filename)

            # get information about channels
            self.channelinfo = CziChannelInfo(filename)

            # get scaling info
            self.scale = CziScaling(filename, dim2none=dim2none)

            # get objective information
            self.objective = CziObjectives(filename)

            # get detector information
            self.detector = CziDetector(filename)

            # get detector information
            self.microscope = CziMicroscope(filename)

            # get information about sample carrier and wells etc.
            self.sample = CziSampleInfo(filename)

            # get additional metainformation
            self.add_metadata = CziAddMetaData(filename)

    # can be also used without creating an instance of the class
    @staticmethod
    def get_dtype_fromstring(pixeltype: str) -> Tuple[Optional[np.dtype], Optional[int]]:

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
        dims = ["R", "I", "M", "H", "V", "B",
                "S", "T", "C", "Z", "Y", "X", "A"]
        dims_dict = {}

        # loop over all dimensions and find the index
        for d in dims:
            dims_dict[d] = dim_string.find(d)
            dimindex_list.append(dim_string.find(d))

        # check if a dimension really exists
        numvalid_dims = sum(i >= 0 for i in dimindex_list)

        return dims_dict, dimindex_list, numvalid_dims

    @staticmethod
    def get_dim_string(dim_order: Dict, num_dims: int = 6) -> str:
        """Create dimension 5d or 6d string based on the dictionary with the dimension order

        Args:
            dim_order (Dict): Dictionary with all dimensions and their indices
            num_dims (int): Number of dimensions contained inside the string

        Returns:
            str: dimension string for a 5d or 6d array, e.g. "TCZYX" or STCZYX"
        """

        dim_string = ""

        for d in range(num_dims):
            # get the key from dim_orders and add to string
            k = [key for key, value in dim_order.items() if value == d][0]
            dim_string = dim_string + k

        return dim_string

    @staticmethod
    def check_scenes_shape(czidoc: pyczi.CziReader,
                           size_s: Union[int, None]) -> bool:
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
    filepath: Union[str, os.PathLike[str]]
    SizeX: Optional[int] = field(init=False, default=None)
    SizeY: Optional[int] = field(init=False, default=None)
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
    SizeX_sf: Optional[int] = field(init=False, default=None)
    SizeY_sf: Optional[int] = field(init=False, default=None)

    # Information official CZI Dimension Characters:
    # "X":"Width"        :
    # "Y":"Height"       :
    # "C":"Channel"      : number of channels
    # "Z":"Slice"        : number of z-planes
    # "T":"Time"         : number of time points
    # "R":"Rotation"     :
    # "S":"Scene"        : contiguous regions of interest in a mosaic image
    # "I":"Illumination" : SPIM direction for LightSheet
    # "B":"Block"        : acquisition
    # "M":"Mosaic"       : index of tile for compositing a scene
    # "H":"Phase"        : e.g. Airy detector fibers
    # "V":"View"         : e.g. for SPIM

    def __post_init__(self):

        self.set_dimensions()

    def set_dimensions(self):
        """Populate the image dimensions with the detected values from the metadata

        Returns:
            The image dimensions dataclass
        """

        # get the Box and extract the relevant metadata
        czi_box = get_czimd_box(self.filepath)
        dimensions = czi_box.ImageDocument.Metadata.Information.Image

        # define the image dimensions to check for
        dims = ["SizeX", "SizeY", "SizeS", "SizeT", "SizeZ", "SizeC", "SizeM", "SizeR", "SizeH", "SizeI", "SizeV", "SizeB"]

        cls_fields: Tuple[Field, ...] = fields(self.__class__)
        for fd in cls_fields:
            if fd.name in dims:
                if dimensions[fd.name] is not None:
                    setattr(self, fd.name, int(dimensions[fd.name]))

        # get the field names and their type hints
        # for field_name, field_type in get_type_hints(self).items():
        #    if field_name in dims:
        #        if dimensions[field_name] is not None:
        #            setattr(self, field_name, int(dimensions[field_name]))


@dataclass
class CziBoundingBox:
    filename: Union[str, os.PathLike[str]]
    scenes_bounding_rect: Optional[Dict[int, pyczi.Rectangle]] = field(init=False)
    total_rect: Optional[pyczi.Rectangle] = field(init=False)
    total_bounding_box: Optional[Dict[str, tuple]] = field(init=False)

    def __post_init__(self):

        if not isinstance(self.filename, str):
            self.filename = self.filename.as_posix()

        with pyczi.open_czi(self.filename) as czidoc:

            try:
                self.scenes_bounding_rect = czidoc.scenes_bounding_rectangle
            except Exception as e:
                self.scenes_bounding_rect = None
                logger.warning("Scenes Bounding rectangle not found.")

            try:
                self.total_rect = czidoc.total_bounding_rectangle
            except Exception as e:
                self.total_rect = None
                logger.warning("Total Bounding rectangle not found.")

            try:
                self.total_bounding_box = czidoc.total_bounding_box
            except Exception as e:
                self.total_bounding_box = None
                logger.warning("Total Bounding Box not found.")


@dataclass
class CziChannelInfo:
    filepath: Union[str, os.PathLike[str]]
    names: List[str] = field(init=False, default_factory=lambda: [])
    dyes: List[str] = field(init=False, default_factory=lambda: [])
    colors: List[str] = field(init=False, default_factory=lambda: [])
    clims: List[List[float]] = field(init=False, default_factory=lambda: [])
    gamma: List[float] = field(init=False, default_factory=lambda: [])

    def __post_init__(self):

        czi_box = get_czimd_box(self.filepath)

        # get channels part of dict
        if czi_box.ImageDocument.Metadata.Information.Image.Dimensions is not None:
        #if czi_box.ImageDocument.Metadata.Information.Image.Dimensions.Channels.Channel is not None:
            try:
                channels = czi_box.ImageDocument.Metadata.Information.Image.Dimensions.Channels.Channel
            except AttributeError:
                channels = None
            
            try:
                disp = czi_box.ImageDocument.Metadata.DisplaySetting.Channels.Channel
            except AttributeError:
                disp = None

            
            if isinstance(channels, Box):

                self.names.append('CH1') if channels.Name is None else self.names.append(channels.Name)
                
                if disp is not None:
                    self.dyes.append('Dye-CH1') if disp.ShortName is None else self.dyes.append(disp.ShortName)
                    self.colors.append('#80808000') if disp.Color is None else self.colors.append(disp.Color)

                    low = 0.0 if disp.Low is None else float(disp.Low)
                    high = 0.5 if disp.High is None else float(disp.High)

                    self.clims.append([low, high])
                    self.gamma.append(0.85) if disp.Gamma is None else self.gamma.append(float(disp.Gamma))
                else:
                    self.dyes.append('Dye-CH1')
                    self.colors.append('#80808000')
                    self.clims.append([0.0, 0.5])
                    self.gamma.append(0.85)


            elif isinstance(channels, BoxList):

                for ch in range(len(channels)):
                    self.names.append('CH' + str(ch + 1)) if channels[ch].Name is None else self.names.append(channels[ch].Name)

                    if disp is not None:

                        self.dyes.append('Dye-CH' + str(ch + 1)) if disp[ch].ShortName is None else self.dyes.append(disp[ch].ShortName)
                        self.colors.append('#80808000') if disp[ch].Color is None else self.colors.append(disp[ch].Color)

                        low = 0.0 if disp[ch].Low is None else float(disp[ch].Low)
                        high = 0.5 if disp[ch].High is None else float(disp[ch].High)

                        self.clims.append([low, high])
                        self.gamma.append(0.85) if disp[ch].Gamma is None else self.gamma.append(float(disp[ch].Gamma))
                    else:
                        self.dyes.append('Dye-CH' + str(ch + 1))
                        self.colors.append('#80808000')
                        self.clims.append([0.0, 0.5])
                        self.gamma.append(0.85)

        #elif czi_box.ImageDocument.Metadata.Information.Image.Dimensions.Channels.Channel is None:
        elif czi_box.ImageDocument.Metadata.Information.Image.Dimensions is None:
            logger.warning("Channel information not found.")
            pass


class CziScaling:
    def __init__(self, filename: Union[str, os.PathLike[str]], dim2none: bool = False) -> None:

        if not isinstance(filename, str):
            filename = filename.as_posix()

        # get metadata dictionary using pylibCZIrw
        self.scalefactorXY = None
        self.ratio_sf = None
        self.Y_sf = None
        self.X_sf = None

        with pyczi.open_czi(filename) as czidoc:
            md_dict = czidoc.metadata

        def _safe_get_scale(distances_: List[Dict[Any, Any]], idx: int) -> Optional[float]:
            try:
                return float(distances_[idx]["Value"]) * 1000000 if distances_[idx]["Value"] is not None else None
            except IndexError:
                logger.warning("No Z-Scaling found. Using defaults = 1.0.")
                return 1.0

        try:
            distances = md_dict["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"]

            # get the scaling in [micron] - inside CZI the default is [m]
            self.X = _safe_get_scale(distances, 0)
            self.Y = _safe_get_scale(distances, 1)
            self.Z = _safe_get_scale(distances, 2)

        except KeyError:
            if dim2none:
                self.X = None
                self.Y = None
                self.Z = None
            if not dim2none:
                self.X = 1.0
                self.Y = 1.0
                self.Z = 1.0

        # safety check in case a scale = 0
        if self.X == 0.0:
            self.X = 1.0
            logger.warning("Detected ScalingX = 0. Use 1.0 as fallback.")
        if self.Y == 0.0:
            self.Y = 1.0
            logger.warning("Detected ScalingY = 0. Use 1.0 as fallback.")
        if self.Z == 0.0:
            self.Z = 1.0
            logger.warning("Detected ScalingZ = 0. Use 1.0 as fallback.")

        # set the scaling unit to [micron]
        self.Unit = "micron"

        # get scaling ratio
        self.ratio = self.get_scale_ratio(scalex=self.X,
                                          scaley=self.Y,
                                          scalez=self.Z)

    @staticmethod
    def get_scale_ratio(scalex: float = 1.0,
                        scaley: float = 1.0,
                        scalez: float = 1.0) -> Dict:

        if scalex is None or scaley is None or scalez is None:
            scale_ratio = {"xy": None, "zx": None}
        else:
            # set default scale factor to 1.0
            scale_ratio = {"xy": np.round(scalex / scaley, 3), "zx": np.round(scalez / scalex, 3)}

        # get the factor between XY scaling
        # get the scale factor between XZ scaling

        return scale_ratio


@dataclass
class CziInfo:
    filepath: Union[str, os.PathLike[str]]
    dirname: str = field(init=False)
    filename: str = field(init=False)
    software_name: Optional[str] = field(init=False, default=None)
    softname_version: Optional[str] = field(init=False, default=None)
    acquisition_date: Optional[str] = field(init=False, default=None)

    def __post_init__(self):
        czimd_box = get_czimd_box(self.filepath)

        # get directory and filename etc.
        self.dirname = Path(self.filepath).parent
        self.filename = Path(self.filepath).name

        # get acquisition data and SW version
        if czimd_box.ImageDocument.Metadata.Information.Application is not None:
            self.software_name = czimd_box.ImageDocument.Metadata.Information.Application.Name
            self.software_version = czimd_box.ImageDocument.Metadata.Information.Application.Version
        
        if czimd_box.ImageDocument.Metadata.Information.Image is not None:
            self.acquisition_date = czimd_box.ImageDocument.Metadata.Information.Image.AcquisitionDateAndTime


@dataclass
class CziObjectives:
    filepath: Union[str, os.PathLike[str]]
    NA: Optional[float] = field(init=False, default=None)
    objmag: Optional[float] = field(init=False, default=None)
    ID: Optional[str] = field(init=False, default=None)
    name: Optional[str] = field(init=False, default=None)
    model: Optional[str] = field(init=False, default=None)
    immersion: Optional[str] = field(init=False, default=None)
    tubelensmag: Optional[float] = field(init=False, default=None)
    totalmg: Optional[float] = field(init=False, default=None) 
    
    
    def __post_init__(self):

        czi_box = get_czimd_box(self.filepath)

        # check if objective metadata actually exist
        if czi_box.has_objectives:
            
            # get objective data
            objective =  czi_box.ImageDocument.Metadata.Information.Instrument.Objectives.Objective

            self.name = objective.Name
            self.immersion = objective.Immersion
            self.NA = float(objective.LensNA)
            self.ID = objective.Id
            self.objmag = float(objective.NominalMagnification)

            if self.name is None:
                self.name =  objective.Manufacturer.Model
        
        elif not czi_box.has_objectives:
            logger.warning("No Objective Information found.")

        # chek if tublens metadata exist
        if czi_box.has_tubelenses:
            
            # get tubelenes data
            tubelens = czi_box.ImageDocument.Metadata.Information.Instrument.TubeLenses.TubeLens
            
            self.tubelensmag = float(tubelens.Magnification)

        elif not czi_box.has_tubelens:
            logger.warning("No Tublens Information found.")

        # some additional checks to clac the total magnification
        if self.objmag is not None and self.tubelensmag is not None:
            self.totalmg = self.objmag * self.tubelensmag

        if self.objmag is not None and self.tubelensmag is None:
            self.totalmg = self.objmag


@dataclass
class CziDetector:
    filepath: Union[str, os.PathLike[str]]
    model: List[str] = field(init=False, default_factory=lambda: [])
    name: List[str] = field(init=False, default_factory=lambda: [])
    ID: List[str] = field(init=False, default_factory=lambda: [])
    modeltype: List[str] = field(init=False, default_factory=lambda: [])

    def __post_init__(self):

        czi_box = get_czimd_box(self.filepath)

        # check if there are any detector entries inside the dictionary
        if czi_box.ImageDocument.Metadata.Information.Instrument is not None:

            # get the data for the detectors
            detectors = czi_box.ImageDocument.Metadata.Information.Instrument.Detectors.Detector

            # check for detector ID, Name, Model and Type
            if isinstance(detectors, Box):
                self.ID.append(detectors.Id)
                self.name.append(detectors.Name)
                self.model.append(detectors.Model)
                self.modeltype.append(detectors.Type)

            # and do this differently in case of a list of detectors
            elif isinstance(detectors, BoxList):

                for d in range(len(detectors)):
                    self.ID.append(detectors[d].Id)
                    self.name.append(detectors[d].Name)
                    self.model.append(detectors[d].Model)
                    self.modeltype.append(detectors[d].Type)

        elif czi_box.ImageDocument.Metadata.Information.Instrument is None:

            logger.warning("No Detetctor(s) information found.")
            self.model = [None]
            self.name = [None]
            self.ID = [None]
            self.modeltype = [None]


@dataclass
class CziMicroscope:
    filepath: Union[str, os.PathLike[str]]
    ID: Optional[str] = field(init=False)
    Name: Optional[str] = field(init=False)

    def __post_init__(self):

        czi_box = get_czimd_box(self.filepath)

        if czi_box.ImageDocument.Metadata.Information.Instrument is None:
            self.ID = None
            self.Name = None
            logger.warning("No Microscope information found.")
        else:
            self.ID = czi_box.ImageDocument.Metadata.Information.Instrument.Microscopes.Microscope.Id
            self.Name = czi_box.ImageDocument.Metadata.Information.Instrument.Microscopes.Microscope.Name


class CziSampleInfo:
    def __init__(self, filename: Union[str, os.PathLike[str]]) -> None:

        if not isinstance(filename, str):
            filename = filename.as_posix()

        # get metadata dictionary using pylibCZIrw
        with pyczi.open_czi(filename) as czidoc:
            md_dict = czidoc.metadata

        size_s = CziDimensions(filename).SizeS
        #size_s = dim_dict["SizeS"]

        # check for well information
        self.well_array_names = []
        self.well_indices = []
        self.well_position_names = []
        self.well_colID = []
        self.well_rowID = []
        self.well_counter = []
        self.scene_stageX = []
        self.scene_stageY = []
        self.image_stageX = None
        self.image_stageY = None

        if size_s is not None:
            # logger.info("Trying to extract Scene and Well information if existing ...")

            # extract well information from the dictionary
            allscenes: Union[Dict, List]
            well: Union[Dict, List]

            # get the information from the dictionary (based on the XML)
            try:
                allscenes = md_dict["ImageDocument"]["Metadata"]["Information"]["Image"]["Dimensions"]["S"]["Scenes"][
                    "Scene"]

                # loop over all detected scenes
                for s in range(size_s):

                    if size_s == 1:
                        well = allscenes
                        try:
                            self.well_array_names.append(allscenes["ArrayName"])
                        except (KeyError, TypeError) as e:
                            try:
                                self.well_array_names.append(well["@Name"])
                            except (KeyError, TypeError) as e:
                                try:
                                    self.well_array_names.append(well["Name"])
                                except (KeyError, TypeError) as e:
                                    logger.warning("Well Name not found. Using A1 instead.")
                                    self.well_array_names.append("A1")

                        try:
                            self.well_indices.append(allscenes["@Index"])
                        except (KeyError, TypeError) as e:
                            try:
                                self.well_indices.append(allscenes["Index"])
                            except (KeyError, TypeError) as e:
                                logger.warning("Well Index not found.")
                                self.well_indices.append(1)

                        try:
                            self.well_position_names.append(allscenes["@Name"])
                        except (KeyError, TypeError) as e:
                            try:
                                self.well_position_names.append(allscenes["Name"])
                            except (KeyError, TypeError) as e:
                                logger.warning("Well Position Names not found.")
                                self.well_position_names.append("P1")

                        try:
                            self.well_colID.append(
                                int(allscenes["Shape"]["ColumnIndex"]))
                        except (KeyError, TypeError) as e:
                            logger.warning("Well ColumnIDs not found.")
                            self.well_colID.append(0)

                        try:
                            self.well_rowID.append(
                                int(allscenes["Shape"]["RowIndex"]))
                        except (KeyError, TypeError) as e:
                            logger.warning("Well RowIDs not found.")
                            self.well_rowID.append(0)

                        try:
                            # count the content of the list, e.g. how many times a certain well was detected
                            self.well_counter = Counter(self.well_array_names)
                        except (KeyError, TypeError):
                            self.well_counter.append(Counter({"A1": 1}))

                        try:
                            # get the SceneCenter Position
                            sx = allscenes["CenterPosition"].split(",")[0]
                            sy = allscenes["CenterPosition"].split(",")[1]
                            self.scene_stageX.append(np.double(sx))
                            self.scene_stageY.append(np.double(sy))
                        except (TypeError, KeyError) as e:
                            logger.warning("Stage Positions XY not found.")
                            self.scene_stageX.append(0.0)
                            self.scene_stageY.append(0.0)

                    if size_s > 1:
                        well = allscenes[s]
                        try:
                            self.well_array_names.append(well["ArrayName"])
                        except (KeyError, TypeError) as e:
                            try:
                                self.well_array_names.append(well["@Name"])
                            except (KeyError, TypeError) as e:
                                try:
                                    self.well_array_names.append(well["Name"])
                                except (KeyError, TypeError) as e:
                                    logger.warning("Well Name not found. Using A1 instead.")
                                    self.well_array_names.append("A1")

                        # get the well information
                        try:
                            self.well_indices.append(well["@Index"])
                        except (KeyError, TypeError) as e:
                            try:
                                self.well_indices.append(well["Index"])
                            except (KeyError, TypeError) as e:
                                logger.warning("Well Index not found.")
                                self.well_indices.append(None)
                        try:
                            self.well_position_names.append(well["@Name"])
                        except (KeyError, TypeError) as e:
                            try:
                                self.well_position_names.append(well["Name"])
                            except (KeyError, TypeError) as e:
                                logger.warning("Well Position Names not found.")
                                self.well_position_names.append(None)

                        try:
                            self.well_colID.append(
                                int(well["Shape"]["ColumnIndex"]))
                        except (KeyError, TypeError) as e:
                            logger.warning("Well ColumnIDs not found.")
                            self.well_colID.append(None)

                        try:
                            self.well_rowID.append(
                                int(well["Shape"]["RowIndex"]))
                        except (KeyError, TypeError) as e:
                            logger.warning("Well RowIDs not found.")
                            self.well_rowID.append(None)

                        # count the content of the list, e.g. how many times a certain well was detected
                        self.well_counter = Counter(self.well_array_names)

                        # try:
                        if isinstance(allscenes, list):
                            try:
                                # get the SceneCenter Position
                                sx = allscenes[s]["CenterPosition"].split(",")[0]
                                sy = allscenes[s]["CenterPosition"].split(",")[1]
                                self.scene_stageX.append(np.double(sx))
                                self.scene_stageY.append(np.double(sy))
                            except (KeyError, TypeError) as e:
                                logger.warning("Stage Positions XY not found.")
                                self.scene_stageX.append(0.0)
                                self.scene_stageY.append(0.0)
                        if not isinstance(allscenes, list):
                            self.scene_stageX.append(0.0)
                            self.scene_stageY.append(0.0)

                    # count the number of different wells
                    self.number_wells = len(self.well_counter.keys())
            except KeyError as e:
                logger.warning("CZI contains no scene metadata.")

        else:
            logger.info(
                "No Scene or Well information found. Try to read XY Stage Coordinates from subblocks.")
            try:
                # read the data from CSV file
                planetable, csvfile = misc.get_planetable(filename,
                                                          read_one_only=True,
                                                          savetable=False)

                self.image_stageX = float(planetable["X[micron]"][0])
                self.image_stageY = float(planetable["Y[micron]"][0])

            except Exception as e:
                logger.error(e)


@dataclass
class CziAddMetaData:
    filepath: Union[str, os.PathLike[str]]
    experiment: Optional[Box] = field(init=False, default=None)
    hardwaresetting: Optional[Box] = field(init=False, default=None)
    customattributes: Optional[Box] = field(init=False, default=None)
    displaysetting: Optional[Box] = field(init=False, default=None)
    layers: Optional[Box] = field(init=False, default=None)
    
    def __post_init__(self):

        czi_box = get_czimd_box(self.filepath)

        if czi_box.has_experiment:
            self.experiment = czi_box.ImageDocument.Metadata.Experiment
        else:
            logger.warning("No Experiment information found.")

        if czi_box.has_hardware:
            self.hardwaresetting = czi_box.ImageDocument.Metadata.HardwareSetting
        else:
            logger.warning("No HardwareSetting information found.")

        if czi_box.has_customattr:
            self.customattributes = czi_box.ImageDocument.Metadata.CustomAttributes
        else:
            logger.warning("No CustomAttributes information found.")

        if czi_box.has_disp:
            self.displaysetting = czi_box.ImageDocument.Metadata.DisplaySetting
        else:
            logger.warning("No DisplaySetting information found.")

        if czi_box.has_layers:
            self.layers = czi_box.ImageDocument.Metadata.Layers
        else:
            logger.warning("No Layers information found.")


@dataclass
class CziScene:
    filepath: Union[str, os.PathLike[str]]
    index: int
    bbox: Optional[pyczi.Rectangle]= field(init=False, default=None)
    xstart: Optional[int] = field(init=False, default=None)
    ystart: Optional[int] = field(init=False, default=None)
    width: Optional[int] = field(init=False, default=None)
    height: Optional[int] = field(init=False, default=None)
    
    def __post_init__(self):
        
        if not isinstance(self.filepath, str):
            self.filepath = self.filepath.as_posix()

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
                logger.warning("No Scenes detected.")


def obj2dict(obj: Any, sort: bool = True) -> Dict[str, Any]:
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

    if sort:
        return misc.sort_dict_by_key(result)
    if not sort:
        return result


def writexml(filename: Union[str, os.PathLike[str]], xmlsuffix: str = '_CZI_MetaData.xml') -> str:
    """Write XML information of CZI to disk

    :param filename: CZI image filename
    :type filename: str
    :param xmlsuffix: suffix for the XML file that will be created, defaults to '_CZI_MetaData.xml'
    :type xmlsuffix: str, optional
    :return: filename of the XML file
    :rtype: str
    """

    if not isinstance(filename, str):
        filename = filename.as_posix()

    # get the raw metadata as XML or dictionary
    with pyczi.open_czi(filename) as czidoc:
        metadata_xmlstr = czidoc.raw_metadata

    # change file name
    xmlfile = filename.replace('.czi', xmlsuffix)

    # get tree from string
    tree = ET.ElementTree(ET.fromstring(metadata_xmlstr))

    # write XML file to same folder
    tree.write(xmlfile, encoding='utf-8', method='xml')

    return xmlfile


def create_mdict_red(metadata: CziMetadata,
                     sort: bool = True) -> Dict:
    """
    Created a metadata dictionary to be displayed in napari

    Args:
        metadata: CziMetadata class
        sort: sort the dictionary

    Returns: dictionary with the metadata

    """

    # create a dictionary with the metadata
    md_dict = {'Directory': metadata.info.dirname,
               'Filename': metadata.info.filename,
               'AcqDate': metadata.info.acquisition_date,
               'SizeX': metadata.image.SizeX,
               'SizeY': metadata.image.SizeY,
               'SizeZ': metadata.image.SizeZ,
               'SizeC': metadata.image.SizeC,
               'SizeT': metadata.image.SizeT,
               'SizeS': metadata.image.SizeS,
               'SizeB': metadata.image.SizeB,
               'SizeM': metadata.image.SizeM,
               'SizeH': metadata.image.SizeH,
               'SizeI': metadata.image.SizeI,
               'isRGB': metadata.isRGB,
               'ismosaic': metadata.ismosaic,
               'ObjNA': metadata.objective.NA,
               'ObjMag': metadata.objective.mag,
               'TubelensMag': metadata.objective.tubelensmag,
               'XScale': metadata.scale.X,
               'YScale': metadata.scale.Y,
               'ZScale': metadata.scale.Z,
               'ChannelsNames': metadata.channelinfo.names,
               'ChannelDyes': metadata.channelinfo.dyes,
               'WellArrayNames': metadata.sample.well_array_names,
               'WellIndicies': metadata.sample.well_indices,
               'WellPositionNames': metadata.sample.well_position_names,
               'WellRowID': metadata.sample.well_rowID,
               'WellColumnID': metadata.sample.well_colID,
               'WellCounter': metadata.sample.well_counter,
               'SceneCenterStageX': metadata.sample.scene_stageX,
               'SceneCenterStageY': metadata.sample.scene_stageX,
               'ImageStageX': metadata.sample.image_stageX,
               'ImageStageY': metadata.sample.image_stageX
               }

    # check for extra entries when reading mosaic file with a scale factor
    if hasattr(metadata.image, "SizeX_sf"):
        md_dict['SizeX sf'] = metadata.image.SizeX_sf
        md_dict['SizeY sf'] = metadata.image.SizeY_sf
        md_dict['XScale sf'] = metadata.scale.X_sf
        md_dict['YScale sf'] = metadata.scale.Y_sf
        md_dict['ratio sf'] = metadata.scale.ratio_sf
        md_dict['scalefactorXY'] = metadata.scale.scalefactorXY

    if sort:
        return misc.sort_dict_by_key(md_dict)
    if not sort:
        return md_dict


def get_czimd_box(filename: Union[str, os.PathLike[str]]) -> Box:
    if not isinstance(filename, str):
        filename = filename.as_posix()

    # get metadata dictionary using pylibCZIrw
    with pyczi.open_czi(filename) as czi_document:
        metadata_dict = czi_document.metadata

    czimd_box = Box(metadata_dict,
                    conversion_box=True,
                    default_box=True,
                    default_box_attr=None,
                    default_box_create_on_get=True,
                    #default_box_no_key_error=True
                    )

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

    if 'Experiment' in czimd_box.ImageDocument.Metadata:
        czimd_box.has_exp = True

    if 'HardwareSetting' in czimd_box.ImageDocument.Metadata:
        czimd_box.has_hardware = True

    if 'CustomAttributes' in czimd_box.ImageDocument.Metadata:
        czimd_box.has_customattr = True

    if 'Information' in czimd_box.ImageDocument.Metadata:
        czimd_box.has_info = True

        if 'Application' in czimd_box.ImageDocument.Metadata.Information:
            czimd_box.has_app = True
        
        if 'Document' in czimd_box.ImageDocument.Metadata.Information:
            czimd_box.has_doc = True
        
        if 'Image' in czimd_box.ImageDocument.Metadata.Information:
            czimd_box.has_image = True

            if 'Dimensions' in czimd_box.ImageDocument.Metadata.Information.Image:
                czimd_box.has_dims = True
        
                if 'Channels' in czimd_box.ImageDocument.Metadata.Information.Image.Dimensions:
                    czimd_box.has_channels = True

                if 'S' in czimd_box.ImageDocument.Metadata.Information.Image.Dimensions:
                    czimd_box.has_scenes = True


        if 'Instrument' in czimd_box.ImageDocument.Metadata.Information:
            czimd_box.has_instrument = True

            if 'Detectors' in czimd_box.ImageDocument.Metadata.Information.Instrument:
                czimd_box.has_detectors = True

            if 'Microscopes' in czimd_box.ImageDocument.Metadata.Information.Instrument:
                czimd_box.has_microscopes = True

            if 'Objectives' in czimd_box.ImageDocument.Metadata.Information.Instrument:
                czimd_box.has_objectives = True

            if 'TubeLenses' in czimd_box.ImageDocument.Metadata.Information.Instrument:
                czimd_box.has_tubelenses = True

    if 'Scaling' in czimd_box.ImageDocument.Metadata:
        czimd_box.has_scale = True

    if 'DisplaySetting' in czimd_box.ImageDocument.Metadata:
        czimd_box.has_disp = True

    if 'Layers' in czimd_box.ImageDocument.Metadata:
        czimd_box.has_layers = True

    return czimd_box
