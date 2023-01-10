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
from dataclasses import dataclass, field, fields, Field
from pathlib import Path
from box import Box, BoxList


@dataclass
class test:
    filepath: Union[str, os.PathLike[str]]
    info: CziInfo
    pyczi_dims: Dict[str, tuple]
    aics_dimstring: str
    aics_dims_shape: List[Dict[str, tuple]]
    aics_size: Tuple[int]
    aics_ismosaic: bool
    aics_dim_order: Dict[str, int]
    aics_dim_index: List[int]
    aics_dim_valid: int
    aics_posC: int
    pixeltypes: Dict[int, str]
    isRGB: bool
    npdtype: List[Any]
    maxvalue: List[int]
    image: CziDimensions
    bbox: CziBoundingBox
    channelinfo: CziChannelInfo
    scale: CziScaling
    objective: CziObjectives
    detector: CziDetector
    microscope: CziMicroscope
    sample: CziSampleInfo
    add_metadata: CziAddMetaData

    def __post_init__(self):
        pass


class CziMetadata:
    """
    Create a CziMetadata object from the filename
    """

    def __init__(self, filepath: Union[str, os.PathLike[str]]) -> None:

        if isinstance(filepath, Path):
            # convert to string
            filepath = str(filepath)

        # testing
        czi_box = get_czimd_box(filepath)

        # get metadata dictionary using pylibCZIrw
        with pyczi.open_czi(filepath) as czidoc:
            md_dict = czidoc.metadata

            # get directory, filename, SW version and acquisition data
            self.info = CziInfo(filepath)

            # get dimensions
            self.pyczi_dims = czidoc.total_bounding_box

            # get some additional metadata using aicspylibczi
            try:
                from aicspylibczi import CziFile

                # get the general CZI object using aicspylibczi
                aicsczi = CziFile(filepath)

                self.aics_dimstring = aicsczi.dims
                self.aics_dims_shape = aicsczi.get_dims_shape()
                self.aics_size = aicsczi.size
                self.aics_ismosaic = aicsczi.is_mosaic()
                self.aics_dim_order, self.aics_dim_index, self.aics_dim_valid = self.get_dimorder(
                    aicsczi.dims)
                self.aics_posC = self.aics_dim_order["C"]

            except ImportError as e:
                print("Package aicspylibczi not found. Use Fallback values.")
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

            # determine pixel type for CZI array from first channel
            self.npdtype = []
            self.maxvalue = []

            for ch, px in self.pixeltypes.items():
                npdtype, maxvalue = self.get_dtype_fromstring(px)
                self.npdtype.append(npdtype)
                self.maxvalue.append(maxvalue)

            # self.npdtype, self.maxvalue = self.get_dtype_fromstring(self.pixeltype)

            # get the dimensions and order
            self.image = CziDimensions(czi_box)

            # try to guess if the CZI is a mosaic file
            if self.image.SizeM is None or self.image.SizeM == 1:
                self.ismosaic = False
            else:
                self.ismosaic = True

            # check for consistent scene shape
            self.scene_shape_is_consistent = self.check_scenes_shape(
                czidoc, size_s=self.image.SizeS)

            # get the bounding boxes
            self.bbox = CziBoundingBox(filepath)

            # get information about channels
            self.channelinfo = CziChannelInfo(czi_box)

            # get scaling info
            self.scale = CziScaling(czi_box)

            # get objective information
            self.objective = CziObjectives(czi_box)

            # get detector information
            self.detector = CziDetector(czi_box)

            # get detector information
            self.microscope = CziMicroscope(czi_box)

            # get information about sample carrier and wells etc.
            self.sample = CziSampleInfo(filepath)

            # get additional metainformation
            self.add_metadata = CziAddMetaData(czi_box)

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

    # @staticmethod
    # def get_dim_string(dim_order: Dict, num_dims: int = 6) -> str:
    #     """Create dimension 5d or 6d string based on the dictionary with the dimension order

    #     Args:
    #         dim_order (Dict): Dictionary with all dimensions and their indices
    #         num_dims (int): Number of dimensions contained inside the string

    #     Returns:
    #         str: dimension string for a 5d or 6d array, e.g. "TCZYX" or STCZYX"
    #     """

    #     dim_string = ""

    #     for d in range(num_dims):
    #         # get the key from dim_orders and add to string
    #         k = [key for key, value in dim_order.items() if value == d][0]
    #         dim_string = dim_string + k

    #     return dim_string

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
    """Dataclass containing the image dimensions.
    """

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
        """set_dimensions: Populate the image dimensions with the detected values from the metadata
        """

        # get the Box and extract the relevant dimension metadata
        if isinstance(self.czisource, Box):
            czi_box = self.czisource
        else:
            czi_box = get_czimd_box(self.czisource)
        dimensions = czi_box.ImageDocument.Metadata.Information.Image

        # define the image dimensions to check for
        dims = ["SizeX", "SizeY", "SizeS", "SizeT", "SizeZ", "SizeC",
                "SizeM", "SizeR", "SizeH", "SizeI", "SizeV", "SizeB"]

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
    filepath: Union[str, os.PathLike[str]]
    scenes_bounding_rect: Optional[Dict[int, pyczi.Rectangle]] = field(init=False)
    total_rect: Optional[pyczi.Rectangle] = field(init=False)
    total_bounding_box: Optional[Dict[str, tuple]] = field(init=False)

    def __post_init__(self):

        if isinstance(self.filepath, Path):
            # convert to string
            self.filepath = str(self.filepath)

        with pyczi.open_czi(self.filepath) as czidoc:

            try:
                self.scenes_bounding_rect = czidoc.scenes_bounding_rectangle
            except Exception as e:
                self.scenes_bounding_rect = None
                print("Scenes Bounding rectangle not found.")

            try:
                self.total_rect = czidoc.total_bounding_rectangle
            except Exception as e:
                self.total_rect = None
                print("Total Bounding rectangle not found.")

            try:
                self.total_bounding_box = czidoc.total_bounding_box
            except Exception as e:
                self.total_bounding_box = None
                print("Total Bounding Box not found.")


@dataclass
class CziChannelInfo:
    czisource: Union[str, os.PathLike[str], Box]
    names: List[str] = field(init=False, default_factory=lambda: [])
    dyes: List[str] = field(init=False, default_factory=lambda: [])
    colors: List[str] = field(init=False, default_factory=lambda: [])
    clims: List[List[float]] = field(init=False, default_factory=lambda: [])
    gamma: List[float] = field(init=False, default_factory=lambda: [])

    def __post_init__(self):

        if isinstance(self.czisource, Box):
            czi_box = self.czisource
        else:
            czi_box = get_czimd_box(self.czisource)

        # get channels part of dict
        if czi_box.has_channels:

            try:
                # extract the relevant dimension metadata
                channels = czi_box.ImageDocument.Metadata.Information.Image.Dimensions.Channels.Channel
                if isinstance(channels, Box):
                    # get the data in case of only one channel
                    self.names.append(
                        'CH1') if channels.Name is None else self.names.append(channels.Name)
                elif isinstance(channels, BoxList):
                    # get the data in case multiple channels
                    for ch in range(len(channels)):
                        self.names.append('CH1') if channels[ch].Name is None else self.names.append(
                            channels[ch].Name)
            except AttributeError:
                channels = None
        elif not czi_box.has_channels:
            print("Channel(s) information not found.")

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
            print("DisplaySetting(s) not found.")

    def get_channel_info(self, display: Box):

        if display is not None:
            self.dyes.append(
                'Dye-CH1') if display.ShortName is None else self.dyes.append(display.ShortName)
            self.colors.append(
                '#80808000') if display.Color is None else self.colors.append(display.Color)

            low = 0.0 if display.Low is None else float(display.Low)
            high = 0.5 if display.High is None else float(display.High)

            self.clims.append([low, high])
            self.gamma.append(0.85) if display.Gamma is None else self.gamma.append(
                float(display.Gamma))
        else:
            self.dyes.append('Dye-CH1')
            self.colors.append('#80808000')
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
    ratio_sf: Optional[Dict[float, float]] = field(init=False, default=None)
    scalefactorXY: Optional[float] = field(init=False, default=None)
    unit: Optional[str] = field(init=True, default='micron')

    def __post_init__(self):

        if isinstance(self.czisource, Box):
            czi_box = self.czisource
        else:
            czi_box = get_czimd_box(self.czisource)

        if czi_box.has_scale:

            distances = czi_box.ImageDocument.Metadata.Scaling.Items.Distance

            # get the scaling values for X,Y and Z
            self.X = self.safe_get_scale(distances, 0)
            self.Y = self.safe_get_scale(distances, 1)
            self.Z = self.safe_get_scale(distances, 2)

            # calc the scaling ratio
            self.ratio = {"xy": np.round(self.X / self.Y, 3),
                          "zx": np.round(self.Z / self.X, 3)
                          }

        elif not czi_box.has_scale:
            print("No scaling information found.")

    @staticmethod
    def safe_get_scale(dist: BoxList, idx: int) -> Optional[float]:

        scales = ['X', 'Y', 'Z']

        try:
            # get the scaling value in [micron]
            sc = float(dist[idx].Value) * 1000000

            # check for the value = 0.0
            if sc == 0.0:
                sc = 1.0
                print("Detected Scaling = 0.0 for " +
                      scales[idx] + " Using default = 1.0 [micron].")
            return sc

        except (IndexError, TypeError, AttributeError):

            print("No " + scales[idx] + "-Scaling found. Using default = 1.0 [micron].")
            return 1.0


@dataclass
class CziInfo:
    filepath: Union[str, os.PathLike[str]]
    dirname: str = field(init=False)
    filename: str = field(init=False)
    software_name: Optional[str] = field(init=False, default=None)
    softname_version: Optional[str] = field(init=False, default=None)
    acquisition_date: Optional[str] = field(init=False, default=None)

    # TODO Check if Box can be used

    def __post_init__(self):
        czimd_box = get_czimd_box(self.filepath)

        # get directory and filename etc.
        if isinstance(self.filepath, str):
            self.dirname = str(Path(self.filepath).parent)
            self.filename = str(Path(self.filepath).name)
        if isinstance(self.filepath, Path):
            self.dirname = str(self.filepath.parent)
            self.filename = str(self.filepath.name)

        # get acquisition data and SW version
        if czimd_box.ImageDocument.Metadata.Information.Application is not None:
            self.software_name = czimd_box.ImageDocument.Metadata.Information.Application.Name
            self.software_version = czimd_box.ImageDocument.Metadata.Information.Application.Version

        if czimd_box.ImageDocument.Metadata.Information.Image is not None:
            self.acquisition_date = czimd_box.ImageDocument.Metadata.Information.Image.AcquisitionDateAndTime


@dataclass
class CziObjectives:
    czisource: Union[str, os.PathLike[str], Box]
    NA: Optional[float] = field(init=False, default=None)
    objmag: Optional[float] = field(init=False, default=None)
    ID: Optional[str] = field(init=False, default=None)
    name: Optional[str] = field(init=False, default=None)
    model: Optional[str] = field(init=False, default=None)
    immersion: Optional[str] = field(init=False, default=None)
    tubelensmag: Optional[float] = field(init=False, default=None)
    totalmag: Optional[float] = field(init=False, default=None)

    def __post_init__(self):

        if isinstance(self.czisource, Box):
            czi_box = self.czisource
        else:
            czi_box = get_czimd_box(self.czisource)

        # check if objective metadata actually exist
        if czi_box.has_objectives:

            # get objective data
            objective = czi_box.ImageDocument.Metadata.Information.Instrument.Objectives.Objective

            self.name = objective.Name
            self.immersion = objective.Immersion
            self.NA = float(objective.LensNA)
            self.ID = objective.Id
            self.objmag = float(objective.NominalMagnification)

            if self.name is None:
                self.name = objective.Manufacturer.Model

        elif not czi_box.has_objectives:
            print("No Objective Information found.")

        # check if tubelens metadata exist
        if czi_box.has_tubelenses:

            # get tubelenes data
            tubelens = czi_box.ImageDocument.Metadata.Information.Instrument.TubeLenses.TubeLens

            self.tubelensmag = float(tubelens.Magnification)

        elif not czi_box.has_tubelens:
            print("No Tublens Information found.")

        # some additional checks to clac the total magnification
        if self.objmag is not None and self.tubelensmag is not None:
            self.totalmag = self.objmag * self.tubelensmag

        if self.objmag is not None and self.tubelensmag is None:
            self.totalmag = self.objmag


@dataclass
class CziDetector:
    czisource: Union[str, os.PathLike[str], Box]
    model: List[str] = field(init=False, default_factory=lambda: [])
    name: List[str] = field(init=False, default_factory=lambda: [])
    ID: List[str] = field(init=False, default_factory=lambda: [])
    modeltype: List[str] = field(init=False, default_factory=lambda: [])

    def __post_init__(self):

        if isinstance(self.czisource, Box):
            czi_box = self.czisource
        else:
            czi_box = get_czimd_box(self.czisource)

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

            print("No Detetctor(s) information found.")
            self.model = [None]
            self.name = [None]
            self.ID = [None]
            self.modeltype = [None]


@dataclass
class CziMicroscope:
    czisource: Union[str, os.PathLike[str], Box]
    ID: Optional[str] = field(init=False)
    Name: Optional[str] = field(init=False)

    def __post_init__(self):

        if isinstance(self.czisource, Box):
            czi_box = self.czisource
        else:
            czi_box = get_czimd_box(self.czisource)

        if czi_box.ImageDocument.Metadata.Information.Instrument is None:
            self.ID = None
            self.Name = None
            print("No Microscope information found.")
        else:
            self.ID = czi_box.ImageDocument.Metadata.Information.Instrument.Microscopes.Microscope.Id
            self.Name = czi_box.ImageDocument.Metadata.Information.Instrument.Microscopes.Microscope.Name


@dataclass
class CziSampleInfo:
    filepath: Union[str, os.PathLike[str]]
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

        czi_box = get_czimd_box(self.filepath)
        size_s = CziDimensions(self.filepath).SizeS

        if size_s is not None:

            try:
                allscenes = czi_box.ImageDocument.Metadata.Information.Image.Dimensions.S.Scenes.Scene

                if isinstance(allscenes, Box):
                    self.get_well_info(allscenes)

                if isinstance(allscenes, BoxList):
                    for well in range(len(allscenes)):

                        self.get_well_info(allscenes[well])

            except AttributeError:
                print("CZI contains no scene metadata.")

        elif size_s is None:
            print("No Scene or Well information found. Try to read XY Stage Coordinates from subblocks.")
            try:
                # read the data from CSV file
                planetable, csvfile = misc.get_planetable(self.filepath,
                                                          read_one_only=True,
                                                          savetable=False)

                self.image_stageX = float(planetable["X[micron]"][0])
                self.image_stageY = float(planetable["Y[micron]"][0])

            except Exception as e:
                print(e)

    def get_well_info(self, well: Box):

        # check the ArrayName
        if well.ArrayName is not None:
            self.well_array_names.append(well.ArrayName)
            # count the well instances
            self.well_counter = Counter(self.well_array_names)

        if well.Index is not None:
            self.well_indices.append(int(well.Index))
        elif well.Index is None:
            print("Well Index not found.")
            self.well_indices.append(1)

        if well.Name is not None:
            self.well_position_names.append(well.Name)
        elif well.Name is None:
            print("Well Position Names not found.")
            self.well_position_names.append("P1")

        if well.Shape is not None:
            self.well_colID.append(int(well.Shape.ColumnIndex))
            self.well_rowID.append(int(well.Shape.RowIndex))
        elif well.Shape is None:
            print("Well Column or Row IDs not found.")
            self.well_colID.append(0)
            self.well_rowID.append(0)

        if well.CenterPosition is not None:
            # get the SceneCenter Position
            sx = well.CenterPosition.split(",")[0]
            sy = well.CenterPosition.split(",")[1]
            self.scene_stageX.append(np.double(sx))
            self.scene_stageY.append(np.double(sy))
        if well.CenterPosition is None:
            print("Stage Positions XY not found.")
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

        if isinstance(self.czisource, Box):
            czi_box = self.czisource
        else:
            czi_box = get_czimd_box(self.czisource)

        if czi_box.has_experiment:
            self.experiment = czi_box.ImageDocument.Metadata.Experiment
        else:
            print("No Experiment information found.")

        if czi_box.has_hardware:
            self.hardwaresetting = czi_box.ImageDocument.Metadata.HardwareSetting
        else:
            print("No HardwareSetting information found.")

        if czi_box.has_customattr:
            self.customattributes = czi_box.ImageDocument.Metadata.CustomAttributes
        else:
            print("No CustomAttributes information found.")

        if czi_box.has_disp:
            self.displaysetting = czi_box.ImageDocument.Metadata.DisplaySetting
        else:
            print("No DisplaySetting information found.")

        if czi_box.has_layers:
            self.layers = czi_box.ImageDocument.Metadata.Layers
        else:
            print("No Layers information found.")


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
                print("No Scenes detected.")


class CziMetadataComplete:
    """
    Get the complete CZI metadata as an object created based on the
    dictionary created from the XML data.
    """

    def __init__(self, filepath: Union[str, os.PathLike[str]]) -> None:

        if isinstance(filepath, Path):
            # convert to string
            filepath = str(filepath)

        # get metadata dictionary using pylibCZIrw
        with pyczi.open_czi(filepath) as czidoc:
            md_dict = czidoc.metadata

        self.md = DictObj(md_dict)

        # TODO convert to dataclass and check if needed at all. Replace by Box?


class DictObj:
    """
    Create an object based on a dictionary. See https://joelmccune.com/python-dictionary-as-object/
    """

    def __init__(self, in_dict: dict) -> None:

        assert isinstance(in_dict, dict)

        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
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

    if sort:
        return misc.sort_dict_by_key(result)

    elif not sort:
        return result


def writexml(filepath: Union[str, os.PathLike[str]], xmlsuffix: str = '_CZI_MetaData.xml') -> str:
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
    xmlfile = filepath.replace('.czi', xmlsuffix)

    # get tree from string
    tree = ET.ElementTree(ET.fromstring(metadata_xmlstr))

    # write XML file to same folder
    tree.write(xmlfile, encoding='utf-8', method='xml')

    return xmlfile


def create_mdict_red(metadata: CziMetadata, sort: bool = True) -> Dict:
    """
    create_mdict_red: Created a metadata dictionary to be displayed in napari

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
               'ObjMag': metadata.objective.totalmag,
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


def get_czimd_box(filepath: Union[str, os.PathLike[str]]) -> Box:
    """
    get_czimd_box: Get CZI metadata as a python-box. For details: https://pypi.org/project/python-box/

    Args:
        filepath (Union[str, os.PathLike[str]]): Filepath of the CZI file

    Returns:
        Box: CZI metaadat as a Box object
    """

    if isinstance(filepath, Path):
        # convert to string
        filepath = str(filepath)

    # get metadata dictionary using pylibCZIrw
    with pyczi.open_czi(filepath) as czi_document:
        metadata_dict = czi_document.metadata

    czimd_box = Box(metadata_dict,
                    conversion_box=True,
                    default_box=True,
                    default_box_attr=None,
                    default_box_create_on_get=True,
                    # default_box_no_key_error=True
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
        czimd_box.has_experiment = True

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