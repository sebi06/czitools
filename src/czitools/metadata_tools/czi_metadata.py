# -*- coding: utf-8 -*-

#################################################################
# File        : czi_metadata.py
# Author      : sebi06
#
# Disclaimer: The code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

#from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any, Union
import os
import xml.etree.ElementTree as ET
from pylibCZIrw import czi as pyczi
from czitools.utils import logging_tools, misc
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from box import Box
import validators
from dataclasses import asdict
from czitools.metadata_tools.dimension import CziDimensions
from czitools.metadata_tools.boundingbox import CziBoundingBox
from czitools.metadata_tools.attachment import CziAttachments
from czitools.metadata_tools.channel import CziChannelInfo
from czitools.metadata_tools.scaling import CziScaling
from czitools.metadata_tools.sample import CziSampleInfo
from czitools.metadata_tools.objective import CziObjectives
from czitools.metadata_tools.microscope import CziMicroscope
from czitools.metadata_tools.add_metadata import CziAddMetaData
from czitools.metadata_tools.detector import CziDetector
from czitools.utils.box import get_czimd_box
from czitools.metadata_tools.helper import DictObj

logger = logging_tools.set_logging()


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
    array6d_size: Optional[Tuple[int]] = field(init=False, default_factory=lambda: ())
    scene_size_consistent: Optional[Tuple[int]] = field(
        init=False, default_factory=lambda: ()
    )
    verbose: bool = False
    """
    Create a CziMetadata object from the filename of the CZI image file.
    """

    def __post_init__(self):
        if validators.url(str(self.filepath)):
            self.pyczi_readertype = pyczi.ReaderFileInputTypes.Curl
            self.is_url = True
            logger.info(
                "FilePath is a valid link. Only pylibCZIrw functionality is available."
            )
        else:
            self.pyczi_readertype = pyczi.ReaderFileInputTypes.Standard
            self.is_url = False

            if isinstance(self.filepath, Path):
                # convert to string
                self.filepath = str(self.filepath)

        # get directory and filename etc.
        self.dirname = str(Path(self.filepath).parent)
        self.filename = str(Path(self.filepath).name)

        # get the metadata_tools as box
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
        self.image = CziDimensions(self.czi_box, verbose=self.verbose)

        # get metadata_tools using pylibCZIrw
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

        if not self.is_url:
            # get some additional metadata_tools using aicspylibczi
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
        self.bbox = CziBoundingBox(self.czi_box, verbose=self.verbose)

        # get information about channels
        self.channelinfo = CziChannelInfo(self.czi_box, verbose=self.verbose)

        # get scaling info
        self.scale = CziScaling(self.czi_box, verbose=self.verbose)

        # get objective information
        self.objective = CziObjectives(self.czi_box, verbose=self.verbose)

        # get detector information
        self.detector = CziDetector(self.czi_box, verbose=self.verbose)

        # get detector information
        self.microscope = CziMicroscope(self.czi_box, verbose=self.verbose)

        # get information about sample carrier and wells etc.
        self.sample = CziSampleInfo(self.czi_box, verbose=self.verbose)

        # get additional metainformation
        self.add_metadata = CziAddMetaData(self.czi_box, verbose=self.verbose)

        # check for attached label or preview image
        self.attachments = CziAttachments(self.czi_box, verbose=self.verbose)

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


def get_metadata_as_object(filepath: Union[str, os.PathLike[str]]) -> DictObj:
    """
    Get the complete CZI metadata_tools as an object created based on the
    dictionary created from the XML data.
    """

    if isinstance(filepath, Path):
        # convert to string
        filepath = str(filepath)

    # get metadata_tools dictionary using pylibCZIrw
    with pyczi.open_czi(filepath) as czidoc:
        md_dict = czidoc.metadata

    return DictObj(md_dict)





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
        return misc.sort_dict_by_key(result)

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

    # get the raw metadata_tools as XML or dictionary
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
    create_mdict_red: Created a reduced metadata_tools dictionary

    Args:
        metadata: CziMetadata class
        sort: sort the dictionary
        remove_none: Remove None values from dictionary

    Returns: dictionary with the metadata_tools

    """

    # create a dictionary with the metadata_tools
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
        "TotalBoundingBox": metadata.bbox.total_bounding_box,
    }

    # check for extra entries when reading mosaic file with a scale factor
    if hasattr(metadata.image, "SizeX_sf"):
        md_dict["XScale_sf"] = metadata.scale.X_sf
        md_dict["YScale_sf"] = metadata.scale.Y_sf

    if metadata.has_scenes:
        md_dict["SizeX_scene"] = metadata.image.SizeX_scene
        md_dict["SizeY_scene"] = metadata.image.SizeY_scene

    if remove_none:
        md_dict = misc.remove_none_from_dict(md_dict)

    if sort:
        return misc.sort_dict_by_key(md_dict)
    if not sort:
        return md_dict


# def get_czimd_box(
#     filepath: Union[str, os.PathLike[str]]
# ) -> Box:
#     """
#     get_czimd_box: Get CZI metadata_tools as a python-box. For details: https://pypi.org/project/python-box/
#
#     Args:
#         filepath (Union[str, os.PathLike[str]]): Filepath of the CZI file
#
#     Returns:
#         Box: CZI metadata_tools as a Box object
#     """
#
#     readertype = pyczi.ReaderFileInputTypes.Standard
#
#     if validators.url(str(filepath)):
#         readertype = pyczi.ReaderFileInputTypes.Curl
#
#     # get metadata_tools dictionary using pylibCZIrw
#     with pyczi.open_czi(str(filepath), readertype) as czi_document:
#         metadata_dict = czi_document.metadata_tools
#
#     czimd_box = Box(
#         metadata_dict,
#         conversion_box=True,
#         default_box=True,
#         default_box_attr=None,
#         default_box_create_on_get=True,
#         # default_box_no_key_error=True
#     )
#
#     # add the filepath
#     czimd_box.filepath = filepath
#     czimd_box.is_url = validators.url(str(filepath))
#     czimd_box.czi_open_arg = readertype
#
#     # set the defaults to False
#     czimd_box.has_customattr = False
#     czimd_box.has_experiment = False
#     czimd_box.has_disp = False
#     czimd_box.has_hardware = False
#     czimd_box.has_scale = False
#     czimd_box.has_instrument = False
#     czimd_box.has_microscopes = False
#     czimd_box.has_detectors = False
#     czimd_box.has_objectives = False
#     czimd_box.has_tubelenses = False
#     czimd_box.has_disp = False
#     czimd_box.has_channels = False
#     czimd_box.has_info = False
#     czimd_box.has_app = False
#     czimd_box.has_doc = False
#     czimd_box.has_image = False
#     czimd_box.has_scenes = False
#     czimd_box.has_dims = False
#     czimd_box.has_layers = False
#
#     if "Experiment" in czimd_box.ImageDocument.Metadata:
#         czimd_box.has_experiment = True
#
#     if "HardwareSetting" in czimd_box.ImageDocument.Metadata:
#         czimd_box.has_hardware = True
#
#     if "CustomAttributes" in czimd_box.ImageDocument.Metadata:
#         czimd_box.has_customattr = True
#
#     if "Information" in czimd_box.ImageDocument.Metadata:
#         czimd_box.has_info = True
#
#         if "Application" in czimd_box.ImageDocument.Metadata.Information:
#             czimd_box.has_app = True
#
#         if "Document" in czimd_box.ImageDocument.Metadata.Information:
#             czimd_box.has_doc = True
#
#         if "Image" in czimd_box.ImageDocument.Metadata.Information:
#             czimd_box.has_image = True
#
#             if "Dimensions" in czimd_box.ImageDocument.Metadata.Information.Image:
#                 czimd_box.has_dims = True
#
#                 if (
#                     "Channels"
#                     in czimd_box.ImageDocument.Metadata.Information.Image.Dimensions
#                 ):
#                     czimd_box.has_channels = True
#
#                 if "S" in czimd_box.ImageDocument.Metadata.Information.Image.Dimensions:
#                     czimd_box.has_scenes = True
#
#         if "Instrument" in czimd_box.ImageDocument.Metadata.Information:
#             czimd_box.has_instrument = True
#
#             if "Detectors" in czimd_box.ImageDocument.Metadata.Information.Instrument:
#                 czimd_box.has_detectors = True
#
#             if "Microscopes" in czimd_box.ImageDocument.Metadata.Information.Instrument:
#                 czimd_box.has_microscopes = True
#
#             if "Objectives" in czimd_box.ImageDocument.Metadata.Information.Instrument:
#                 czimd_box.has_objectives = True
#
#             if "TubeLenses" in czimd_box.ImageDocument.Metadata.Information.Instrument:
#                 czimd_box.has_tubelenses = True
#
#     if "Scaling" in czimd_box.ImageDocument.Metadata:
#         czimd_box.has_scale = True
#
#     if "DisplaySetting" in czimd_box.ImageDocument.Metadata:
#         czimd_box.has_disp = True
#
#     if "Layers" in czimd_box.ImageDocument.Metadata:
#         czimd_box.has_layers = True
#
#     return czimd_box


def create_md_dict_nested(
    metadata: CziMetadata, sort: bool = True, remove_none: bool = True
) -> Dict:
    """
    Create nested dictionary from metadata_tools

    Args:
        metadata (CziMetadata): CzIMetaData object_
        sort (bool, optional): Sort the dictionary_. Defaults to True.
        remove_none (bool, optional): Remove None values from dictionary. Defaults to True.

    Returns:
        Dict: Nested dictionary with reduced set of metadata_tools
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

    md_box_bbox = Box(metadata.bbox.total_bounding_box)
    # del md_box_bbox.czisource

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

    IDs = [
        "array6d",
        "image",
        "scale",
        "sample",
        "objectives",
        "channels",
        "bbox",
        "info",
    ]

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
        md_dict = misc.remove_none_from_dict(md_dict)

    if sort:
        return misc.sort_dict_by_key(md_dict)
    if not sort:
        return md_dict

    return md_dict
