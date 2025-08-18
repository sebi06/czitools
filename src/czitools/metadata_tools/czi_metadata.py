# -*- coding: utf-8 -*-

#################################################################
# File        : czi_metadata.py
# Author      : sebi06
#
# Disclaimer: The code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

# from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any, Union
import os
import xml.etree.ElementTree as ET
from pylibCZIrw import czi as pyczi
from czitools.utils import logging_tools, misc, pixels

# import numpy as np
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

import contextlib

logger = logging_tools.set_logging()


@dataclass
class CziMetadata:
    """
    CziMetadata class for handling metadata of CZI image files.

    Attributes:
        filepath (Union[str, os.PathLike[str]]): Path to the CZI image file.
        filename (Optional[str]): Name of the CZI image file.
        dirname (Optional[str]): Directory of the CZI image file.
        is_url (Optional[bool]): Indicates if the filepath is a URL.
        software_name (Optional[str]): Name of the software used for acquisition.
        software_version (Optional[str]): Version of the software used for acquisition.
        acquisition_date (Optional[str]): Date and time of image acquisition.
        creation_date (Optional[str]): Date and time of image creation.
        user_name (Optional[str]): Name of the user who created the image.
        czi_box (Optional[Box]): Metadata box of the CZI file.
        pyczi_dims (Optional[Dict[str, tuple]]): Dimensions of the CZI file.
        aics_dimstring (Optional[str]): Dimension string from aicspylibczi.
        aics_dims_shape (Optional[List[Dict[str, tuple]]]): Dimension shapes from aicspylibczi.
        aics_size (Optional[Tuple[int]]): Size of the CZI file from aicspylibczi.
        aics_ismosaic (Optional[bool]): Indicates if the CZI file is a mosaic.
        aics_dim_order (Optional[Dict[str, int]]): Dimension order from aicspylibczi.
        aics_dim_index (Optional[List[int]]): Dimension indices from aicspylibczi.
        aics_dim_valid (Optional[int]): Number of valid dimensions from aicspylibczi.
        aics_posC (Optional[int]): Position of the 'C' dimension from aicspylibczi.
        pixeltypes (Optional[Dict[int, str]]): Pixel types for all channels.
        consistent_pixeltypes (Optional[bool]): Indicates if pixel types are consistent across channels.
        isRGB (Optional[Dict[int, bool]]): Indicates if the image is RGB.
        has_scenes (Optional[bool]): Indicates if the CZI file has scenes.
        has_label (Optional[bool]): Indicates if the CZI file has a label image.
        has_preview (Optional[bool]): Indicates if the CZI file has a preview image.
        attachments (Optional[CziAttachments]): Attachments in the CZI file.
        npdtype_list (Optional[List[Any]]): Numpy data types for pixel values.
        maxvalue_list (Optional[List[int]]): Maximum values for pixel types.
        image (Optional[CziDimensions]): Dimensions of the CZI image.
        bbox (Optional[CziBoundingBox]): Bounding box of the CZI image.
        channelinfo (Optional[CziChannelInfo]): Information about channels.
        scale (Optional[CziScaling]): Scaling information.
        objective (Optional[CziObjectives]): Objective information.
        detector (Optional[CziDetector]): Detector information.
        microscope (Optional[CziMicroscope]): Microscope information.
        sample (Optional[CziSampleInfo]): Sample information.
        add_metadata (Optional[CziAddMetaData]): Additional metadata.
        scene_size_consistent (Optional[Tuple[int]]): Consistency of scene sizes.
        verbose (bool): Verbose output for logging.

    Methods:
        __post_init__(): Initializes the CziMetadata object after dataclass initialization.
    """

    filepath: Union[str, os.PathLike[str]]
    filename: Optional[str] = field(init=False, default=None)
    dirname: Optional[str] = field(init=False, default=None)
    is_url: Optional[bool] = field(init=False, default=False)
    software_name: Optional[str] = field(init=False, default=None)
    software_version: Optional[str] = field(init=False, default=None)
    acquisition_date: Optional[str] = field(init=False, default=None)
    creation_date: Optional[str] = field(init=False, default=None)
    user_name: Optional[str] = field(init=False, default=None)
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
    consistent_pixeltypes: Optional[bool] = field(init=False, default=None)
    isRGB: Optional[Dict[int, bool]] = field(init=False, default_factory=lambda: {})
    has_scenes: Optional[bool] = field(init=False, default=False)
    has_label: Optional[bool] = field(init=False, default=False)
    has_preview: Optional[bool] = field(init=False, default=False)
    attachments: Optional[CziAttachments] = field(init=False, default=None)
    npdtype_list: Optional[List[Any]] = field(init=False, default_factory=lambda: [])
    maxvalue_list: Optional[List[int]] = field(init=False, default_factory=lambda: [])
    image: Optional[CziDimensions] = field(init=False, default=None)
    bbox: Optional[CziBoundingBox] = field(init=False, default=None)
    channelinfo: Optional[CziChannelInfo] = field(init=False, default=None)
    scale: Optional[CziScaling] = field(init=False, default=None)
    objective: Optional[CziObjectives] = field(init=False, default=None)
    detector: Optional[CziDetector] = field(init=False, default=None)
    microscope: Optional[CziMicroscope] = field(init=False, default=None)
    sample: Optional[CziSampleInfo] = field(init=False, default=None)
    add_metadata: Optional[CziAddMetaData] = field(init=False, default=None)
    scene_size_consistent: Optional[Tuple[int]] = field(
        init=False, default_factory=lambda: ()
    )
    verbose: bool = False

    def __post_init__(self):
        if validators.url(str(self.filepath)):
            self.pyczi_readertype = pyczi.ReaderFileInputTypes.Curl
            self.is_url = True
            if self.verbose:
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
            self.isRGB, self.consistent_pixeltypes = pixels.check_if_rgb(
                self.pixeltypes
            )

            # check for consistent scene shape
            self.scene_shape_is_consistent = pixels.check_scenes_shape(
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
                ) = pixels.get_dimorder(aicsczi.dims)
                self.aics_posC = self.aics_dim_order["C"]

            except ImportError as e:
                # print("Package aicspylibczi not found. Use Fallback values.")
                logger.warning(
                    f" {e} : Package aicspylibczi not found. Use Fallback values."
                )

        self.npdtype_list = []
        self.maxvalue_list = []

        for ch, px in self.pixeltypes.items():
            npdtype, maxvalue = pixels.get_dtype_fromstring(px)
            self.npdtype_list.append(npdtype)
            self.maxvalue_list.append(maxvalue)

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


def get_metadata_as_object(filepath: Union[str, os.PathLike[str]]) -> DictObj:
    """
    Get the complete CZI metadata as an object.
    This function reads the metadata from a CZI file and converts it into a
    dictionary, which is then used to create an object of type DictObj.
    Args:
        filepath (Union[str, os.PathLike[str]]): The path to the CZI file.
            This can be a string or an os.PathLike object.
    Returns:
        DictObj: An object containing the metadata extracted from the CZI file.
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
        "ChannelDyesShort": metadata.channelinfo.dyes_short,
        "ChannelColors": metadata.channelinfo.colors,
        "WellArrayNames": metadata.sample.well_array_names,
        "ChannelDescriptions": metadata.channelinfo.channel_descriptions,
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

    # Convert all numpy values to native Python types for better display
    with contextlib.suppress(Exception):  # Catch any conversion errors
        md_dict = convert_numpy_types(md_dict)

    # check for extra entries when reading mosaic file with a scale factor
    if hasattr(metadata.image, "SizeX_sf"):
        md_dict["XScale_sf"] = metadata.scale.X_sf
        md_dict["YScale_sf"] = metadata.scale.Y_sf

    if metadata.has_scenes:
        md_dict["SizeX_scene"] = metadata.image.SizeX_scene
        md_dict["SizeY_scene"] = metadata.image.SizeY_scene

    if remove_none:
        # md_dict = misc.remove_none_from_dict(md_dict)
        md_dict = misc.clean_dict(md_dict)

    if sort:
        return misc.sort_dict_by_key(md_dict)
    if not sort:
        return md_dict


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
        "image",
        "scale",
        "sample",
        "objectives",
        "channels",
        "bbox",
        "info",
    ]

    mds = [
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
        # md_dict = misc.remove_none_from_dict(md_dict)
        md_dict = misc.clean_dict(md_dict)

    if sort:
        return misc.sort_dict_by_key(md_dict)
    if not sort:
        return md_dict

    return md_dict


def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python types.

    This function handles numpy scalars, arrays, dictionaries, lists, and tuples.
    It converts numpy scalars to their native Python equivalents, numpy arrays
    to Python lists, and recursively processes dictionaries, lists, and tuples.

    Args:
        obj (Any): The object to convert. Can be a numpy scalar, numpy array,
                   dictionary, list, tuple, or any other type.

    Returns:
        Any: The converted object with numpy types replaced by native Python types.
    """
    if hasattr(obj, "dtype"):  # numpy scalar or array
        if obj.ndim == 0:  # scalar
            return obj.item()  # Convert to native Python type
        else:
            return obj.tolist()  # Convert array to list
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(convert_numpy_types(item) for item in obj)
    else:
        return obj
