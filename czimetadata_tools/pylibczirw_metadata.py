# -*- coding: utf-8 -*-

#################################################################
# File        : pylibczirw_metadata.py
# Version     : 0.1.8
# Author      : sebi06
# Date        : 22.04.2022
#
# Disclaimer: The code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from __future__ import annotations
import os
from collections import Counter
import xml.etree.ElementTree as ET
from pylibCZIrw import czi as pyczi
from tqdm.contrib.itertools import product
try:
    from czimetadata_tools import misc
except ImportError:
    import misc
import numpy as np
import pydash
from typing import List, Dict, Tuple, Optional, Type, Any, Union
from dataclasses import dataclass, field


class CziMetadataComplete:
    """Get the complete CZI metadata as an object created based on the
    dictionary created from the XML data.
    """

    def __init__(self, filename: str):

        # get metadata dictionary using pylibCZIrw
        with pyczi.open_czi(filename) as czidoc:
            md_dict = czidoc.metadata

        self.md = DictObj(md_dict)


class DictObj:
    """Create an object based on a dictionary
    """
    # based upon: https://joelmccune.com/python-dictionary-as-object/

    def __init__(self, in_dict: dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, DictObj(val) if isinstance(val, dict) else val)


class CziMetadata:

    def __init__(self, filename: str, dim2none: bool = False) -> None:

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
                self.aics_dim_order, self.aics_dim_index, self.aics_dim_valid = self.get_dimorder(aicsczi.dims)
                self.aics_posC = self.aics_dim_order["C"]

            except ImportError as e:
                print("Use Fallback values because:", e)
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
            self.isRGB = False

            # determine pixel type for CZI array by reading XML metadata
            self.pixeltype = md_dict["ImageDocument"]["Metadata"]["Information"]["Image"]["PixelType"]

            # check if CZI is a RGB file
            if self.pixeltype in ["Bgr24", "Bgr48", "Bgr96Float"]:
                self.isRGB = True

            # determine pixel type for CZI array
            self.npdtype, self.maxvalue = self.get_dtype_fromstring(self.pixeltype)

            # get the dimensions and order
            self.image = CziDimensions(filename)

            # try to guess if the CZI is a mosaic file
            if self.image.SizeM is None or self.image.SizeM == 1:
                self.ismosaic = False
            else:
                self.ismosaic = True

            # get the bounding boxes
            self.bbox = CziBoundingBox(filename)

            # get information about channels
            self.channelinfo = CziChannelInfo(filename)

            # get scaling info
            self.scale = CziScaling(filename, dim2none=dim2none)

            # get objetive information
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
            maxvalue = 265535

        return dtype, maxvalue

    @staticmethod
    def get_dimorder(dimstring: str) -> Tuple[Dict, List, int]:
        """Get the order of dimensions from dimension string

        :param dimstring: string containing the dimensions
        :type dimstring: str
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
            dims_dict[d] = dimstring.find(d)
            dimindex_list.append(dimstring.find(d))

        # check if a dimension really exists
        numvalid_dims = sum(i > 0 for i in dimindex_list)

        return dims_dict, dimindex_list, numvalid_dims


class CziDimensions:

    def __init__(self, filename: str) -> None:

        self.SizeX = None
        self.SizeY = None
        self.SizeS = None
        self.SizeT = None
        self.SizeZ = None
        self.SizeC = None
        self.SizeM = None
        self.SizeR = None
        self.SizeH = None
        self.SizeI = None
        self.SizeV = None
        self.SizeB = None
        self.SizeX_sf = None
        self.SizeY_sf = None

        with pyczi.open_czi(filename) as czidoc_r:
            dim_dict = self.get_image_dimensions(czidoc_r.metadata)

            for key in dim_dict:
                setattr(self, key, dim_dict[key])

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

    @staticmethod
    def get_image_dimensions(raw_metadata: Dict[Any, Any],
                             dim2none: bool = True) -> Dict[Any, Union[int, None]]:
        """Determine the image dimensions.

        Arguments:
            raw_metadata: The CZI meta-data to derive the image dimensions from.

        Returns:
            The dimension dictionary.
        """

        def _safe_get(key: str) -> Optional[int]:
            try:
                extracted_value = raw_metadata["ImageDocument"]["Metadata"]["Information"]["Image"][key]
                return int(extracted_value) if extracted_value is not None else None
            except KeyError:
                return None

        dimensions = ["SizeX",
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
                      "SizeB"]

        dim_dict = {}

        for dim in dimensions:
            dim_dict[dim] = _safe_get(dim)

        return dim_dict


@dataclass
class CziBoundingBox:
    filename: str
    all_scenes: Optional[Dict[int, pyczi.Rectangle]] = field(init=False)
    total_rect: Optional[pyczi.Rectangle] = field(init=False)
    total_bounding_box: Optional[Dict[str, tuple]] = field(init=False)

    def __post_init__(self):
        with pyczi.open_czi(self.filename) as czidoc:

            try:
                self.all_scenes = czidoc.scenes_bounding_rectangle
            except Exception as e:
                self.all_scenes = None
                print("Scenes Bounding rectangle not found.", e)

            try:
                self.total_rect = czidoc.total_bounding_rectangle
            except Exception as e:
                self.total_rect = None
                print("Total Bounding rectangle not found.", e)

            try:
                self.total_bounding_box = czidoc.total_bounding_box
            except Exception as e:
                self.total_bounding_box = None
                print("Total Bounding Box not found.", e)


class CziChannelInfo:
    def __init__(self, filename: str) -> None:

        # get metadata dictionary using pylibCZIrw
        with pyczi.open_czi(filename) as czidoc:
            md_dict = czidoc.metadata

        # create empty lists for channel related information
        channels = []
        channels_names = []
        channels_colors = []
        channels_contrast = []
        channels_gamma = []

        try:
            sizeC = np.int(md_dict["ImageDocument"]["Metadata"]["Information"]["Image"]["SizeC"])
        except (KeyError, TypeError) as e:
            sizeC = 1

        # in case of only one channel
        if sizeC == 1:
            # get name for dye
            try:
                channels.append(
                    md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"]["ShortName"])
            except (KeyError, TypeError) as e:
                print("Channel shortname not found :", e)
                try:
                    channels.append(md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"]["DyeName"])
                except (KeyError, TypeError) as e:
                    print("Channel dye not found :", e)
                    channels.append("Dye-CH1")

            # get channel name
            try:
                channels_names.append(
                    md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"]["Name"])
            except (KeyError, TypeError) as e:
                try:
                    channels_names.append(
                        md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"]["@Name"])
                except (KeyError, TypeError) as e:
                    print("Channel name found :", e)
                    channels_names.append("CH1")

            # get channel color
            try:
                channels_colors.append(
                    md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"]["Color"])
            except (KeyError, TypeError) as e:
                print("Channel color not found :", e)
                channels_colors.append("#80808000")

            # get contrast setting fro DisplaySetting
            try:
                low = np.float(md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"]["Low"])
            except (KeyError, TypeError) as e:
                low = 0.1
            try:
                high = np.float(md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"]["High"])
            except (KeyError, TypeError) as e:
                high = 0.5

            channels_contrast.append([low, high])

            # get the gamma values
            try:
                channels_gamma.append(
                    np.float(md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"]["Gamma"]))
            except (KeyError, TypeError) as e:
                channels_gamma.append(0.85)

        # in case of two or more channels
        if sizeC > 1:
            # loop over all channels
            for ch in range(sizeC):
                # get name for dyes
                try:
                    channels.append(
                        md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"][ch]["ShortName"])
                except (KeyError, TypeError) as e:
                    print("Channel shortname not found :", e)
                    try:
                        channels.append(
                            md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"][ch][
                                "DyeName"])
                    except (KeyError, TypeError) as e:
                        print("Channel dye not found :", e)
                        channels.append("Dye-CH" + str(ch))

                # get channel names
                try:
                    channels_names.append(
                        md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"][ch]["Name"])
                except (KeyError, TypeError) as e:
                    try:
                        channels_names.append(
                            md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"][ch]["@Name"])
                    except (KeyError, TypeError) as e:
                        print("Channel name not found :", e)
                        channels_names.append("CH" + str(ch))

                # get channel colors
                try:
                    channels_colors.append(
                        md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"][ch]["Color"])
                except (KeyError, TypeError) as e:
                    print("Channel color not found :", e)
                    # use grayscale instead
                    channels_colors.append("80808000")

                # get contrast setting fro DisplaySetting
                try:
                    low = np.float(
                        md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"][ch]["Low"])
                except (KeyError, TypeError) as e:
                    low = 0.0
                try:
                    high = np.float(
                        md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"][ch]["High"])
                except (KeyError, TypeError) as e:
                    high = 0.5

                channels_contrast.append([low, high])

                # get the gamma values
                try:
                    channels_gamma.append(np.float(
                        md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"][ch]["Gamma"]))
                except (KeyError, TypeError) as e:
                    channels_gamma.append(0.85)

        # write channels information (as lists) into metadata dictionary
        self.shortnames = channels
        self.names = channels_names
        self.colors = channels_colors
        self.clims = channels_contrast
        self.gamma = channels_gamma


class CziScaling:
    def __init__(self, filename: str, dim2none: bool = True) -> None:

        # get metadata dictionary using pylibCZIrw
        self.scalefactorXY = None
        self.ratio_sf = None
        self.Y_sf = None
        self.X_sf = None

        with pyczi.open_czi(filename) as czidoc:
            md_dict = czidoc.metadata

        def _safe_get_scale(distances_: List[Dict[Any, Any]], idx: int) -> Optional[float]:
            try:
                # return np.round(float(distances_[idx]["Value"]) * 1000000, 6) if distances_[idx]["Value"] is not None else None
                return float(distances_[idx]["Value"]) * 1000000 if distances_[idx]["Value"] is not None else None
            except IndexError:
                if dim2none:
                    return None
                if not dim2none:
                    # use scaling = 1.0 micron as fallback
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

        ## get the scaling in [micron] - inside CZI the default is [m]
        #self.X = _safe_get_scale(distances, 0)
        #self.Y = _safe_get_scale(distances, 1)
        #self.Z = _safe_get_scale(distances, 2)

        # safety check in case a scale = 0
        if self.X == 0.0:
            self.X = 1.0
        if self.Y == 0.0:
            self.Y = 1.0
        if self.Z == 0.0:
            self.Z = 1.0

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

        # set default scale factor to 1.0
        scale_ratio = {"xy": 1.0,
                       "zx": 1.0
                       }
        try:
            # get the factor between XY scaling
            scale_ratio["xy"] = np.round(scalex / scaley, 3)
            # get the scalefactor between XZ scaling
            scale_ratio["zx"] = np.round(scalez / scalex, 3)
        except (KeyError, TypeError) as e:
            print(e, "Using defaults = 1.0")

        return scale_ratio


@dataclass
class CziInfo:
    filepath: str
    dirname: str = field(init=False)
    filename: str = field(init=False)
    software_name: str = field(init=False)
    softname_version: str = field(init=False)
    acquisition_date: str = field(init=False)

    def __post_init__(self):

        # get directory and filename etc.
        self.dirname = os.path.dirname(self.filepath)
        self.filename = os.path.basename(self.filepath)

        # get metadata dictionary using pylibCZIrw
        with pyczi.open_czi(self.filepath) as czidoc:
            md_dict = czidoc.metadata

        # get acquisition data and SW version
        try:
            self.software_name = md_dict["ImageDocument"]["Metadata"]["Information"]["Application"]["Name"]
            self.software_version = md_dict["ImageDocument"]["Metadata"]["Information"]["Application"]["Version"]
        except (KeyError, TypeError) as e:
            print("Key not found:", e)
            self.software_name = None
            self.software_version = None

        try:
            self.acquisition_date = md_dict["ImageDocument"]["Metadata"]["Information"]["Image"]["AcquisitionDateAndTime"]
        except (KeyError, TypeError) as e:
            print("Key not found:", e)
            self.acquisition_date = None


class CziObjectives:
    def __init__(self, filename: str) -> None:

        # get metadata dictionary using pylibCZIrw
        with pyczi.open_czi(filename) as czidoc:
            md_dict = czidoc.metadata

        self.NA = []
        self.mag = []
        self.ID = []
        self.name = []
        self.immersion = []
        self.tubelensmag = []
        self.nominalmag = []

        # check if Instrument metadata actually exist
        if pydash.objects.has(md_dict, ["ImageDocument", "Metadata", "Information", "Instrument", "Objectives"]):
            # get objective data
            try:
                if isinstance(md_dict["ImageDocument"]["Metadata"]["Information"]["Instrument"]["Objectives"]["Objective"], list):
                    num_obj = len(md_dict["ImageDocument"]["Metadata"]["Information"]["Instrument"]["Objectives"]["Objective"])
                else:
                    num_obj = 1
            except (KeyError, TypeError) as e:
                num_obj = 0  # no objective found

            # if there is only one objective found
            if num_obj == 1:
                try:
                    self.name.append(
                        md_dict["ImageDocument"]["Metadata"]["Information"]["Instrument"]["Objectives"]["Objective"]["Name"])
                except (KeyError, TypeError) as e:
                    print("No Objective Name :", e)
                    self.name.append(None)

                try:
                    self.immersion = md_dict["ImageDocument"]["Metadata"]["Information"]["Instrument"]["Objectives"]["Objective"]["Immersion"]
                except (KeyError, TypeError) as e:
                    print("No Objective Immersion :", e)
                    self.immersion = None

                try:
                    self.NA = np.float(md_dict["ImageDocument"]["Metadata"]["Information"]
                                       ["Instrument"]["Objectives"]["Objective"]["LensNA"])
                except (KeyError, TypeError) as e:
                    print("No Objective NA :", e)
                    self.NA = None

                try:
                    self.ID = md_dict["ImageDocument"]["Metadata"]["Information"]["Instrument"]["Objectives"]["Objective"]["Id"]
                except (KeyError, TypeError) as e:
                    print("No Objective ID :", e)
                    self.ID = None

                try:
                    self.tubelensmag = np.float(
                        md_dict["ImageDocument"]["Metadata"]["Information"]["Instrument"]["TubeLenses"]["TubeLens"]["Magnification"])
                except (KeyError, TypeError) as e:
                    print("No Tubelens Mag. :", e, "Using Default Value = 1.0.")
                    self.tubelensmag = 1.0

                try:
                    self.nominalmag = np.float(
                        md_dict["ImageDocument"]["Metadata"]["Information"]["Instrument"]["Objectives"]["Objective"][
                            "NominalMagnification"])
                except (KeyError, TypeError) as e:
                    print("No Nominal Mag.:", e, "Using Default Value = 1.0.")
                    self.nominalmag = 1.0

                try:
                    if self.tubelensmag is not None:
                        self.mag = self.nominalmag * self.tubelensmag
                    if self.tubelensmag is None:
                        print("Using Tublens Mag = 1.0 for calculating Objective Magnification.")
                        self.mag = self.nominalmag * 1.0

                except (KeyError, TypeError) as e:
                    print("No Objective Magnification :", e)
                    self.mag = None

            if num_obj > 1:
                for o in range(num_obj):

                    try:
                        self.name.append(
                            md_dict["ImageDocument"]["Metadata"]["Information"]["Instrument"]["Objectives"]["Objective"][o][
                                "Name"])
                    except (KeyError, TypeError) as e:
                        print("No Objective Name :", e)
                        self.name.append(None)

                    try:
                        self.immersion.append(
                            md_dict["ImageDocument"]["Metadata"]["Information"]["Instrument"]["Objectives"]["Objective"][o][
                                "Immersion"])
                    except (KeyError, TypeError) as e:
                        print("No Objective Immersion :", e)
                        self.immersion.append(None)

                    try:
                        self.NA.append(np.float(
                            md_dict["ImageDocument"]["Metadata"]["Information"]["Instrument"]["Objectives"]["Objective"][o][
                                "LensNA"]))
                    except (KeyError, TypeError) as e:
                        print("No Objective NA :", e)
                        self.NA.append(None)

                    try:
                        self.ID.append(
                            md_dict["ImageDocument"]["Metadata"]["Information"]["Instrument"]["Objectives"]["Objective"][o][
                                "Id"])
                    except (KeyError, TypeError) as e:
                        print("No Objective ID :", e)
                        self.ID.append(None)

                    try:
                        self.tubelensmag.append(np.float(
                            md_dict["ImageDocument"]["Metadata"]["Information"]["Instrument"]["TubeLenses"]["TubeLens"][o][
                                "Magnification"]))
                    except (KeyError, TypeError) as e:
                        print("No Tubelens Mag. :", e, "Using Default Value = 1.0.")
                        self.tubelensmag.append(1.0)

                    try:
                        self.nominalmag.append(np.float(
                            md_dict["ImageDocument"]["Metadata"]["Information"]["Instrument"]["Objectives"]["Objective"][o][
                                "NominalMagnification"]))
                    except (KeyError, TypeError) as e:
                        print("No Nominal Mag. :", e, "Using Default Value = 1.0.")
                        self.nominalmag.append(1.0)

                    try:
                        if self.tubelensmag is not None:
                            self.mag.append(self.nominalmag[o] * self.tubelensmag[o])
                        if self.tubelensmag is None:
                            print("Using Tublens Mag = 1.0 for calculating Objective Magnification.")
                            self.mag.append(self.nominalmag[o] * 1.0)

                    except (KeyError, TypeError) as e:
                        print("No Objective Magnification :", e)
                        self.mag.append(None)


class CziDetector:
    def __init__(self, filename: str) -> None:

        # get metadata dictionary using pylibCZIrw
        with pyczi.open_czi(filename) as czidoc:
            md_dict = czidoc.metadata

        # get detector information
        self.model = []
        self.name = []
        self.ID = []
        self.modeltype = []
        self.instrumentID = []

        # check if there are any detector entries inside the dictionary
        if pydash.objects.has(md_dict, ["ImageDocument", "Metadata", "Information", "Instrument", "Detectors"]):

            if isinstance(md_dict["ImageDocument"]["Metadata"]["Information"]["Instrument"]["Detectors"]["Detector"], list):
                num_detectors = len(
                    md_dict["ImageDocument"]["Metadata"]["Information"]["Instrument"]["Detectors"]["Detector"])
            else:
                num_detectors = 1

            # if there is only one detector found
            if num_detectors == 1:

                # check for detector ID
                try:
                    self.ID.append(
                        md_dict["ImageDocument"]["Metadata"]["Information"]["Instrument"]["Detectors"]["Detector"]["Id"])
                except (KeyError, TypeError) as e:
                    print("DetectorID not found :", e)
                    self.ID.append(None)

                # check for detector Name
                try:
                    self.name.append(
                        md_dict["ImageDocument"]["Metadata"]["Information"]["Instrument"]["Detectors"]["Detector"]["Name"])
                except (KeyError, TypeError) as e:
                    print("DetectorName not found :", e)
                    self.name.append(None)

                # check for detector model
                try:
                    self.model.append(
                        md_dict["ImageDocument"]["Metadata"]["Information"]["Instrument"]["Detectors"]["Detector"][
                            "Manufacturer"]["Model"])
                except (KeyError, TypeError) as e:
                    print("DetectorModel not found :", e)
                    self.model.append(None)

                # check for detector modeltype
                try:
                    self.modeltype.append(
                        md_dict["ImageDocument"]["Metadata"]["Information"]["Instrument"]["Detectors"]["Detector"]["Type"])
                except (KeyError, TypeError) as e:
                    print("DetectorType not found :", e)
                    self.modeltype.append(None)

            if num_detectors > 1:
                for d in range(num_detectors):

                    # check for detector ID
                    try:
                        self.ID.append(
                            md_dict["ImageDocument"]["Metadata"]["Information"]["Instrument"]["Detectors"]["Detector"][d][
                                "Id"])
                    except (KeyError, TypeError) as e:
                        print("DetectorID not found :", e)
                        self.ID.append(None)

                    # check for detector Name
                    try:
                        self.name.append(
                            md_dict["ImageDocument"]["Metadata"]["Information"]["Instrument"]["Detectors"]["Detector"][d][
                                "Name"])
                    except (KeyError, TypeError) as e:
                        print("DetectorName not found :", e)
                        self.name.append(None)

                    # check for detector model
                    try:
                        self.model.append(
                            md_dict["ImageDocument"]["Metadata"]["Information"]["Instrument"]["Detectors"]["Detector"][d][
                                "Manufacturer"]["Model"])
                    except (KeyError, TypeError) as e:
                        print("DetectorModel not found :", e)
                        self.model.append(None)

                    # check for detector modeltype
                    try:
                        self.modeltype.append(
                            md_dict["ImageDocument"]["Metadata"]["Information"]["Instrument"]["Detectors"]["Detector"][d][
                                "Type"])
                    except (KeyError, TypeError) as e:
                        print("DetectorType not found :", e)
                        self.modeltype.append(None)


class CziMicroscope:
    def __init__(self, filename: str) -> None:

        # get metadata dictionary using pylibCZIrw
        with pyczi.open_czi(filename) as czidoc:
            md_dict = czidoc.metadata

        self.ID = None
        self.Name = None

        # check if there are any microscope entry inside the dictionary
        if pydash.objects.has(md_dict, ["ImageDocument", "Metadata", "Information", "Instrument", "Microscopes"]):

            # check for detector ID
            try:
                self.ID = md_dict["ImageDocument"]["Metadata"]["Information"]["Instrument"]["Microscopes"]["Microscope"][
                    "Id"]
            except (KeyError, TypeError) as e:
                try:
                    self.ID = md_dict["ImageDocument"]["Metadata"]["Information"]["Instrument"]["Microscopes"]["Microscope"][
                        "@Id"]
                except (KeyError, TypeError) as e:
                    print("Microscope ID not found :", e)
                    self.ID = None

            # check for microscope system name
            try:
                self.Name = md_dict["ImageDocument"]["Metadata"]["Information"]["Instrument"]["Microscopes"]["Microscope"][
                    "System"]
            except (KeyError, TypeError) as e:
                print("Microscope System Name not found :", e)
                self.Name = None


class CziSampleInfo:
    def __init__(self, filename: str) -> None:

        # get metadata dictionary using pylibCZIrw
        with pyczi.open_czi(filename) as czidoc:
            md_dict = czidoc.metadata

        dim_dict = CziDimensions.get_image_dimensions(md_dict)
        sizeS = dim_dict["SizeS"]

        # check for well information
        self.well_array_names = []
        self.well_indices = []
        self.well_position_names = []
        self.well_colID = []
        self.well_rowID = []
        self.well_counter = []
        self.scene_stageX = []
        self.scene_stageY = []

        if sizeS is not None:
            print("Trying to extract Scene and Well information if existing ...")

            # extract well information from the dictionary
            allscenes = md_dict["ImageDocument"]["Metadata"]["Information"]["Image"]["Dimensions"]["S"]["Scenes"]["Scene"]

            # loop over all detected scenes
            for s in range(sizeS):

                if sizeS == 1:
                    well = allscenes
                    try:
                        self.well_array_names.append(allscenes["ArrayName"])
                    except (KeyError, TypeError) as e:
                        try:
                            self.well_array_names.append(well["Name"])
                        except (KeyError, TypeError) as e:
                            # print("Well Name not found :", e)
                            try:
                                self.well_array_names.append(well["@Name"])
                            except (KeyError, TypeError) as e:
                                # print("Well @Name not found :", e)
                                print("Well Name not found :", e, "Using A1 instead")
                                self.well_array_names.append("A1")

                    try:
                        self.well_indices.append(allscenes["Index"])
                    except (KeyError, TypeError) as e:
                        try:
                            self.well_indices.append(allscenes["@Index"])
                        except (KeyError, TypeError) as e:
                            print("Well Index not found :", e)
                            self.well_indices.append(1)

                    try:
                        self.well_position_names.append(allscenes["Name"])
                    except (KeyError, TypeError) as e:
                        try:
                            self.well_position_names.append(allscenes["@Name"])
                        except (KeyError, TypeError) as e:
                            print("Well Position Names not found :", e)
                            self.well_position_names.append("P1")

                    try:
                        self.well_colID.append(np.int(allscenes["Shape"]["ColumnIndex"]))
                    except (KeyError, TypeError) as e:
                        print("Well ColumnIDs not found :", e)
                        self.well_colID.append(0)

                    try:
                        self.well_rowID.append(np.int(allscenes["Shape"]["RowIndex"]))
                    except (KeyError, TypeError) as e:
                        print("Well RowIDs not found :", e)
                        self.well_rowID.append(0)

                    try:
                        # count the content of the list, e.g. how many time a certain well was detected
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
                        print("Stage Positions XY not found :", e)
                        self.scene_stageX.append(0.0)
                        self.scene_stageY.append(0.0)

                if sizeS > 1:
                    try:
                        well = allscenes[s]
                        self.well_array_names.append(well["ArrayName"])
                    except (KeyError, TypeError) as e:
                        try:
                            self.well_array_names.append(well["Name"])
                        except (KeyError, TypeError) as e:
                            # print("Well Name not found :", e)
                            try:
                                self.well_array_names.append(well["@Name"])
                            except (KeyError, TypeError) as e:
                                # print("Well @Name not found :", e)
                                print("Well Name not found. Using A1 instead")
                                self.well_array_names.append("A1")

                    # get the well information
                    try:
                        self.well_indices.append(well["Index"])
                    except (KeyError, TypeError) as e:
                        try:
                            self.well_indices.append(well["@Index"])
                        except (KeyError, TypeError) as e:
                            print("Well Index not found :", e)
                            self.well_indices.append(None)
                    try:
                        self.well_position_names.append(well["Name"])
                    except (KeyError, TypeError) as e:
                        try:
                            self.well_position_names.append(well["@Name"])
                        except (KeyError, TypeError) as e:
                            print("Well Position Names not found :", e)
                            self.well_position_names.append(None)

                    try:
                        self.well_colID.append(np.int(well["Shape"]["ColumnIndex"]))
                    except (KeyError, TypeError) as e:
                        print("Well ColumnIDs not found :", e)
                        self.well_colID.append(None)

                    try:
                        self.well_rowID.append(np.int(well["Shape"]["RowIndex"]))
                    except (KeyError, TypeError) as e:
                        print("Well RowIDs not found :", e)
                        self.well_rowID.append(None)

                    # count the content of the list, e.g. how many time a certain well was detected
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
                            print("Stage Positions XY not found :", e)
                            self.scene_stageX.append(0.0)
                            self.scene_stageY.append(0.0)
                    if not isinstance(allscenes, list):
                        self.scene_stageX.append(0.0)
                        self.scene_stageY.append(0.0)

                # count the number of different wells
                self.number_wells = len(self.well_counter.keys())

        else:
            print("No Scene or Well information found.")


class CziAddMetaData:
    def __init__(self, filename: str) -> None:

        # get metadata dictionary using pylibCZIrw
        with pyczi.open_czi(filename) as czidoc:
            md_dict = czidoc.metadata

        try:
            self.experiment = md_dict["ImageDocument"]["Metadata"]["Experiment"]
        except (KeyError, TypeError) as e:
            print("Key not found :", e)
            self.experiment = None

        try:
            self.hardwaresetting = md_dict["ImageDocument"]["Metadata"]["HardwareSetting"]
        except (KeyError, TypeError) as e:
            print("Key not found :", e)
            self.hardwaresetting = None

        try:
            self.customattributes = md_dict["ImageDocument"]["Metadata"]["CustomAttributes"]
        except (KeyError, TypeError) as e:
            print("Key not found :", e)
            self.customattributes = None

        try:
            self.displaysetting = md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]
        except (KeyError, TypeError) as e:
            print("Key not found :", e)
            self.displaysetting = None

        try:
            self.layers = md_dict["ImageDocument"]["Metadata"]["Layers"]
        except (KeyError, TypeError) as e:
            print("Key not found :", e)
            self.layers = None


class CziScene:
    def __init__(self, filename: str, sceneindex: int) -> None:

        # get metadata dictionary using pylibCZIrw
        with pyczi.open_czi(filename) as czidoc:
            md_dict = czidoc.metadata

            self.bbox = czidoc.scenes_bounding_rectangle[sceneindex]
            self.xstart = self.bbox.x
            self.ystart = self.bbox.y
            self.width = self.bbox.w
            self.height = self.bbox.h
            self.index = sceneindex

        # TODO : And scene dimensions to CziScene class


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


def writexml(filename: str, xmlsuffix: str = '_CZI_MetaData.xml') -> str:
    """Write XML information of CZI to disk

    :param filename: CZI image filename
    :type filename: str
    :param xmlsuffix: suffix for the XML file that will be created, defaults to '_CZI_MetaData.xml'
    :type xmlsuffix: str, optional
    :return: filename of the XML file
    :rtype: str
    """

    # get the raw metadata as a XML or dictionary
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
               'ChannelShortNames': metadata.channelinfo.shortnames,
               'bbox_all_scenes': metadata.bbox.all_scenes,
               'WellArrayNames': metadata.sample.well_array_names,
               'WellIndicies': metadata.sample.well_indices,
               'WellPositionNames': metadata.sample.well_position_names,
               'WellRowID': metadata.sample.well_rowID,
               'WellColumnID': metadata.sample.well_colID,
               'WellCounter': metadata.sample.well_counter,
               'SceneCenterStageX': metadata.sample.scene_stageX,
               'SceneCenterStageY': metadata.sample.scene_stageX
               }

    # check fro extra entries when reading mosaic file with a scale factor
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
