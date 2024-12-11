# -*- coding: utf-8 -*-

#################################################################
# File        : pixels.py
# Author      : sebi06
#
# Disclaimer: The code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from pylibCZIrw import czi as pyczi
import numpy as np
from typing import List, Dict, Tuple, Optional, Union


def check_if_rgb(pixeltypes: Dict) -> Tuple[Dict[int, bool], bool]:
    """
    Check if the pixel types are RGB and if they are consistent.
    Args:
        pixeltypes (Dict): A dictionary where keys are pixel identifiers and values are pixel type strings.
    Returns:
        Tuple[Dict, bool]: A tuple containing:
            - A dictionary where keys are pixel identifiers and values are booleans indicating if the pixel type is RGB.
            - A boolean flag indicating if all pixel types in the input dictionary are the same.
    """

    is_rgb = {}

    for k, v in pixeltypes.items():
        if "Bgr" in v:
            is_rgb[k] = True
        else:
            is_rgb[k] = False

    # flag to check if all elements are same
    is_consistent = True

    # extracting value to compare
    test_val = list(pixeltypes.values())[0]

    is_consistent = all(value == test_val for value in pixeltypes.values())

    return is_rgb, is_consistent


def get_dtype_fromstring(
    pixeltype: str,
) -> Tuple[Optional[np.dtype], Optional[int]]:
    """
    Determine the numpy data type and maximum value based on the given pixel type string.
    Parameters:
    pixeltype (str): The pixel type as a string. Possible values include "gray16", "Gray16",
                     "gray8", "Gray8", "bgr48", "Bgr48", "bgr24", "Bgr24", "bgr96float", "Bgr96Float".
    Returns:
    Tuple[Optional[np.dtype], Optional[int]]: A tuple containing the numpy data type and the maximum value
                                              for the given pixel type. If the pixel type is not recognized,
                                              both elements of the tuple will be None.
    """
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


def get_dimorder(dim_string: str) -> Tuple[Dict, List, int]:
    """
    Extracts the order and indices of dimensions from a given dimension string.
    Args:
        dim_string (str): A string representing the dimensions.
    Returns:
        Tuple[Dict, List, int]: A tuple containing:
            - A dictionary with dimensions as keys and their indices in the string as values.
            - A list of indices corresponding to the dimensions in the order they appear in the predefined list.
            - An integer representing the number of valid dimensions found in the string.
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
