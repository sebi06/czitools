# -*- coding: utf-8 -*-

#################################################################
# File        : misc.py
# Author      : sebi06
#
# Disclaimer: The code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

import os
import zarr
import pandas as pd
import dask.array as da
import numpy as np
import time
from pathlib import Path
import dateutil.parser as dt
from tqdm.contrib.itertools import product
from typing import Dict, Tuple, Any, Union, Annotated, Optional
import validators
from aicspylibczi import CziFile
from czitools.metadata_tools.helper import ValueRange
from czitools.utils import logging_tools
import requests
import tracemalloc

logger = logging_tools.set_logging()


def slicedim(array: Union[np.ndarray, da.Array, zarr.Array], dimindex: int, posdim: int) -> np.ndarray:
    """Slice out a specific dimension without (!) dropping the dimension
    of the array to conserve the dimorder string
    This works for Numpy.Array, Dask and ZARR.

    Example:
        array.shape = (1, 3, 2, 5, 170, 240) and dim_order is STCZYX.
        Index for C inside array is 2.
        Task: Cut out the fist channel = 0.
        channel = slicedim(array, 0, 2).
        The resulting channel.shape = (1, 3, 1, 5, 170, 240)

    Args:
        array: Input array.
        dimindex: Index of the slice dimension to be kept.
        posdim: Position of the dimension to be sliced.

    Returns:
        Sliced array.

    """

    idl_all = [slice(None, None, None)] * (len(array.shape) - 2)
    idl_all[posdim] = slice(dimindex, dimindex + 1, None)
    array_sliced = array[tuple(idl_all)]

    return array_sliced


def calc_scaling(
    data: Union[np.ndarray, da.array],
    corr_min: float = 1.0,
    offset_min: int = 0,
    corr_max: float = 0.85,
    offset_max: int = 0,
) -> Tuple[int, int]:
    """Calculate the scaling for better display

    Args:
        data (Union[numpy.ndarray, dask.array, zarr.array]): Array to calculate scaling.
        corr_min (float, optional): Correction factor for minimum value. Defaults to 1.0.
        offset_min (int, optional): Offset for minimum value. Defaults to 0.
        corr_max (float, optional): Correction factor for maximum value. Defaults to 0.85.
        offset_max (int, optional): Offset for maximum value. Defaults to 0.

    Returns:
        Tuple[int, int]: Tuple with minimum value and maximum value.

    Raises:
        ValueError: If the data array is not of type numpy.ndarray, dask.array or zarr.array.

    """

    start = time.time()

    # get min-max values for initial scaling
    if isinstance(data, zarr.Array):
        minvalue, maxvalue = np.min(data), np.max(data)
    elif isinstance(data, da.Array):
        # compute only once since this is faster
        minvalue, maxvalue = da.compute(data.min(), data.max())
    else:
        minvalue, maxvalue = np.min(data), np.max(data)

    end = time.time()

    minvalue = int(np.round((minvalue + offset_min) * corr_min, 0))
    maxvalue = int(np.round((maxvalue + offset_max) * corr_max, 0))

    logger.info(f"Scaling: {minvalue}, {maxvalue}")
    logger.info(f"Calculation of Min-Max [s] : {end - start}")

    return minvalue, maxvalue


def md2dataframe(md_dict: Dict, paramcol: str = "Parameter", keycol: str = "Value") -> pd.DataFrame:
    """Converts the given metadata_tools dictionary to a Pandas DataFrame.

    Args:
        md_dict (dict): A dictionary containing metadata_tools.
        paramcol (str, optional): The name of the column for metadata_tools parameters. Defaults to "Parameter".
        keycol (str, optional): The name of the column for metadata_tools values. Defaults to "Value".

    Returns:
        pd.DataFrame: A Pandas DataFrame containing all the metadata_tools.
    """
    mdframe = pd.DataFrame(columns=[paramcol, keycol])

    for k in md_dict.keys():
        d = {"Parameter": k, "Value": md_dict[k]}
        df = pd.DataFrame([d], index=[0])
        mdframe = pd.concat([mdframe, df], ignore_index=True)

    return mdframe


def sort_dict_by_key(unsorted_dict: Dict) -> Dict:
    """Sort a dictionary by key names

    Args:
        unsorted_dict: the unsorted dictionary where the keys should be sorted

    Returns:
        Dictionary with keys sorted by name
    """

    sorted_keys = sorted(unsorted_dict.keys(), key=lambda x: x.lower())
    sorted_dict = {}
    for key in sorted_keys:
        sorted_dict.update({key: unsorted_dict[key]})

    return sorted_dict


def addzeros(number: int) -> str:
    """Convert a number into a string and add leading zeros.
    Typically used to construct filenames with equal lengths.

    :param number: the number
    :type number: int
    :return: zerostring - string with leading zeros
    :rtype: str
    """

    zerostring = None

    if number < 10:
        zerostring = "0000" + str(number)
    if 10 <= number < 100:
        zerostring = "000" + str(number)
    if 100 <= number < 1000:
        zerostring = "00" + str(number)
    if 1000 <= number < 10000:
        zerostring = "0" + str(number)

    return zerostring


def get_fname_woext(filepath: Union[str, os.PathLike[str]]) -> str:
    """
    Extracts the filename without its extension from a given file path.
    It also works for extensions like myfile.abc.xyz
    The output will be: myfile

    Args:
        filepath (Union[str, os.PathLike[str]]): The path to the file.
    Returns:
        str: The filename without its extension.
    """

    # create empty string
    real_extension = ""

    # get all part of the file extension
    sufs = Path(filepath).suffixes
    for s in sufs:
        real_extension = real_extension + s

    # remove real extension from filepath
    filepath_woext = filepath.replace(real_extension, "")

    return filepath_woext


def check_dimsize(mdata_entry: Union[Any, None], set2value: Any = 1) -> Union[Any, None]:
    """Check the entries for None.

    Args:
        mdata_entry: The entry to be checked.
        set2value: The value to replace None.

    Returns:
        Union of Any and None
    """

    if mdata_entry is None:
        return set2value
    if mdata_entry is not None:
        return mdata_entry


# def get_planetable(
#     czifile: Union[str, os.PathLike[str]],
#     norm_time: Optional[bool] = True,
#     save_table: Optional[bool] = False,
#     table_separator: Optional[str] = ";",
#     table_index: Optional[bool] = True,
#     planes: Optional[Dict[str, int]] = None,
# ) -> Tuple[pd.DataFrame, Optional[str]]:
#     """
#     Extracts the plane table from the individual subblocks of a CZI file.

#     Args:
#         czifile (Union[str, os.PathLike[str]]): The path to the CZI image file.
#         norm_time (Optional[bool]): Whether to normalize the timestamps. Defaults to True.
#         save_table (Optional[bool]): Whether to save the plane table to a CSV file. Defaults to False.
#         table_separator (Optional[str]): The separator for the CSV file. Defaults to ";".
#         table_index (Optional[bool]): Whether to include the index in the CSV file. Defaults to True.
#         planes (Optional[Dict[str, int]]): Optional dictionary for filtering planes by dimensions.
#             Valid keys: 'scene', 'tile', 'time', 'channel', 'zplane'

#     Returns:
#         Tuple[pd.DataFrame, Optional[str]]: A tuple containing:
#             - A Pandas DataFrame representing the plane table.
#             - The file path of the saved CSV file, if `save_table` is True; otherwise, None.
#     """

#     if isinstance(czifile, Path):
#         # convert to string
#         czifile = str(czifile)

#     if validators.url(czifile):
#         logger.warning("Reading PlaneTable from CZI via a link is not supported yet.")
#         return pd.DataFrame(), None

#     # initialize the plane table
#     df_czi = pd.DataFrame(
#         columns=[
#             "Subblock",
#             "S",
#             "M",
#             "T",
#             "C",
#             "Z",
#             "X[micron]",
#             "Y[micron]",
#             "Z[micron]",
#             "Time[s]",
#             "xstart",
#             "ystart",
#             "width",
#             "height",
#         ]
#     )

#     # define subblock counter
#     sbcount = -1

#     try:
#         aicsczi = CziFile(czifile)
#         dims = aicsczi.get_dims_shape()
#     except Exception as e:
#         logger.error(f"Failed to read CZI file: {e}")
#         return pd.DataFrame(), None

#     # has_s = False
#     # has_m = False
#     # has_t = False
#     # has_c = False
#     # has_z = False

#     # if "S" in dims[0].keys():
#     #     size_s = dims[0]["S"][1]
#     #     has_s = True
#     # else:
#     #     size_s = 1

#     # if "M" in dims[0].keys():
#     #     size_m = dims[0]["M"][1]
#     #     has_m = True
#     # else:
#     #     size_m = 1

#     # if "T" in dims[0].keys():
#     #     size_t = dims[0]["T"][1]
#     #     has_t = True
#     # else:
#     #     size_t = 1

#     # if "C" in dims[0].keys():
#     #     size_c = dims[0]["C"][1]
#     #     has_c = True
#     # else:
#     #     size_c = 1

#     # if "Z" in dims[0].keys():
#     #     size_z = dims[0]["Z"][1]
#     #     has_z = True
#     # else:
#     #     size_z = 1

#     # Get dimension sizes and check which dimensions are present in the CZI file
#     # This replaces the repetitive if/else blocks with a more concise approach
#     dimensions = ["S", "M", "T", "C", "Z"]
#     dim_info = {}

#     # Extract dimension information from the CZI file's dimension structure
#     for dim in dimensions:
#         if dim in dims[0].keys():
#             # If dimension exists, store its size and mark as present
#             dim_info[dim] = {"size": dims[0][dim][1], "present": True}
#         else:
#             # If dimension doesn't exist, set default size of 1 and mark as absent
#             dim_info[dim] = {"size": 1, "present": False}

#     # Unpack the dimension information into individual variables for backward compatibility
#     size_s, has_s = dim_info["S"]["size"], dim_info["S"]["present"]
#     size_m, has_m = dim_info["M"]["size"], dim_info["M"]["present"]
#     size_t, has_t = dim_info["T"]["size"], dim_info["T"]["present"]
#     size_c, has_c = dim_info["C"]["size"], dim_info["C"]["present"]
#     size_z, has_z = dim_info["Z"]["size"], dim_info["Z"]["present"]

#     # cast data types
#     df_czi = df_czi.astype(
#         {
#             "Subblock": "int32",
#             "S": "int32",
#             "M": "int32",
#             "T": "int32",
#             "C": "int32",
#             "Z": "int32",
#             "X[micron]": "float",
#             "Y[micron]": "float",
#             "Z[micron]": "float",
#             "Time[s]": "float",
#             "xstart": "int32",
#             "ystart": "int32",
#             "width": "int32",
#             "height": "int32",
#         },
#         copy=False,
#         errors="ignore",
#     )

#     # check for scenes argument
#     if planes is not None and "scene" in planes:
#         s_start = planes["scene"]
#         s_end = planes["scene"] + 1
#     else:
#         s_start = 0
#         s_end = size_s

#     # check for tile argument
#     if planes is not None and "tile" in planes:
#         m_start = planes["tile"]
#         m_end = planes["tile"] + 1
#     else:
#         m_start = 0
#         m_end = size_m

#     # check for time argument
#     if planes is not None and "time" in planes:
#         t_start = planes["time"]
#         t_end = planes["time"] + 1
#     else:
#         t_start = 0
#         t_end = size_t

#     # check for channel argument
#     if planes is not None and "channel" in planes:
#         c_start = planes["channel"]
#         c_end = planes["channel"] + 1
#     else:
#         c_start = 0
#         c_end = size_c

#     # check for zplane argument
#     if planes is not None and "zplane" in planes:
#         z_start = planes["zplane"]
#         z_end = planes["zplane"] + 1
#     else:
#         z_start = 0
#         z_end = size_z

#     for s, m, t, c, z in product(
#         enumerate(range(s_start, s_end)),
#         enumerate(range(m_start, m_end)),
#         enumerate(range(t_start, t_end)),
#         enumerate(range(c_start, c_end)),
#         enumerate(range(z_start, z_end)),
#         desc="Reading sublocks planes",
#         unit=" 2Dplanes",
#     ):
#         sbcount += 1

#         args = {"S": s[1], "M": m[1], "T": t[1], "Z": z[1], "C": c[1]}

#         # Remove dimensions from args dictionary if they are not present in the CZI file
#         # This ensures we only pass valid dimension arguments to the CZI reading functions
#         dimensions_to_check = [("T", has_t), ("Z", has_z), ("S", has_s), ("M", has_m), ("C", has_c)]

#         for dim_name, is_present in dimensions_to_check:
#             if not is_present:
#                 args.pop(dim_name, None)  # Use pop with default to avoid KeyError

#         # read information from subblock and bounding box
#         if has_m:
#             # get x, y, width and height for a specific tile
#             bbox = aicsczi.get_mosaic_tile_bounding_box(**args)
#             sb = aicsczi.read_subblock_metadata(unified_xml=True, B=0, S=s[1], M=m[1], T=t[1], Z=z[1], C=c[1])
#         elif not has_m:
#             bbox = aicsczi.get_tile_bounding_box(**args)
#             sb = aicsczi.read_subblock_metadata(unified_xml=True, B=0, S=s[1], T=t[1], Z=z[1], C=c[1])

#         # get information from subblock
#         timestamp, xpos, ypos, zpos = _getsbinfo(sb)

#         # assemble data for a single plane
#         plane = [
#             {
#                 "Subblock": sbcount,
#                 "S": s[1],
#                 "M": m[1],
#                 "T": t[1],
#                 "C": c[1],
#                 "Z": z[1],
#                 "X[micron]": xpos,
#                 "Y[micron]": ypos,
#                 "Z[micron]": zpos,
#                 "Time[s]": timestamp,
#                 "xstart": bbox.x,
#                 "ystart": bbox.y,
#                 "width": bbox.w,
#                 "height": bbox.h,
#             }
#         ]

#         # concatenate the plane data to the dataframe
#         df_czi = pd.concat(
#             [df_czi if not df_czi.empty else None, pd.DataFrame(plane)],
#             ignore_index=True,
#         )

#     # normalize time stamps
#     if norm_time:
#         df_czi = norm_columns(df_czi, colname="Time[s]", mode="min")

#     if save_table:
#         try:
#             planetable_savepath = save_planetable(df_czi, czifile, separator=table_separator, index=table_index)
#             logger.info(f"Planetable saved successfully at: {planetable_savepath}")
#             return df_czi, planetable_savepath
#         except Exception as e:
#             logger.error(f"Failed to save planetable: {e}")
#             return df_czi, None

#     return df_czi, None


# def get_planetable(
#     czifile: Union[str, os.PathLike[str]],
#     norm_time: Optional[bool] = True,
#     save_table: Optional[bool] = False,
#     table_separator: Optional[str] = ";",
#     table_index: Optional[bool] = True,
#     planes: Optional[Dict[str, int]] = None,
# ) -> Tuple[pd.DataFrame, Optional[str]]:
#     """
#     Extracts the plane table from the individual subblocks of a CZI file.

#     Args:
#         czifile (Union[str, os.PathLike[str]]): The path to the CZI image file.
#         norm_time (Optional[bool]): Whether to normalize the timestamps. Defaults to True.
#         save_table (Optional[bool]): Whether to save the plane table to a CSV file. Defaults to False.
#         table_separator (Optional[str]): The separator for the CSV file. Defaults to ";".
#         table_index (Optional[bool]): Whether to include the index in the CSV file. Defaults to True.
#         planes (Optional[Dict[str, int]]): Optional dictionary for filtering planes by dimensions.
#             Valid keys: 'scene', 'tile', 'time', 'channel', 'zplane'

#     Returns:
#         Tuple[pd.DataFrame, Optional[str]]: A tuple containing:
#             - A Pandas DataFrame representing the plane table.
#             - The file path of the saved CSV file, if `save_table` is True; otherwise, None.
#     """

#     # Convert Path object to string if necessary
#     if isinstance(czifile, Path):
#         czifile = str(czifile)

#     # Check if input is URL - not supported for plane table extraction
#     if validators.url(czifile):
#         logger.warning("Reading PlaneTable from CZI via a link is not supported yet.")
#         return pd.DataFrame(), None

#     # Initialize the plane table with predefined columns and data types
#     df_czi = _initialize_planetable_dataframe()

#     # Read CZI file and extract dimension information
#     try:
#         aicsczi = CziFile(czifile)
#         dims = aicsczi.get_dims_shape()
#     except Exception as e:
#         logger.error(f"Failed to read CZI file: {e}")
#         return pd.DataFrame(), None

#     # Extract dimension information in a more organized way
#     dim_info = _extract_dimension_info(dims)

#     # Calculate iteration ranges based on available dimensions and user-specified planes
#     iteration_ranges = _calculate_iteration_ranges(dim_info, planes)

#     # Process each subblock and extract plane information
#     df_czi = _process_subblocks(aicsczi, dim_info, iteration_ranges, df_czi)

#     # Normalize time stamps if requested
#     if norm_time:
#         df_czi = norm_columns(df_czi, colname="Time[s]", mode="min")

#     # Save table to CSV if requested
#     if save_table:
#         return _save_planetable_if_requested(df_czi, czifile, table_separator, table_index)

#     return df_czi, None


# def _getsbinfo(subblock: Any) -> Tuple[float, float, float, float]:
#     """
#     Extracts metadata information from a given subblock element.
#     This function retrieves the acquisition timestamp, stage X position,
#     stage Y position, and focus position from the provided subblock.
#     If any of the elements are missing, their corresponding values are
#     set to 0.0.
#     Args:
#         subblock (Any): The subblock element containing metadata information.
#     Returns:
#         Tuple[float, float, float, float]: A tuple containing:
#             - timestamp (float): The acquisition time in seconds since the epoch.
#             - xpos (float): The stage X position.
#             - ypos (float): The stage Y position.
#             - zpos (float): The focus position.
#     """
#     try:
#         time = subblock.findall(".//AcquisitionTime")[0].text
#         timestamp = dt.parse(time).timestamp()
#     except IndexError as e:
#         timestamp = 0.0

#     try:
#         xpos = np.double(subblock.findall(".//StageXPosition")[0].text)
#     except IndexError as e:
#         xpos = 0.0

#     try:
#         ypos = np.double(subblock.findall(".//StageYPosition")[0].text)
#     except IndexError as e:
#         ypos = 0.0

#     try:
#         zpos = np.double(subblock.findall(".//FocusPosition")[0].text)
#     except IndexError as e:
#         zpos = 0.0

#     return timestamp, xpos, ypos, zpos


# def norm_columns(df: pd.DataFrame, colname: str = "Time [s]", mode: str = "min") -> pd.DataFrame:
#     """Normalize a specific column inside a Pandas dataframe
#     Args:
#         df: DataFrame
#         colname: Name of the column to be normalized, defaults to 'Time [s]'
#         mode: Mode of Normalization, defaults to 'min'

#     Returns:
#         Dataframe with normalized columns
#     """

#     # normalize columns according to min or max value
#     if mode == "min":
#         min_value = df[colname].min()
#         df[colname] = np.round((df[colname] - min_value), 3)

#     if mode == "max":
#         max_value = df[colname].max()
#         df[colname] = np.round((df[colname] - max_value), 3)

#     return df


# def filter_planetable(planetable: pd.DataFrame, planes: Optional[Dict[str, int]] = None) -> pd.DataFrame:
#     """
#     Filters the input planetable DataFrame based on specified dimension entries.

#     The function uses the `planes` dictionary to filter the planetable on
#     various dimensions such as scene index, time index, z-plane index, and channel index.
#     If a dimension is not specified in the `planes` dictionary, it will not be used for filtering.
#     Only valid keys are accepted, which are 'scene', 'time', 'zplane', and 'channel'.
#     An error is raised for invalid keys.

#     Args:
#         planetable (pd.DataFrame): The DataFrame to be filtered.
#             It should contain columns "S", "M", "T", "Z", and "C".
#         planes (Optional[Dict[str, int]]): A dictionary specifying the indices to filter on.
#             Valid keys include:
#                 - 'scene': The scene index.
#                 - 'time': The time index.
#                 - 'zplane': The z-plane index.
#                 - 'channel': The channel index.

#     Returns:
#         pd.DataFrame: The filtered planetable.

#     Raises:
#         ValueError: If an invalid key is passed in the `planes` dictionary.

#     Examples:
#         >>> planetable = pd.DataFrame({'S': [0, 1, 1], 'T': [0, 1, 0], 'Z': [0, 1, 1], 'C': [0, 1, 1]})
#         >>> filter_planetable(planetable, planes={'scene': 1, 'time': 0})
#         S  T  Z  C
#         1  0  1  1
#     """

#     # Define valid keys for filtering
#     valid_args = ["scene", "time", "zplane", "channel"]

#     # If no filtering criteria are provided, return the unfiltered planetable
#     if planes is None:
#         return planetable
#     elif planes is not None:
#         # Check for invalid keys in the `planes` dictionary
#         for k, v in planes.items():
#             if k not in valid_args:
#                 raise ValueError(f"Invalid keyword argument: {k}")

#         # Apply filtering for each valid key if it exists in `planes`
#         if "scene" in planes:
#             planetable = planetable[planetable["S"] == planes["scene"]]

#         if "time" in planes:
#             planetable = planetable[planetable["T"] == planes["time"]]

#         if "zplane" in planes:
#             planetable = planetable[planetable["Z"] == planes["zplane"]]

#         if "channel" in planes:
#             planetable = planetable[planetable["C"] == planes["channel"]]

#         # Return the filtered planetable
#         return planetable


# def save_planetable(df: pd.DataFrame, filepath: str, separator: str = ",", index: bool = True) -> str:
#     """Saves a pandas dataframe as a CSV file.

#     Args:
#         df (pd.DataFrame): The dataframe to be saved as CSV.
#         filepath (str): The filepath of the CSV file to be written.
#         separator (str, optional): The separator character for the CSV file. Defaults to ','.
#         index (bool, optional): Whether to include the index in the CSV file. Defaults to True.

#     Returns:
#         str: The filepath of the CSV file that was written.
#     """
#     # Generate the filepath for the planetable CSV.
#     csvfile = os.path.splitext(filepath)[0] + "_planetable.csv"

#     # Write the dataframe to the planetable CSV file.
#     df.to_csv(csvfile, sep=separator, index=index)

#     return csvfile


def expand5d(array: np.ndarray) -> np.ndarray:
    """Expands a multi-dimensional numpy array to 5 dimensions.

    Args:
        array (np.ndarray): The numpy array to be extended to 5 dimensions.

    Returns:
        np.ndarray: The 5-dimensional numpy array.
    """
    # Expand the input array along the third-to-last dimension.
    array = np.expand_dims(array, axis=-3)
    # Expand the result along the fourth-to-last dimension.
    array = np.expand_dims(array, axis=-4)
    # Expand the result along the fifth-to-last dimension.
    array5d = np.expand_dims(array, axis=-5)

    return array5d


# def clean_dict(d: Dict) -> Dict:
#     """
#     Recursively cleans a dictionary by removing keys with values that are None, empty lists, or empty dictionaries.
#     Args:
#         d (Dict): The dictionary to be cleaned.
#     Returns:
#         Dict: A new dictionary with the same structure as the input, but without keys that have None, empty lists, or empty dictionaries as values.
#     """

#     def _clean_dict(d: Dict) -> Dict:
#         # Initialize an empty dictionary to store cleaned key-value pairs
#         cleaned = {}

#         # Iterate over each key-value pair in the dictionary
#         for k, v in d.items():

#             # Check if the value is a dictionary
#             if isinstance(v, dict):

#                 # Recursively clean the nested dictionary
#                 nested = _clean_dict(v)

#                 # If the nested dictionary is not empty
#                 if nested:
#                     # Add the cleaned nested dictionary to the cleaned dictionary
#                     cleaned[k] = nested

#             # Check if the value is an array and ensure it is not empty
#             elif isinstance(v, (np.ndarray, da.Array)):
#                 if v.size > 0:
#                     cleaned[k] = v

#             # Check if the value is not None, an empty list, or an empty dictionary
#             elif v is not None and not isinstance(v, (np.ndarray, da.Array)) and v != [] and v != {}:
#                 cleaned[k] = v

#         return cleaned

#     return _clean_dict(d)  # Call the inner function and return its result


def clean_dict(d: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Recursively cleans a dictionary by removing keys with values that are None, empty lists, empty dictionaries,
    or empty NumPy arrays.

    Args:
        d (Dict): The dictionary to be cleaned.

    Returns:
        Dict: A new dictionary with the same structure as the input, but without keys that have None, empty lists,
        empty dictionaries, or empty NumPy arrays as values.
    """
    if not isinstance(d, dict):
        raise ValueError("Input must be a dictionary.")

    cleaned = {}
    for key, value in d.items():
        # Handle None values
        if value is None:
            continue

        # Handle empty lists
        if isinstance(value, list) and len(value) == 0:
            continue

        # Handle empty dictionaries
        if isinstance(value, dict) and len(value) == 0:
            continue

        # Handle empty NumPy arrays
        if isinstance(value, np.ndarray) and value.size == 0:
            continue

        # Recursively clean nested dictionaries
        if isinstance(value, dict):
            nested_cleaned = clean_dict(value)
            if nested_cleaned:  # Only include if the cleaned nested dictionary is not empty
                cleaned[key] = nested_cleaned
        else:
            cleaned[key] = value

    return cleaned


def download_zip(source_link: str) -> str:

    if not validators.url(source_link):
        logger.warning("Not a valid link.")
        return ""

    import io
    import zipfile

    response = requests.get(source_link, stream=True)
    compressed_data = io.BytesIO(response.content)

    with zipfile.ZipFile(compressed_data, "r") as zip_accessor:
        zip_accessor.extractall("./")

    return compressed_data[:-4]


def check_zoom(zoom: Annotated[float, ValueRange(0.01, 1.0)] = 1.0) -> float:

    # check zoom factor
    if zoom > 1.0:
        logger.warning(f"Zoom factor f{zoom} is not in valid range [0.01 - 1.0]. Using 1.0 instead.")
        zoom = 1.0
    if zoom < 0.01:
        logger.warning(f"Zoom factor f{zoom} is not in valid range [0.01 - 1.0]. Using 0.01 instead.")
        zoom = 0.01

    return zoom


def measure_memory_usage(target_function):
    """
    A decorator that measures and logs the memory usage of the decorated function.
    This decorator uses the `tracemalloc` module to track memory allocations and logs
    the top memory-consuming lines after the function execution.
    Args:
        target_function (function): The function to be decorated.
    Returns:
        function: The wrapped function with memory usage measurement.
    Example:
        @measure_memory_usage
        def my_function():
            # Function implementation
            pass
    """

    def wrapper(*args, **kwargs):
        tracemalloc.start()

        # Call the original function
        result = target_function(*args, **kwargs)

        snapshot = tracemalloc.take_snapshot()

        top_stats = snapshot.statistics("lineno")
        # # Print the top memory-consuming lines
        logger.info(f"Memory usage of {target_function.__name__}:")
        for stat in top_stats[:3]:
            logger.info(stat)

        # top_stats = snapshot.statistics("traceback")
        ## pick the biggest memory block
        # stat = top_stats[0]
        # print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
        # for line in stat.traceback.format():
        #    logger.info(line)

        # Return the result
        return result

    return wrapper


def measure_execution_time(func):
    """
    Decorator that measures the execution time of a function and logs it.
    Args:
        func (callable): The function to be decorated.
    Returns:
        callable: The wrapped function with execution time measurement.
    Example:
        @measure_execution_time
        def my_function():
            # Function implementation
            pass
    The execution time will be logged using the logger with an info level.
    """

    def timed_execution(*args, **kwargs):
        start_timestamp = time.time()
        result = func(*args, **kwargs)
        end_timestamp = time.time()
        execution_duration = end_timestamp - start_timestamp
        logger.info(f"Function: {func.__name__} --> {execution_duration:.2f} [s]")
        return result

    return timed_execution


def is_valid_czi_url(myurl: str) -> Tuple[bool, str]:
    """
    Validates whether a given URL points to a valid .czi file.
    Args:
        myurl (str): The URL to validate.
    Returns:
        tuple:
            - bool: True if the URL is valid and points to a .czi file, False otherwise.
            - str: A message providing details about the validation result.
    Validation Steps:
        1. Checks if the URL has a valid format using a URL validator.
        2. Ensures the URL ends with the '.czi' file extension.
        3. Sends an HTTP HEAD request to verify the file's existence and accessibility.
           - If accessible, optionally checks the Content-Type header for additional validation.
    Raises:
        None: Any exceptions during the HTTP request are caught and included in the return message.
    """
    # Step 1: Validate the URL format
    if not validators.url(myurl):
        return False, "Invalid URL format"

    # Step 3: Verify the file exists by making a HEAD request
    try:
        response = requests.head(myurl, allow_redirects=True)
        if response.status_code == 200:
            # Optionally, check the Content-Type header for additional validation
            content_type = response.headers.get("Content-Type", "")
            if "application/octet-stream" in content_type or "czi" in content_type:
                return True, "Valid .czi file URL"
            else:
                return True, "File exists but Content-Type is not specific to .czi"
        else:
            return False, f"File not accessible, HTTP status code: {response.status_code}"
    except requests.RequestException as e:
        return False, f"Error during HTTP request: {e}"


# def _initialize_planetable_dataframe() -> pd.DataFrame:
#     """
#     Initialize an empty DataFrame with the correct structure and data types for the plane table.

#     Returns:
#         pd.DataFrame: Empty DataFrame with predefined columns and data types.
#     """
#     df_czi = pd.DataFrame(
#         columns=[
#             "Subblock",
#             "S",
#             "M",
#             "T",
#             "C",
#             "Z",
#             "X[micron]",
#             "Y[micron]",
#             "Z[micron]",
#             "Time[s]",
#             "xstart",
#             "ystart",
#             "width",
#             "height",
#         ]
#     )

#     # Set appropriate data types for better memory efficiency and performance
#     df_czi = df_czi.astype(
#         {
#             "Subblock": "int32",
#             "S": "int32",
#             "M": "int32",
#             "T": "int32",
#             "C": "int32",
#             "Z": "int32",
#             "X[micron]": "float",
#             "Y[micron]": "float",
#             "Z[micron]": "float",
#             "Time[s]": "float",
#             "xstart": "int32",
#             "ystart": "int32",
#             "width": "int32",
#             "height": "int32",
#         },
#         copy=False,
#         errors="ignore",
#     )

#     return df_czi


# def _extract_dimension_info(dims: list) -> Dict[str, Dict[str, Union[int, bool]]]:
#     """
#     Extract dimension information from CZI file dimensions in a structured way.

#     Args:
#         dims (list): Dimension information from CZI file.

#     Returns:
#         Dict[str, Dict[str, Union[int, bool]]]: Dictionary containing size and presence info for each dimension.
#     """
#     dimensions = ["S", "M", "T", "C", "Z"]
#     dim_info = {}

#     for dim in dimensions:
#         if dim in dims[0].keys():
#             # Dimension exists in the CZI file
#             dim_info[dim] = {"size": dims[0][dim][1], "present": True}
#         else:
#             # Dimension doesn't exist, set default values
#             dim_info[dim] = {"size": 1, "present": False}

#     logger.info(f"CZI dimensions found: {[dim for dim in dimensions if dim_info[dim]['present']]}")

#     return dim_info


# def _calculate_iteration_ranges(
#     dim_info: Dict[str, Dict[str, Union[int, bool]]], planes: Optional[Dict[str, int]] = None
# ) -> Dict[str, Tuple[int, int]]:
#     """
#     Calculate the iteration ranges for each dimension based on user input and available dimensions.

#     Args:
#         dim_info (Dict): Dimension information with size and presence data.
#         planes (Optional[Dict[str, int]]): User-specified plane constraints.

#     Returns:
#         Dict[str, Tuple[int, int]]: Dictionary with start and end indices for each dimension.
#     """
#     # Mapping between user-friendly names and internal dimension names
#     plane_mapping = {
#         "scene": ("S", "size_s"),
#         "tile": ("M", "size_m"),
#         "time": ("T", "size_t"),
#         "channel": ("C", "size_c"),
#         "zplane": ("Z", "size_z"),
#     }

#     ranges = {}

#     for user_key, (dim_key, _) in plane_mapping.items():
#         max_size = dim_info[dim_key]["size"]

#         if planes is not None and user_key in planes:
#             # User specified a specific index for this dimension
#             start_idx = planes[user_key]
#             end_idx = start_idx + 1

#             # Validate that the specified index is within bounds
#             if start_idx >= max_size:
#                 logger.warning(f"Specified {user_key} index {start_idx} exceeds maximum {max_size-1}. Using maximum.")
#                 start_idx = max_size - 1
#                 end_idx = max_size
#         else:
#             # Use full range for this dimension
#             start_idx = 0
#             end_idx = max_size

#         ranges[dim_key] = (start_idx, end_idx)

#     return ranges


# def _process_subblocks(
#     aicsczi: CziFile,
#     dim_info: Dict[str, Dict[str, Union[int, bool]]],
#     ranges: Dict[str, Tuple[int, int]],
#     df_czi: pd.DataFrame,
# ) -> pd.DataFrame:
#     """
#     Process all subblocks and extract plane information.

#     Args:
#         aicsczi (CziFile): The opened CZI file object.
#         dim_info (Dict): Dimension information.
#         ranges (Dict): Iteration ranges for each dimension.
#         df_czi (pd.DataFrame): The DataFrame to populate.

#     Returns:
#         pd.DataFrame: Populated DataFrame with plane information.
#     """
#     sbcount = -1  # Subblock counter

#     # Extract ranges for cleaner code
#     s_start, s_end = ranges["S"]
#     m_start, m_end = ranges["M"]
#     t_start, t_end = ranges["T"]
#     c_start, c_end = ranges["C"]
#     z_start, z_end = ranges["Z"]

#     # Iterate through all combinations of dimensions
#     for s, m, t, c, z in product(
#         enumerate(range(s_start, s_end)),
#         enumerate(range(m_start, m_end)),
#         enumerate(range(t_start, t_end)),
#         enumerate(range(c_start, c_end)),
#         enumerate(range(z_start, z_end)),
#         desc="Reading subblock planes",
#         unit=" 2Dplanes",
#     ):
#         sbcount += 1

#         # Prepare arguments for CZI file reading, including only present dimensions
#         args = _prepare_czi_args(dim_info, s[1], m[1], t[1], c[1], z[1])

#         # Read bounding box and subblock metadata
#         bbox, sb = _read_subblock_data(aicsczi, dim_info["M"]["present"], args, s[1], m[1], t[1], c[1], z[1])

#         # Extract information from subblock metadata
#         timestamp, xpos, ypos, zpos = _getsbinfo(sb)

#         # Create plane data dictionary
#         plane_data = {
#             "Subblock": sbcount,
#             "S": s[1],
#             "M": m[1],
#             "T": t[1],
#             "C": c[1],
#             "Z": z[1],
#             "X[micron]": xpos,
#             "Y[micron]": ypos,
#             "Z[micron]": zpos,
#             "Time[s]": timestamp,
#             "xstart": bbox.x,
#             "ystart": bbox.y,
#             "width": bbox.w,
#             "height": bbox.h,
#         }

#         # Add plane to DataFrame
#         df_czi = pd.concat([df_czi, pd.DataFrame([plane_data])], ignore_index=True)

#     return df_czi


# def _prepare_czi_args(
#     dim_info: Dict[str, Dict[str, Union[int, bool]]], s: int, m: int, t: int, c: int, z: int
# ) -> Dict[str, int]:
#     """
#     Prepare arguments for CZI file reading, including only dimensions that are present.

#     Args:
#         dim_info (Dict): Dimension information.
#         s, m, t, c, z (int): Dimension indices.

#     Returns:
#         Dict[str, int]: Arguments dictionary with only present dimensions.
#     """
#     args = {"S": s, "M": m, "T": t, "Z": z, "C": c}

#     # Remove dimensions that are not present in the CZI file
#     dimensions_to_check = ["T", "Z", "S", "M", "C"]

#     for dim_name in dimensions_to_check:
#         if not dim_info[dim_name]["present"]:
#             args.pop(dim_name, None)

#     return args


# def _read_subblock_data(
#     aicsczi: CziFile, has_m: bool, args: Dict[str, int], s: int, m: int, t: int, c: int, z: int
# ) -> Tuple[Any, Any]:
#     """
#     Read bounding box and subblock metadata from CZI file.

#     Args:
#         aicsczi (CziFile): The opened CZI file object.
#         has_m (bool): Whether the M dimension (mosaic/tile) is present.
#         args (Dict[str, int]): Arguments for CZI reading functions.
#         s, m, t, c, z (int): Dimension indices.

#     Returns:
#         Tuple[Any, Any]: Bounding box and subblock metadata.
#     """
#     if has_m:
#         # Handle mosaic/tile data
#         bbox = aicsczi.get_mosaic_tile_bounding_box(**args)
#         sb = aicsczi.read_subblock_metadata(unified_xml=True, B=0, S=s, M=m, T=t, Z=z, C=c)
#     else:
#         # Handle regular (non-mosaic) data
#         bbox = aicsczi.get_tile_bounding_box(**args)
#         sb = aicsczi.read_subblock_metadata(unified_xml=True, B=0, S=s, T=t, Z=z, C=c)

#     return bbox, sb


# def _save_planetable_if_requested(
#     df_czi: pd.DataFrame, czifile: str, table_separator: str, table_index: bool
# ) -> Tuple[pd.DataFrame, Optional[str]]:
#     """
#     Save the planetable to CSV if requested.

#     Args:
#         df_czi (pd.DataFrame): The planetable DataFrame.
#         czifile (str): Original CZI file path.
#         table_separator (str): CSV separator.
#         table_index (bool): Whether to include index in CSV.

#     Returns:
#         Tuple[pd.DataFrame, Optional[str]]: DataFrame and path to saved CSV file.
#     """
#     try:
#         planetable_savepath = save_planetable(df_czi, czifile, separator=table_separator, index=table_index)
#         logger.info(f"Planetable saved successfully at: {planetable_savepath}")
#         return df_czi, planetable_savepath
#     except Exception as e:
#         logger.error(f"Failed to save planetable: {e}")
#         return df_czi, None
