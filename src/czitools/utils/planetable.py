# -*- coding: utf-8 -*-

#################################################################
# File        : planetable.py
# Author      : sebi06
#
# Disclaimer: The code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

import os
import pandas as pd
import numpy as np
from pathlib import Path
import dateutil.parser as dt
from tqdm.contrib.itertools import product
from typing import Dict, Tuple, Any, Union, Optional
import validators
from aicspylibczi import CziFile
from czitools.utils import logging_tools

logger = logging_tools.set_logging()


def get_planetable(
    czifile: Union[str, os.PathLike[str]],
    norm_time: Optional[bool] = True,
    save_table: Optional[bool] = False,
    table_separator: Optional[str] = ";",
    table_index: Optional[bool] = True,
    planes: Optional[Dict[str, int]] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Extracts the plane table from the individual subblocks of a CZI file.

    Args:
        czifile (Union[str, os.PathLike[str]]): The path to the CZI image file.
        norm_time (Optional[bool]): Whether to normalize the timestamps. Defaults to True.
        save_table (Optional[bool]): Whether to save the plane table to a CSV file. Defaults to False.
        table_separator (Optional[str]): The separator for the CSV file. Defaults to ";".
        table_index (Optional[bool]): Whether to include the index in the CSV file. Defaults to True.
        planes (Optional[Dict[str, int]]): Optional dictionary for filtering planes by dimensions.
            Valid keys: 'scene', 'tile', 'time', 'channel', 'zplane'

    Returns:
        Tuple[pd.DataFrame, Optional[str]]: A tuple containing:
            - A Pandas DataFrame representing the plane table.
            - The file path of the saved CSV file, if `save_table` is True; otherwise, None.
    """

    # Convert Path object to string if necessary
    if isinstance(czifile, Path):
        czifile = str(czifile)

    # Check if input is URL - not supported for plane table extraction
    if validators.url(czifile):
        logger.warning("Reading PlaneTable from CZI via a link is not supported yet.")
        return pd.DataFrame(), None

    # Initialize the plane table with predefined columns and data types
    df_czi = _initialize_planetable_dataframe()

    # Read CZI file and extract dimension information
    try:
        aicsczi = CziFile(czifile)
        dims = aicsczi.get_dims_shape()
    except Exception as e:
        logger.error(f"Failed to read CZI file: {e}")
        return pd.DataFrame(), None

    # Extract dimension information in a more organized way
    dim_info = _extract_dimension_info(dims)

    # Calculate iteration ranges based on available dimensions and user-specified planes
    iteration_ranges = _calculate_iteration_ranges(dim_info, planes)

    # Process each subblock and extract plane information
    df_czi = _process_subblocks(aicsczi, dim_info, iteration_ranges, df_czi)

    # Normalize time stamps if requested
    if norm_time:
        df_czi = norm_columns(df_czi, colname="Time[s]", mode="min")

    # Save table to CSV if requested
    if save_table:
        return _save_planetable_if_requested(df_czi, czifile, table_separator, table_index)

    return df_czi, None


def _getsbinfo(subblock: Any) -> Tuple[float, float, float, float]:
    """
    Extracts metadata information from a given subblock element.
    This function retrieves the acquisition timestamp, stage X position,
    stage Y position, and focus position from the provided subblock.
    If any of the elements are missing, their corresponding values are
    set to 0.0.
    Args:
        subblock (Any): The subblock element containing metadata information.
    Returns:
        Tuple[float, float, float, float]: A tuple containing:
            - timestamp (float): The acquisition time in seconds since the epoch.
            - xpos (float): The stage X position.
            - ypos (float): The stage Y position.
            - zpos (float): The focus position.
    """
    try:
        time = subblock.findall(".//AcquisitionTime")[0].text
        timestamp = dt.parse(time).timestamp()
    except IndexError:
        timestamp = 0.0

    try:
        xpos = np.double(subblock.findall(".//StageXPosition")[0].text)
    except IndexError:
        xpos = 0.0

    try:
        ypos = np.double(subblock.findall(".//StageYPosition")[0].text)
    except IndexError:
        ypos = 0.0

    try:
        zpos = np.double(subblock.findall(".//FocusPosition")[0].text)
    except IndexError:
        zpos = 0.0

    return timestamp, xpos, ypos, zpos


def norm_columns(df: pd.DataFrame, colname: str = "Time [s]", mode: str = "min") -> pd.DataFrame:
    """Normalize a specific column inside a Pandas dataframe
    Args:
        df: DataFrame
        colname: Name of the column to be normalized, defaults to 'Time [s]'
        mode: Mode of Normalization, defaults to 'min'

    Returns:
        Dataframe with normalized columns
    """

    # normalize columns according to min or max value
    if mode == "min":
        min_value = df[colname].min()
        df[colname] = np.round((df[colname] - min_value), 3)

    if mode == "max":
        max_value = df[colname].max()
        df[colname] = np.round((df[colname] - max_value), 3)

    return df


def filter_planetable(planetable: pd.DataFrame, planes: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """
    Filters the input planetable DataFrame based on specified dimension entries.

    The function uses the `planes` dictionary to filter the planetable on
    various dimensions such as scene index, time index, z-plane index, and channel index.
    If a dimension is not specified in the `planes` dictionary, it will not be used for filtering.
    Only valid keys are accepted, which are 'scene', 'time', 'zplane', and 'channel'.
    An error is raised for invalid keys.

    Args:
        planetable (pd.DataFrame): The DataFrame to be filtered.
            It should contain columns "S", "M", "T", "Z", and "C".
        planes (Optional[Dict[str, int]]): A dictionary specifying the indices to filter on.
            Valid keys include:
                - 'scene': The scene index.
                - 'time': The time index.
                - 'zplane': The z-plane index.
                - 'channel': The channel index.

    Returns:
        pd.DataFrame: The filtered planetable.

    Raises:
        ValueError: If an invalid key is passed in the `planes` dictionary.

    Examples:
        >>> planetable = pd.DataFrame({'S': [0, 1, 1], 'T': [0, 1, 0], 'Z': [0, 1, 1], 'C': [0, 1, 1]})
        >>> filter_planetable(planetable, planes={'scene': 1, 'time': 0})
        S  T  Z  C
        1  0  1  1
    """

    # Define valid keys for filtering
    valid_args = ["scene", "time", "zplane", "channel"]

    # If no filtering criteria are provided, return the unfiltered planetable
    if planes is None:
        return planetable
    elif planes is not None:
        # Check for invalid keys in the `planes` dictionary
        for k, v in planes.items():
            if k not in valid_args:
                raise ValueError(f"Invalid keyword argument: {k}")

        # Apply filtering for each valid key if it exists in `planes`
        if "scene" in planes:
            planetable = planetable[planetable["S"] == planes["scene"]]

        if "time" in planes:
            planetable = planetable[planetable["T"] == planes["time"]]

        if "zplane" in planes:
            planetable = planetable[planetable["Z"] == planes["zplane"]]

        if "channel" in planes:
            planetable = planetable[planetable["C"] == planes["channel"]]

        # Return the filtered planetable
        return planetable


def save_planetable(df: pd.DataFrame, filepath: str, separator: str = ",", index: bool = True) -> str:
    """Saves a pandas dataframe as a CSV file.

    Args:
        df (pd.DataFrame): The dataframe to be saved as CSV.
        filepath (str): The filepath of the CSV file to be written.
        separator (str, optional): The separator character for the CSV file. Defaults to ','.
        index (bool, optional): Whether to include the index in the CSV file. Defaults to True.

    Returns:
        str: The filepath of the CSV file that was written.
    """
    # Generate the filepath for the planetable CSV.
    csvfile = os.path.splitext(filepath)[0] + "_planetable.csv"

    # Write the dataframe to the planetable CSV file.
    df.to_csv(csvfile, sep=separator, index=index)

    return csvfile


def _initialize_planetable_dataframe() -> pd.DataFrame:
    """
    Initialize an empty DataFrame with the correct structure and data types for the plane table.

    Returns:
        pd.DataFrame: Empty DataFrame with predefined columns and data types.
    """
    df_czi = pd.DataFrame(
        columns=[
            "Subblock",
            "S",
            "M",
            "T",
            "C",
            "Z",
            "X[micron]",
            "Y[micron]",
            "Z[micron]",
            "Time[s]",
            "xstart",
            "ystart",
            "width",
            "height",
        ]
    )

    # Set appropriate data types for better memory efficiency and performance
    df_czi = df_czi.astype(
        {
            "Subblock": "int32",
            "S": "int32",
            "M": "int32",
            "T": "int32",
            "C": "int32",
            "Z": "int32",
            "X[micron]": "float",
            "Y[micron]": "float",
            "Z[micron]": "float",
            "Time[s]": "float",
            "xstart": "int32",
            "ystart": "int32",
            "width": "int32",
            "height": "int32",
        },
        copy=False,
        errors="ignore",
    )

    return df_czi


def _extract_dimension_info(dims: list) -> Dict[str, Dict[str, Union[int, bool]]]:
    """
    Extract dimension information from CZI file dimensions in a structured way.

    Args:
        dims (list): Dimension information from CZI file.

    Returns:
        Dict[str, Dict[str, Union[int, bool]]]: Dictionary containing size and presence info for each dimension.
    """
    dimensions = ["S", "M", "T", "C", "Z"]
    dim_info = {}

    for dim in dimensions:
        if dim in dims[0].keys():
            # Dimension exists in the CZI file
            dim_info[dim] = {"size": dims[0][dim][1], "present": True}
        else:
            # Dimension doesn't exist, set default values
            dim_info[dim] = {"size": 1, "present": False}

    logger.info(f"CZI dimensions found: {[dim for dim in dimensions if dim_info[dim]['present']]}")

    return dim_info


def _calculate_iteration_ranges(
    dim_info: Dict[str, Dict[str, Union[int, bool]]], planes: Optional[Dict[str, int]] = None
) -> Dict[str, Tuple[int, int]]:
    """
    Calculate the iteration ranges for each dimension based on user input and available dimensions.

    Args:
        dim_info (Dict): Dimension information with size and presence data.
        planes (Optional[Dict[str, int]]): User-specified plane constraints.

    Returns:
        Dict[str, Tuple[int, int]]: Dictionary with start and end indices for each dimension.
    """
    # Mapping between user-friendly names and internal dimension names
    plane_mapping = {
        "scene": ("S", "size_s"),
        "tile": ("M", "size_m"),
        "time": ("T", "size_t"),
        "channel": ("C", "size_c"),
        "zplane": ("Z", "size_z"),
    }

    ranges = {}

    for user_key, (dim_key, _) in plane_mapping.items():
        max_size = dim_info[dim_key]["size"]

        if planes is not None and user_key in planes:
            # User specified a specific index for this dimension
            start_idx = planes[user_key]
            end_idx = start_idx + 1

            # Validate that the specified index is within bounds
            if start_idx >= max_size:
                logger.warning(f"Specified {user_key} index {start_idx} exceeds maximum {max_size-1}. Using maximum.")
                start_idx = max_size - 1
                end_idx = max_size
        else:
            # Use full range for this dimension
            start_idx = 0
            end_idx = max_size

        ranges[dim_key] = (start_idx, end_idx)

    return ranges


def _process_subblocks(
    aicsczi: CziFile,
    dim_info: Dict[str, Dict[str, Union[int, bool]]],
    ranges: Dict[str, Tuple[int, int]],
    df_czi: pd.DataFrame,
) -> pd.DataFrame:
    """
    Process all subblocks and extract plane information.

    Args:
        aicsczi (CziFile): The opened CZI file object.
        dim_info (Dict): Dimension information.
        ranges (Dict): Iteration ranges for each dimension.
        df_czi (pd.DataFrame): The DataFrame to populate.

    Returns:
        pd.DataFrame: Populated DataFrame with plane information.
    """
    sbcount = -1  # Subblock counter

    # Extract ranges for cleaner code
    s_start, s_end = ranges["S"]
    m_start, m_end = ranges["M"]
    t_start, t_end = ranges["T"]
    c_start, c_end = ranges["C"]
    z_start, z_end = ranges["Z"]

    # Iterate through all combinations of dimensions
    for s, m, t, c, z in product(
        enumerate(range(s_start, s_end)),
        enumerate(range(m_start, m_end)),
        enumerate(range(t_start, t_end)),
        enumerate(range(c_start, c_end)),
        enumerate(range(z_start, z_end)),
        desc="Reading subblock planes",
        unit=" 2Dplanes",
    ):
        sbcount += 1

        # Prepare arguments for CZI file reading, including only present dimensions
        args = _prepare_czi_args(dim_info, s[1], m[1], t[1], c[1], z[1])

        # Read bounding box and subblock metadata
        bbox, sb = _read_subblock_data(aicsczi, dim_info["M"]["present"], args, s[1], m[1], t[1], c[1], z[1])

        # Extract information from subblock metadata
        timestamp, xpos, ypos, zpos = _getsbinfo(sb)

        # Create plane data dictionary
        plane_data = {
            "Subblock": sbcount,
            "S": s[1],
            "M": m[1],
            "T": t[1],
            "C": c[1],
            "Z": z[1],
            "X[micron]": xpos,
            "Y[micron]": ypos,
            "Z[micron]": zpos,
            "Time[s]": timestamp,
            "xstart": bbox.x,
            "ystart": bbox.y,
            "width": bbox.w,
            "height": bbox.h,
        }

        # Add plane to DataFrame
        df_czi = pd.concat([df_czi, pd.DataFrame([plane_data])], ignore_index=True)

    return df_czi


def _prepare_czi_args(
    dim_info: Dict[str, Dict[str, Union[int, bool]]], s: int, m: int, t: int, c: int, z: int
) -> Dict[str, int]:
    """
    Prepare arguments for CZI file reading, including only dimensions that are present.

    Args:
        dim_info (Dict): Dimension information.
        s, m, t, c, z (int): Dimension indices.

    Returns:
        Dict[str, int]: Arguments dictionary with only present dimensions.
    """
    args = {"S": s, "M": m, "T": t, "Z": z, "C": c}

    # Remove dimensions that are not present in the CZI file
    dimensions_to_check = ["T", "Z", "S", "M", "C"]

    for dim_name in dimensions_to_check:
        if not dim_info[dim_name]["present"]:
            args.pop(dim_name, None)

    return args


def _read_subblock_data(
    aicsczi: CziFile, has_m: bool, args: Dict[str, int], s: int, m: int, t: int, c: int, z: int
) -> Tuple[Any, Any]:
    """
    Read bounding box and subblock metadata from CZI file.

    Args:
        aicsczi (CziFile): The opened CZI file object.
        has_m (bool): Whether the M dimension (mosaic/tile) is present.
        args (Dict[str, int]): Arguments for CZI reading functions.
        s, m, t, c, z (int): Dimension indices.

    Returns:
        Tuple[Any, Any]: Bounding box and subblock metadata.
    """
    if has_m:
        # Handle mosaic/tile data
        bbox = aicsczi.get_mosaic_tile_bounding_box(**args)
        sb = aicsczi.read_subblock_metadata(unified_xml=True, B=0, S=s, M=m, T=t, Z=z, C=c)
    else:
        # Handle regular (non-mosaic) data
        bbox = aicsczi.get_tile_bounding_box(**args)
        sb = aicsczi.read_subblock_metadata(unified_xml=True, B=0, S=s, T=t, Z=z, C=c)

    return bbox, sb


def _save_planetable_if_requested(
    df_czi: pd.DataFrame, czifile: str, table_separator: str, table_index: bool
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Save the planetable to CSV if requested.

    Args:
        df_czi (pd.DataFrame): The planetable DataFrame.
        czifile (str): Original CZI file path.
        table_separator (str): CSV separator.
        table_index (bool): Whether to include index in CSV.

    Returns:
        Tuple[pd.DataFrame, Optional[str]]: DataFrame and path to saved CSV file.
    """
    try:
        planetable_savepath = save_planetable(df_czi, czifile, separator=table_separator, index=table_index)
        logger.info(f"Planetable saved successfully at: {planetable_savepath}")
        return df_czi, planetable_savepath
    except Exception as e:
        logger.error(f"Failed to save planetable: {e}")
        return df_czi, None
