# -*- coding: utf-8 -*-

#################################################################
# File        : planetable.py
# Author      : sebi06
#
# Disclaimer: The code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

"""Plane-table (stage position) utilities for CZI files.

Provides functions to extract per-plane stage-position data from CZI
metadata and return it as a `pandas.DataFrame`.
"""
import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from pathlib import Path
import dateutil.parser as dt
from typing import Dict, Tuple, Any, Union, Optional
import validators
import czifile as czifile_module
from czitools.utils import logging_tools

# Import progressbar2
try:
    import progressbar

    HAS_PROGRESSBAR = True
except ImportError:
    HAS_PROGRESSBAR = False

logger = logging_tools.set_logging()


def _get_dim_start(de: Any, dim: str, default: int = 0) -> int:
    """Return the start index of a dimension from a czifile directory_entry, or default if absent."""
    if dim in de.dims:
        return de.start[de.dims.index(dim)]
    return default


def _get_bbox(de: Any) -> Tuple[int, int, int, int]:
    """Extract (xstart, ystart, width, height) from a czifile directory_entry."""
    xstart = _get_dim_start(de, "X")
    ystart = _get_dim_start(de, "Y")
    width = de.shape[de.dims.index("X")] if "X" in de.dims else 0
    height = de.shape[de.dims.index("Y")] if "Y" in de.dims else 0
    return xstart, ystart, width, height


def get_planetable(
    czipath: Union[str, os.PathLike[str]],
    norm_time: Optional[bool] = True,
    save_table: Optional[bool] = False,
    table_separator: Optional[str] = ";",
    table_index: Optional[bool] = True,
    planes: Optional[Dict[str, int]] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Extracts the plane table from the individual subblocks of a CZI file.

    Iterates all subblocks directly from the CZI file, extracting per-plane
    position, timing, and bounding-box data.  Mosaic tile indices (M) are
    derived by counting subblocks sharing the same (S, T, C, Z) combination.

    Args:
        czipath (Union[str, os.PathLike[str]]): The path to the CZI image file.
        norm_time (Optional[bool]): Whether to normalize the timestamps. Defaults to True.
        save_table (Optional[bool]): Whether to save the plane table to a CSV file. Defaults to False.
        table_separator (Optional[str]): The separator for the CSV file. Defaults to ";".
        table_index (Optional[bool]): Whether to include the index in the CSV file. Defaults to True.
        planes (Optional[Dict[str, int]]): Optional dictionary for filtering planes by dimensions.
            Valid keys: 'scene', 'tile', 'time', 'channel', 'zplane'

    Returns:
        Tuple[pd.DataFrame, Optional[str]]: A tuple containing:
            - A Pandas DataFrame representing the plane table.
            - The file path of the saved CSV file, if ``save_table`` is True; otherwise, None.
    """
    if isinstance(czipath, Path):
        czipath = str(czipath)

    if validators.url(czipath):
        logger.warning("Reading PlaneTable from CZI via a link is not supported yet.")
        return pd.DataFrame(), None

    rows = []
    # Mosaic tile counter: maps (S, T, C, Z) → next M index
    tile_counters: Dict[Tuple[int, int, int, int], int] = {}

    with czifile_module.CziFile(czipath) as czi:
        subblocks = list(czi.subblocks())

        if HAS_PROGRESSBAR:
            widgets = [
                progressbar.Percentage(),
                " ",
                progressbar.Bar(),
                " ",
                progressbar.ETA(),
                " ",
                progressbar.SimpleProgress(),
            ]
            iterator = progressbar.progressbar(
                enumerate(subblocks),
                widgets=widgets,
                max_value=len(subblocks),
                term_width=80,
            )
        else:
            iterator = enumerate(subblocks)

        for sbcount, sb in iterator:
            de = sb.directory_entry

            s = _get_dim_start(de, "S")
            t = _get_dim_start(de, "T")
            c = _get_dim_start(de, "C")
            z = _get_dim_start(de, "Z")

            # Derive mosaic tile (M) index by counting subblocks per (S, T, C, Z) group
            key = (s, t, c, z)
            m = tile_counters.get(key, 0)
            tile_counters[key] = m + 1

            xstart, ystart, width, height = _get_bbox(de)

            # Parse subblock metadata (czifile returns XML as a string)
            md_raw = sb.metadata()
            if isinstance(md_raw, str):
                try:
                    md_xml = ET.fromstring(md_raw)
                except ET.ParseError:
                    md_xml = None
            else:
                md_xml = md_raw

            timestamp, xpos, ypos, zpos = _getsbinfo(md_xml)

            rows.append(
                {
                    "Subblock": sbcount,
                    "S": s,
                    "M": m,
                    "T": t,
                    "C": c,
                    "Z": z,
                    "X[micron]": xpos,
                    "Y[micron]": ypos,
                    "Z[micron]": zpos,
                    "Time[s]": timestamp,
                    "xstart": xstart,
                    "ystart": ystart,
                    "width": width,
                    "height": height,
                }
            )

    df_czi = _initialize_planetable_dataframe()
    if rows:
        df_czi = pd.concat([df_czi, pd.DataFrame(rows)], ignore_index=True)

    # Sort by dimension indices to produce the same ordered output as the old
    # nested-loop implementation, then renumber Subblock sequentially.
    df_czi = df_czi.sort_values(["S", "M", "T", "C", "Z"]).reset_index(drop=True)

    # Apply planes filter
    if planes is not None:
        df_czi = filter_planetable(df_czi, planes)
        df_czi = df_czi.reset_index(drop=True)

    # Sequential Subblock numbering (0-based) on the final ordered, filtered set
    df_czi["Subblock"] = np.arange(len(df_czi), dtype="int32")

    # Enforce column dtypes
    df_czi = df_czi.astype(
        {
            "Subblock": "int32",
            "S": "int32",
            "M": "int32",
            "T": "int32",
            "C": "int32",
            "Z": "int32",
            "xstart": "int32",
            "ystart": "int32",
            "width": "int32",
            "height": "int32",
        },
        errors="ignore",
    )

    if norm_time:
        df_czi = norm_columns(df_czi, colname="Time[s]", mode="min")

    if save_table:
        return _save_planetable_if_requested(df_czi, czipath, table_separator, table_index)

    return df_czi, None


def _getsbinfo(subblock: Optional[Any]) -> Tuple[float, float, float, float]:
    """
    Extracts metadata information from a parsed subblock XML element.

    Args:
        subblock: An ``xml.etree.ElementTree.Element`` parsed from subblock
            metadata, or ``None`` if parsing failed.

    Returns:
        Tuple[float, float, float, float]: timestamp, xpos, ypos, zpos.
        All values default to 0.0 when the corresponding XML element is absent.
    """
    if subblock is None:
        return 0.0, 0.0, 0.0, 0.0

    try:
        time_text = subblock.findtext(".//AcquisitionTime")
        timestamp = dt.parse(time_text).timestamp() if time_text else 0.0
    except (ValueError, OverflowError):
        timestamp = 0.0

    try:
        xpos = float(np.double(subblock.findtext(".//StageXPosition") or "0"))
    except (ValueError, TypeError):
        xpos = 0.0

    try:
        ypos = float(np.double(subblock.findtext(".//StageYPosition") or "0"))
    except (ValueError, TypeError):
        ypos = 0.0

    try:
        zpos = float(np.double(subblock.findtext(".//FocusPosition") or "0"))
    except (ValueError, TypeError):
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
    valid_args = ["scene", "tile", "time", "zplane", "channel"]

    # If no filtering criteria are provided, return the unfiltered planetable
    if planes is None:
        return planetable

    # Check for invalid keys in the `planes` dictionary
    for k in planes:
        if k not in valid_args:
            raise ValueError(f"Invalid keyword argument: {k}")

    # Apply filtering for each valid key if it exists in `planes`
    if "scene" in planes:
        planetable = planetable[planetable["S"] == planes["scene"]]

    if "tile" in planes:
        planetable = planetable[planetable["M"] == planes["tile"]]

    if "time" in planes:
        planetable = planetable[planetable["T"] == planes["time"]]

    if "zplane" in planes:
        planetable = planetable[planetable["Z"] == planes["zplane"]]

    if "channel" in planes:
        planetable = planetable[planetable["C"] == planes["channel"]]

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
        errors="ignore",
    )

    return df_czi


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
