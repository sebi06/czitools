# -*- coding: utf-8 -*-

#################################################################
# File        : misc.py
# Author      : sebi06
#
# Disclaimer: The code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

"""Miscellaneous utility functions for czitools.

Contains helpers for path normalisation, zoom validation, zarr/dask
array utilities, and other small cross-cutting concerns.
"""

import os
import time
import tracemalloc
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Dict, Tuple, Union

if TYPE_CHECKING:
    from czitools.metadata_tools.czi_metadata import CziMetadata

import dask.array as da
import numpy as np
import pandas as pd
import requests
import validators
import zarr
from pylibCZIrw import czi as pyczi

from czitools.metadata_tools.helper import ValueRange
from czitools.utils import logging_tools

logger = logging_tools.set_logging()


def get_pyczi_readertype(
    filepath: Union[str, os.PathLike[str]],
) -> Tuple[pyczi.ReaderFileInputTypes, bool]:
    """Determine the appropriate pylibCZIrw reader type for a CZI file path.

    This utility function checks whether the filepath is a URL or a local file
    and returns the appropriate reader type for pylibCZIrw.

    Args:
        filepath: Path to the CZI file. Can be a local path or a URL.

    Returns:
        Tuple of (reader_type, is_url) where:
            - reader_type: pyczi.ReaderFileInputTypes.Curl for URLs,
                          pyczi.ReaderFileInputTypes.Standard for local files.
            - is_url: True if the filepath is a valid URL, False otherwise.
    """
    # Convert Path to string if needed
    if isinstance(filepath, Path):
        filepath = str(filepath)

    if validators.url(str(filepath)):
        return pyczi.ReaderFileInputTypes.Curl, True
    else:
        return pyczi.ReaderFileInputTypes.Standard, False


def _slicedim(array: Union[np.ndarray, da.Array, zarr.Array], dimindex: int, posdim: int) -> np.ndarray:
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

    # TODO: explian this function much better

    idl_all = [slice(None, None, None)] * (len(array.shape) - 2)
    idl_all[posdim] = slice(dimindex, dimindex + 1, None)
    array_sliced = array[tuple(idl_all)]

    return array_sliced


def calc_scaling(
    data: Union[np.ndarray, da.Array, zarr.Array],
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
        # Wrap in dask so reduction is done chunk-by-chunk without loading the
        # full array into memory; also avoids zarr version API differences
        _dask_view = da.from_zarr(data)
        minvalue, maxvalue = da.compute(_dask_view.min(), _dask_view.max())
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


def md2dataframe(
    mdata: "CziMetadata", reduced_params: bool = True, paramcol: str = "Parameter", keycol: str = "Value"
) -> pd.DataFrame:
    """Convert a CziMetadata object into a two-column pandas DataFrame.

    Flattens the metadata object into a key-value dictionary and represents
    each entry as a row in the returned DataFrame.

    Args:
        mdata: The CziMetadata instance to convert.
        reduced_params: If True, only include a subset of key metadata parameters.
        paramcol: Column name for the parameter (key) column. Defaults to "Parameter".
        keycol: Column name for the value column. Defaults to "Value".

    Returns:
        DataFrame with one row per metadata entry and columns ``paramcol`` / ``keycol``.
    """
    # Deferred import to avoid circular dependency between utils and metadata_tools
    from czitools.metadata_tools.czi_metadata import _obj2dict  # noqa: PLC0415
    from czitools.metadata_tools.czi_metadata import create_md_dict_red  # noqa: PLC0415

    if reduced_params:
        md_dict = create_md_dict_red(mdata)
    else:
        # Flatten the CziMetadata object into a plain key-value dictionary
        md_dict = _obj2dict(mdata)

    # Initialise an empty DataFrame with the desired column names
    mdframe = pd.DataFrame(columns=[paramcol, keycol])

    for k in md_dict.keys():
        # Build a single-row DataFrame for each metadata entry and append it
        d = {paramcol: k, keycol: md_dict[k]}
        df = pd.DataFrame([d], index=[0])
        mdframe = pd.concat([mdframe, df], ignore_index=True)

    return mdframe


def _sort_dict_by_key(unsorted_dict: Dict) -> Dict:
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


def _addzeros(number: int) -> str:
    """Convert a number into a string and add leading zeros.

    Typically used to construct filenames with equal lengths.

    Args:
        number (int): The number.

    Returns:
        str: String with leading zeros.
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


def _check_dimsize(mdata_entry: Union[Any, None], set2value: Any = 1) -> Union[Any, None]:
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


def _clean_dict(d: Dict[Any, Any]) -> Dict[Any, Any]:
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
            nested_cleaned = _clean_dict(value)
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


def _check_zoom(zoom: Annotated[float, ValueRange(0.01, 1.0)] = 1.0) -> float:

    # check zoom factor
    if zoom > 1.0:
        logger.warning(f"Zoom factor f{zoom} is not in valid range [0.01 - 1.0]. Using 1.0 instead.")
        zoom = 1.0
    if zoom < 0.01:
        logger.warning(f"Zoom factor f{zoom} is not in valid range [0.01 - 1.0]. Using 0.01 instead.")
        zoom = 0.01

    return zoom


def _measure_memory_usage(target_function):
    """
    A decorator that measures and logs the memory usage of the decorated function.
    This decorator uses the `tracemalloc` module to track memory allocations and logs
    the top memory-consuming lines after the function execution.
    Args:
        target_function (function): The function to be decorated.
    Returns:
        function: The wrapped function with memory usage measurement.
    Example:
        @_measure_memory_usage
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


def _measure_execution_time(func):
    """
    Decorator that measures the execution time of a function and logs it.
    Args:
        func (callable): The function to be decorated.
    Returns:
        callable: The wrapped function with execution time measurement.
    Example:
        @_measure_execution_time
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
            return (
                False,
                f"File not accessible, HTTP status code: {response.status_code}",
            )
    except requests.RequestException as e:
        return False, f"Error during HTTP request: {e}"


# ---------------------------------------------------------------------------
# czifile directory-entry helpers (czifile >= 2026, Python >= 3.12)
# ---------------------------------------------------------------------------
# czifile's CziDirectoryEntryDV exposes: dims, start, shape, stored_shape,
# scene_index, mosaic_index, is_pyramid.


def _de_dim_start(de: Any, dim: str, default: int = 0) -> int:
    """Return the start index for *dim* from a czifile directory entry."""
    if dim in de.dims:
        return de.start[de.dims.index(dim)]
    return default


def _de_dim_size(de: Any, dim: str, default: int = 0) -> int:
    """Return the size (shape) of *dim* from a czifile directory entry."""
    if dim in de.dims:
        return de.shape[de.dims.index(dim)]
    return default


def _de_dim_chars(de: Any) -> tuple:
    """Return all dimension names as a tuple from a czifile directory entry."""
    return tuple(de.dims)


def _de_scene_idx(de: Any) -> int:
    """Return the scene index from a czifile directory entry, or -1 if absent."""
    return int(de.scene_index)


def _de_mosaic_idx(de: Any) -> int:
    """Return the mosaic (tile) index from a czifile directory entry, or -1 if absent."""
    return int(de.mosaic_index)
