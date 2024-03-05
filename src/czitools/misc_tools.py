# -*- coding: utf-8 -*-

#################################################################
# File        : misc_tools.py
# Author      : sebi06
#
# Disclaimer: The code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

import os
from tkinter import filedialog
from tkinter import *
import zarr
import pandas as pd
import dask.array as da
import numpy as np
import time
from pathlib import Path
import dateutil.parser as dt
from tqdm.contrib.itertools import product
from czitools import metadata_tools as czimd
from typing import List, Dict, Tuple, Optional, Type, Any, Union, Mapping
from dataclasses import make_dataclass, fields, dataclass
from czitools import logger as LOGGER
import validators
import re
from aicspylibczi import CziFile

# from urllib.parse import urlparse

logger = LOGGER.get_logger()


def openfile(
    directory: Union[str, os.PathLike[str]],
    title: str = "Open CZI Image File",
    ftypename: str = "CZI Files",
    extension: str = "*.czi",
) -> str:
    """Open a simple Tk dialog to select a file.

    Args:
        directory: Default directory.
        title: Title of the dialog window, defaults to "Open CZI Image File".
        ftypename: Name of allowed file type, defaults to "CZI Files".
        extension: Extension of allowed file type, defaults to "*.czi".

    Returns:
        Filepath object for the selected file.
    """

    if isinstance(directory, Path):
        # convert to string
        directory = str(directory)

    # request input and output image path from user
    root = Tk()
    root.withdraw()
    input_path = filedialog.askopenfile(
        title=title, initialdir=directory, filetypes=[(ftypename, extension)]
    )
    if input_path is not None:
        return input_path.name
    if input_path is None:
        return ""


def slicedim(
    array: Union[np.ndarray, da.Array, zarr.Array], dimindex: int, posdim: int
) -> np.ndarray:
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


def md2dataframe(
    md_dict: Dict, paramcol: str = "Parameter", keycol: str = "Value"
) -> pd.DataFrame:
    """Converts the given metadata dictionary to a Pandas DataFrame.

    Args:
        md_dict (dict): A dictionary containing metadata.
        paramcol (str, optional): The name of the column for metadata parameters. Defaults to "Parameter".
        keycol (str, optional): The name of the column for metadata values. Defaults to "Value".

    Returns:
        pd.DataFrame: A Pandas DataFrame containing all the metadata.
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
    """Get the complete path of a file without the extension
    It also works for extensions like myfile.abc.xyz
    The output will be: myfile

    :param filepath: complete filepath
    :type filepath: str
    :return: complete filepath without extension
    :rtype: str
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


def check_dimsize(
    mdata_entry: Union[Any, None], set2value: Any = 1
) -> Union[Any, None]:
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


def get_planetable(
    czifile: Union[str, os.PathLike[str]],
    norm_time: bool = True,
    pt_complete: bool = True,
    t: int = 0,
    c: int = 0,
    z: int = 0,
) -> pd.DataFrame:
    """Get the planetable from the individual subblocks
    Args:
        czifile: the source for the CZI image file
        norm_time: normalize the timestamps
        pt_complete: Read data from all subblocks or only use the ones to be shown in the plot

    Returns:
        Planetable as pd.DataFrame
    """

    if isinstance(czifile, Path):
        # convert to string
        czifile = str(czifile)

    if validators.url(czifile):
        logger.warning("Reading PlaneTable from CZI via a link is not supported.")
        return None, None

    # initialize the plane table
    df_czi = pd.DataFrame(
        columns=[
            "Subblock",
            "Scene",
            "Tile",
            "T",
            "Z",
            "C",
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

    # define subblock counter
    sbcount = -1

    aicsczi = CziFile(czifile)
    dims = aicsczi.get_dims_shape()

    if "S" in dims[0].keys():
        size_s = dims[0]["S"][1]
    else:
        size_s = 1

    if "M" in dims[0].keys():
        size_m = dims[0]["M"][1]
    else:
        size_m = 1

    if "T" in dims[0].keys():
        size_t = dims[0]["T"][1]
    else:
        size_t = 1

    if "C" in dims[0].keys():
        size_c = dims[0]["C"][1]
    else:
        size_c = 1

    if "Z" in dims[0].keys():
        size_z = dims[0]["Z"][1]
    else:
        size_z = 1

    def getsbinfo(subblock: Any) -> Tuple[float, float, float, float]:
        try:
            time = subblock.findall(".//AcquisitionTime")[0].text
            timestamp = dt.parse(time).timestamp()
        except IndexError as e:
            timestamp = 0.0

        try:
            xpos = np.double(subblock.findall(".//StageXPosition")[0].text)
        except IndexError as e:
            xpos = 0.0

        try:
            ypos = np.double(subblock.findall(".//StageYPosition")[0].text)
        except IndexError as e:
            ypos = 0.0

        try:
            zpos = np.double(subblock.findall(".//FocusPosition")[0].text)
        except IndexError as e:
            zpos = 0.0

        return timestamp, xpos, ypos, zpos

    if pt_complete:
        t_start = 0
        t_end = size_t
        c_start = 0
        c_end = size_c
        z_start = 0
        z_end = size_z

    elif not pt_complete:
        t_start = t
        t_end = t + 1
        c_start = c
        c_end = c + 1
        z_start = 0
        z_end = z + 1

    # do if the data is not a mosaic
    if size_m > 1:
        # for s, m, t, z, c in product(
        #    range(size_s),
        #    range(size_m),
        #    range(size_t),
        #    range(size_z),
        #    range(size_c),
        # ):
        for s, m, t, c, z in product(
            range(size_s),
            range(size_m),
            enumerate(range(t_start, t_end)),
            enumerate(range(c_start, c_end)),
            enumerate(range(z_start, z_end)),
            desc="Reading sublocks planes",
            unit=" 2Dplanes",
        ):
            sbcount += 1

            # get x, y, width and height for a specific tile
            tilebbox = aicsczi.get_mosaic_tile_bounding_box(
                S=s, M=m, T=t[1], Z=z[1], C=c[1]
            )

            # read information from subblock
            sb = aicsczi.read_subblock_metadata(
                unified_xml=True, B=0, S=s, M=m, T=t[1], Z=z[1], C=c[1]
            )

            # get information from subblock
            timestamp, xpos, ypos, zpos = getsbinfo(sb)

            plane = pd.DataFrame(
                {
                    "Subblock": sbcount,
                    "Scene": s,
                    "Tile": m,
                    "T": t[1],
                    "Z": z[1],
                    "C": c[1],
                    "X[micron]": xpos,
                    "Y[micron]": ypos,
                    "Z[micron]": zpos,
                    "Time[s]": timestamp,
                    "xstart": tilebbox.x,
                    "ystart": tilebbox.y,
                    "width": tilebbox.w,
                    "height": tilebbox.h,
                },
                index=[0],
            )

            df_czi = pd.concat([df_czi if not df_czi.empty else None, plane])

    # do if the data is not a mosaic
    if size_m == 1:
        # for s, t, z, c in product(
        #    range(size_s), range(size_t), range(size_z), range(size_c)
        # ):
        for s, t, c, z in product(
            range(size_s),
            enumerate(range(t_start, t_end)),
            enumerate(range(c_start, c_end)),
            enumerate(range(z_start, z_end)),
            desc="Reading sublocks planes",
            unit=" 2Dplanes",
        ):
            sbcount += 1

            # get x, y, width and height for a specific tile
            tilebbox = aicsczi.get_tile_bounding_box(S=s, T=t[1], Z=z[1], C=c[1])

            # read information from subblocks
            sb = aicsczi.read_subblock_metadata(
                unified_xml=True, B=0, S=s, T=t[1], Z=z[1], C=c[1]
            )

            # get information from subblock
            timestamp, xpos, ypos, zpos = getsbinfo(sb)

            plane = pd.DataFrame(
                {
                    "Subblock": sbcount,
                    "Scene": s,
                    "Tile": 0,
                    "T": t[1],
                    "Z": z[1],
                    "C": c[1],
                    "X[micron]": xpos,
                    "Y[micron]": ypos,
                    "Z[micron]": zpos,
                    "Time[s]": timestamp,
                    "xstart": tilebbox.x,
                    "ystart": tilebbox.y,
                    "width": tilebbox.w,
                    "height": tilebbox.h,
                },
                index=[0],
            )

            df_czi = pd.concat([df_czi if not df_czi.empty else None, plane])

    # cast data  types
    df_czi = df_czi.astype(
        {
            "Subblock": "int32",
            "Scene": "int32",
            "Tile": "int32",
            "T": "int32",
            "Z": "int32",
            "C": "int16",
            "X[micron]": "float",
            "Y[micron]": "float",
            "Z[micron]": "float",
            "xstart": "int32",
            "ystart": "int32",
            "width": "int32",
            "height": "int32",
        },
        copy=False,
        errors="ignore",
    )

    # normalize time stamps
    if norm_time:
        df_czi = norm_columns(df_czi, colname="Time[s]", mode="min")

    return df_czi


def norm_columns(
    df: pd.DataFrame, colname: str = "Time [s]", mode: str = "min"
) -> pd.DataFrame:
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
        df[colname] = df[colname] - min_value

    if mode == "max":
        max_value = df[colname].max()
        df[colname] = df[colname] - max_value

    return df


def filter_planetable(planetable: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Filters the input planetable DataFrame based on specified dimension entries.

    The function uses optional keyword arguments to filter the planetable on
    various dimensions such as scene index, time index, z-plane index, and channel index.
    If a dimension is not specified in the keyword arguments, it will not be used for filtering.
    Only valid arguments are accepted, which are 's', 't', 'z', and 'c'. An error is raised for invalid arguments.

    Args:
        planetable (pd.DataFrame): The DataFrame to be filtered.
            It should contain columns "Scene", "T", "Z" and "C".

        **kwargs: Optional keyword arguments specifying the indices to filter on.
            These can include:
                'S': The scene index.
                'T': The time index.
                'Z': The z-plane index.
                'C': The channel index.

    Returns:
        pd.DataFrame: The filtered planetable.

    Raises:
        KeyError: If a specified index is not a column in the DataFrame.
        ValueError: If an invalid keyword argument is passed.

    Examples:
        >>> planetable = pd.DataFrame({'Scene': [0, 1, 1], 'T': [0, 1, 0], 'Z': [0, 1, 1], 'C': [0, 1, 1]})
        >>> filter_planetable(planetable, S=1, T=0)
        Scene  T  Z  C
        2      1  0  1  1
    """

    valid_args = ["S", "T", "Z", "C"]

    # check for invalid arguments
    for k, v in kwargs.items():

        if k not in valid_args:
            raise ValueError(f"Invalid keyword argument: {k}")

    if "S" in kwargs:
        # filter planetable for specific scene
        planetable = planetable[planetable["Scene"] == kwargs["S"]]

    if "T" in kwargs:
        # filter planetable for specific timepoint
        planetable = planetable[planetable["T"] == kwargs["T"]]

    if "Z" in kwargs:
        # filter resulting planetable pt for a specific z-plane
        planetable = planetable[planetable["Z"] == kwargs["Z"]]

    if "C" in kwargs:
        # filter planetable for specific channel
        planetable = planetable[planetable["C"] == kwargs["C"]]

    # return filtered planetable
    return planetable


def save_planetable(
    df: pd.DataFrame, filename: str, separator: str = ",", index: bool = True
) -> str:
    """Saves a pandas dataframe as a CSV file.

    Args:
        df (pd.DataFrame): The dataframe to be saved as CSV.
        filename (str): The filename of the CSV file to be written.
        separator (str, optional): The separator character for the CSV file. Defaults to ','.
        index (bool, optional): Whether to include the index in the CSV file. Defaults to True.

    Returns:
        str: The filename of the CSV file that was written.
    """
    # Generate the filename for the planetable CSV.
    csvfile = os.path.splitext(filename)[0] + "_planetable.csv"

    # Write the dataframe to the planetable CSV file.
    df.to_csv(csvfile, sep=separator, index=index)

    return csvfile


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


# def dc_factory(
#     base: dataclass, name: str, exclusions: tuple = ("field1", "fields")
# ) -> dataclass:
#     """
#     Adapted from: https://stackoverflow.com/questions/69289547/how-to-remove-dynamically-fields-from-a-dataclass

#     Args:
#         base (dataclass): Base Dataclass
#         name (str): name for new dataclass
#         exclusions (tuple, optional): Fields to be removed. Defaults to ("field1", "field2").

#     Returns:
#         dataclass: New dataclass with the fields that were removed
#     """

#     new_fields = [(i.name, i.type, i) for i in fields(base) if i.name not in exclusions]

#     return make_dataclass(name, new_fields)


def remove_none_from_dict(dictionary: Dict) -> Dict:
    """
    Remove values equal to: None, [] or {} from dictionary

    Args:
        dictionary (Dict): Dictionary to be checked

    Returns:
        Dict: Cleaned up dictionary
    """
    for key, value in list(dictionary.items()):
        # if value is None or value == [] or value == {}:
        if (
            value is None
            or (isinstance(value, list) and not bool(value))
            or value == {}
        ):
            del dictionary[key]
        elif isinstance(value, dict):
            remove_none_from_dict(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    remove_none_from_dict(item)

    return dictionary


def download_zip(source_link: str) -> str:

    if not validators.url(source_link):
        logger.warning("Not a valid link.")
        return ""

    import io
    import zipfile

    compressed_data = os.path.join(os.getcwd(), os.path.basename(source_link))

    if not os.path.isfile(compressed_data):
        response = requests.get(GITHUB_IMAGES_PATH, stream=True)
        compressed_data = io.BytesIO(response.content)

    with zipfile.ZipFile(compressed_data, "r") as zip_accessor:
        zip_accessor.extractall("./")

    return compressed_data[:-4]
