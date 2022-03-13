# -*- coding: utf-8 -*-

#################################################################
# File        : misc.py
# Version     : 0.0.9
# Author      : sebi06
# Date        : 13.03.2022
#
# Disclaimer: The code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from __future__ import annotations
import os
from tkinter import filedialog
from tkinter import *
#from types import NoneType
import zarr
import pandas as pd
import dask
import dask.array as da
import numpy as np
import time
from pathlib import Path
import xml.etree.ElementTree as ET
from aicspylibczi import CziFile
from aicsimageio import AICSImage
import dateutil.parser as dt
from czimetadata_tools import pylibczirw_metadata as czimd
from tqdm.contrib.itertools import product
from typing import List, Dict, Tuple, Optional, Type, Any, Union


def openfile(directory: str,
             title: str = "Open CZI Image File",
             ftypename: str = "CZI Files",
             extension: str = "*.czi") -> str:
    """ Open a simple Tk dialog to select a file.

    :param directory: default directory
    :param title: title of the dialog window
    :param ftypename: name of allowed file type
    :param extension: extension of allowed file type
    :return: filepath object for the selected
    """

    # request input and output image path from user
    root = Tk()
    root.withdraw()
    input_path = filedialog.askopenfile(title=title,
                                        initialdir=directory,
                                        filetypes=[(ftypename, extension)])
    if input_path is not None:
        return input_path.name
    if input_path is None:
        return ""


def slicedim(array: Union[np.ndarray, dask.array.Array, zarr.Array],
             dimindex: int,
             posdim: int) -> np.ndarray:
    """Slice out a specific dimension without (!) dropping the dimension
    of the array to conserve the dimorder string
    this should work for Numpy.Array, Dask and ZARR ...

    :param array: input array
    :param dimindex: index of the slice dimension to be kept
    :param posdim: position of the dimension to be sliced
    :return: sliced array
    """

    # if posdim == 0:
    #    array_sliced = array[dimindex:dimindex + 1, ...]
    # if posdim == 1:
    #    array_sliced = array[:, dimindex:dimindex + 1, ...]
    # if posdim == 2:
    #    array_sliced = array[:, :, dimindex:dimindex + 1, ...]
    # if posdim == 3:
    #    array_sliced = array[:, :, :, dimindex:dimindex + 1, ...]
    # if posdim == 4:
    #    array_sliced = array[:, :, :, :, dimindex:dimindex + 1, ...]
    # if posdim == 5:
    #    array_sliced = array[:, :, :, :, :, dimindex:dimindex + 1, ...]

    idl_all = [slice(None, None, None)] * (len(array.shape) - 2)
    idl_all[posdim] = slice(dimindex, dimindex + 1, None)
    array_sliced = array[tuple(idl_all)]

    return array_sliced


def calc_scaling(data: Union[np.ndarray, da.array],
                 corr_min: float = 1.0,
                 offset_min: int = 0,
                 corr_max: float = 0.85,
                 offset_max: int = 0) -> Tuple[int, int]:
    """Calculate the scaling for better display

    :param data: Calculate min / max scaling
    :type data: Numpy.Array or dask.array or zarr.array
    :param corr_min: correction factor for minvalue, defaults to 1.0
    :type corr_min: float, optional
    :param offset_min: offset for min value, defaults to 0
    :type offset_min: int, optional
    :param corr_max: correction factor for max value, defaults to 0.85
    :type corr_max: float, optional
    :param offset_max: offset for max value, defaults to 0
    :type offset_max: int, optional
    :return: list with [minvalue, maxvalue]
    :rtype: list
    """

    start = time.time()

    # get min-max values for initial scaling
    if isinstance(data, zarr.Array):
        minvalue, maxvalue = np.min(data), np.max(data)
    elif isinstance(data, da.Array):
        # use dask.compute only once since this is faster
        minvalue, maxvalue = da.compute(data.min(), data.max())
    else:
        minvalue, maxvalue = np.min(data), np.max(data)

    end = time.time()

    minvalue = np.round((minvalue + offset_min) * corr_min, 0)
    maxvalue = np.round((maxvalue + offset_max) * corr_max, 0)

    print("Scaling:", minvalue, maxvalue)
    print("Calculation of Min-Max [s] : ", end - start)

    return minvalue, maxvalue


def md2dataframe(md_dict: Dict,
                 paramcol: str = "Parameter",
                 keycol: str = "Value") -> pd.DataFrame:
    """Convert the metadata dictionary to a Pandas DataFrame.

    :param metadata: MeteData dictionary
    :type metadata: dict
    :param paramcol: Name of Columns for the MetaData Parameters, defaults to "Parameter"
    :type paramcol: str, optional
    :param keycol: Name of Columns for the MetaData Values, defaults to "Value"
    :type keycol: str, optional
    :return: Pandas DataFrame containing all the metadata
    :rtype: Pandas.DataFrame
    """
    mdframe = pd.DataFrame(columns=[paramcol, keycol])

    for k in md_dict.keys():
        d = {"Parameter": k, "Value": md_dict[k]}
        df = pd.DataFrame([d], index=[0])
        mdframe = pd.concat([mdframe, df], ignore_index=True)

    return mdframe


def sort_dict_by_key(unsorted_dict: Dict) -> Dict:
    """Sort a dictionary by its keys.

    :param unsorted_dict: the unsorted dictionary
    :type unsorted_dict: dict
    :return: sorted_dict - sorted dictionary
    :rtype: dict
    """
    sorted_keys = sorted(unsorted_dict.keys(), key=lambda x: x.lower())
    sorted_dict = {}
    for key in sorted_keys:
        sorted_dict.update({key: unsorted_dict[key]})

    return sorted_dict


def addzeros(number: int) -> str:
    """Convert a number into a string and add leading zeros.
    Used to construct filenames with equal lengths.

    :param number: the number
    :type number: int
    :return: zerostring - string with leading zeros
    :rtype: str
    """

    if number < 10:
        zerostring = '0000' + str(number)
    if 10 <= number < 100:
        zerostring = '000' + str(number)
    if 100 <= number < 1000:
        zerostring = '00' + str(number)
    if 1000 <= number < 10000:
        zerostring = '0' + str(number)

    return zerostring


def get_fname_woext(filepath: str) -> str:
    """Get the complete path of a file without the extension
    It also will works for extensions like c:\myfile.abc.xyz
    The output will be: c:\myfile

    :param filepath: complete fiepath
    :type filepath: str
    :return: complete filepath without extension
    :rtype: str
    """
    # create empty string
    real_extension = ''

    # get all part of the file extension
    sufs = Path(filepath).suffixes
    for s in sufs:
        real_extension = real_extension + s

    # remove real extension from filepath
    filepath_woext = filepath.replace(real_extension, '')

    return filepath_woext


def check_dimsize(mdata_entry: Union[int, None], set2value: int = 1) -> int:
    """Check is a value is None and replace with a specific number

    Args:
        mdata_entry (Union[int, None]): value to be checked
        set2value (int, optional): value used to replace None with. Defaults to 1.

    Returns:
        int: new_value
    """

    # check if the dimension entry is None
    if mdata_entry is None:
        new_value = set2value
    if mdata_entry is not None:
        new_value = mdata_entry

    return new_value


def get_daskstack(aics_img: AICSImage) -> List:
    """Get all scenes of an image as dask stack

    Args:
        aics_img (AICSImage): the AICSImageIO object

    Returns:
        List: List of dask arrays
    """

    stacks = []
    for scene in aics_img.scenes:
        aics_img.set_scene(scene)
        stacks.append(aics_img.dask_data)

    stacks = da.stack(stacks)

    return stacks


def get_planetable(czifile: str,
                   norm_time: bool = True,
                   savetable: bool = False,
                   separator: str = ',',
                   index: bool = True) -> Tuple[pd.DataFrame, Optional[str]]:

    # get the czi metadata
    metadata = czimd.CziMetadata(czifile)
    aicsczi = CziFile(czifile)

    # initialize the plane table
    df_czi = pd.DataFrame(columns=['Subblock',
                                   'Scene',
                                   'Tile',
                                   'T',
                                   'Z',
                                   'C',
                                   'X[micron]',
                                   'Y[micron]',
                                   'Z[micron]',
                                   'Time[s]',
                                   'xstart',
                                   'ystart',
                                   'width',
                                   'height'])

    # define subblock counter
    sbcount = -1

    # check if dimensions are None (because they do not exist for that image)
    sizeC = check_dimsize(metadata.image.SizeC, set2value=1)
    sizeZ = check_dimsize(metadata.image.SizeZ, set2value=1)
    sizeT = check_dimsize(metadata.image.SizeT, set2value=1)
    sizeS = check_dimsize(metadata.image.SizeS, set2value=1)
    sizeM = check_dimsize(metadata.image.SizeM, set2value=1)

    def getsbinfo(subblock: Any) -> Tuple[float, float, float, float]:
        try:
            # time = sb.xpath('//AcquisitionTime')[0].text
            time = subblock.findall(".//AcquisitionTime")[0].text
            timestamp = dt.parse(time).timestamp()
        except IndexError as e:
            timestamp = 0.0

        try:
            # xpos = np.double(sb.xpath('//StageXPosition')[0].text)
            xpos = np.double(subblock.findall(".//StageXPosition")[0].text)
        except IndexError as e:
            xpos = 0.0

        try:
            # ypos = np.double(sb.xpath('//StageYPosition')[0].text)
            ypos = np.double(subblock.findall(".//StageYPosition")[0].text)
        except IndexError as e:
            ypos = 0.0

        try:
            # zpos = np.double(sb.xpath('//FocusPosition')[0].text)
            zpos = np.double(subblock.findall(".//FocusPosition")[0].text)
        except IndexError as e:
            zpos = 0.0

        return timestamp, xpos, ypos, zpos

    # in case the CZI has the M-Dimension
    if metadata.ismosaic:

        for s, m, t, z, c in product(range(sizeS),
                                     range(sizeM),
                                     range(sizeT),
                                     range(sizeZ),
                                     range(sizeC)):
            sbcount += 1
            print("Reading sublock : ", sbcount)

            # get x, y, width and height for a specific tile
            tilebbox = aicsczi.get_mosaic_tile_bounding_box(S=s,
                                                            M=m,
                                                            T=t,
                                                            Z=z,
                                                            C=c)

            # read information from subblock
            sb = aicsczi.read_subblock_metadata(unified_xml=True,
                                                B=0,
                                                S=s,
                                                M=m,
                                                T=t,
                                                Z=z,
                                                C=c)

            # get information from subblock
            timestamp, xpos, ypos, zpos = getsbinfo(sb)

            df_czi = df_czi.append({'Subblock': sbcount,
                                    'Scene': s,
                                    'Tile': m,
                                    'T': t,
                                    'Z': z,
                                    'C': c,
                                    'X[micron]': xpos,
                                    'Y[micron]': ypos,
                                    'Z[micron]': zpos,
                                    'Time[s]': timestamp,
                                    'xstart': tilebbox.x,
                                    'ystart': tilebbox.y,
                                    'width': tilebbox.w,
                                    'height': tilebbox.h},
                                   ignore_index=True)

    if not metadata.ismosaic:

        for s, t, z, c in product(range(sizeS),
                                  range(sizeT),
                                  range(sizeZ),
                                  range(sizeC)):
            sbcount += 1

            # get x, y, width and height for a specific tile
            tilebbox = aicsczi.get_tile_bounding_box(S=s,
                                                     T=t,
                                                     Z=z,
                                                     C=c)

            # read information from subblocks
            sb = aicsczi.read_subblock_metadata(unified_xml=True,
                                                B=0,
                                                S=s,
                                                T=t,
                                                Z=z,
                                                C=c)

            # get information from subblock
            timestamp, xpos, ypos, zpos = getsbinfo(sb)

            df_czi = df_czi.append({'Subblock': sbcount,
                                    'Scene': s,
                                    'Tile': 0,
                                    'T': t,
                                    'Z': z,
                                    'C': c,
                                    'X[micron]': xpos,
                                    'Y[micron]': ypos,
                                    'Z[micron]': zpos,
                                    'Time[s]': timestamp,
                                    'xstart': tilebbox.x,
                                    'ystart': tilebbox.y,
                                    'width': tilebbox.w,
                                    'height': tilebbox.h},
                                   ignore_index=True)

    # cast data  types
    df_czi = df_czi.astype({'Subblock': 'int32',
                            'Scene': 'int32',
                            'Tile': 'int32',
                            'T': 'int32',
                            'Z': 'int32',
                            'C': 'int16',
                            'X[micron]': 'float',
                            'Y[micron]': 'float',
                            'Z[micron]': 'float',
                            'xstart': 'int32',
                            'ystart': 'int32',
                            'width': 'int32',
                            'height': 'int32'},
                           copy=False,
                           errors='ignore')

    # normalize time stamps
    if norm_time:
        df_czi = norm_columns(df_czi, colname='Time[s]', mode='min')

    # save planetable as CSV file
    if savetable:
        csvfile = save_planetable(df_czi, czifile, separator=separator, index=index)
    if not savetable:
        csvfile = None

    return df_czi, csvfile


def norm_columns(df: pd.DataFrame,
                 colname: str = 'Time [s]',
                 mode: str = 'min') -> pd.DataFrame:
    """Normalize a specific column inside a Pandas dataframe

    :param df: DataFrame
    :type df: pf.DataFrame
    :param colname: Name of the column to be normalized, defaults to 'Time [s]'
    :type colname: str, optional
    :param mode: Mode of Normalization, defaults to 'min'
    :type mode: str, optional
    :return: Dataframe with normalized column
    :rtype: pd.DataFrame
    """
    # normalize columns according to min or max value
    if mode == 'min':
        min_value = df[colname].min()
        df[colname] = df[colname] - min_value

    if mode == 'max':
        max_value = df[colname].max()
        df[colname] = df[colname] - max_value

    return df


def filter_planetable(planetable: pd.DataFrame,
                      s: int = 0,
                      t: int = 0,
                      z: int = 0,
                      c: int = 0) -> pd.DataFrame:

    # filter planetable for specific scene
    if s > planetable['Scene'].max():
        print('Scene Index was invalid. Using Scene = 0.')
        s = 0
    pt = planetable[planetable['Scene'] == s]

    # filter planetable for specific timepoint
    if t > planetable['T'].max():
        print('Time Index was invalid. Using T = 0.')
        t = 0
    pt = planetable[planetable['T'] == t]

    # filter resulting planetable pt for a specific z-plane
    try:
        if z > planetable['Z[micron]'].max():
            print('Z-Plane Index was invalid. Using Z = 0.')
            zplane = 0
            pt = pt[pt['Z[micron]'] == z]
    except KeyError as e:
        if z > planetable['Z [micron]'].max():
            print('Z-Plane Index was invalid. Using Z = 0.')
            zplane = 0
            pt = pt[pt['Z [micron]'] == z]

    # filter planetable for specific channel
    if c > planetable['C'].max():
        print('Channel Index was invalid. Using C = 0.')
        c = 0
    pt = planetable[planetable['C'] == c]

    # return filtered planetable
    return pt


def save_planetable(df: pd.DataFrame,
                    filename: str,
                    separator: str = ',',
                    index: bool = True) -> str:
    """Save dataframe as CSV table

    :param df: Dataframe to be saved as CSV.
    :type df: pd.DataFrame
    :param filename: filename of the CSV to be written
    :type filename: str
    :param separator: separator for the CSV file, defaults to ','
    :type separator: str, optional
    :param index: option write the index into the CSV file, defaults to True
    :type index: bool, optional
    :return: filename of the CSV
    :rtype: str
    """
    csvfile = os.path.splitext(filename)[0] + '_planetable.csv'

    # write the CSV data table
    df.to_csv(csvfile, sep=separator, index=index)

    return csvfile
