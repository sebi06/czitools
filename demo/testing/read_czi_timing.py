# -*- coding: utf-8 -*-

#################################################################
# File        : read_czi_simple.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from czitools.metadata_tools.czi_metadata import CziMetadata, create_md_dict_red, CziSampleInfo
from czitools.read_tools import read_tools
from czitools.utils import misc

filepath1 = r"F:\Testdata_Zeiss\LLS7\LS_Mitosis_T=10_Z=50_CH=2.czi"
filepath2 = r"F:\Testdata_Zeiss\LLS7\LS_Mitosis_T=50_Z=10_CH=2.czi"
filepath3 = r"F:\Testdata_Zeiss\LLS7\LS_Mitosis_T=150-300sm_ZSTD.czi"


# determine shape of combines stack
@misc.measure_execution_time
def read_mdata(filepath) -> CziMetadata:
    """Read metadata from CZI file."""
    return CziMetadata(filepath)


@misc.measure_execution_time
def read_mdata_dict(mdata: CziMetadata) -> dict:
    """Create a dictionary from CziMetadata."""
    return create_md_dict_red(mdata)


@misc.measure_execution_time
def read_array(filepath: str):
    return read_tools.read_6darray(filepath)


@misc.measure_execution_time
def read_sample(filepath: str):
    return CziSampleInfo(filepath)


# You need to actually call the functions to see timing
mdata1 = read_mdata(filepath1)
mdata2 = read_mdata(filepath2)
mdata3 = read_sample(filepath3)
mdata_dict = read_mdata_dict(mdata1)
array_data = read_array(filepath1)

print("Done")
