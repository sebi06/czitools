from bioio import BioImage
import bioio_czi
from pathlib import Path
from matplotlib import pyplot as plt
from box import Box
import xmltodict
import xml.etree.ElementTree as ET
from czitools.utils import misc
from czitools.utils import planetable


def get_subblock_metadata(bioimage: BioImage, attribute_filters: dict = {"S": "0"}):
    # Filter subblocks based on the specified attributes
    specific_subblocks = [
        subblock
        for subblock in bioimage.metadata.findall("./Subblocks/Subblock")
        if all(subblock.attrib.get(attr) == value for attr, value in attribute_filters.items())
    ]

    # Convert each subblock XML element to a dictionary
    specific_subblocks_list = [xmltodict.parse(ET.tostring(subblock)) for subblock in specific_subblocks]
    combined_subblocks_dict = {f"{i}": subblock for i, subblock in enumerate(specific_subblocks_list)}

    # Recursive function to remove "@" symbols from keys in nested dictionaries
    def clean_dict_keys(data):
        if isinstance(data, dict):
            return {key.lstrip("@"): clean_dict_keys(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [clean_dict_keys(item) for item in data]
        else:
            return data

    # Clean "@" symbols from keys in the nested dictionary
    cleaned_subblocks_dict = clean_dict_keys(combined_subblocks_dict)

    # Remove "Subblock" keys to simplify the dictionary
    simplified_subblocks_dict = {
        key: value["Subblock"] if "Subblock" in value else value for key, value in cleaned_subblocks_dict.items()
    }

    sb_box = Box(
        simplified_subblocks_dict,
        conversion_box=True,
        default_box_attr=None,
        default_box_create_on_get=True,
    )

    # # loop of all subblocks
    # for key, value in sb_box.items():
    #     print(f"Key: {key}, Value: {value}")

    return specific_subblocks, combined_subblocks_dict, sb_box


# filepath = r"F:\Testdata_Zeiss\CZI_Testfiles\testwell96.czi"
# filepath = r"F:\Testdata_Zeiss\CZI_Testfiles\W96_B2+B4_S=2_T=2=Z=4_C=3_Tile=5x9.czi"
# filepath = r"f:\Github\czitools\data\CellDivision_T3_Z5_CH2_X240_Y170.czi"
# filepath = r"f:\Github\czitools\data\S2_3x3_CH2.czi"
filepath = r"f:\Github\czitools\data\WP96_4Pos_B4-10_DAPI.czi"

# aics = BioImage(
#     filepath, reader=bioio_czi.Reader, reconstruct_mosaic=True, use_aicspylibczi=True, include_subblock_metadata=True
# )

# # Define a list of attributes and their desired values to filter subblocks
# attribute_filters = {"S": "0"}  # Add more attributes as needed

# sbs, sb_dict, sb_box = get_subblock_metadata(aics, attribute_filters=attribute_filters)


# get the planetable for the CZI file
pt, savepath = planetable.get_planetable(
    filepath,
    norm_time=True,
    save_table=True,
    table_separator=";",
    table_index=True,
    # scene=0,
    # time=0,
    # channel=0,
    # zplane=0,
)

print(pt[:])

print(f"Planetable saved to: {savepath}")

# print(f"File: {filepath}")
# print(f"Shape: {aics.shape}")
# print(f"Scenes (all): {aics.scenes}")
# print(f"Dims: {aics.dims}")
# print(f"Dims Order: {aics.dims.order}")
# print(f"Resolution Levels: {aics.resolution_levels}")
