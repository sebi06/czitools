# -*- coding: utf-8 -*-

#################################################################
# File        : _extract_model.py
# Version     : 0.0.1
# Author      : Team Enchilada
# Date        : 28.12.2021
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

import os
import zipfile
import xmltodict
from pathlib import Path
import xml.etree.ElementTree as ET
import json
from typing import List, Dict, Tuple, Optional, Type, Any, Union


def _load_xml_metadata(path: str) -> dict:
    """Get the metadata from an XML file as a dictionary

    Args:
        path (str): Path to the XML file

    Returns:
        dict: Dictionary with the XML data
    """
    data = xmltodict.parse(ET.tostring(ET.parse(path).getroot()))
    return data


def _load_json_metadata(path: str) -> dict:
    """Get the metadata from a JSON file as a dictionary 

    Args:
        path (str): Path to the JSON file

    Returns:
        dict: Dictionary with the JSON data
    """
    with open(path, mode="r") as open_file:
        content = open_file.read()
    return json.loads(content)


def _load_metadata(path: str) -> Dict[str, str]:
    """Get the metadata from a XML or JSON file

    Args:
        path (str): Path to the XML or JSON file

    Returns:
        Dict[str, str]: Dictionary with the data
    """

    # in case of a JSON file
    if path.endswith(".json"):
        print(f"> Loading JSON from '{path}'")
        return _load_json_metadata(path)
    # in case of a XML file
    elif path.endswith(".xml"):
        print(f"> Loading XML as JSON from '{path}'")
        return _load_xml_metadata(path)

    print(f"> Loading failed for '{path}'")
    return {}


def _extract(zip_filename: os.PathLike, targetdir: os.PathLike) -> None:
    """Extract a ZIP file to a target diretcory

    Args:
        zip_filename (os.PathLike): Path of ZIP file
        targetdir (os.PathLike): Target directory
    """

    # extract the model file into a temporary directory
    with zipfile.ZipFile(zip_filename) as z:
        z.extractall(str(targetdir))


def extract_model(path: os.PathLike, target_dir: Path) -> Tuple[Dict[str, str], Path]:
    """Extract the metadata from an CZANN model

    Args:
        path (os.PathLike): Path of the CZANN (or CZMODEL) file
        target_dir (Path): Target directory for the extraction

    Raises:
        file_not_found_in_zip: File with specific extension was not found inside the ZIP file
        file_not_found_in_zip: File with specific extension was not found inside the ZIP file

    Returns:
        Tuple[Dict[str, str], Path]: Tuple containing the the model metadata and the model itself
    """

    # extract the model file
    _extract(path, target_dir)

    # get the metadata file
    metadata_file = _find_metadata_file(target_dir)
    if metadata_file is None:
        raise file_not_found_in_zip(path, "'.json' or '.xml'")

    # get the model file
    model_file = _find_model_file(target_dir)
    if model_file is None:
        raise file_not_found_in_zip(path, "'.model'")

    # get the information of the model as a dictionary
    model_dict = _load_metadata(str(metadata_file))
    return model_dict, model_file


def file_not_found_in_zip(zip_file_path: os.PathLike, extension: str) -> FileNotFoundError:
    """Check if the ZIP file contains the specified file

    Args:
        zip_file_path (os.PathLike): ZIP file path
        extension (str): file extension to check

    Returns:
        FileNotFoundError: error message
    """
    return FileNotFoundError(f"{zip_file_path} does not contain a file with {extension} extension.")


def _find_model_file(target_dir: os.PathLike) -> Optional[os.PathLike]:
    """Find the model file itself

    Args:
        target_dir (os.PathLike): Directory to search

    Returns:
        Optional[os.PathLike]: Model file path
    """
    return next(target_dir.glob('*.model'), None)


def _find_metadata_file(target_dir: os.PathLike) -> Optional[os.PathLike]:
    """Find the metadata file of a model (*.json or *.xml)

    Args:
        target_dir (os.PathLike): Diectory to search

    Returns:
        Optional[os.PathLike]: File path of the model file
    """
    files = target_dir.glob('*')
    metadata_file = next((target_dir / file for file in files if str(file).endswith(".json") or str(file).endswith(".xml")), None)
    return metadata_file
