"""Utilities for loading CZI metadata as a `python-box` `Box` object.

Provides `get_czimd_box`, which opens a CZI file (local path or URL) and
returns its XML metadata as an attribute-accessible `Box` together with a set
of convenience boolean flags (e.g. `has_channels`, `has_scenes`) that indicate
which metadata sections are present.
"""

from __future__ import annotations
from typing import Any, List, Union
import os
from pylibCZIrw import czi as pyczi
from box import Box
import validators
from pydantic import BaseModel, create_model, Field


def get_czimd_box(filepath: Union[str, os.PathLike[str]]) -> Box:
    """
    get_czimd_box: Get CZI metadata_tools as a python-box. For details: https://pypi.org/project/python-box/

    Args:
        filepath (Union[str, os.PathLike[str]]): Filepath of the CZI file

    Returns:
        Box: CZI metadata_tools as a Box object
    """

    readertype = pyczi.ReaderFileInputTypes.Standard

    if validators.url(str(filepath)):
        readertype = pyczi.ReaderFileInputTypes.Curl

    # get metadata_tools dictionary using pylibCZIrw
    with pyczi.open_czi(str(filepath), readertype) as czi_document:
        metadata_dict = czi_document.metadata
        # total_bounding_box_no_pyramid = czi_document.total_bounding_box_no_pyramid
        scenes_bounding_rectangle = czi_document.scenes_bounding_rectangle

    czimd_box = Box(
        metadata_dict,
        conversion_box=True,
        default_box=True,
        default_box_attr=None,
        default_box_create_on_get=True,
        # default_box_no_key_error=True
    )

    # add the filepath
    czimd_box.filepath = filepath
    czimd_box.is_url = validators.url(str(filepath))
    czimd_box.czi_open_arg = readertype

    # set the defaults to False
    czimd_box.has_customattr = False
    czimd_box.has_experiment = False
    czimd_box.has_disp = False
    czimd_box.has_hardware = False
    czimd_box.has_scale = False
    czimd_box.has_instrument = False
    czimd_box.has_microscopes = False
    czimd_box.has_detectors = False
    czimd_box.has_objectives = False
    czimd_box.has_tubelenses = False
    czimd_box.has_disp = False
    czimd_box.has_channels = False
    czimd_box.has_info = False
    czimd_box.has_app = False
    czimd_box.has_doc = False
    czimd_box.has_image = False
    czimd_box.has_scenes = False
    czimd_box.has_T = False
    czimd_box.has_Z = False
    czimd_box.has_dims = False
    czimd_box.has_layers = False

    if "Experiment" in czimd_box.ImageDocument.Metadata:
        czimd_box.has_experiment = True

    if "HardwareSetting" in czimd_box.ImageDocument.Metadata:
        czimd_box.has_hardware = True

    if "CustomAttributes" in czimd_box.ImageDocument.Metadata:
        czimd_box.has_customattr = True

    if "Information" in czimd_box.ImageDocument.Metadata:
        czimd_box.has_info = True

        if "Application" in czimd_box.ImageDocument.Metadata.Information:
            czimd_box.has_app = True

        if "Document" in czimd_box.ImageDocument.Metadata.Information:
            czimd_box.has_doc = True

        if "Image" in czimd_box.ImageDocument.Metadata.Information:
            czimd_box.has_image = True

            if "Dimensions" in czimd_box.ImageDocument.Metadata.Information.Image:
                czimd_box.has_dims = True

                if "Channels" in czimd_box.ImageDocument.Metadata.Information.Image.Dimensions:
                    # if "C" in total_bounding_box_no_pyramid.keys():
                    czimd_box.has_channels = True

                if "T" in czimd_box.ImageDocument.Metadata.Information.Image.Dimensions:
                    # if "T" in total_bounding_box_no_pyramid.keys():
                    czimd_box.has_T = True

                if "Z" in czimd_box.ImageDocument.Metadata.Information.Image.Dimensions:
                    # if "Z" in total_bounding_box_no_pyramid.keys():
                    czimd_box.has_Z = True

                # if "S" in czimd_box.ImageDocument.Metadata.Information.Image.Dimensions:
                if len(scenes_bounding_rectangle) > 0:
                    czimd_box.has_scenes = True

        if "Instrument" in czimd_box.ImageDocument.Metadata.Information:
            czimd_box.has_instrument = True

            if "Detectors" in czimd_box.ImageDocument.Metadata.Information.Instrument:
                czimd_box.has_detectors = True

            if "Microscopes" in czimd_box.ImageDocument.Metadata.Information.Instrument:
                czimd_box.has_microscopes = True

            if "Objectives" in czimd_box.ImageDocument.Metadata.Information.Instrument:
                czimd_box.has_objectives = True

            if "TubeLenses" in czimd_box.ImageDocument.Metadata.Information.Instrument:
                czimd_box.has_tubelenses = True

    if "Scaling" in czimd_box.ImageDocument.Metadata:
        czimd_box.has_scale = True

    if "DisplaySetting" in czimd_box.ImageDocument.Metadata:
        czimd_box.has_disp = True

    if "Layers" in czimd_box.ImageDocument.Metadata:
        czimd_box.has_layers = True

    return czimd_box


def box_to_pydantic(box_obj: Box, model_name: str = "DynamicModel") -> BaseModel:
    """
    Convert Box to Pydantic model (better than dataclass for validation).

    Args:
        box_obj: Box object
        model_name: Name for Pydantic model

    Returns:
        Pydantic model instance
    """
    field_definitions = {}

    for key, value in box_obj.items():
        # Infer type and set as field
        if isinstance(value, Box):
            # Nested Box -> recursive conversion
            nested_model = box_to_pydantic(value, f"{model_name}{key.capitalize()}")
            field_definitions[key] = (type(nested_model), Field(default=nested_model))
        elif isinstance(value, list):
            if value and isinstance(value[0], Box):
                item_model = box_to_pydantic(value[0], f"{model_name}{key.capitalize()}Item")
                nested_items = [box_to_pydantic(item, f"{model_name}{key.capitalize()}Item") for item in value]
                field_definitions[key] = (List[type(item_model)], Field(default=nested_items))
            else:
                field_definitions[key] = (type(value), Field(default=value))
        else:
            field_type = type(value) if value is not None else Any
            field_definitions[key] = (field_type, Field(default=value))

    # Create Pydantic model dynamically
    DynamicModel = create_model(model_name, **field_definitions)

    return DynamicModel()
