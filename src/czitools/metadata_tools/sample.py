"""CZI sample information utilities.

This module provides `CziSampleInfo`, a small helper dataclass that
extracts sample-carrier (well/scene) and stage position information
from CZI metadata. It uses the `python-box` (`Box`) representation for
embedded XML metadata and falls back to scanning subblock plane
information (via `get_planetable`) when scene metadata is not present.

The code must handle multiple metadata sources and many missing fields
gracefully; comments explain key fallbacks and edge cases.
"""

from typing import Union, List, Dict, Optional
from dataclasses import dataclass, field
from box import Box, BoxList
import os
import numpy as np
from collections import Counter
from czitools.utils import logging_tools
from czitools.utils.box import get_czimd_box
from czitools.utils.planetable import get_planetable
from czitools.metadata_tools.dimension import CziDimensions
import traceback

logger = logging_tools.set_logging()


@dataclass
class CziSampleInfo:
    """
    A class to represent and extract sample information from CZI image data.
    Attributes:
    -----------
    czisource : Union[str, os.PathLike[str], Box]
        The source of the CZI image data.
    well_array_names : List[str]
        Names of the well arrays.
    well_indices : List[int]
        Indices of the wells.
    well_position_names : List[str]
        Position names of the wells.
    well_colID : List[int]
        Column IDs of the wells.
    well_rowID : List[int]
        Row IDs of the wells.
    well_counter : Dict[str, int]
        Counter for the well instances.
    well_scene_indices : Dict[str, int]
        Dictionary to store scene indices for each well.
    well_total_number : int
        Total number of wells.
    scene_stageX : List[float]
        X coordinates of the scene stages.
    scene_stageY : List[float]
        Y coordinates of the scene stages.
    image_stageX : float
        X coordinate of the image stage.
    image_stageY : float
        Y coordinate of the image stage.
    multipos_per_well : bool
        Flag to indicate multiple positions per well.
    verbose (bool):
        Flag to enable verbose logging.
    Methods:
    --------
    __post_init__():
        Initializes the CziSampleInfo object and reads sample carrier information from CZI image data.
    get_well_info(well: Box):
        Extracts well information from the given well Box object.
    """

    czisource: Union[str, os.PathLike[str], Box]
    well_array_names: List[str] = field(init=False, default_factory=lambda: [])
    well_indices: List[int] = field(init=False, default_factory=lambda: [])
    well_position_names: List[str] = field(init=False, default_factory=lambda: [])
    well_colID: List[int] = field(init=False, default_factory=lambda: [])
    well_rowID: List[int] = field(init=False, default_factory=lambda: [])
    well_counter: Dict[str, int] = field(init=False, default_factory=lambda: {})
    well_scene_indices: Dict[str, List[int]] = field(init=False, default_factory=lambda: {})
    well_total_number: Optional[int] = field(init=False, default=None)
    scene_stageX: List[float] = field(init=False, default_factory=lambda: [])
    scene_stageY: List[float] = field(init=False, default_factory=lambda: [])
    image_stageX: Optional[float] = field(init=False, default=None)
    image_stageY: Optional[float] = field(init=False, default=None)
    multipos_per_well: bool = False
    verbose: bool = False

    def __post_init__(self):
        if self.verbose:
            logger.info("Reading SampleCarrier Information from CZI image data.")

        if isinstance(self.czisource, Box):
            czi_box = self.czisource
        else:
            czi_box = get_czimd_box(self.czisource)

        # Determine whether the CZI contains explicit scene/well information
        # using the pylibCZIrw-backed `CziDimensions` helper. If `SizeS` is
        # set the file contains scenes that may include well metadata.
        size_s = CziDimensions(czi_box).SizeS

        if size_s is not None:
            try:
                allscenes = czi_box.ImageDocument.Metadata.Information.Image.Dimensions.S.Scenes.Scene

                # get well information
                if isinstance(allscenes, Box):
                    self.get_well_info(allscenes)

                if isinstance(allscenes, BoxList):
                    for well in range(len(allscenes)):
                        self.get_well_info(allscenes[well])

                # loop over all wells and store the scene indices for each well
                for well_id in self.well_counter.keys():
                    self.well_scene_indices[well_id] = get_scenes_for_well(self, well_id)

            except AttributeError:
                if self.verbose:
                    logger.info("CZI contains no scene metadata_tools.")

        else:
            # If there is no scene metadata, attempt a fallback: read a
            # single-plane planetable (reads subblock metadata) and extract
            # the XY stage coordinates for the first plane. This is a best-
            # effort fallback for files saved without scene/well XML.

            if self.verbose:
                logger.info("No Scene or Well information found. Try to read XY Stage Coordinates from subblocks.")

            try:
                # read the data from planetable for the first plane; keys
                # come from `planetable.get_planetable()` which returns a
                # dataframe. We guard this call in try/except because the
                # fallback may fail on malformed files.
                planetable, savepath = get_planetable(
                    czi_box.filepath, planes={"scene": 0, "tile": 0, "time": 0, "channel": 0, "zplane": 0}
                )

                # If planetable produced rows, extract the first XY stage
                # coordinates and store as floats. These values remain
                # Optional[float] if the fallback fails.
                self.image_stageX = float(planetable["X[micron]"][0])
                self.image_stageY = float(planetable["Y[micron]"][0])

            except Exception as e:
                if self.verbose:
                    traceback.print_exc()
                    logger.error(e)

    def get_well_info(self, well: Box):
        """
        Extracts and processes well information from a given Box object.
        Parameters:
        well (Box): A Box object containing well information.
        The method performs the following actions:
        - Appends the well's ArrayName to self.well_array_names and updates self.well_counter.
        - Appends the well's Index to self.well_indices. If Index is None, logs a message and appends 1.
        - Appends the well's Name to self.well_position_names. If Name is None, logs a message and appends "P1".
        - Appends the well's Shape ColumnIndex and RowIndex to self.well_colID and self.well_rowID respectively. If Shape is None, logs a message and appends 0 to both.
        - Appends the well's CenterPosition coordinates to self.scene_stageX and self.scene_stageY. If CenterPosition is None, logs a message and appends 0.0 to both.
        """

        # Primary source: ArrayName. Many CZIs have ArrayName; others don't.
        # We handle missing attributes (AttributeError) and fall back to
        # other fields where possible.
        #
        # Note: we intentionally avoid raising for missing values; callers
        # expect a best-effort population of the dataclass fields.
        # check the ArrayName
        if well.ArrayName is not None:
            self.well_array_names.append(well.ArrayName)
            # count the well instances
            self.well_counter = Counter(self.well_array_names)
            # store the total number of wells
            self.well_total_number = len(self.well_array_names)
        elif well.ArrayName is None:
            try:
                id = well.Shape.Name
                self.well_array_names.append(id)
                # count the well instances
                self.well_counter = Counter(self.well_array_names)
                # store the total number of wells
                self.well_total_number = len(self.well_array_names)
            except AttributeError:
                if self.verbose:
                    logger.info("Well Array Names not found.")

        if well.Index is not None:
            self.well_indices.append(int(well.Index))
        elif well.Index is None:
            if self.verbose:
                logger.info("Well Index not found.")
                self.well_indices.append(1)

        if well.Name is not None:
            self.well_position_names.append(well.Name)
        elif well.Name is None:
            if self.verbose:
                logger.info("Well Position Names not found.")
                self.well_position_names.append("P1")

        if well.Shape is not None:
            self.well_colID.append(int(well.Shape.ColumnIndex))
            self.well_rowID.append(int(well.Shape.RowIndex))
        elif well.Shape is None:
            if self.verbose:
                logger.info("Well Column or Row IDs not found.")
                self.well_colID.append(0)
                self.well_rowID.append(0)

        if well.CenterPosition is not None:
            # get the SceneCenter Position stored as a comma-separated
            # pair: "x,y". Guarding the split ensures malformed strings
            # don't raise unhandled exceptions.
            try:
                sx = well.CenterPosition.split(",")[0]
                sy = well.CenterPosition.split(",")[1]
                self.scene_stageX.append(np.double(sx))
                self.scene_stageY.append(np.double(sy))
            except Exception:
                if self.verbose:
                    logger.warning("Malformed CenterPosition value; using 0.0 for scene stage XY.")
                self.scene_stageX.append(0.0)
                self.scene_stageY.append(0.0)
        elif well.CenterPosition is None:
            if self.verbose:
                logger.info("Stage Positions XY not found.")
                self.scene_stageX.append(0.0)
                self.scene_stageY.append(0.0)

        # check if there multiple positions per well
        if Counter(self.well_counter.values())[1] == len(self.well_counter):
            self.multipos_per_well = False
        else:
            self.multipos_per_well = True


def get_scenes_for_well(sample: CziSampleInfo, well_id: str) -> List[int]:
    """
    Returns a list of scene indices for a given well ID.

    Args:
        sample: The CziSampleInfo object containing well information.
        well_id: The ID of the well.

    Returns:
        list[int]: List of scene indices corresponding to the given well ID.
    """

    scene_indices = [i for i, x in enumerate(sample.well_array_names) if x == well_id]

    return scene_indices
