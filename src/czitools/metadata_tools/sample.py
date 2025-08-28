from typing import Union, List, Dict, Tuple
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
    well_scene_indices: Dict[str, int] = field(init=False, default_factory=lambda: {})
    well_total_number: int = field(init=False, default=None)
    scene_stageX: List[float] = field(init=False, default_factory=lambda: [])
    scene_stageY: List[float] = field(init=False, default_factory=lambda: [])
    image_stageX: float = field(init=False, default=None)
    image_stageY: float = field(init=False, default=None)
    multipos_per_well: bool = False
    verbose: bool = False

    def __post_init__(self):
        if self.verbose:
            logger.info("Reading SampleCarrier Information from CZI image data.")

        if isinstance(self.czisource, Box):
            czi_box = self.czisource
        else:
            czi_box = get_czimd_box(self.czisource)

        size_s = CziDimensions(czi_box).SizeS

        if size_s is not None:
            try:
                allscenes = czi_box.ImageDocument.Metadata.Information.Image.Dimensions.S.Scenes.Scene

                # check if there are multiple positions per well
                # self.multipos_per_well = check_multipos_well(allscenes[0])

                if isinstance(allscenes, Box):
                    # check if there are multiple positions per well
                    self.multipos_per_well = check_multipos_well(allscenes)
                    self.get_well_info(allscenes)

                if isinstance(allscenes, BoxList):
                    # check if there are multiple positions per well
                    self.multipos_per_well = check_multipos_well(allscenes[0])
                    for well in range(len(allscenes)):
                        self.get_well_info(allscenes[well])

                # loop over all wells and store the scene indices for each well
                for well_id in self.well_counter.keys():
                    self.well_scene_indices[well_id] = get_scenes_for_well(self, well_id)

            except AttributeError:
                if self.verbose:
                    logger.info("CZI contains no scene metadata_tools.")

        elif size_s is None:
            if self.verbose:
                logger.info("No Scene or Well information found. Try to read XY Stage Coordinates from subblocks.")

            try:
                # read the data from CSV file from a single plane
                planetable, savepath = get_planetable(
                    czi_box.filepath, {"scene": 0, "tile": 0, "time": 0, "channel": 0, "zplane": 0}
                )

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

        # check the ArrayName
        if well.ArrayName is not None:
            self.well_array_names.append(well.ArrayName)
            # count the well instances
            self.well_counter = Counter(self.well_array_names)
            # store the total number of wells
            self.well_total_number = len(self.well_array_names)

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
            # get the SceneCenter Position
            sx = well.CenterPosition.split(",")[0]
            sy = well.CenterPosition.split(",")[1]
            self.scene_stageX.append(np.double(sx))
            self.scene_stageY.append(np.double(sy))
        elif well.CenterPosition is None:
            if self.verbose:
                logger.info("Stage Positions XY not found.")
                self.scene_stageX.append(0.0)
                self.scene_stageY.append(0.0)


def get_scenes_for_well(sample: CziSampleInfo, well_id: str) -> list[int]:
    """
    Returns a list of scene indices for a given well ID.

    Args:
        sample: The CziSampleInfo object containing well information.
        well_id: The ID of the well.

    Returns:
        list[int]: List of scene indices corresponding to the given well ID.
    """

    if sample.multipos_per_well:
        scene_indices = [i for i, x in enumerate(sample.well_array_names) if x == well_id]
    else:
        scene_indices = [i for i, x in enumerate(sample.well_position_names) if x == well_id]

    return scene_indices


def check_multipos_well(scene: Box):

    if scene.ArrayName is None and scene.Name is not None:
        return False
    else:
        return True
