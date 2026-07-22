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
from pydantic import BaseModel
import os
from collections import Counter
from czitools.utils import logging_tools
from czitools.utils.box import get_czimd_box, box_to_pydantic
from czitools.utils.planetable import get_planetable
from czitools.metadata_tools.dimension import CziDimensions
import traceback

logger = logging_tools.set_logging()


@dataclass
class CziSampleInfo:
    """A class to represent and extract sample information from CZI image data.

    Attributes:
        czisource (Union[str, os.PathLike[str], Box]): The source of the CZI image data.
        well_array_names (List[str]): Names of the well arrays.
        well_indices (List[int]): Indices of the wells.
        well_position_names (List[str]): Position names of the wells.
        well_colID (List[int]): Column IDs of the wells.
        well_rowID (List[int]): Row IDs of the wells.
        well_counter (Dict[str, int]): Counter for the well instances.
        well_scene_indices (Dict[str, List[int]]): Dictionary to store scene indices for each well.
        scene_count (int): Number of parsed scene entries.
        well_unique_number (int): Number of unique, non-empty well names.
        well_total_number (Optional[int]): Deprecated alias for scene count.
        field_centerX (List[Optional[float]]): Scene center X values in micrometers.
            A real metadata value of ``0.0`` remains ``0.0``; missing or malformed
            metadata is represented as ``None`` so the two cases remain distinguishable.
        field_centerY (List[Optional[float]]): Scene center Y values in micrometers.
            It follows the same missing-value behavior as ``field_centerX``.
        well_region_ids (List[Optional[str]]): Source-scoped CZI RegionId values.
        image_stageX (Optional[float]): X coordinate of the image stage.
        image_stageY (Optional[float]): Y coordinate of the image stage.
        multipos_per_well (bool): Flag to indicate multiple positions per well.
        sample_carrier (Optional[BaseModel]): Pydantic model representing the sample carrier information.
        specimen (Optional[BaseModel]): Pydantic model representing the specimen information.
        verbose (bool): Flag to enable verbose logging.
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
    scene_count: int = field(init=False, default=0)
    well_unique_number: int = field(init=False, default=0)
    field_centerX: List[Optional[float]] = field(init=False, default_factory=lambda: [])
    field_centerY: List[Optional[float]] = field(init=False, default_factory=lambda: [])
    well_region_ids: List[Optional[str]] = field(init=False, default_factory=lambda: [])
    image_stageX: Optional[float] = field(init=False, default=None)
    image_stageY: Optional[float] = field(init=False, default=None)
    multipos_per_well: bool = False
    sample_carrier: Optional[BaseModel] = field(init=False, default=None)
    specimen: Optional[BaseModel] = field(init=False, default=None)
    verbose: bool = False

    def __post_init__(self):
        if isinstance(self.czisource, os.PathLike):
            self.czisource = str(self.czisource)

        if self.verbose:
            logger.info("Reading SampleCarrier Information from CZI image data.")

        if isinstance(self.czisource, Box):
            czi_box = self.czisource
        else:
            czi_box = get_czimd_box(self.czisource)

        # try to extract sample holder information
        try:
            self.sample_carrier = box_to_pydantic(
                czi_box.ImageDocument.Metadata.Experiment.ExperimentBlocks.AcquisitionBlock.SubDimensionSetups.RegionsSetup.SampleHolder.Template,
                "SampleCarrierInfo",
            )
        except AttributeError:
            if self.verbose:
                logger.info("CZI metadata do not contain sample carrier information.")

        # try to extract the specimen information
        try:
            # self.specimen = czi_box.ImageDocument.Metadata.Information.Image.Specimen

            self.specimen = box_to_pydantic(czi_box.ImageDocument.Metadata.Information.Image.Specimen, "SpecimenInfo")

        except AttributeError:
            if self.verbose:
                logger.info("CZI metadata do not contain specimen information.")

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

                self._finalize_scene_info()

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

    @property
    def scene_stageX(self) -> List[float]:
        """Deprecated compatibility view of :attr:`field_centerX`.

        Missing values remain represented as ``0.0`` here, matching the legacy
        API. New code should use ``field_centerX``, where missing values are
        represented by ``None``. Consequently, this compatibility view cannot
        distinguish an explicit coordinate of ``0.0`` from unavailable metadata.
        """

        return [0.0 if value is None else value for value in self.field_centerX]

    @property
    def scene_stageY(self) -> List[float]:
        """Deprecated compatibility view of :attr:`field_centerY`.

        As with :attr:`scene_stageX`, both a real zero coordinate and a missing
        coordinate appear as ``0.0``. Use ``field_centerY`` when that distinction
        matters.
        """

        return [0.0 if value is None else value for value in self.field_centerY]

    def _finalize_scene_info(self) -> None:
        """Compute scene/well aggregates after all scene entries are parsed."""

        self.scene_count = len(self.well_array_names)
        # Keep the historical attribute usable while documenting its real meaning.
        self.well_total_number = self.scene_count
        self.well_counter = Counter(name for name in self.well_array_names if name)
        self.well_unique_number = len(self.well_counter)
        self.multipos_per_well = any(count > 1 for count in self.well_counter.values())

    def get_well_info(self, well: Box):
        """Extracts and processes well information from a given Box object.

        Args:
            well (Box): A Box object containing well information.

        The method performs the following actions:

        - Appends the well's ArrayName to `self.well_array_names` and updates `self.well_counter`.
        - Appends the well's Index to `self.well_indices`. If Index is None, logs a message and appends 1.
        - Appends the well's Name to `self.well_position_names`. If Name is None, logs a message and appends "P1".
        - Appends the well's Shape ColumnIndex and RowIndex to `self.well_colID` and `self.well_rowID` respectively. If Shape is None, logs a message and appends 0 to both.
        - Appends the scene's CenterPosition coordinates to `field_centerX/Y`.
          Missing or malformed values are represented as ``None``.
        - Appends the source-scoped CZI RegionId to `well_region_ids`.
        """

        # Primary source: ArrayName. Many CZIs have ArrayName; others don't.
        # We handle missing attributes (AttributeError) and fall back to
        # other fields where possible.
        #
        # Note: we intentionally avoid raising for missing values; callers
        # expect a best-effort population of the dataclass fields.
        # check the ArrayName
        array_name = getattr(well, "ArrayName", None)
        if array_name is not None:
            self.well_array_names.append(str(array_name))
        else:
            try:
                shape_name = well.Shape.Name
                if shape_name is None:
                    raise AttributeError
                self.well_array_names.append(str(shape_name))
            except AttributeError:
                if self.verbose:
                    logger.info("Well Array Names not found.")
                # Preserve one entry per scene so all parallel lists stay aligned.
                self.well_array_names.append("")

        index = getattr(well, "Index", None)
        if index is not None:
            try:
                self.well_indices.append(int(index))
            except (TypeError, ValueError):
                if self.verbose:
                    logger.info("Malformed Well Index; using legacy default 1.")
                self.well_indices.append(1)
        else:
            if self.verbose:
                logger.info("Well Index not found.")
            self.well_indices.append(1)

        name = getattr(well, "Name", None)
        if name is not None:
            self.well_position_names.append(str(name))
        else:
            if self.verbose:
                logger.info("Well Position Names not found.")
            self.well_position_names.append("P1")

        shape = getattr(well, "Shape", None)
        if shape is not None:
            try:
                self.well_colID.append(int(shape.ColumnIndex))
                self.well_rowID.append(int(shape.RowIndex))
            except (AttributeError, TypeError, ValueError):
                if self.verbose:
                    logger.info("Well Column or Row IDs not found.")
                self.well_colID.append(0)
                self.well_rowID.append(0)
        else:
            if self.verbose:
                logger.info("Well Column or Row IDs not found.")
            self.well_colID.append(0)
            self.well_rowID.append(0)

        center_position = getattr(well, "CenterPosition", None)
        if center_position is not None:
            # get the SceneCenter Position stored as a comma-separated
            # pair: "x,y". Guarding the split ensures malformed strings
            # don't raise unhandled exceptions.
            try:
                sx, sy = str(center_position).split(",", maxsplit=1)
                self.field_centerX.append(float(sx))
                self.field_centerY.append(float(sy))
            except (TypeError, ValueError):
                if self.verbose:
                    logger.warning("Malformed CenterPosition value; using None for field center XY.")
                self.field_centerX.append(None)
                self.field_centerY.append(None)
        else:
            if self.verbose:
                logger.info("Scene CenterPosition XY not found.")
            self.field_centerX.append(None)
            self.field_centerY.append(None)

        region_id = getattr(well, "RegionId", None)
        self.well_region_ids.append(None if region_id is None else str(region_id))


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
