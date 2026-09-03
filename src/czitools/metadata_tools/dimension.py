"""Image dimension metadata for CZI files.

Provides `CziDimensions`, a validated Pydantic dataclass that reads
STCZYX(A) image-dimension sizes from full-resolution CZI subblocks.
"""

import os
from dataclasses import field

from box import Box
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from pylibCZIrw import czi as pyczi

from czitools.utils import logging_tools
from czitools.utils.box import get_czimd_box

logger = logging_tools.set_logging()


def _string_to_float_list(string: str) -> list[float]:
    """Convert a space-separated string of numbers to floats.

    Args:
                string (str): Space-separated numeric values.

    Returns:
                list[float]: Converted values.
    """

    return [float(number) for number in string.split()]


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class CziDimensions:
    """Represent the dimensions of a CZI image.

    Attributes:
        czisource (str | os.PathLike[str] | Box): Source of the CZI image.
        SizeX (int | None): Total size in X, including scenes.
        SizeY (int | None): Total size in Y, including scenes.
        SizeX_scene (int | None): X size of the first stored scene.
        SizeY_scene (int | None): Y size of the first stored scene.
        SizeS (int | None): Number of stored scenes.
        SizeT (int | None): Size in T.
        SizeZ (int | None): Size in Z.
        SizeC (int | None): Size in C.
        SizeM (int | None): Number of mosaic indices.
        SizeR (int | None): Size in R.
        SizeH (int | None): Size in H.
        SizeI (int | None): Size in I.
        SizeV (int | None): Size in V.
        SizeB (int | None): Size in B.
        posZ (list[float] | None): Z positions in microns, when available.
        posT (list[float] | None): T positions in seconds, when available.
        verbose (bool): Flag to enable verbose logging.

    Notes:
        The class contains information about official CZI Dimension Characters:

        - "X": "Width" - width of image [pixel]
        - "Y": "Height" - height of image [pixel]
        - "C": "Channel" - number of channels
        - "Z": "Slice" - number of z-planes
        - "T": "Time" - number of time points
        - "R": "Rotation"
        - "S": "Scene" - contiguous regions of interest in a mosaic image
        - "I": "Illumination" - SPIM direction for LightSheet
        - "B": "Block" - acquisition
        - "M": "Mosaic" - index of tile for compositing a scene
        - "V": "View" - e.g. for SPIM
        - "H": "Phase" - e.g. Airy detector fibers

        In addition, it contains the Z-Positions [microns] and T-Positions [s]
        if they exist. Otherwise, they are set to None.
    """

    czisource: str | os.PathLike[str] | Box
    SizeX: int | None = field(init=False, default=None)  # total size X including scenes
    SizeY: int | None = field(init=False, default=None)  # total size Y including scenes
    SizeX_scene: int | None = field(init=False, default=None)  # size X per scene (if equal scene sizes)
    SizeY_scene: int | None = field(init=False, default=None)  # size Y per scene (if equal scene sizes)
    SizeS: int | None = field(init=False, default=None)
    SizeT: int | None = field(init=False, default=None)
    SizeZ: int | None = field(init=False, default=None)
    SizeC: int | None = field(init=False, default=None)
    SizeM: int | None = field(init=False, default=None)
    SizeR: int | None = field(init=False, default=None)
    SizeH: int | None = field(init=False, default=None)
    SizeI: int | None = field(init=False, default=None)
    SizeV: int | None = field(init=False, default=None)
    SizeB: int | None = field(init=False, default=None)
    posZ: list[float] | None = field(init=False, default=None)
    posT: list[float] | None = field(init=False, default=None)
    verbose: bool = False

    def __post_init__(self) -> None:

        if self.verbose:
            logger.info("Reading Dimensions from CZI image data.")
        self.set_dimensions()

        # set dimensions in XY with respect to possible down scaling
        self.SizeX_sf = self.SizeX
        self.SizeY_sf = self.SizeY

    def set_dimensions(self) -> None:
        """Populate dimensions from full-resolution CZI subblocks.

        Dimension sizes are derived from pylibCZIrw bounding boxes, which are
        calculated from stored subblock headers rather than XML ``Size*`` values.
        Time and Z positions remain optional XML metadata because they are physical
        coordinates rather than index ranges.

        """

        # get the Box and extract the relevant dimension metadata_tools
        if isinstance(self.czisource, Box):
            czi_box = self.czisource
        else:
            czi_box = get_czimd_box(self.czisource)

        dimensions = czi_box.ImageDocument.Metadata.Information.Image

        with pyczi.open_czi(str(czi_box.filepath), czi_box.czi_open_arg) as czidoc:
            bounding_box = czidoc.total_bounding_box_no_pyramid
            for dim in ("X", "Y", "T", "Z", "C", "R", "H", "I", "V", "B"):
                if dim in bounding_box:
                    start, end = bounding_box[dim]
                    setattr(self, f"Size{dim}", end - start)

            scene_rectangles = czidoc.scenes_bounding_rectangle_no_pyramid
            declared_dimensions = dimensions.Dimensions
            has_scene_dimension = (
                declared_dimensions is not None and "S" in declared_dimensions
            ) or dimensions.SizeS is not None
            if has_scene_dimension and scene_rectangles:
                self.SizeS = len(scene_rectangles)
                first_scene = scene_rectangles[min(scene_rectangles)]
                self.SizeX_scene = first_scene.w
                self.SizeY_scene = first_scene.h

            mosaic_indices: set[int] = set()

            def collect_mosaic_index(_index, info):
                if info.is_mindex_valid():
                    mosaic_indices.add(info.mIndex)
                return True

            enumerate_layer0 = getattr(czidoc, "enumerate_subblocks_subset", None)
            if enumerate_layer0 is not None:
                enumerate_layer0(collect_mosaic_index, only_layer0=True)
            if mosaic_indices:
                self.SizeM = max(mosaic_indices) + 1

        if czi_box.has_T:
            # check if there is a list with timepoints (is not in very CZI)
            if dimensions.Dimensions.T.Positions is not None:
                if dimensions.Dimensions.T.Positions.List is not None:
                    try:
                        self.posT = _string_to_float_list(dimensions.Dimensions.T.Positions.List.Offsets)
                    except Exception as e:
                        if self.verbose:
                            logger.error(f"{e}")
                else:
                    if self.verbose:
                        logger.warning("No posT list found under 'dimensions.Dimensions.T.Positions.List'")
            else:
                if self.verbose:
                    logger.warning("No posT list found under 'dimensions.Dimensions.T.Positions'")

        if czi_box.has_Z:
            # check if there is a list with z-positions (is not in very CZI)
            if dimensions.Dimensions.Z.Positions is not None:
                if dimensions.Dimensions.Z.Positions.List is not None:
                    try:
                        self.posZ = _string_to_float_list(dimensions.Dimensions.Z.Positions.List.Offsets)
                    except Exception as e:
                        if self.verbose:
                            logger.error(f"{e}")
                else:
                    if self.verbose:
                        logger.warning("No posZ list found under 'dimensions.Dimensions.Z.Positions.List'")
            else:
                if self.verbose:
                    logger.warning("No posZ list found under 'dimensions.Dimensions.Z.Positions'")
