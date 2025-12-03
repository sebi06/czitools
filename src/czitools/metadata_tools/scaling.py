"""Scaling helpers for CZI files.

Provides `CziScaling` which extracts physical scaling (X/Y/Z) from
CZI metadata and computes downsampled values and simple ratios. The
implementation is defensive: missing values fall back to sensible
defaults and ratio computations avoid division-by-zero.
"""

from typing import Union, Optional, Annotated, Dict
from dataclasses import dataclass, field
from box import Box, BoxList
import os
import numpy as np
from czitools.utils import logging_tools
from czitools.utils.box import get_czimd_box
from czitools.metadata_tools.helper import ValueRange

logger = logging_tools.set_logging()


@dataclass
class CziScaling:
    """
    A class to handle scaling information from CZI image data.
    Attributes:
        czisource (Union[str, os.PathLike[str], Box]): The source of the CZI image data.
        X (Optional[float]): The scaling value for the X dimension in microns.
        Y (Optional[float]): The scaling value for the Y dimension in microns.
        Z (Optional[float]): The scaling value for the Z dimension in microns.
        X_sf (Optional[float]): The downscaled scaling value for the X dimension in microns.
        Y_sf (Optional[float]): The downscaled scaling value for the Y dimension in microns.
        ratio (Optional[Dict[str, float]]): The scaling ratios for XY, ZX, and ZX_sf.
        unit (Optional[str]): The unit of measurement for scaling, default is "micron".
        zoom (Annotated[float, ValueRange(0.01, 1.0)]): The zoom factor, default is 1.0.
        verbose (bool): Flag to enable verbose logging.
    Methods:
        __post_init__(): Initializes the scaling values from the CZI image data.
        _safe_get_scale(dist: BoxList, idx: int) -> Optional[float]: Safely retrieves the scaling value for a given dimension.
    """

    czisource: Union[str, os.PathLike[str], Box]
    X: Optional[float] = field(init=False, default=None)
    Y: Optional[float] = field(init=False, default=None)
    Z: Optional[float] = field(init=False, default=None)
    X_sf: Optional[float] = field(init=False, default=None)
    Y_sf: Optional[float] = field(init=False, default=None)
    ratio: Optional[Dict[str, float]] = field(init=False, default=None)
    unit: Optional[str] = field(init=True, default="micron")
    zoom: Annotated[float, ValueRange(0.01, 1.0)] = field(init=True, default=1.0)
    verbose: bool = False

    def __post_init__(self):
        if self.verbose:
            logger.info("Reading Scaling from CZI image data.")

        if isinstance(self.czisource, Box):
            czi_box = self.czisource
        else:
            czi_box = get_czimd_box(self.czisource)

        if getattr(czi_box, "has_scale", False):
            distances = czi_box.ImageDocument.Metadata.Scaling.Items.Distance

            # get the scaling values for X,Y and Z (safe_get returns 1.0 fallback)
            self.X = np.round(self._safe_get_scale(distances, 0, verbose=self.verbose), 3)
            self.Y = np.round(self._safe_get_scale(distances, 1, verbose=self.verbose), 3)
            self.Z = np.round(self._safe_get_scale(distances, 2, verbose=self.verbose), 3)

            # calc the scaling values for X,Y when applying downscaling (guard zoom)
            if self.X is not None and self.zoom:
                self.X_sf = np.round(self.X * (1.0 / float(self.zoom)), 3)
            if self.Y is not None and self.zoom:
                self.Y_sf = np.round(self.Y * (1.0 / float(self.zoom)), 3)

            # calc the scaling ratios only when denom isn't zero or None
            xy = zx = zx_sf = None
            if self.X and self.Y:
                xy = float(np.round(self.X / self.Y, 3))
            if self.X and self.Z:
                zx = float(np.round(self.Z / self.X, 3))
            if self.X_sf and self.Z:
                zx_sf = float(np.round(self.Z / self.X_sf, 3))

            self.ratio = {"xy": xy, "zx": zx, "zx_sf": zx_sf}
        else:
            if self.verbose:
                logger.warning("No scaling information found.")

    @staticmethod
    def _safe_get_scale(dist: BoxList, idx: int, verbose: bool = False) -> Optional[float]:
        scales = ["X", "Y", "Z"]

        try:
            # get the scaling value in [micron]
            sc = float(dist[idx].Value) * 1000000

            # check for the value = 0.0
            if sc == 0.0:
                sc = 1.0
                if verbose:
                    logger.info("Detected Scaling = 0.0 for " + scales[idx] + " Using default = 1.0 [micron].")
            return sc

        except (IndexError, TypeError, AttributeError):
            if verbose:
                logger.info("No " + scales[idx] + "-Scaling found. Using default = 1.0 [micron].")
            return 1.0
