# -*- coding: utf-8 -*-
"""2D array processing utilities for czitools analysis tools.

Provides the :class:`ArrayProcessor` class for filtering, thresholding, and
object counting on 2D NumPy arrays.

Note:
    These features require optional dependencies. Install them with:

    ```
    pip install "czitools[analysis]"
    ```
"""

from typing import Tuple, Optional, Literal
import numpy as np
import pandas as pd
from skimage.filters import threshold_triangle, threshold_otsu, median, gaussian
from skimage.measure import label, regionprops_table
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    disk,
    ball,
    white_tophat,
    black_tophat,
)
from skimage import segmentation
from skimage.color import label2rgb
from skimage.util import invert

from czitools.utils import logging_tools

logger = logging_tools.set_logging()


class ArrayProcessor:
    """Process 2D arrays with filtering, thresholding, and object counting.

    Attributes:
        array (np.ndarray): The 2D NumPy array being processed.
    """

    def __init__(self, array: np.ndarray) -> None:
        if isinstance(array, np.ndarray) and len(array.shape) == 2:
            self.array = array
        else:
            raise TypeError("Input should be a 2D array")

    def apply_gaussian_filter(self, sigma: int) -> np.ndarray:
        """Apply Gaussian filter to the array.

        Args:
            sigma (int): Sigma value for the Gaussian kernel (must be > 1).

        Returns:
            np.ndarray: Filtered array with the same dtype as input.

        Raises:
            ValueError: If sigma is not a valid integer > 1.
        """
        if isinstance(sigma, int) and sigma > 1:
            return gaussian(self.array, sigma=sigma, preserve_range=True, mode="nearest").astype(self.array.dtype)
        raise ValueError("Sigma parameter is invalid.")

    def apply_median_filter(self, filter_size: int) -> np.ndarray:
        """Apply median filter to the array.

        Args:
            filter_size (int): Radius of the disk-shaped footprint.

        Returns:
            np.ndarray: Filtered array with the same dtype as input.

        Raises:
            ValueError: If filter_size is not an integer.
        """
        if isinstance(filter_size, int):
            return median(self.array, footprint=disk(filter_size)).astype(self.array.dtype)
        raise ValueError("Filter Size parameter is invalid.")

    def apply_triangle_threshold(self) -> np.ndarray:
        """Apply triangle threshold to the array.

        Returns:
            np.ndarray: Boolean array (array >= threshold).
        """
        thresh = threshold_triangle(self.array)
        return self.array >= thresh

    def apply_otsu_threshold(self) -> np.ndarray:
        """Apply Otsu threshold to the array.

        Returns:
            np.ndarray: Boolean array (array >= threshold).
        """
        thresh = threshold_otsu(self.array)
        return self.array >= thresh

    def apply_threshold(self, value: int, invert_result: bool = False) -> np.ndarray:
        """Apply a fixed threshold to the array.

        Args:
            value (int): Threshold value (must be >= 0).
            invert_result (bool): If True, invert the thresholded result.

        Returns:
            np.ndarray: Thresholded (and optionally inverted) array.

        Raises:
            ValueError: If threshold parameters are invalid.
        """
        if isinstance(value, int) and value >= 0 and isinstance(invert_result, bool):
            self.array = self.array >= value
            if invert_result:
                self.array = invert(self.array)
            return self.array
        raise ValueError("Threshold parameters are invalid.")

    def label_objects(
        self,
        min_size: int = 10,
        max_size: int = 100_000_000,
        fill_holes: bool = True,
        max_holesize: int = 1,
        label_rgb: bool = True,
        orig_image: Optional[np.ndarray] = None,
        bg_label: int = 0,
        measure_params: bool = False,
        measure_properties: Optional[Tuple[str, ...]] = (
            "label",
            "area",
            "centroid",
            "bbox",
        ),
    ) -> Tuple[np.ndarray, int, Optional[pd.DataFrame]]:
        """Label objects in the thresholded array and optionally measure properties.

        Args:
            min_size (int): Minimum object size in pixels (default: 10).
            max_size (int): Maximum object size in pixels (default: 100 000 000).
            fill_holes (bool): Fill small holes before labelling (default: True).
            max_holesize (int): Maximum hole size to fill (default: 1).
            label_rgb (bool): Generate an RGB-labelled image overlay (default: True).
            orig_image (Optional[np.ndarray]): Original image for RGB overlay (default: None).
            bg_label (int): Background label value (default: 0).
            measure_params (bool): Run regionprops measurement (default: False).
            measure_properties (Optional[Tuple[str, ...]]): Property names for regionprops
                (default: label/area/centroid/bbox).

        Returns:
            Tuple[np.ndarray, int, Optional[pd.DataFrame]]: Tuple of
                (labelled_array, object_count, props_dataframe_or_None).

        Raises:
            ValueError: If parameters are invalid.
        """
        if not (isinstance(min_size, int) and min_size >= 1 and max_holesize >= 1 and isinstance(fill_holes, bool)):
            raise ValueError("Parameters are invalid.")

        if not np.issubdtype(self.array.dtype, bool):
            self.array = remove_small_holes(self.array.astype(bool), max_size=max_holesize, connectivity=1)
        else:
            self.array = remove_small_holes(self.array, max_size=max_holesize, connectivity=1)

        # scikit-image 0.26+ changed API: min_size → max_size with different semantics
        # Old: min_size=N removes objects with area < N
        # New: max_size=N removes objects with area ≤ N
        # To keep objects ≥ min_size, use max_size=min_size-1
        if not np.issubdtype(self.array.dtype, bool):
            self.array = remove_small_objects(self.array.astype(bool), max_size=min_size - 1)
        else:
            self.array = remove_small_objects(self.array, max_size=min_size - 1)

        self.array = segmentation.clear_border(self.array, bgval=bg_label)
        self.array, num_label = label(self.array, background=bg_label, return_num=True, connectivity=2)

        props: Optional[pd.DataFrame] = None
        if measure_params and measure_properties is not None:
            if orig_image is None:
                props = pd.DataFrame(
                    regionprops_table(self.array.astype(np.uint16), properties=measure_properties)
                ).set_index("label")
            else:
                props = pd.DataFrame(
                    regionprops_table(
                        self.array.astype(np.uint16),
                        intensity_image=orig_image,
                        properties=measure_properties,
                    )
                ).set_index("label")
            props = props[(props["area"] >= min_size) & (props["area"] <= max_size)]

        if label_rgb:
            if orig_image is None:
                self.array = label2rgb(self.array, image=None, bg_label=bg_label)
            else:
                self.array = label2rgb(self.array, image=orig_image, bg_label=bg_label)

        return self.array, num_label, props

    @staticmethod
    def subtract_background(
        image: np.ndarray,
        elem: Literal["disk", "ball"],
        radius: int = 50,
        light_bg: bool = False,
    ) -> np.ndarray:
        """Subtract background using morphological top-hat filtering.

        Args:
            image (np.ndarray): 2D grayscale image.
            elem (Literal["disk", "ball"]): Structuring element shape, either "disk" or "ball".
            radius (int): Radius of the structuring element (must be > 0).
            light_bg (bool): If True, use black top-hat (light background); otherwise white top-hat.

        Returns:
            np.ndarray: Background-subtracted image.

        Raises:
            ValueError: If parameters are invalid.
        """
        if not (isinstance(radius, int) and elem in ("disk", "ball") and radius > 0):
            raise ValueError("Parameters are invalid.")

        str_el = disk(radius) if elem == "disk" else ball(radius)
        return black_tophat(image, str_el) if light_bg else white_tophat(image, str_el)
