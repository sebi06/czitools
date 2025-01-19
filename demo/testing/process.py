pip# -*- coding: utf-8 -*-

#################################################################
# File        : process.py
# Author      : sebi06
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

import dask.array as da
from skimage.measure import label
import numpy as np
import itertools
import inspect
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import List, NamedTuple, Union, Tuple, Callable
from czitools.utils.misc import measure_execution_time, measure_memory_usage


def process_nd(func):

    def wrapper(md_array: np.ndarray, *args, **kwargs):

        # print("Before function call")
        arg_names = inspect.getfullargspec(func).args
        print(f"Arguments: {arg_names}")

        shape_woXY = md_array.shape[:-2]

        # processed_md = da.zeros_like(md_array, chunks=md_array.shape)
        processed_md = np.zeros_like(md_array)

        # create the "values" each for-loop iterates over
        loopover = [range(s) for s in shape_woXY]
        prod = itertools.product(*loopover)

        # loop over all dimensions
        for idx in prod:

            # create list of slice objects based on the shape
            sl = len(shape_woXY) * [np.s_[0:1]]

            # insert the correct index into the respective slice objects for all dimensions
            for nd in range(len(shape_woXY)):
                sl[nd] = idx[nd]

            # extract the 2D image from the n-dims stack using the list of slice objects
            array2d = np.squeeze(md_array[tuple(sl)])

            # process the whole 2d image - make sure to use the correct **kwargs

            # insert new 2D after tile-wise processing into nd array
            processed_md[tuple(sl)] = func(array2d, *args, **kwargs)

        return processed_md

    return wrapper


def process_array_nd(func):
    """Processes an n-dimensional NumPy array by applying the given function to each 2D slice.

    Args:
        func: The function to apply to each 2D slice.

    Returns:
        A function that takes an n-dimensional NumPy array and returns the processed array.
    """

    def wrapper(array: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Applies the function to each 2D slice of the n-dimensional array.

        Args:
            array: The n-dimensional NumPy array to process.
            *args: Additional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            The processed n-dimensional NumPy array.
        """

        # Determine the last two dimensions to apply the filter
        last_two_dims = array.shape[-2:]

        # Reshape the array to a 2D array of shape (N, -1)
        reshaped_array = array.reshape(-1, last_two_dims[0], last_two_dims[1])

        # Apply the function to each 2D slice
        processed_array = np.apply_along_axis(func, 1, reshaped_array, *args, **kwargs)

        # Reshape the processed array back to the original shape
        return processed_array.reshape(array.shape)

    return wrapper


# @measure_memory_usage
@measure_execution_time
@process_nd
def apply_filter(array: np.ndarray, sigma: float) -> np.ndarray:
    """Applies a 2D Gaussian filter to a 2D NumPy array.

    Args:
      array: The 2D NumPy array to filter.
      sigma: The standard deviation of the Gaussian kernel.

    Returns:
      The filtered 2D NumPy array.
    """

    # return gaussian_filter(array, sigma=sigma)
    return array**2


# @measure_memory_usage
@measure_execution_time
@process_array_nd
def apply_gauss(array: np.ndarray, sigma: float) -> np.ndarray:
    """Applies a 2D Gaussian filter to a 2D NumPy array.

    Args:
      array: The 2D NumPy array to filter.
      sigma: The standard deviation of the Gaussian kernel.

    Returns:
      The filtered 2D NumPy array.
    """

    # return gaussian_filter(array, sigma=sigma)
    return array**2


# Example usage:
array = np.random.randint(0, high=100, size=(1, 3, 5, 2000, 1000), dtype=int)

# Apply function to array
filtered_array1 = apply_filter(array, sigma=2)
filtered_array2 = apply_gauss(array, sigma=2)

# print(f"Equal: {np.array_equal(filtered_array1, filtered_array2)}")
