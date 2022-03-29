# -*- coding: utf-8 -*-

#################################################################
# File        : _tiling.py
# Version     : 0.0.6
# Author      : Team Enchilada
# Date        : 14.01.2022
#
# Disclaimer: This code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

import numpy as np
from cztile.fixed_total_area_strategy import AlmostEqualBorderFixedTotalAreaStrategy2D
from cztile.tiling_strategy import Tile2D
from typing import List, NamedTuple, Union, Tuple, Callable
import dask.array as da
from tqdm import tqdm


def create_tiler(array2d_width: int,
                 array2d_height: int,
                 tile_width: int = 1024,
                 tile_height: int = 1024,
                 min_border_width: int = 128) -> List:
    """Create a tiler object

    Args:
        array2d_width (int): width of 2d array to be tiles
        array2d_height (int): height of the 2d array to be tiled
        tile_width (int, optional): desired tile width. Defaults to 1024.
        tile_height (int, optional): desired tile height. Defaults to 1024.
        min_border_width (int, optional): minimum border width for the tiling. Defaults to 128.

    Returns:
        List: List containing all created tile rectangles
    """

    # create the "tiler"
    tiler = AlmostEqualBorderFixedTotalAreaStrategy2D(total_tile_width=tile_width,
                                                      total_tile_height=tile_height,
                                                      min_border_width=min_border_width)

    # tile a specific scene
    Rectangle = NamedTuple("Rectangle", [("x", int), ("y", int), ("w", int), ("h", int)])

    # create the rectangle to be tiles
    rectangle_to_tile = Rectangle(x=0, y=0, w=array2d_width, h=array2d_height)

    # create the tiles
    tiles = tiler.tile_rectangle(rectangle_to_tile)

    return tiles


def get_tile(array2d: Union[np.ndarray, da.Array],
             tile: Tile2D,
             verbose: bool = False) -> Union[np.ndarray, da.Array]:
    """Get an individual 2D tile from a bigger 2D image

    Args:
        array2d (Union[np.ndarray, da.array]): 2D array from which the tile will be extracted
        tile (Tile2D): Tile properties
        verbose (bool, optional): Option to sho the shape of every created tile. Defaults to False.

    Returns:
        Union[np.ndarray, da.array]: 2D Tile image
    """

    frame = array2d[tile.roi.x:tile.roi.x + tile.roi.w, tile.roi.y:tile.roi.y + tile.roi.h]
    if verbose:
        print("Tile Shape : ", frame.shape)

    return frame


def insert_tile(array2d: Union[np.ndarray, da.Array],
                frame: Union[np.ndarray, da.Array],
                tile: Tile2D) -> Union[np.ndarray, da.Array]:
    """Insert a 2D tile into a bigger 2D image array based on the tile properties

    Args:
        array2d (Union[np.ndarray, da.array]): 2D array to insert the tile into
        frame (Union[np.ndarray, da.array]): 2D tile to be inserted
        tile (Tile2D): Tile properties

    Returns:
        Union[np.ndarray, da.array]: 2D image array with the tile inserted
    """

    array2d[tile.roi.x:tile.roi.x + tile.roi.w, tile.roi.y:tile.roi.y + tile.roi.h] = frame

    return array2d


def process2d_tiles(func2d: Callable,
                    img2d: Union[np.ndarray, da.Array],
                    tile_width: int = 64,
                    tile_height: int = 64,
                    min_border_width: int = 8,
                    use_dask=True,
                    **kwargs: int) -> Union[np.ndarray, da.Array]:
    """Function to process a "larger" 2d image using a 2d processing function
    that will be applied to the "larger" image in a tilewise manner

    :param func2d: 2d processing function that will be applied tilewise
    :type func2d: Callable
    :param img2d: the "larger" 2d image to be processed
    :type img2d: Union[np.ndarray, da.array]
    :param tile_width: tile width in pixel, defaults to 64
    :type tile_width: int, optional
    :param tile_height: tile height in pixel, defaults to 64
    :type tile_height: int, optional
    :param min_border_width: minmum border width in pixel, defaults to 8
    :type min_border_width: int, optional
    :param use_dask: out will be dask array, defaults to True
    :type use_dask: bool
    :return: the processed "larger" 2d image
    :rtype: Union[np.ndarray, da.array]
    """

    if img2d.ndim == 2:

        if use_dask:
            new_img2d = da.zeros_like(img2d, chunks=(img2d.shape[0], img2d.shape[1]))
        if not use_dask:
            new_img2d = np.zeros_like(img2d)

        tiles = create_tiler(array2d_width=img2d.shape[0],
                             array2d_height=img2d.shape[1],
                             tile_width=tile_width,
                             tile_height=tile_height,
                             min_border_width=min_border_width)

        # loop over all tiles
        for tile in tqdm(tiles):

            # get a single frame based on the roi
            tile2d = get_tile(img2d, tile, verbose=False)
            tile2d = func2d(tile2d, **kwargs)
            new_img2d = insert_tile(new_img2d, tile2d, tile)

    if img2d.ndim != 2:
        raise tile_has_wrong_dimensionality(img2d.ndim)

    return new_img2d


def tile_has_wrong_dimensionality(num_dim: int) -> ValueError:
    """Check if the array as exactly 2 dimensions

    :param num_dim: number of dimensions
    :type num_dim: int
    :return: error message
    :rtype: ValueError
    """

    return ValueError(f"{str(num_dim)} does not equal 2.")
