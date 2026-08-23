"""Read individual mosaic tiles from CZI files."""

from typing import Any

import czifile as czifile_module
import numpy as np

from czitools.utils import misc

from ._helpers import CziPath, logger


def _sb_dim_start(de: Any, dim: str, default: int = 0) -> int:
    """Return the start index of *dim* from a czifile directory_entry."""
    return misc._de_dim_start(de, dim, default)


def _sb_dim_shape(de: Any, dim: str, default: int = 0) -> int:
    """Return the shape (size) of *dim* from a czifile directory_entry."""
    return misc._de_dim_size(de, dim, default)


def read_tiles(filepath: CziPath, scene: int, tile: int, **kwargs) -> tuple[np.ndarray, list]:
    """Reads a specific tile from a CZI file.

    Uses czifile to iterate subblocks and assemble the requested tile data.
    Thread-safe and compatible with Napari on all platforms.

    Args:
        filepath (Union[str, os.PathLike[str]]): Path to the CZI file.
        scene (int): The scene index to read from the CZI file.
        tile (int): The tile index to read from the CZI file.
        **kwargs (dict): Additional keyword arguments to specify substacks.
            Valid arguments are: 'T' (Time), 'Z' (Z-dimension), 'C' (Channel).

    Returns:
        Tuple[np.ndarray, List]: A tuple containing:
            - tile_stack (np.ndarray): The image data of the specified tile.
            - size (List): A list of tuples representing the dimensions and their sizes.

    Raises:
        ValueError: If an invalid keyword argument is provided in **kwargs, or if
            the requested scene/tile is not found.
    """
    filepath = str(filepath)

    valid_args = ["T", "Z", "C"]
    for k in kwargs:
        if k not in valid_args:
            raise ValueError(f"Invalid keyword argument: {k}")

    with czifile_module.CziFile(filepath) as czi:
        # Only consider non-pyramidal subblocks (scale 1.0)
        subblocks = [sb for sb in czi.subblocks() if not sb.directory_entry.is_pyramid]

        if not subblocks:
            raise ValueError("No non-pyramidal subblocks found in CZI file")

        # Determine which dimensions exist from the first subblock
        file_dims = set(misc._de_dim_chars(subblocks[0].directory_entry))
        has_h = "H" in file_dims
        has_b = "B" in file_dims
        has_t = "T" in file_dims
        has_c = "C" in file_dims
        has_z = "Z" in file_dims

        # Group subblocks by (scene_index, mosaic_index)
        # scene_index and mosaic_index are -1 when not applicable
        scene_tiles: dict[int, dict[int, list]] = {}
        for sb in subblocks:
            de = sb.directory_entry
            _si = misc._de_scene_idx(de)
            s = _si if _si >= 0 else 0
            _mi = misc._de_mosaic_idx(de)
            m = _mi if _mi >= 0 else 0
            scene_tiles.setdefault(s, {}).setdefault(m, []).append(sb)

        has_multi_scenes = len(scene_tiles) > 1
        req_scene = scene
        if req_scene not in scene_tiles:
            req_scene = 0
        if req_scene not in scene_tiles:
            raise ValueError(f"Scene {scene} not found in CZI file")

        tile_indices = sorted(scene_tiles[req_scene].keys())
        is_mosaic = len(tile_indices) > 1

        logger.info(f"Reading File: {filepath} Scene: {scene} - Tile {tile}")

        if not is_mosaic:
            logger.warning("CZI file is not a mosaic. No M-Dimension found.")

        if tile not in scene_tiles[req_scene]:
            raise ValueError(f"Tile {tile} not found in scene {scene}")

        target_sbs = scene_tiles[req_scene][tile]

        # Build lookup: (t, c, z) -> subblock
        sb_lookup: dict[tuple[int, int, int], Any] = {}
        for sb in target_sbs:
            de = sb.directory_entry
            t_val = _sb_dim_start(de, "T") if has_t else 0
            c_val = _sb_dim_start(de, "C") if has_c else 0
            z_val = _sb_dim_start(de, "Z") if has_z else 0
            sb_lookup[(t_val, c_val, z_val)] = sb

        # Determine value ranges and apply kwargs filters
        all_t = sorted({k[0] for k in sb_lookup}) if has_t else [0]
        all_c = sorted({k[1] for k in sb_lookup}) if has_c else [0]
        all_z = sorted({k[2] for k in sb_lookup}) if has_z else [0]

        out_t = [kwargs["T"]] if "T" in kwargs else all_t
        out_c = [kwargs["C"]] if "C" in kwargs else all_c
        out_z = [kwargs["Z"]] if "Z" in kwargs else all_z

        # Get pixel dimensions and dtype from the actual decoded pixel data of
        # the first available subblock.  de.stored_shape holds the stored tile
        # size; we use sb.data() to get the actual decoded array shape.
        first_key = (out_t[0], out_c[0], out_z[0])
        if first_key not in sb_lookup:
            first_key = next(iter(sb_lookup))
        first_sb = sb_lookup[first_key]
        first_data = first_sb.data()
        sample_de = first_sb.directory_entry
        rdl = list(sample_de.dims)
        size_y = first_data.shape[rdl.index("Y")] if "Y" in rdl else first_data.shape[-2]
        size_x = first_data.shape[rdl.index("X")] if "X" in rdl else first_data.shape[-1]
        dtype = first_data.dtype

        # Build output dimension ordering following aicspylibczi convention:
        # Include H/B (if present), S (if multi-scene), then T/C/Z only if
        # they exist in the file's dimension entries, and always Y, X.
        out_dims: list[tuple[str, int]] = []
        if has_h:
            out_dims.append(("H", 1))
        elif has_b:
            out_dims.append(("B", 1))
        if has_multi_scenes:
            out_dims.append(("S", 1))
        if has_t:
            out_dims.append(("T", len(out_t)))
        out_dims.append(("C", len(out_c)))
        if has_z:
            out_dims.append(("Z", len(out_z)))
        out_dims.append(("Y", size_y))
        out_dims.append(("X", size_x))

        out_shape = tuple(s for _, s in out_dims)
        size_list = list(out_dims)
        dim_names = [d[0] for d in out_dims]

        # Allocate and fill output array
        tile_stack = np.zeros(out_shape, dtype=dtype)

        for ti, t_val in enumerate(out_t):
            for ci, c_val in enumerate(out_c):
                for zi, z_val in enumerate(out_z):
                    key = (t_val, c_val, z_val)
                    if key not in sb_lookup:
                        continue

                    sb = sb_lookup[key]
                    pixel_data = sb.data()

                    # Extract the 2D (Y, X) plane from the subblock pixel data.
                    # de.dims gives the physical storage dimension order
                    # (e.g. ('C','Y','X','S')) and pixel_data shape matches it.
                    de = sb.directory_entry
                    de_dims_list = list(de.dims)
                    slicer: list[Any] = [0] * len(de_dims_list)
                    slicer[de_dims_list.index("Y")] = slice(None)
                    slicer[de_dims_list.index("X")] = slice(None)
                    plane = pixel_data[tuple(slicer)]

                    # Build the index into the output array
                    idx: list = []
                    for dim_name in dim_names:
                        if dim_name in ("H", "B", "S"):
                            idx.append(0)
                        elif dim_name == "T":
                            idx.append(ti)
                        elif dim_name == "C":
                            idx.append(ci)
                        elif dim_name == "Z":
                            idx.append(zi)
                        elif dim_name in ("Y", "X"):
                            idx.append(slice(None))

                    tile_stack[tuple(idx)] = plane

    return tile_stack, size_list
