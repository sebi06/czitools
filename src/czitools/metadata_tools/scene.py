"""Scene helper utilities for CZI files.

This module exposes `CziScene`, a small dataclass that reads a scene's
bounding box and coordinates from a CZI file using the `pylibCZIrw`
reader. The implementation keeps the runtime dependency usage minimal
inside `__post_init__` and avoids modifying caller-provided types by
using `os.fspath` for path-like conversion.
"""

from typing import Optional, Union
from dataclasses import dataclass, field
import os
from czitools.utils import logging_tools
from pylibCZIrw import czi as pyczi

logger = logging_tools.set_logging()


@dataclass
class CziScene:
    """Represent a single SizeS scene inside a CZI file.

    Reads the scene bounding box and coordinate information from a CZI file
    using `pylibCZIrw`. Fields are populated during object initialization
    (`__post_init__`). If the scene cannot be read (invalid file or
    index), the attributes remain ``None`` and no exception is raised;
    set ``verbose=True`` to see detailed logs.

    Args:
        filepath (Union[str, os.PathLike]): Path to the CZI file.
        index (int): Zero-based scene index (SizeS index).
        verbose (bool): If True, emit informational and exception logs.

    Attributes:
        bbox (Optional[pyczi.Rectangle]): Scene bounding rectangle if found.
        xstart (Optional[int]): Left coordinate of the scene (pixels).
        ystart (Optional[int]): Top coordinate of the scene (pixels).
        width (Optional[int]): Scene width (pixels).
        height (Optional[int]): Scene height (pixels).

    Notes:
        - ``filepath`` is accepted as a path-like object; ``os.fspath`` is
          used internally so the original object is not mutated.
        - The class uses pylibCZIrw's context manager to ensure file
          handles are closed.
    """

    filepath: Union[str, os.PathLike]
    index: int
    bbox: Optional[pyczi.Rectangle] = field(init=False, default=None)
    xstart: Optional[int] = field(init=False, default=None)
    ystart: Optional[int] = field(init=False, default=None)
    width: Optional[int] = field(init=False, default=None)
    height: Optional[int] = field(init=False, default=None)
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.verbose:
            logger.info("Reading Scene information from CZI image data.")

        # Accept Path-like objects without mutating the original object
        # passed by the caller; `os.fspath` supports str and Path-like.
        czi_path = os.fspath(self.filepath)

        # Use pylibCZIrw context manager which properly closes resources.
        try:
            with pyczi.open_czi(czi_path) as czidoc:
                try:
                    self.bbox = czidoc.scenes_bounding_rectangle[self.index]
                    self.xstart = self.bbox.x
                    self.ystart = self.bbox.y
                    self.width = self.bbox.w
                    self.height = self.bbox.h
                except (KeyError, IndexError) as exc:
                    if self.verbose:
                        logger.info(f"Scene index {self.index} not found: {exc}")
        except Exception:
            # Don't crash on malformed files; caller can inspect attributes
            # to see whether the scene was populated. Log at exception
            # level only when verbose flag is set so normal runs stay quiet.
            if self.verbose:
                logger.exception("Failed to read CZI scene information.")
