from typing import Union, Optional, List
from dataclasses import dataclass, field
from box import Box, BoxList
import os
from czitools.utils import logging_tools
from czitools.utils.box import get_czimd_box

logger = logging_tools.set_logging()


@dataclass
class CziObjectives:
    """Extract objective and tubelens metadata from a CZI file.

    This helper collects objective properties (NA, nominal magnification,
    manufacturer/model, immersion, id) and tubelens magnifications when
    present. All lists are populated during ``__post_init__``. The class
    is defensive about missing fields in the Box metadata and will not
    raise on partially missing metadata; enable ``verbose`` to see
    diagnostic logging.

    Attributes:
        czisource: Path, path-like or pre-parsed ``Box`` containing the
            CZImetadata.
        NA: list of numerical aperture values (float) or empty if not found.
        objmag: list of objective magnifications (float).
        Id: list of objective identifiers (string).
        name: list of objective names or manufacturer/model fallbacks.
        model: list of objective model strings.
        immersion: list of immersion descriptors (string).
        tubelensmag: list of tubelens magnifications (float).
        totalmag: combined magnifications computed as objmag * tubelensmag
            when both lists are present; otherwise contains available mags.
        verbose: emit informational logs when True.
    """

    czisource: Union[str, os.PathLike, Box]
    NA: List[Optional[float]] = field(init=False, default_factory=list)
    objmag: List[Optional[float]] = field(init=False, default_factory=list)
    Id: List[Optional[str]] = field(init=False, default_factory=list)
    name: List[Optional[str]] = field(init=False, default_factory=list)
    model: List[Optional[str]] = field(init=False, default_factory=list)
    immersion: List[Optional[str]] = field(init=False, default_factory=list)
    tubelensmag: List[Optional[float]] = field(init=False, default_factory=list)
    totalmag: List[Optional[float]] = field(init=False, default_factory=list)
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.verbose:
            logger.info("Reading Objective information from CZI image data.")

        czi_box = self.czisource if isinstance(self.czisource, Box) else get_czimd_box(self.czisource)

        # Objectives
        try:
            objectives = czi_box.ImageDocument.Metadata.Information.Instrument.Objectives.Objective
        except Exception:
            objectives = None

        if objectives is None:
            if self.verbose:
                logger.info("No Objective information found.")
        else:
            if isinstance(objectives, Box):
                self.get_objective_info(objectives)
            elif isinstance(objectives, BoxList):
                for obj in objectives:
                    self.get_objective_info(obj)

        # Tube lenses (some files have a single TubeLens entry, others a list)
        try:
            tubelens = czi_box.ImageDocument.Metadata.Information.Instrument.TubeLenses.TubeLens
        except Exception:
            tubelens = None

        if tubelens is None:
            if self.verbose:
                logger.info("No TubeLens information found.")
        else:
            if isinstance(tubelens, Box):
                mag = tubelens.Magnification if hasattr(tubelens, "Magnification") else None
                if mag is None:
                    if self.verbose:
                        logger.warning("No tube lens magnification found; using 1.0x fallback.")
                    self.tubelensmag.append(1.0)
                else:
                    self.tubelensmag.append(float(mag))
            elif isinstance(tubelens, BoxList):
                for tl in tubelens:
                    mag = getattr(tl, "Magnification", None)
                    if mag is not None:
                        self.tubelensmag.append(float(mag))

        # Compute total magnification combinations (Cartesian product)
        if self.objmag and self.tubelensmag:
            self.totalmag = [o * t for o in self.objmag for t in self.tubelensmag]
        elif self.objmag and not self.tubelensmag:
            self.totalmag = list(self.objmag)

    def get_objective_info(self, objective: Box) -> None:
        # Defensive extractors: use getattr to avoid AttributeError on
        # partially present Box objects.
        name = getattr(objective, "Name", None)
        # Only append a readable name when available. If Name is missing,
        # prefer Manufacturer.Model when present. Otherwise omit the entry
        # so the `name` list contains only meaningful strings (this
        # matches historical behavior used by the test-suite).
        if name is not None:
            self.name.append(name)
        else:
            model = getattr(objective.Manufacturer, "Model", None) if hasattr(objective, "Manufacturer") else None
            if model is not None:
                self.name.append(model)

        self.immersion.append(getattr(objective, "Immersion", None))

        lens_na = getattr(objective, "LensNA", None)
        if lens_na is not None:
            try:
                self.NA.append(float(lens_na))
            except Exception:
                if self.verbose:
                    logger.warning("Could not parse LensNA value: %r", lens_na)

        self.Id.append(getattr(objective, "Id", None))

        nominal = getattr(objective, "NominalMagnification", None)
        if nominal is not None:
            try:
                self.objmag.append(float(nominal))
            except Exception:
                if self.verbose:
                    logger.warning("Could not parse NominalMagnification: %r", nominal)

        # If the Name is missing but Manufacturer.Model is present, prefer
        # to use that as a human-readable name.
        if name is None:
            model = getattr(objective.Manufacturer, "Model", None) if hasattr(objective, "Manufacturer") else None
            if model is not None:
                # prefer Manufacturer.Model as a human-readable name when Name is missing
                self.name[-1] = model
