from typing import Union, Optional, List
from dataclasses import dataclass, field
from box import Box, BoxList
import os
from czitools.utils import logging_tools
from czitools.utils.box import get_czimd_box

logger = logging_tools.set_logging()


@dataclass
class CziObjectives:
    czisource: Union[str, os.PathLike[str], Box]
    NA: List[Optional[float]] = field(init=False, default_factory=lambda: [])
    objmag: List[Optional[float]] = field(init=False, default_factory=lambda: [])
    Id: List[Optional[str]] = field(init=False, default_factory=lambda: [])
    name: List[Optional[str]] = field(init=False, default_factory=lambda: [])
    model: List[Optional[str]] = field(init=False, default_factory=lambda: [])
    immersion: List[Optional[str]] = field(init=False, default_factory=lambda: [])
    tubelensmag: List[Optional[float]] = field(init=False, default_factory=lambda: [])
    totalmag: List[Optional[float]] = field(init=False, default_factory=lambda: [])
    verbose: bool = False
    
    def __post_init__(self):
        if self.verbose:
            logger.info("Reading Objective Information from CZI image data.")  

        if isinstance(self.czisource, Box):
            czi_box = self.czisource
        else:
            czi_box = get_czimd_box(self.czisource)

        # check if objective metadata_tools actually exist
        if czi_box.has_objectives:
            try:
                # get objective data
                objective = (
                    czi_box.ImageDocument.Metadata.Information.Instrument.Objectives.Objective
                )
                if isinstance(objective, Box):
                    self.get_objective_info(objective)
                elif isinstance(objective, BoxList):
                    for obj in range(len(objective)):
                        self.get_objective_info(objective[obj])
            except AttributeError:
                objective = None

        elif not czi_box.has_objectives:
            if self.verbose:
                logger.info("No Objective Information found.")

        # check if tubelens metadata_tools exist
        if czi_box.has_tubelenses:
            # get tubelenes data
            tubelens = (
                czi_box.ImageDocument.Metadata.Information.Instrument.TubeLenses.TubeLens
            )

            if isinstance(tubelens, Box):
                if tubelens.Magnification is not None:
                    self.tubelensmag.append(float(tubelens.Magnification))
                elif tubelens.Magnification is None:
                    if self.verbose:
                        logger.warning("No tubelens magnification found. Use 1.0x instead.")
                    self.tubelensmag.append(1.0)

            elif isinstance(tubelens, BoxList):
                for tl in range(len(tubelens)):
                    self.tubelensmag.append(float(tubelens[tl].Magnification))

            # some additional checks to calc the total magnification
            if self.objmag is not None and self.tubelensmag is not None:
                self.totalmag = [i * j for i in self.objmag for j in self.tubelensmag]

        elif not czi_box.has_tubelens:
            if self.verbose:
                logger.info("No Tublens Information found.")

        if self.objmag is not None and self.tubelensmag == []:
            self.totalmag = self.objmag

    def get_objective_info(self, objective: Box):
        self.name.append(objective.Name)
        self.immersion.append(objective.Immersion)

        if objective.LensNA is not None:
            self.NA.append(float(objective.LensNA))

        if objective.Id is not None:
            self.Id.append(objective.Id)

        if objective.NominalMagnification is not None:
            self.objmag.append(float(objective.NominalMagnification))

        if None in self.name and self.name.count(None) == 1:
            self.name.remove(None)
            self.name.append(objective.Manufacturer.Model)


