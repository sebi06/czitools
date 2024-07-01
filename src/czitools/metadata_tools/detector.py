from typing import Union, List
from dataclasses import dataclass, field
from box import Box, BoxList
import os
from czitools.utils import logging_tools
from czitools.utils.box import get_czimd_box

logger = logging_tools.set_logging()


@dataclass
class CziDetector:
    czisource: Union[str, os.PathLike[str], Box]
    model: List[str] = field(init=False, default_factory=lambda: [])
    name: List[str] = field(init=False, default_factory=lambda: [])
    Id: List[str] = field(init=False, default_factory=lambda: [])
    modeltype: List[str] = field(init=False, default_factory=lambda: [])
    gain: List[float] = field(init=False, default_factory=lambda: [])
    zoom: List[float] = field(init=False, default_factory=lambda: [])
    amplificationgain: List[float] = field(init=False, default_factory=lambda: [])

    def __post_init__(self):
        logger.info("Reading Detector Information from CZI image data.")

        if isinstance(self.czisource, Box):
            czi_box = self.czisource
        else:
            czi_box = get_czimd_box(self.czisource)

        # check if there are any detector entries inside the dictionary
        if czi_box.ImageDocument.Metadata.Information.Instrument is not None:
            # get the data for the detectors
            detectors = (
                czi_box.ImageDocument.Metadata.Information.Instrument.Detectors.Detector
            )

            # check for detector Id, Name, Model and Type
            if isinstance(detectors, Box):
                self.Id.append(detectors.Id)
                self.name.append(detectors.Name)
                self.model.append(detectors.Model)
                self.modeltype.append(detectors.Type)
                self.gain.append(detectors.Gain)
                self.zoom.append(detectors.Zoom)
                self.amplificationgain.append(detectors.AmplificationGain)

            # and do this differently in case of a list of detectors
            elif isinstance(detectors, BoxList):
                for d in range(len(detectors)):
                    self.Id.append(detectors[d].Id)
                    self.name.append(detectors[d].Name)
                    self.model.append(detectors[d].Model)
                    self.modeltype.append(detectors[d].Type)
                    self.gain.append(detectors[d].Gain)
                    self.zoom.append(detectors[d].Zoom)
                    self.amplificationgain.append(detectors[d].AmplificationGain)

        elif czi_box.ImageDocument.Metadata.Information.Instrument is None:
            logger.info("No Detetctor(s) information found.")
            self.model = [None]
            self.name = [None]
            self.Id = [None]
            self.modeltype = [None]
            self.gain = [None]
            self.zoom = [None]
            self.amplificationgain = [None]
