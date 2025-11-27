from typing import Union, List, Optional
import os
from dataclasses import dataclass, field
from box import Box, BoxList
from czitools.utils import logging_tools
from czitools.utils.box import get_czimd_box

logger = logging_tools.set_logging()


@dataclass
class CziDetector:
    """Extract detector information from a CZI file.

    The class populates parallel lists for detector attributes. It is
    defensive about missing metadata: if the Instrument block is absent
    it will populate lists with a single ``None`` element to match the
    historical shape expected by callers/tests.
    """

    czisource: Union[str, os.PathLike, Box]
    model: List[Optional[str]] = field(init=False, default_factory=list)
    name: List[Optional[str]] = field(init=False, default_factory=list)
    Id: List[Optional[str]] = field(init=False, default_factory=list)
    modeltype: List[Optional[str]] = field(init=False, default_factory=list)
    gain: List[Optional[float]] = field(init=False, default_factory=list)
    zoom: List[Optional[float]] = field(init=False, default_factory=list)
    amplificationgain: List[Optional[float]] = field(init=False, default_factory=list)
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.verbose:
            logger.info("Reading Detector information from CZI image data.")

        czi_box = self.czisource if isinstance(self.czisource, Box) else get_czimd_box(self.czisource)

        # Guard existence of Instrument block
        instr = getattr(czi_box.ImageDocument.Metadata.Information, "Instrument", None)
        if instr is None:
            if self.verbose:
                logger.info("No Detector(s) information found.")
            # Keep historical single-None placeholders
            self.model = [None]
            self.name = [None]
            self.Id = [None]
            self.modeltype = [None]
            self.gain = [None]
            self.zoom = [None]
            self.amplificationgain = [None]
            return

        # Extract detectors block
        detectors = getattr(instr, "Detectors", None)
        det = getattr(detectors, "Detector", None) if detectors is not None else None

        if det is None:
            if self.verbose:
                logger.info("No Detector entries found under Instrument.Detectors.")
            return

        # Single detector
        if isinstance(det, Box):
            self.Id.append(getattr(det, "Id", None))
            self.name.append(getattr(det, "Name", None))
            self.model.append(getattr(det, "Model", None))
            self.modeltype.append(getattr(det, "Type", None))
            self.gain.append(getattr(det, "Gain", None))
            self.zoom.append(getattr(det, "Zoom", None))
            self.amplificationgain.append(getattr(det, "AmplificationGain", None))

        # Multiple detectors
        elif isinstance(det, BoxList):
            for d in det:
                self.Id.append(getattr(d, "Id", None))
                self.name.append(getattr(d, "Name", None))
                self.model.append(getattr(d, "Model", None))
                self.modeltype.append(getattr(d, "Type", None))
                self.gain.append(getattr(d, "Gain", None))
                self.zoom.append(getattr(d, "Zoom", None))
                self.amplificationgain.append(getattr(d, "AmplificationGain", None))
